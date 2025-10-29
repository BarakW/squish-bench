from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic_settings import BaseSettings, CliApp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import logging
from typing import cast

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from pydantic import Field, model_validator
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, Qwen3ForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.tokenization_utils_base import BatchEncoding

import wandb
from data import format_context_for_training, get_summary_from_generation
from datasets import DatasetDict, load_from_disk
from evaluation import SummaryBudget, evaluate_summaries, get_model_log_probs, total_cross_entropy
from utils import get_attn_mask_from_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# TORCH CONFIG
torch.set_float32_matmul_precision("medium")
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.capture_scalar_outputs = True

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
elif torch.mps.is_available():
    DEVICE = "mps"


class TrainingConfig(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    base_model_name: str = "Qwen/Qwen3-1.7B-Base"
    trained_model_dir: str
    max_document_length: int = 4096
    num_epochs: int = 1
    batch_full_size: int = 32
    batch_micro_size: int = 1
    group_full_size: int = 32
    group_micro_size: int = 1
    group_gen_size: int = 16
    policy_steps_per_group: int = 4
    learning_rate: float = 1e-5
    epsilon_low: float = 0.2
    epsilon_high: float = 0.3

    logging: bool = Field(default=False, exclude=True)

    @model_validator(mode="after")
    def validate_batch_and_group_sizes(self) -> "TrainingConfig":
        if self.batch_full_size % self.batch_micro_size != 0:
            raise ValueError("batch_full_size must be divisible by batch_micro_size")
        if self.group_full_size % self.group_micro_size != 0:
            raise ValueError("group_full_size must be divisible by group_micro_size")
        return self

    @property
    def batch_accum_steps(self) -> int:
        return self.batch_full_size // self.batch_micro_size

    @property
    def group_accum_steps(self) -> int:
        return self.group_full_size // self.group_micro_size

    @property
    def gen_steps(self) -> int:
        return self.group_full_size // self.group_gen_size


@dataclass
class Rollouts:
    tokens_BGT: Tensor  # Int32
    generation_mask_BGT: Tensor  # Int8
    old_policy_log_probs_BGT: Tensor  # Float
    summary_budgets_B: Tensor  # Int32
    # TODO: add advantages as a field for logical consolidation

    def get_group_slice(self, group_idx: int, group_size: int) -> Rollouts:
        return Rollouts(
            tokens_BGT=self.tokens_BGT[:, group_idx : group_idx + group_size, :],
            generation_mask_BGT=self.generation_mask_BGT[:, group_idx : group_idx + group_size, :],
            old_policy_log_probs_BGT=self.old_policy_log_probs_BGT[:, group_idx : group_idx + group_size, :],
            summary_budgets_B=self.summary_budgets_B,
        )


def setup_model_and_tokenizer(config: TrainingConfig) -> tuple[PreTrainedTokenizer, Qwen3ForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    base_model.gradient_checkpointing_enable({"use_reentrant": False})

    if os.path.exists(f"models/{config.trained_model_dir}"):
        logger.info(f"Loading trained model from {config.trained_model_dir}")
        model = PeftModel.from_pretrained(base_model, f"models/{config.trained_model_dir}", is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base_model, peft_config)

    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.forward = torch.compile(model.forward, dynamic=True)
    return tokenizer, model  # type: ignore[reportReturnType]


def get_advantage(rewards: Tensor) -> Tensor:
    mean_reward = rewards.mean()
    return rewards - mean_reward


def total_cross_entropy_from_base_model(
    model: Qwen3ForCausalLM,
    tokenizer: PreTrainedTokenizer,
    input_ids_BT: Tensor,
    /,
    micro_batch_size: int,
    mask_to_zero_BT: Tensor | None = None,
) -> Tensor:
    """Disable LORA while calculating total information as we want to use the base model"""
    with model.disable_adapter():  # type: ignore[reportCallIssue]
        res = total_cross_entropy(model, tokenizer, input_ids_BT, micro_batch_size=micro_batch_size, mask_to_zero_BT=mask_to_zero_BT)
    return res


class StopOnSubsequence(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizer, stop_strings: list[str], device: torch.device):
        self.eos_token_id = cast(int, tokenizer.eos_token_id)
        self.stop_sequences: list[Tensor] = []
        for s in stop_strings:
            ids = tokenizer.encode(s, add_special_tokens=False) if s != tokenizer.eos_token else [tokenizer.eos_token_id]
            self.stop_sequences.append(torch.tensor(ids, device=device))

    def __call__(self, input_ids_BT: Tensor, scores_BTD: Tensor) -> Tensor:
        if not self.stop_sequences:
            return scores_BTD

        for batch_idx in range(input_ids_BT.size(0)):
            sequence_i = input_ids_BT[batch_idx]
            for stop_sequence in self.stop_sequences:
                if sequence_i.size(0) < stop_sequence.size(0):
                    continue
                if torch.equal(sequence_i[-stop_sequence.size(0) :], stop_sequence):
                    scores_BTD[batch_idx].fill_(float("-inf"))
                    scores_BTD[batch_idx, self.eos_token_id] = 0.0
                    break
        return scores_BTD


def generate_outputs(model: Qwen3ForCausalLM, tokenizer: PreTrainedTokenizer, inputs_BT: BatchEncoding, config: TrainingConfig) -> Tensor:
    logits_processor = LogitsProcessorList(
        [StopOnSubsequence(tokenizer, stop_strings=["</summary>", tokenizer.eos_token], device=model.device)]
    )
    generations_uBuGT: list[Tensor] = []
    with torch.inference_mode():
        for _ in range(config.gen_steps):
            batch_completions_uBT = cast(
                Tensor,
                model.generate(
                    do_sample=True,
                    temperature=1.0,
                    num_return_sequences=config.group_gen_size,
                    logits_processor=logits_processor,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.pad_token_id,
                    **inputs_BT,  # type: ignore
                ),
            )
            generations_uBuGT.append(batch_completions_uBT.reshape(config.batch_micro_size, config.group_gen_size, -1))

    max_length = max([generation_uBuGT.shape[2] for generation_uBuGT in generations_uBuGT])
    padded_generations_uBuGT = [F.pad(gen_uBuGT, (0, max_length - gen_uBuGT.shape[2])) for gen_uBuGT in generations_uBuGT]
    batch_completions_uBT = torch.cat(padded_generations_uBuGT, dim=1).reshape(-1, max_length)

    return batch_completions_uBT.clone()  # clone the tensor so it can be saved in a backwards pass downstream


def get_policy_log_probs(
    model: Qwen3ForCausalLM, tokenizer: PreTrainedTokenizer, completions_BGT: Tensor, micro_batch_size: int, forward_only: bool
) -> Tensor:
    B, G, T = completions_BGT.shape
    completions_BT = completions_BGT.reshape(-1, T)

    # Break up forward passes into microbatches to save memory
    microbatch_policy_log_probs = []
    for i in range(0, completions_BT.size(0), micro_batch_size):
        micro_batch_completions = completions_BT[i : i + micro_batch_size]
        attn = get_attn_mask_from_tokens(micro_batch_completions, tokenizer.pad_token_id)
        input_ids = micro_batch_completions[:, :-1]
        target_ids = micro_batch_completions[:, 1:]

        policy_log_probs_uBT = get_model_log_probs(model, input_ids, target_ids, attn, forward_only)
        microbatch_policy_log_probs.append(policy_log_probs_uBT)
    return torch.cat(microbatch_policy_log_probs, dim=0).view(B, G, -1)


def move_optimizer_to_cpu(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.device.type != "cpu":
                state[k] = v.cpu()


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.device != device:
                state[k] = v.to(device)


def main():
    """Main training function."""
    SUMMARY_BUDGET_CHOICES = torch.tensor([budget.value for budget in SummaryBudget], device=DEVICE)
    CONFIG = CliApp.run(TrainingConfig)

    tokenizer, policy_model = setup_model_and_tokenizer(CONFIG)

    dataset = cast(DatasetDict, load_from_disk("datasets/c4_realnewslike_8k"))["train"]
    dataset_iter = dataset.batch(CONFIG.batch_full_size, num_proc=4, drop_last_batch=True)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=CONFIG.learning_rate, fused=True)
    if os.path.exists(f"models/{CONFIG.trained_model_dir}/optimizer.pt"):
        optimizer_state = torch.load(f"models/{CONFIG.trained_model_dir}/optimizer.pt")
        optimizer.load_state_dict(optimizer_state)
    move_optimizer_to_cpu(optimizer)

    if CONFIG.logging:
        wandb.init(
            project="rl-summarization",
            config=CONFIG.model_dump(),
        )

    max_reward_mean = -float("inf")

    for epoch in range(CONFIG.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{CONFIG.num_epochs}")
        for batch_idx, batch in enumerate(dataset_iter):
            rollouts_accum_list: list[Rollouts] = []
            rewards_accum_list: list[list[Tensor]] = []

            for batch_accum_step in range(CONFIG.batch_accum_steps):
                logger.info(f"STEP {batch_accum_step + 1}/{CONFIG.batch_accum_steps}")
                start = batch_accum_step * CONFIG.batch_micro_size
                end = (batch_accum_step + 1) * CONFIG.batch_micro_size
                micro_batch = batch["text"][start:end]

                uB, G = CONFIG.batch_micro_size, CONFIG.group_full_size

                budget_choices_idx = torch.randint(len(SUMMARY_BUDGET_CHOICES), (uB,))
                summary_budgets_uB = SUMMARY_BUDGET_CHOICES[budget_choices_idx]

                # TODO: handle document truncation without doing a tokenizer round trip
                batch_accum_documents_tokens = tokenizer(
                    [cast(str, item) for item in micro_batch],
                    truncation=True,
                    max_length=CONFIG.max_document_length,
                    padding=True,
                    return_tensors="pt",
                )
                batch_accum_documents = tokenizer.batch_decode(batch_accum_documents_tokens.input_ids, skip_special_tokens=True)
                batch_accum_contexts = [
                    cast(str, tokenizer.eos_token)
                    + format_context_for_training(cast(str, doc), SummaryBudget.from_float(budget.item()), is_base_model=True)
                    for doc, budget in zip(batch_accum_documents, summary_budgets_uB)
                ]
                batch_accum_tokens = tokenizer(batch_accum_contexts, padding=True, padding_side="left", return_tensors="pt").to(DEVICE)

                input_ids_uBT = batch_accum_tokens.input_ids
                attn_mask_uBT = batch_accum_tokens.attention_mask

                batch_accum_completions_uBT = generate_outputs(policy_model, tokenizer, batch_accum_tokens, CONFIG)
                generations_accum = tokenizer.batch_decode(batch_accum_completions_uBT)
                batch_accum_completions_uBGT = batch_accum_completions_uBT.reshape(uB, G, -1)

                old_policy_log_probs_uBGT = get_policy_log_probs(
                    policy_model, tokenizer, batch_accum_completions_uBGT, micro_batch_size=CONFIG.group_micro_size, forward_only=True
                )

                _, _, T = old_policy_log_probs_uBGT.shape
                prompt_lens_uB = attn_mask_uBT.sum(dim=1)
                sequence_lens_uBG = (batch_accum_completions_uBGT != tokenizer.pad_token_id).sum(dim=2)
                gen_lens_uBG = sequence_lens_uBG - prompt_lens_uB[:, None]

                # Create generation mask
                # Since batches are left padded, all generations start at the same position
                start_idx = input_ids_uBT.shape[1]
                positions_uBGT = torch.arange(T, device=policy_model.device).repeat(uB, G, 1)
                generation_mask_uBGT = (positions_uBGT > start_idx) & (positions_uBGT <= (start_idx + gen_lens_uBG)[:, :, None])
                generation_mask_uBGT = generation_mask_uBGT.to(torch.int8)

                rollouts_slice_uBGT = Rollouts(
                    tokens_BGT=batch_accum_completions_uBGT,
                    generation_mask_BGT=generation_mask_uBGT,
                    old_policy_log_probs_BGT=old_policy_log_probs_uBGT,
                    summary_budgets_B=summary_budgets_uB,
                )
                rollouts_accum_list.append(rollouts_slice_uBGT)

                # group summaries by document
                summaries = [get_summary_from_generation(text, is_base_model=True) for text in generations_accum]
                logger.info("Generated summaries:")
                for summary in summaries:
                    logger.info(f"\t{summary}")
                summaries_uBG = [summaries[i * G : (i + 1) * G] for i in range(len(batch_accum_documents))]
                rewards_uBG = [
                    evaluate_summaries(
                        policy_model,
                        tokenizer,
                        batch_accum_documents[i],
                        summaries_uBG[i],
                        SummaryBudget.from_float(summary_budgets_uB[i].item()),
                        micro_batch_size=CONFIG.group_micro_size,
                        is_training=False,
                    )
                    for i in range(len(batch_accum_documents))
                ]
                rewards_accum_list.append(rewards_uBG)

            reward_mean = torch.cat([torch.cat(rewards) for rewards in rewards_accum_list], dim=0).mean()

            if CONFIG.logging:
                wandb.log({"train::reward_mean": reward_mean.item()})

            if reward_mean.item() > max_reward_mean:
                max_reward_mean = reward_mean.item()
                logger.info(f"New max reward mean {reward_mean.item()}, saving model")
                policy_model.save_pretrained(f"models/{CONFIG.trained_model_dir}")
                torch.save(optimizer.state_dict(), f"models/{CONFIG.trained_model_dir}/optimizer.pt")

            for step in range(CONFIG.policy_steps_per_group):
                for batch_accum_step in range(CONFIG.batch_accum_steps):
                    rewards = rewards_accum_list[batch_accum_step]
                    advantages_BG = torch.stack([get_advantage(reward) for reward in rewards]).to(device=policy_model.device)

                    # DAPO style length objective normalization (divide by total number of tokens in group)
                    total_gen_lens_B11 = rollouts_accum_list[batch_accum_step].generation_mask_BGT.sum(dim=(1, 2), keepdim=True)

                    # Accumulate gradients over groups as well as batches, as a large group size requires too much VRAM due to retained activations
                    # TODO: get rid of the double accumulation some how (will still need to handle VRAM limits for generation)
                    for group_accum_step in range(CONFIG.group_accum_steps):
                        advantages_micro_BG = advantages_BG[
                            :, group_accum_step * CONFIG.group_micro_size : (group_accum_step + 1) * CONFIG.group_micro_size
                        ]

                        rollouts_slice_BGT = rollouts_accum_list[batch_accum_step].get_group_slice(
                            group_idx=group_accum_step * CONFIG.group_micro_size,
                            group_size=CONFIG.group_micro_size,
                        )

                        policy_log_probs_BGT = get_policy_log_probs(
                            policy_model,
                            tokenizer,
                            rollouts_slice_BGT.tokens_BGT,
                            micro_batch_size=CONFIG.group_micro_size,
                            forward_only=False,
                        )
                        importance_ratio_BGT = (policy_log_probs_BGT - rollouts_slice_BGT.old_policy_log_probs_BGT).exp()

                        # We only care about the probabilities of the generated tokens
                        importance_ratio_BGT = importance_ratio_BGT * rollouts_slice_BGT.generation_mask_BGT
                        term_1_BGT = importance_ratio_BGT * advantages_micro_BG[:, :, None]
                        term_2_BGT = (
                            importance_ratio_BGT.clamp(1 - CONFIG.epsilon_low, 1 + CONFIG.epsilon_high) * advantages_micro_BG[:, :, None]
                        )
                        objective_BGT = torch.min(term_1_BGT, term_2_BGT)
                        objective_BGT = objective_BGT / total_gen_lens_B11.clamp(min=1)  # Don't want to divide by 0 on an empty completion
                        objective = objective_BGT.sum()
                        loss = -objective

                        logger.info(
                            f"\tBatchIdx {batch_idx}, Policy Step {step + 1}/{CONFIG.policy_steps_per_group}, "
                            f"Accum Step: {batch_accum_step + 1}/{CONFIG.batch_accum_steps}, Objective: {objective.item()}"
                        )

                        loss.backward()

                # Offload optimizer state to CPU when not in use (during rollout generation)
                move_optimizer_to_device(optimizer, torch.device(DEVICE))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                move_optimizer_to_cpu(optimizer)


if __name__ == "__main__":
    main()
