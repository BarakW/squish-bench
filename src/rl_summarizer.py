import os

from pydantic_settings import BaseSettings, CliApp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import logging
from typing import cast

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from pydantic import Field, model_validator
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Qwen3ForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.tokenization_utils_base import BatchEncoding

import wandb
from data import format_context_for_training, get_attn_mask_from_tokens, get_context_for_evaluation, get_summary_from_generation
from policy import Rollouts, SummaryBudgetChars

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


@torch.inference_mode()
def evaluate_summaries(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    document: str,
    summaries: list[str | None],
    /,
    summary_budget: int,
    micro_batch_size: int,
    is_training: bool,
) -> Tensor:
    """
    Uses a reference model to evaluate the information within the target documents.

    Step 1: Get the cross-entropy of the document alone
    Step 2: Get the cross-entropy of the document when it is prefixed by the summary
    Step 3: Get the information saved by using the summary
    Step 4: Return the information saved by the summary, conditional on meeting the summary budget

    The return value represents the total
    """
    # Handle all empty summaries, to both save compute time and avoid edge cases
    if all(not summary for summary in summaries):
        return torch.zeros(len(summaries), device=model.device)

    # failing to generate the summary format gets 0 reward
    # Compute rewards only for non-empty summaries, set reward to 0 for empty summaries
    populated_summaries: list[str] = []
    nonempty_indices: list[int] = []
    for i, summary in enumerate(summaries):
        if summary:
            populated_summaries.append(summary)
            nonempty_indices.append(i)
    populated_summary_lens = torch.tensor([len(summary) for summary in populated_summaries], device=DEVICE)

    eval_context = get_context_for_evaluation(tokenizer, populated_summaries, document)

    # Get the cross-entropy of the document alone
    # TODO: cache these as they never change throughout training
    document_tokens = eval_context.document_tokens[None, :].to(device=DEVICE)
    document_loss = total_cross_entropy(model, tokenizer, document_tokens, micro_batch_size=micro_batch_size)

    # Get the cross-entropy of the document when prefixed by the summary
    document_with_summary_loss = total_cross_entropy(
        model,
        tokenizer,
        eval_context.summaries_and_document_tokens.to(device=DEVICE),
        micro_batch_size=micro_batch_size,
        mask_to_zero_BT=eval_context.document_masks.to(device=DEVICE),
    )

    # Get the information saved by using the summary
    information_saved = document_loss - document_with_summary_loss

    rewards_nonempty = information_saved.where(populated_summary_lens <= summary_budget, 0)
    rewards_nonempty /= document_loss  # normalize reward by document information

    # Create a reward tensor for all summaries, setting (mean - (max - mean)) for empty ones
    max_reward = rewards_nonempty.max()
    min_reward = rewards_nonempty.min()
    reward_diff = max_reward - min_reward

    # require at least (min - 0.1) for empty summaries, otherwise if all rewards are the same, empty summaries get too high a reward
    reward_diff = max(reward_diff, 1e-1)

    reward_empty_value = 0
    if is_training:  # We want to discourage format non-adherance during training
        reward_empty_value = min_reward - reward_diff

    rewards = torch.empty(len(summaries), device=rewards_nonempty.device).fill_(reward_empty_value)
    if len(nonempty_indices) > 0:
        rewards[torch.tensor(nonempty_indices, device=rewards.device)] = rewards_nonempty

    return rewards.to(device=model.device)


def get_advantage(rewards: Tensor) -> Tensor:
    mean_reward = rewards.mean()
    return rewards - mean_reward


@torch.compile(dynamic=True)
def get_model_log_probs(
    model: PreTrainedModel,
    input_ids_BT: Tensor,
    target_ids_BT: Tensor,
    attention_mask_BT: Tensor,
    forward_only: bool,
) -> Tensor:
    # We don't want the completions to be retained in the computation graph when computing the old policy
    with torch.autocast(DEVICE, dtype=torch.bfloat16), torch.inference_mode(forward_only):
        outputs = model.forward(input_ids=input_ids_BT, attention_mask=attention_mask_BT, use_cache=False)
        assert outputs.logits is not None
        log_probs_BTV = F.log_softmax(outputs.logits, dim=-1, dtype=torch.float32).to(model.device)
        target_ids_BT = target_ids_BT.clone()  # There's a weird pytorch compile bug where the unsqueeze below fails if we don't clone first
        log_probs_BT = log_probs_BTV.gather(dim=-1, index=target_ids_BT.unsqueeze(-1)).squeeze(-1)
        return log_probs_BT


def total_cross_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids_BT: Tensor,
    /,
    micro_batch_size: int,
    mask_to_zero_BT: Tensor | None = None,
) -> Tensor:
    """Compute per-example summed cross-entropy over causal next-token predictions.

    - Shifts targets by 1 position as usual for causal LM.
    - If `mask_to_zero` is provided (True = masked), zeros those token losses
      after the shift alignment (i.e., `mask_to_zero[:, :-1]`).
    """
    # Micro-batch the forward pass to reduce memory, matching get_policy_log_probs
    per_example_losses: list[Tensor] = []
    batch_size = input_ids_BT.size(0)

    for i in range(0, batch_size, micro_batch_size):
        micro_input = input_ids_BT[i : i + micro_batch_size]
        micro_mask_to_zero = None if mask_to_zero_BT is None else mask_to_zero_BT[i : i + micro_batch_size]

        attn_mask = get_attn_mask_from_tokens(micro_input, tokenizer.pad_token_id)
        input_ids = micro_input[:, :-1]
        target_ids = micro_input[:, 1:]
        losses_BT = -get_model_log_probs(model, input_ids, target_ids, attn_mask, forward_only=True)

        if micro_mask_to_zero is not None:
            losses_BT[micro_mask_to_zero[:, :-1]] = 0.0

        per_example_losses.append(losses_BT.sum(dim=1))

    return torch.cat(per_example_losses, dim=0)


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


def generate_outputs(model: Qwen3ForCausalLM, tokenizer: PreTrainedTokenizer, inputs: BatchEncoding, config: TrainingConfig) -> Tensor:
    logits_processor = LogitsProcessorList(
        [StopOnSubsequence(tokenizer, stop_strings=["</summary>", tokenizer.eos_token], device=model.device)]
    )
    with torch.inference_mode():
        batch_completions_BT = model.generate(
            do_sample=True,
            temperature=1.0,
            num_return_sequences=config.group_full_size,
            logits_processor=logits_processor,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            **inputs,  # type: ignore
        )

    return cast(Tensor, batch_completions_BT.clone())  # clone the tensor so it can be saved in a backwards pass downstream


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

        policy_log_probs_BT = get_model_log_probs(model, input_ids, target_ids, attn, forward_only)
        microbatch_policy_log_probs.append(policy_log_probs_BT)
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
    SUMMARY_BUDGET_CHOICES = torch.tensor([budget.value for budget in SummaryBudgetChars], device=DEVICE)
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

                B, G = CONFIG.batch_micro_size, CONFIG.group_full_size

                budget_choices_idx = torch.randint(len(SUMMARY_BUDGET_CHOICES), (B,))
                summary_budgets_B = SUMMARY_BUDGET_CHOICES[budget_choices_idx]

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
                    + format_context_for_training(cast(str, doc), cast(int, budget.item()), is_base_model=True)
                    for doc, budget in zip(batch_accum_documents, summary_budgets_B)
                ]
                batch_accum_tokens = tokenizer(batch_accum_contexts, padding=True, padding_side="left", return_tensors="pt").to(DEVICE)

                input_ids_BT = batch_accum_tokens.input_ids
                attn_mask_BT = batch_accum_tokens.attention_mask

                batch_accum_completions_flat = generate_outputs(policy_model, tokenizer, batch_accum_tokens, CONFIG)
                generations_accum = tokenizer.batch_decode(batch_accum_completions_flat)
                batch_accum_completions_BGT = batch_accum_completions_flat.reshape(B, G, -1)

                old_policy_log_probs_BGT = get_policy_log_probs(
                    policy_model, tokenizer, batch_accum_completions_BGT, micro_batch_size=CONFIG.group_micro_size, forward_only=True
                )

                _, _, T = old_policy_log_probs_BGT.shape
                prompt_lens_B = attn_mask_BT.sum(dim=1)
                sequence_lens_BG = (batch_accum_completions_BGT != tokenizer.pad_token_id).sum(dim=2)
                gen_lens_BG = sequence_lens_BG - prompt_lens_B[:, None]

                # Create generation mask
                # Since batches are left padded, all generations start at the same position
                start_idx = input_ids_BT.shape[1]
                positions_BGT = torch.arange(T, device=policy_model.device).repeat(B, G, 1)
                generation_mask_BGT = (positions_BGT > start_idx) & (positions_BGT <= (start_idx + gen_lens_BG)[:, :, None])
                generation_mask_BGT = generation_mask_BGT.to(torch.int8)

                rollouts_slice_BGT = Rollouts(
                    tokens_BGT=batch_accum_completions_BGT,
                    generation_mask_BGT=generation_mask_BGT,
                    old_policy_log_probs_BGT=old_policy_log_probs_BGT,
                    summary_budgets_B=summary_budgets_B,
                )
                rollouts_accum_list.append(rollouts_slice_BGT)

                # group summaries by document
                summaries = [get_summary_from_generation(text, is_base_model=True) for text in generations_accum]
                logger.info("Generated summaries:")
                for summary in summaries:
                    logger.info(f"\t{summary}")
                summaries_BG = [summaries[i * G : (i + 1) * G] for i in range(len(batch_accum_documents))]
                rewards_BG = [
                    evaluate_summaries(
                        policy_model,
                        tokenizer,
                        batch_accum_documents[i],
                        summaries_BG[i],
                        cast(int, summary_budgets_B[i].item()),
                        micro_batch_size=CONFIG.group_micro_size,
                        is_training=False,
                    )
                    for i in range(len(batch_accum_documents))
                ]
                rewards_accum_list.append(rewards_BG)

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
