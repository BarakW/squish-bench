from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import get_attn_mask_from_tokens

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
elif torch.mps.is_available():
    DEVICE = "mps"


class SummaryBudget(float, Enum):
    """Summary budget is measured as a ratio of the document length"""

    EXTRA_SMALL = 0.01
    SMALL = 0.02
    MEDIUM = 0.05
    LARGE = 0.1
    EXTRA_LARGE = 0.2

    @classmethod
    def from_float(cls, value: float) -> SummaryBudget:
        return cls(round(value, 3))


@dataclass
class EvalContext:
    document_tokens: torch.Tensor
    summaries_and_document_tokens: torch.Tensor
    document_masks: torch.Tensor


def get_context_for_evaluation(tokenizer: PreTrainedTokenizer, summaries: list[str], document: str) -> EvalContext:
    eos_token = cast(str, tokenizer.eos_token)

    prefix = """The following is a summary of a document followed by the document itself. The summary is within a <summary> tag and the document is within a <document> tag.

<summary>"""
    summary_to_document = "</summary>\n<document>"
    document_suffix = "</document>"

    prefix_tokens = cast(Tensor, tokenizer.encode(eos_token + prefix, return_tensors="pt")[0])
    summary_to_document_tokens = cast(Tensor, tokenizer.encode(summary_to_document, return_tensors="pt")[0])
    document_suffix_tokens = cast(Tensor, tokenizer.encode(document_suffix + eos_token, return_tensors="pt")[0])

    document_tokens = cast(Tensor, tokenizer.encode(eos_token + document + eos_token, return_tensors="pt")[0])
    truncated_document_tokens = document_tokens[1:-1]

    summary_and_documents = []
    document_masks = []
    for i in range(len(summaries)):
        summary_tokens = cast(Tensor, tokenizer.encode(summaries[i], return_tensors="pt")[0])
        prefix_and_summary = torch.cat([prefix_tokens, summary_tokens, summary_to_document_tokens], dim=-1)

        document_start = prefix_and_summary.shape[0]
        document_end = document_start + truncated_document_tokens.shape[0]

        whole_context = torch.cat([prefix_and_summary, truncated_document_tokens, document_suffix_tokens], dim=-1)

        document_mask = torch.ones_like(whole_context, dtype=torch.bool)
        document_mask[document_start:document_end] = False

        summary_and_documents.append(whole_context)
        document_masks.append(document_mask)

    all_summary_and_documents = torch.nn.utils.rnn.pad_sequence(
        summary_and_documents, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    all_document_masks = torch.nn.utils.rnn.pad_sequence(document_masks, batch_first=True, padding_value=True)

    return EvalContext(
        document_tokens=document_tokens,
        summaries_and_document_tokens=all_summary_and_documents,
        document_masks=all_document_masks,
    )


@torch.compile(dynamic=True)
def get_model_log_probs(
    model: PreTrainedModel,
    input_ids_BT: Tensor,
    target_ids_BT: Tensor,
    attention_mask_BT: Tensor,
    forward_only: bool,
) -> Tensor:
    # We don't want the completions to be retained in the computation graph when computing the old policy
    with torch.autocast(model.device.type, dtype=torch.bfloat16), torch.inference_mode(forward_only):
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


@torch.inference_mode()
def evaluate_summaries(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    document: str,
    summaries: list[str | None],
    /,
    summary_budget: SummaryBudget,
    micro_batch_size: int,
    is_training: bool,
) -> Tensor:
    """
    Uses a reference model to evaluate the information within the target documents.

    Step 1: Get the cross-entropy of the document alone
    Step 2: Get the cross-entropy of the document when it is prefixed by the summary
    Step 3: Get the information saved by using the summary
    Step 4: Return the information saved by the summary, conditional on meeting the summary budget
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

    budget_chars = int(summary_budget * len(document))
    rewards_nonempty = information_saved.where(populated_summary_lens <= budget_chars, 0)
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
