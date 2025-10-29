import re
from dataclasses import dataclass
from typing import TypedDict, cast

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from policy import SummaryBudget


class SummarizationData(TypedDict):
    text: str
    summary: str


def format_context_for_training(document: str, budget: SummaryBudget, is_base_model: bool) -> str:
    budget_chars = int(len(document) * budget)
    maybe_prefill = "\n<summary>" if is_base_model else ""
    res = f"""Generate a summary of the provided document in strictly fewer than {budget_chars} characters. Target a length of around {int(budget_chars * 0.9)} characters. The document is within a <document> tag and the summary is within a <summary> tag.

<document>{document}</document>{maybe_prefill}"""
    return res


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


# There are 2 <summary> tags, the first one is for the instructions, the second one is for the actual summary
# TODO: get generated tokens from base model generations so there's only *one* regex
SUMMARY_EXTRACTOR_CHAT_MODEL = re.compile(r"<summary>(?P<summary>[\s\S]*?)</summary>")
SUMMARY_EXTRACTOR_BASE_MODEL = re.compile(r"<summary>[\s\S]*<summary>(?P<summary>[\s\S]*?)</summary>")


def get_summary_from_generation(text: str, is_base_model: bool) -> str | None:
    if is_base_model:
        match = SUMMARY_EXTRACTOR_BASE_MODEL.search(text)
    else:
        match = SUMMARY_EXTRACTOR_CHAT_MODEL.search(text)
    return match.group("summary") if match else None


def get_attn_mask_from_tokens(tokens_BT: Tensor, pad_token_id: int) -> Tensor:
    """Get the attention mask from the input tokens.

    Leave the first BOS token unmasked to allow it to act as an attention sink.
    """
    B, _ = tokens_BT.shape
    pad_tokens_BT = tokens_BT == pad_token_id
    next_is_non_pad = torch.cat([~pad_tokens_BT[:, 1:], torch.zeros((B, 1), dtype=torch.bool, device=tokens_BT.device)], dim=-1)
    last_pad_before_doc = pad_tokens_BT & next_is_non_pad
    return (~pad_tokens_BT) | last_pad_before_doc
