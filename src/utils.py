import torch
from torch import Tensor


def get_attn_mask_from_tokens(tokens_BT: Tensor, pad_token_id: int) -> Tensor:
    """Get the attention mask from the input tokens.

    Leave the first BOS token unmasked to allow it to act as an attention sink.
    """
    B, _ = tokens_BT.shape
    pad_tokens_BT = tokens_BT == pad_token_id
    next_is_non_pad = torch.cat([~pad_tokens_BT[:, 1:], torch.zeros((B, 1), dtype=torch.bool, device=tokens_BT.device)], dim=-1)
    last_pad_before_doc = pad_tokens_BT & next_is_non_pad
    return (~pad_tokens_BT) | last_pad_before_doc
