"""PyTorch SDPA helpers used by native speculative attention adapters."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    sliding_window: int | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """Apply SDPA to Q/K/V in ``[batch, heads, sequence, dim]`` layout.

    ``attention_mask`` follows PyTorch SDPA semantics: boolean ``True`` marks
    visible positions, while floating masks are additive. The caller supplies
    ``is_causal=False`` for cached single-token decoding, whose query may attend
    to every key in the prefix.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("SDPA Q/K/V tensors must have four dimensions")
    if key.shape != value.shape:
        raise ValueError("SDPA key and value tensor shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("SDPA query and key batch/head dimensions are incompatible")
    if sliding_window is not None:
        raise ValueError("SDPA native benchmark does not support sliding-window attention")
    if attention_mask is not None and is_causal:
        # A supplied mask can encode tree, cache, or padding constraints. Do not
        # combine it with SDPA's top-left causal mask.
        is_causal = False
    if (
        attention_mask is not None
        and attention_mask.dtype != torch.bool
        and attention_mask.dtype != query.dtype
    ):
        attention_mask = attention_mask.to(dtype=query.dtype)

    enable_gqa = query.shape[1] != key.shape[1]
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=float(scaling),
        enable_gqa=enable_gqa,
    )
