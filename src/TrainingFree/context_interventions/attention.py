"""Causal target attention collectors for all context positions."""

from __future__ import annotations

from typing import Any

import torch

from TrainingFree.context_adaptive.signals import _cache_key


def full_key_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    query_index: int,
    query_position: int,
    head_groups: int,
    scaling: float,
    sliding_window: int | None = None,
) -> dict[int, float]:
    """Mean-head softmax over allowed causal keys, indexed by original position."""
    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != key.shape[0]:
        raise ValueError("query and key must have [batch, heads, sequence, head_dim] shape")
    if not 0 <= query_index < query.shape[-2]:
        raise ValueError("query_index is outside captured query rows")
    if head_groups < 1 or query.shape[1] != key.shape[1] * head_groups:
        raise ValueError("head_groups do not match query/key head counts")
    allowed = min(int(query_position) + 1, key.shape[-2])
    if allowed < 1:
        return {}
    q = query[:, :, query_index : query_index + 1, :].float()
    k = key[:, :, :allowed, :].float().repeat_interleave(head_groups, dim=1)
    logits = torch.matmul(q, k.transpose(-1, -2)) * float(scaling)
    if sliding_window is not None:
        positions = torch.arange(allowed, device=key.device)
        valid = positions >= max(0, int(query_position) - int(sliding_window) + 1)
        logits = logits.masked_fill(~valid.view(1, 1, 1, -1), torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=-1).mean(dim=(0, 1, 2))
    if not torch.isfinite(probabilities).all().item():
        raise ValueError("target attention scores contain non-finite values")
    values = probabilities.detach().cpu().tolist()
    return {position: float(value) for position, value in enumerate(values)}


def target_full_key_scores(
    query: torch.Tensor,
    target_cache: Any,
    layer_index: int,
    query_index: int,
    query_position: int,
    attention: Any,
) -> dict[int, float]:
    key = _cache_key(target_cache, layer_index)
    return full_key_attention_scores(
        query,
        key,
        query_index=query_index,
        query_position=query_position,
        head_groups=int(attention.num_key_value_groups),
        scaling=float(attention.scaling),
        sliding_window=getattr(attention, "sliding_window", None),
    )


class MultiLayerTargetQueryCapture:
    """Install target Q hooks on multiple layers and preserve RoPE positions."""

    def __init__(self, target: Any, layer_indices: tuple[int, ...] | list[int]) -> None:
        from TrainingFree.context_adaptive.signals import TargetQueryCapture

        indices = tuple(dict.fromkeys(int(index) for index in layer_indices))
        if not indices:
            raise ValueError("at least one target layer is required")
        self.captures = {index: TargetQueryCapture(target, index) for index in indices}

    def enable(self, *, last_only: bool) -> None:
        for capture in self.captures.values():
            capture.enable(last_only=last_only)

    def disable(self) -> dict[int, torch.Tensor]:
        rows: dict[int, torch.Tensor] = {}
        for index, capture in self.captures.items():
            query = capture.disable()
            if query is not None:
                rows[index] = query
        return rows

    def close(self) -> None:
        for capture in self.captures.values():
            capture.close()
