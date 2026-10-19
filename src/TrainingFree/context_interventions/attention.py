"""Causal target attention collectors for all context positions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from TrainingFree.context_adaptive.signals import _cache_key, causal_query_positions


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


def multi_query_full_key_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    query_positions: Sequence[int] | torch.Tensor | int,
    head_groups: int,
    scaling: float,
    sliding_window: int | None = None,
    pooling_mode: str = "max",
) -> dict[int, float]:
    """Mean-head softmax across verified queries with max/mean pooling over queries."""
    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != key.shape[0]:
        raise ValueError("query and key must have [batch, heads, sequence, head_dim] shape")
    num_queries = query.shape[-2]
    if head_groups < 1 or query.shape[1] != key.shape[1] * head_groups:
        raise ValueError("head_groups do not match query/key head counts")
    if pooling_mode not in {"max", "mean"}:
        raise ValueError("pooling_mode must be max or mean")
    positions_tensor = causal_query_positions(query_positions, num_queries, device=key.device)
    max_allowed = min(int(positions_tensor[-1].item()) + 1, key.shape[-2])
    if max_allowed < 1 or num_queries < 1:
        return {}
    q = query[:, :, :num_queries, :].float()
    k = key[:, :, :max_allowed, :].float().repeat_interleave(head_groups, dim=1)
    logits = torch.matmul(q, k.transpose(-1, -2)) * float(scaling)
    key_positions = torch.arange(max_allowed, device=key.device)
    causal_mask = key_positions.view(1, 1, 1, -1) <= positions_tensor.view(1, 1, -1, 1)
    if sliding_window is not None:
        window_mask = key_positions.view(1, 1, 1, -1) >= (positions_tensor.view(1, 1, -1, 1) - int(sliding_window) + 1)
        causal_mask = causal_mask & window_mask
    logits = logits.masked_fill(~causal_mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=-1).mean(dim=(0, 1))
    if pooling_mode == "mean":
        pooled = probabilities.mean(dim=0)
    else:
        pooled = probabilities.amax(dim=0)
    total = pooled.sum()
    if total > 0:
        pooled = pooled / total
    values = pooled.detach().cpu().tolist()
    return {position: float(value) for position, value in enumerate(values)}


def duo_ensemble_full_key_attention_scores(
    query_mid: torch.Tensor,
    key_mid: torch.Tensor,
    query_deep: torch.Tensor,
    key_deep: torch.Tensor,
    *,
    query_positions: Sequence[int] | torch.Tensor | int,
    head_groups: int,
    scaling: float,
    alpha: float = 0.5,
    sliding_window: int | None = None,
    pooling_mode: str = "max",
) -> dict[int, float]:
    """Full-key attention blending multi-query scores from Middle and Deep layers."""
    scores_mid = multi_query_full_key_attention_scores(
        query_mid,
        key_mid,
        query_positions=query_positions,
        head_groups=head_groups,
        scaling=scaling,
        sliding_window=sliding_window,
        pooling_mode=pooling_mode,
    )
    scores_deep = multi_query_full_key_attention_scores(
        query_deep,
        key_deep,
        query_positions=query_positions,
        head_groups=head_groups,
        scaling=scaling,
        sliding_window=sliding_window,
        pooling_mode=pooling_mode,
    )
    all_pos = set(scores_mid.keys()).union(scores_deep.keys())
    return {
        pos: float(alpha * scores_mid.get(pos, 0.0) + (1.0 - alpha) * scores_deep.get(pos, 0.0))
        for pos in all_pos
    }


def target_multi_query_full_key_scores(
    query: torch.Tensor,
    target_cache: Any,
    layer_index: int,
    query_positions: Sequence[int] | torch.Tensor | int,
    attention: Any,
    pooling_mode: str = "max",
) -> dict[int, float]:
    key = _cache_key(target_cache, layer_index)
    return multi_query_full_key_attention_scores(
        query,
        key,
        query_positions=query_positions,
        head_groups=int(attention.num_key_value_groups),
        scaling=float(attention.scaling),
        sliding_window=getattr(attention, "sliding_window", None),
        pooling_mode=pooling_mode,
    )


def target_duo_ensemble_full_key_scores(
    queries_by_layer: Mapping[int, torch.Tensor],
    target_cache: Any,
    layer_mid: int,
    layer_deep: int,
    query_positions: Sequence[int] | torch.Tensor | int,
    attention: Any,
    alpha: float = 0.5,
    pooling_mode: str = "max",
) -> dict[int, float]:
    key_mid = _cache_key(target_cache, layer_mid)
    key_deep = _cache_key(target_cache, layer_deep)
    return duo_ensemble_full_key_attention_scores(
        queries_by_layer[layer_mid],
        key_mid,
        queries_by_layer[layer_deep],
        key_deep,
        query_positions=query_positions,
        head_groups=int(attention.num_key_value_groups),
        scaling=float(attention.scaling),
        alpha=alpha,
        sliding_window=getattr(attention, "sliding_window", None),
        pooling_mode=pooling_mode,
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
