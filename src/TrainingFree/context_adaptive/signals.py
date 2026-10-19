"""Causal entropy and target-parent attention collection without backend swaps."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from .cache import _rotate_half


def normalized_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    return normalized_entropy_and_confidence(logits, temperature)[0]


def normalized_entropy_and_confidence(logits: torch.Tensor, temperature: float = 1.0) -> tuple[float, float]:
    if temperature <= 0:
        raise ValueError("entropy temperature must be positive")
    values = logits.float() / float(temperature)
    log_probs = torch.log_softmax(values, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    scale = max(math.log(logits.shape[-1]), 1e-12)
    values = torch.stack(((entropy / scale).reshape(-1)[-1], probs.amax(dim=-1).reshape(-1)[-1]))
    return tuple(float(value) for value in values.detach().cpu().tolist())


def causal_query_positions(query_positions, num_queries: int, *, device) -> torch.Tensor:
    """A scalar denotes the LAST position of consecutive causal query rows."""
    if isinstance(query_positions, int):
        positions = torch.arange(query_positions - num_queries + 1, query_positions + 1, device=device)
    else:
        positions = torch.as_tensor(query_positions, dtype=torch.long, device=device)
    if positions.ndim != 1 or positions.numel() != num_queries or num_queries < 1:
        raise ValueError("query_positions must contain one position per query")
    if bool((positions < 0).any()) or bool((positions[1:] <= positions[:-1]).any()):
        raise ValueError("query_positions must be nonnegative and strictly increasing")
    return positions


class TargetQueryCapture:
    """Capture normalized, RoPE'd Q from one target attention module only."""

    def __init__(self, target: Any, layer_index: int) -> None:
        layers = target.model.layers
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"target attention layer {layer_index} is out of range")
        self.module = layers[layer_index].self_attn
        self.layer_index = layer_index
        self.query: torch.Tensor | None = None
        self._position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
        self._enabled = False
        self._last_only = True
        self._handles = []
        self._handles.append(self.module.register_forward_pre_hook(self._capture_inputs, with_kwargs=True))
        self._handles.append(self.module.q_proj.register_forward_hook(self._capture_query))

    def _capture_inputs(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self._enabled:
            self._position_embeddings = kwargs.get("position_embeddings")

    def _capture_query(self, module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
        if not self._enabled:
            return
        if self._position_embeddings is None:
            raise RuntimeError("target Q hook did not receive position_embeddings")
        attention = self.module
        batch, sequence, _ = output.shape
        q = attention.q_norm(output.view(batch, sequence, -1, attention.head_dim)).transpose(1, 2)
        cos, sin = self._position_embeddings
        q = q * cos.unsqueeze(1) + _rotate_half(q) * sin.unsqueeze(1)
        self.query = q[:, :, -1:, :] if self._last_only else q

    def enable(self, *, last_only: bool) -> None:
        self.query = None
        self._position_embeddings = None
        self._last_only = last_only
        self._enabled = True

    def disable(self) -> torch.Tensor | None:
        self._enabled = False
        query = self.query
        self.query = None
        self._position_embeddings = None
        return query

    def close(self) -> None:
        self._enabled = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _cache_key(cache: Any, layer_index: int) -> torch.Tensor:
    if hasattr(cache, "layers") and len(cache.layers) > layer_index:
        key = getattr(cache.layers[layer_index], "keys", None)
        if key is not None:
            return key
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_index]
    raise RuntimeError("target cache does not expose full attention keys")


def target_parent_scores(
    query: torch.Tensor,
    target_cache: Any,
    layer_index: int,
    query_index: int,
    query_position: int,
    source_chunks: Sequence[Sequence[int]],
    *,
    head_groups: int,
    scaling: float,
    sliding_window: int | None = None,
) -> tuple[dict[int, float], float | None]:
    if query_index < 0 or query_index >= query.shape[-2]:
        raise ValueError("parent query index is outside captured Q rows")
    key = _cache_key(target_cache, layer_index)
    allowed = min(query_position + 1, key.shape[-2])
    if allowed < 1:
        return {}, None
    q = query[:, :, query_index : query_index + 1, :].float()
    key = key[:, :, :allowed, :].float()
    if head_groups > 1:
        key = key.repeat_interleave(head_groups, dim=1)
    scores = torch.matmul(q, key.transpose(-1, -2)) * float(scaling)
    key_positions = torch.arange(allowed, device=key.device)
    if sliding_window is not None:
        mask = key_positions >= max(0, query_position - sliding_window + 1)
        scores = scores.masked_fill(~mask.view(1, 1, 1, -1), torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1).mean(dim=(0, 1, 2)).detach().cpu().tolist()
    token_scores: dict[int, float] = {}
    chunk_masses: list[float] = []
    for chunk in source_chunks:
        valid = [position for position in chunk if 0 <= position < allowed]
        mass = sum(float(probabilities[position]) for position in valid) if valid else 0.0
        chunk_masses.append(mass)
        for position in valid:
            token_scores[position] = float(probabilities[position])
    total_source = sum(chunk_masses)
    concentration = sum(sorted(chunk_masses, reverse=True)[:8]) / total_source if total_source > 0 else None
    if concentration is None:
        token_scores = {}
    return token_scores, concentration


def target_multi_query_scores(
    query: torch.Tensor,
    target_cache: Any,
    layer_index: int,
    query_positions: Sequence[int] | torch.Tensor | int,
    source_chunks: Sequence[Sequence[int]],
    *,
    head_groups: int,
    scaling: float,
    sliding_window: int | None = None,
    pooling_mode: str = "max",
) -> tuple[dict[int, float], float | None]:
    """Causal multi-query attention with max or mean pooling across verified queries."""
    key = _cache_key(target_cache, layer_index)
    num_queries = query.shape[-2]
    positions_tensor = causal_query_positions(query_positions, num_queries, device=key.device)
    max_allowed = min(int(positions_tensor[-1].item()) + 1, key.shape[-2])

    if max_allowed < 1 or num_queries < 1:
        return {}, None

    q = query[:, :, :num_queries, :].float()
    k = key[:, :, :max_allowed, :].float()
    if head_groups > 1:
        k = k.repeat_interleave(head_groups, dim=1)

    scores = torch.matmul(q, k.transpose(-1, -2)) * float(scaling)
    key_positions = torch.arange(max_allowed, device=key.device)
    causal_mask = key_positions.view(1, 1, 1, -1) <= positions_tensor.view(1, 1, -1, 1)
    if sliding_window is not None:
        window_mask = key_positions.view(1, 1, 1, -1) >= (positions_tensor.view(1, 1, -1, 1) - int(sliding_window) + 1)
        causal_mask = causal_mask & window_mask

    scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1)
    mean_prob = probabilities.mean(dim=(0, 1))

    if pooling_mode == "mean":
        pooled_prob = mean_prob.mean(dim=0)
    else:
        pooled_prob = mean_prob.amax(dim=0)

    total_mass = pooled_prob.sum()
    if total_mass > 0:
        norm_prob = (pooled_prob / total_mass).detach().cpu().tolist()
    else:
        norm_prob = pooled_prob.detach().cpu().tolist()

    token_scores: dict[int, float] = {}
    chunk_masses: list[float] = []
    for chunk in source_chunks:
        valid = [position for position in chunk if 0 <= position < max_allowed]
        mass = sum(float(norm_prob[position]) for position in valid) if valid else 0.0
        chunk_masses.append(mass)
        for position in valid:
            token_scores[position] = float(norm_prob[position])

    total_source = sum(chunk_masses)
    concentration = sum(sorted(chunk_masses, reverse=True)[:8]) / total_source if total_source > 0 else None
    if concentration is None:
        token_scores = {}
    return token_scores, concentration


def target_duo_ensemble_scores(
    queries_by_layer: Mapping[int, torch.Tensor],
    target_cache: Any,
    layer_mid: int,
    layer_deep: int,
    query_positions: Sequence[int] | torch.Tensor | int,
    source_chunks: Sequence[Sequence[int]],
    *,
    head_groups: int,
    scaling: float,
    alpha: float = 0.5,
    sliding_window: int | None = None,
    pooling_mode: str = "max",
) -> tuple[dict[int, float], float | None]:
    """Duo-Layer Ensemble combining Middle Retrieval Core and Deep Output Sink."""
    if layer_mid not in queries_by_layer or layer_deep not in queries_by_layer:
        raise ValueError(f"Duo-Ensemble requires queries for layers {layer_mid} and {layer_deep}")

    scores_mid, _ = target_multi_query_scores(
        queries_by_layer[layer_mid],
        target_cache,
        layer_mid,
        query_positions,
        source_chunks,
        head_groups=head_groups,
        scaling=scaling,
        sliding_window=sliding_window,
        pooling_mode=pooling_mode,
    )
    scores_deep, _ = target_multi_query_scores(
        queries_by_layer[layer_deep],
        target_cache,
        layer_deep,
        query_positions,
        source_chunks,
        head_groups=head_groups,
        scaling=scaling,
        sliding_window=sliding_window,
        pooling_mode=pooling_mode,
    )

    if not scores_mid and not scores_deep:
        return {}, None
    if not scores_mid:
        return scores_deep, None
    if not scores_deep:
        return scores_mid, None

    all_positions = set(scores_mid.keys()).union(scores_deep.keys())
    token_scores: dict[int, float] = {}
    for pos in all_positions:
        val_mid = scores_mid.get(pos, 0.0)
        val_deep = scores_deep.get(pos, 0.0)
        token_scores[pos] = float(alpha * val_mid + (1.0 - alpha) * val_deep)

    chunk_masses: list[float] = []
    for chunk in source_chunks:
        mass = sum(token_scores.get(pos, 0.0) for pos in chunk)
        chunk_masses.append(mass)

    total_source = sum(chunk_masses)
    concentration = sum(sorted(chunk_masses, reverse=True)[:8]) / total_source if total_source > 0 else None
    return token_scores, concentration


def refresh_concentration(layer_scores: Mapping[int, Mapping[int, float]], source_chunks: Sequence[Sequence[int]]) -> float | None:
    values: list[float] = []
    for scores in layer_scores.values():
        masses = [sum(float(scores.get(position, 0.0)) for position in chunk) for chunk in source_chunks]
        total = sum(masses)
        if total > 0:
            values.append(sum(sorted(masses, reverse=True)[:8]) / total)
    return sum(values) / len(values) if values else None
