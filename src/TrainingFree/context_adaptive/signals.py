"""Causal entropy and target-parent attention collection without backend swaps."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from .cache import _rotate_half


def normalized_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    if temperature <= 0:
        raise ValueError("entropy temperature must be positive")
    values = logits.float() / float(temperature)
    log_probs = torch.log_softmax(values, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    scale = max(math.log(logits.shape[-1]), 1e-12)
    return float((entropy / scale).reshape(-1)[-1].item())


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


def refresh_concentration(layer_scores: Mapping[int, Mapping[int, float]], source_chunks: Sequence[Sequence[int]]) -> float | None:
    values: list[float] = []
    for scores in layer_scores.values():
        masses = [sum(float(scores.get(position, 0.0)) for position in chunk) for chunk in source_chunks]
        total = sum(masses)
        if total > 0:
            values.append(sum(sorted(masses, reverse=True)[:8]) / total)
    return sum(values) / len(values) if values else None
