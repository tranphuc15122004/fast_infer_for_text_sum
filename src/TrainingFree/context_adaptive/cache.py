"""Preallocated target-feature K/V bank used only by the DFlash drafter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .types import LayerContext, Selection


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


@dataclass
class _LayerStorage:
    key: torch.Tensor
    value: torch.Tensor


class DraftContextBank:
    """Append target hidden features once and cache their projected layer KV.

    Stored keys have already passed ``k_norm`` and RoPE at their original
    logical positions. Gathering never rotates or renumbers them.
    """

    def __init__(self, draft_model: Any, capacity: int, *, batch_size: int = 1) -> None:
        if capacity < 1 or batch_size != 1:
            raise ValueError("V1 bank requires positive capacity and batch size one")
        self.model = draft_model
        self.capacity = int(capacity)
        self.batch_size = batch_size
        self.length = 0
        self.device = next(draft_model.parameters()).device
        self.dtype = next(draft_model.parameters()).dtype
        self.layer_storage: list[_LayerStorage] = []
        self.positions = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self.features: torch.Tensor | None = None
        for layer in draft_model.layers:
            attention = layer.self_attn
            head_dim = int(attention.head_dim)
            kv_heads = int(attention.config.num_key_value_heads)
            self.layer_storage.append(
                _LayerStorage(
                    key=torch.empty((1, kv_heads, capacity, head_dim), dtype=self.dtype, device=self.device),
                    value=torch.empty((1, kv_heads, capacity, head_dim), dtype=self.dtype, device=self.device),
                )
            )

    @property
    def allocated_bytes(self) -> int:
        total = self.positions.numel() * self.positions.element_size()
        total += sum(item.key.numel() * item.key.element_size() + item.value.numel() * item.value.element_size() for item in self.layer_storage)
        if self.features is not None:
            total += self.features.numel() * self.features.element_size()
        return total

    def append(self, target_features: torch.Tensor, positions: torch.Tensor) -> None:
        if target_features.ndim != 3 or target_features.shape[0] != 1:
            raise ValueError("target_features must have shape [1, delta, feature_dim]")
        delta = target_features.shape[1]
        positions = positions.to(device=self.device, dtype=torch.long).reshape(-1)
        if positions.numel() != delta:
            raise ValueError("positions must contain one logical position per feature row")
        expected = torch.arange(self.length, self.length + delta, device=self.device)
        if not torch.equal(positions, expected):
            raise ValueError(f"bank append must be contiguous: expected {expected.tolist()}, got {positions.tolist()}")
        if self.length + delta > self.capacity:
            raise ValueError(f"draft bank capacity {self.capacity} exceeded")
        features = target_features.to(device=self.device, dtype=self.dtype)
        latent = self.model.hidden_norm(self.model.fc(features))
        cos, sin = self.model.rotary_emb(latent, positions.unsqueeze(0))
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        for index, layer in enumerate(self.model.layers):
            attention = layer.self_attn
            shape = (1, delta, -1, attention.head_dim)
            key = attention.k_norm(attention.k_proj(latent).view(shape)).transpose(1, 2)
            value = attention.v_proj(latent).view(shape).transpose(1, 2)
            key = key * cos + _rotate_half(key) * sin
            storage = self.layer_storage[index]
            storage.key[:, :, self.length : self.length + delta, :].copy_(key)
            storage.value[:, :, self.length : self.length + delta, :].copy_(value)
        if self.features is None:
            self.features = torch.empty(
                (1, self.capacity, latent.shape[-1]), dtype=latent.dtype, device=self.device
            )
        self.features[:, self.length : self.length + delta].copy_(latent)
        self.positions[self.length : self.length + delta].copy_(positions)
        self.length += delta

    def gather(self, selection: Selection) -> tuple[LayerContext, ...]:
        if len(selection.positions_by_layer) != len(self.layer_storage):
            raise ValueError("selection layer count does not match DFlash checkpoint")
        layers: list[LayerContext] = []
        for storage, position_list in zip(self.layer_storage, selection.positions_by_layer):
            if not position_list:
                raise ValueError("each draft layer must retain at least one context position")
            positions = torch.tensor(position_list, dtype=torch.long, device=self.device)
            if positions.min().item() < 0 or positions.max().item() >= self.length:
                raise ValueError("selection references an unprocessed logical position")
            if positions.unique().numel() != positions.numel() or not torch.all(positions[1:] > positions[:-1]):
                raise ValueError("selected positions must be sorted and unique")
            layers.append(
                LayerContext(
                    key=storage.key.index_select(2, positions),
                    value=storage.value.index_select(2, positions),
                    positions=positions,
                )
            )
        return tuple(layers)

    def feature_rows(self, positions: Sequence[int]) -> torch.Tensor:
        if self.features is None:
            raise ValueError("draft bank has no target features")
        index = torch.tensor(tuple(positions), dtype=torch.long, device=self.device)
        return self.features.index_select(1, index)

    def draft_kv_bytes(self, positions_by_layer: Sequence[Sequence[int]]) -> int:
        return sum(
            int(len(positions)) * (storage.key.shape[1] * storage.key.shape[-1] + storage.value.shape[1] * storage.value.shape[-1]) * storage.key.element_size()
            for positions, storage in zip(positions_by_layer, self.layer_storage)
        )
