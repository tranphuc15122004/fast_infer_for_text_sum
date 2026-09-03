"""CPU-testable feature interface and matched probe for E1."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def wrong_document_indices(document_ids: Sequence[str]) -> list[int]:
    """Map each row to the first row of the next document for a negative control."""

    if not document_ids:
        raise ValueError("document_ids must not be empty")
    groups: list[list[int]] = []
    for index, document_id in enumerate(document_ids):
        if not groups or document_ids[groups[-1][0]] != document_id:
            groups.append([])
        groups[-1].append(index)
    result = [0] * len(document_ids)
    for group_index, group in enumerate(groups):
        next_group = groups[(group_index + 1) % len(groups)]
        for row_offset, row_index in enumerate(group):
            result[row_index] = next_group[min(row_offset, len(next_group) - 1)]
    return result


def required_capture_layers(layer_ids: Sequence[int], *, num_layers: int) -> list[int]:
    """Add the final decoder layer needed by R2's token-wise hidden control."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    selected = sorted({int(layer_id) for layer_id in layer_ids})
    if not selected or selected[0] < 0 or selected[-1] >= num_layers:
        raise ValueError("layer ids are outside target model")
    return sorted(set(selected) | {num_layers - 1})


def probe_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Return position-wise CE, top-k accuracy, and exact-prefix survival."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [batch,horizon,vocab] and labels [batch,horizon]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have matching batch/horizon")
    predictions = logits.argmax(dim=-1)
    top_k = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
    correct = predictions.eq(labels)
    top5_correct = top_k.eq(labels.unsqueeze(-1)).any(dim=-1)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    prefix = correct.cumprod(dim=1).float()
    return {
        "ce_by_position": ce.mean(dim=0).detach().cpu().tolist(),
        "acc1_by_position": correct.float().mean(dim=0).detach().cpu().tolist(),
        "acc5_by_position": top5_correct.float().mean(dim=0).detach().cpu().tolist(),
        "prefix_exact_by_position": prefix.mean(dim=0).detach().cpu().tolist(),
        "ce_mean": float(ce.mean().item()),
    }


def anchor_positions(
    input_length: int,
    *,
    count: int = 4,
    minimum_prefix: int = 8,
) -> list[int]:
    """Choose deterministic prefix lengths for one document."""

    if input_length <= 0:
        raise ValueError("input_length must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if minimum_prefix <= 0:
        raise ValueError("minimum_prefix must be positive")
    if input_length < minimum_prefix:
        return [input_length]
    positions = {
        max(minimum_prefix, min(input_length, round(input_length * i / count)))
        for i in range(1, count + 1)
    }
    return sorted(positions)


def _pool_token_axis(values: torch.Tensor, count: int) -> torch.Tensor:
    length = int(values.shape[0])
    if length <= count:
        return values
    chunks = []
    for index in range(count):
        start = math.floor(index * length / count)
        end = max(start + 1, math.floor((index + 1) * length / count))
        chunks.append(values[start:end].mean(dim=0))
    return torch.stack(chunks, dim=0)


def _pool_channel_axis(values: torch.Tensor, dimension: int) -> torch.Tensor:
    if dimension <= 0:
        raise ValueError("interface_dim must be positive")
    if values.shape[-1] == dimension:
        return values
    return F.adaptive_avg_pool1d(values.unsqueeze(0), dimension).squeeze(0)


def pool_sequence_to_interface(
    values: torch.Tensor,
    *,
    max_memory_tokens: int,
    interface_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map any [tokens, features] representation to fixed [M, d] memory."""

    if values.ndim != 2:
        raise ValueError("values must have shape [tokens, features]")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("values must be non-empty")
    if max_memory_tokens <= 0:
        raise ValueError("max_memory_tokens must be positive")
    pooled = _pool_token_axis(values.float(), max_memory_tokens)
    pooled = _pool_channel_axis(pooled, interface_dim)
    output = values.new_zeros((max_memory_tokens, interface_dim), dtype=torch.float32)
    mask = values.new_zeros((max_memory_tokens,), dtype=torch.float32)
    output[: pooled.shape[0]] = pooled
    mask[: pooled.shape[0]] = 1.0
    return output.to(dtype=values.dtype), mask


def pool_representation_dict(
    representations: Mapping[str, torch.Tensor],
    *,
    max_memory_tokens: int,
    interface_dim: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Pool representations while retaining each representation's valid mask."""

    features: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for name, values in representations.items():
        features[name], masks[name] = pool_sequence_to_interface(
            values,
            max_memory_tokens=max_memory_tokens,
            interface_dim=interface_dim,
        )
    return features, masks


def split_feature_rows_by_document(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split all anchors from a document into one partition."""

    if not rows:
        raise ValueError("rows must not be empty")
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= dev_fraction < 1.0:
        raise ValueError("invalid split fractions")
    if train_fraction + dev_fraction >= 1.0:
        raise ValueError("train and dev fractions must leave a test split")
    documents = sorted({str(row["document_id"]) for row in rows})
    train_count = max(1, int(len(documents) * train_fraction))
    dev_count = int(len(documents) * dev_fraction)
    if len(documents) >= 3:
        dev_count = max(1, min(dev_count, len(documents) - train_count - 1))
    train_ids = set(documents[:train_count])
    dev_ids = set(documents[train_count : train_count + dev_count])
    train = [row for row in rows if str(row["document_id"]) in train_ids]
    dev = [row for row in rows if str(row["document_id"]) in dev_ids]
    test = [
        row
        for row in rows
        if str(row["document_id"]) not in train_ids
        and str(row["document_id"]) not in dev_ids
    ]
    return train, dev, test


class MemoryBlockProbe(nn.Module):
    """Identical non-autoregressive block predictor used for every E1 variant."""

    def __init__(
        self,
        *,
        interface_dim: int,
        hidden_dim: int,
        horizon: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        if min(interface_dim, hidden_dim, horizon, vocab_size) <= 0:
            raise ValueError("probe dimensions must be positive")
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.input = nn.Linear(interface_dim, hidden_dim)
        self.hidden = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, horizon * vocab_size)

    def forward(self, memory: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError("memory must have shape [batch, memory, dimension]")
        if mask is None:
            pooled = memory.mean(dim=1)
        else:
            if mask.shape != memory.shape[:2]:
                raise ValueError("mask must match [batch, memory]")
            weights = mask.to(dtype=memory.dtype).unsqueeze(-1)
            pooled = (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        hidden = self.hidden(self.input(pooled))
        return self.output(hidden).view(-1, self.horizon, self.vocab_size)
