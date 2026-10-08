"""Assemble dense, selection-only, or hybrid AMR memory for one draft block."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .core import (
    AcceptanceSelector,
    ComplementaryCompressor,
    MemoryConfig,
    build_sparse_attention_mask,
)


@dataclass
class DraftMemory:
    features: torch.Tensor
    positions: torch.Tensor
    raw_positions: torch.Tensor
    slot_positions: torch.Tensor
    slot_valid_mask: torch.Tensor
    attention_mask: torch.Tensor
    selection_latency_ms: float
    bypassed: bool


class AMRMemory(nn.Module):
    """Trainable selector/compressor wrapper over frozen DFlash projections."""

    def __init__(self, hidden_size: int, config: MemoryConfig, index_dim: int = 64) -> None:
        super().__init__()
        self.config = config
        self.selector = AcceptanceSelector(hidden_size, index_dim=index_dim)
        self.compressor = ComplementaryCompressor(hidden_size, num_slots=config.num_slots)
        self.slot_gate = nn.Parameter(torch.tensor(float(config.slot_gate_init)))

    def build(
        self,
        projected_context: torch.Tensor,
        anchor_embedding: torch.Tensor,
        live_positions: torch.Tensor,
        *,
        mode: str = "amr",
        use_cost_gate: bool = True,
        candidate_positions: torch.Tensor | None = None,
    ) -> DraftMemory:
        if projected_context.ndim != 3 or projected_context.shape[0] != 1:
            raise ValueError("AMR-DFlash V0 supports one [1,N,H] context at a time")
        batch, context_length, _ = projected_context.shape
        if anchor_embedding.shape != (batch, projected_context.shape[-1]):
            raise ValueError("anchor embedding must have shape [1,H]")
        if live_positions.ndim != 2 or live_positions.shape[0] != batch:
            raise ValueError("live_positions must have shape [1,Q]")
        if mode not in {"dense", "selection", "amr"}:
            raise ValueError("mode must be one of: dense, selection, amr")
        start = torch.cuda.Event(enable_timing=True) if projected_context.is_cuda else None
        end = torch.cuda.Event(enable_timing=True) if projected_context.is_cuda else None
        if start is not None:
            start.record()
        cpu_start = torch.cuda.synchronize if projected_context.is_cuda else None
        if cpu_start is not None:
            cpu_start(projected_context.device)
        wall_start = __import__("time").perf_counter()

        all_positions = torch.arange(
            context_length, dtype=torch.long, device=projected_context.device
        ).unsqueeze(0)
        bypassed = bool(
            mode != "dense"
            and use_cost_gate
            and context_length < self.config.min_context_tokens
        )
        effective_mode = "dense" if mode == "dense" or bypassed else mode
        if candidate_positions is not None:
            raw_positions = candidate_positions.to(
                device=projected_context.device, dtype=torch.long
            )
            if raw_positions.ndim == 1:
                raw_positions = raw_positions.unsqueeze(0)
            if raw_positions.shape[0] != batch:
                raise ValueError("candidate positions must have one row per context")
            if raw_positions.shape[1] > self.config.raw_budget:
                raise ValueError("candidate positions exceed configured raw budget")
            if raw_positions.numel() and (
                int(raw_positions.min()) < 0 or int(raw_positions.max()) >= context_length
            ):
                raise ValueError("candidate position is outside the observed prefix")
            if raw_positions.shape[1] > 1 and torch.any(
                raw_positions[:, 1:] <= raw_positions[:, :-1]
            ):
                raise ValueError("candidate positions must be sorted and unique")
        elif effective_mode == "dense":
            raw_positions = all_positions
        else:
            raw_positions = self.selector.select(
                projected_context,
                all_positions,
                anchor_embedding,
                raw_budget=self.config.raw_budget,
                local_window=self.config.local_window,
            )

        raw_features = projected_context[:, raw_positions[0], :]
        if effective_mode == "amr" and self.config.num_slots:
            compressed = self.compressor(projected_context, all_positions)
            slot_features = compressed.values
            slot_positions = compressed.positions
            slot_valid = compressed.valid_mask
        else:
            slot_features = projected_context[:, :0, :]
            slot_positions = all_positions[:, :0]
            slot_valid = torch.zeros((batch, 0), dtype=torch.bool, device=projected_context.device)
        live_count = live_positions.shape[1]
        key_features = torch.cat((raw_features, slot_features), dim=1)
        position_ids = torch.cat((raw_positions, slot_positions, live_positions), dim=1)
        mask = build_sparse_attention_mask(
            raw_positions,
            slot_positions,
            live_positions,
            query_length=live_count,
            slot_gate=self.slot_gate,
            slot_valid_mask=slot_valid,
            dtype=projected_context.dtype,
        )
        if end is not None:
            end.record()
            torch.cuda.synchronize(projected_context.device)
            latency_ms = float(start.elapsed_time(end))
        else:
            latency_ms = (__import__("time").perf_counter() - wall_start) * 1000.0
        return DraftMemory(
            features=key_features,
            positions=position_ids,
            raw_positions=raw_positions,
            slot_positions=slot_positions,
            slot_valid_mask=slot_valid,
            attention_mask=mask,
            selection_latency_ms=latency_ms,
            bypassed=bypassed,
        )
