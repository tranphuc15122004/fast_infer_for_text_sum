"""AMR-DFlash memory, selection, compression, and training primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryConfig:
    """Limits for the draft-side working memory; live block is additional."""

    raw_budget: int = 4096
    num_slots: int = 128
    local_window: int = 128
    slot_gate_init: float = -2.0
    min_context_tokens: int = 4096

    def __post_init__(self) -> None:
        if self.raw_budget < 0 or self.num_slots < 0 or self.local_window < 0:
            raise ValueError("memory budgets must be non-negative")
        if self.local_window > self.raw_budget:
            raise ValueError("local_window must fit inside raw_budget")
        if self.min_context_tokens < 0:
            raise ValueError("min_context_tokens must be non-negative")


@dataclass
class CompressedMemory:
    values: torch.Tensor
    positions: torch.Tensor
    valid_mask: torch.Tensor


@dataclass
class CompressorState:
    weighted_values: torch.Tensor
    weight_sum: torch.Tensor
    weighted_positions: torch.Tensor
    observed_min: torch.Tensor
    observed_max: torch.Tensor


@dataclass(frozen=True)
class Acceptance:
    accepted: int
    first_mismatch: int | None


class AcceptanceSelector(nn.Module):
    """Pre-draft state-conditioned score over projected context features."""

    def __init__(self, hidden_size: int, index_dim: int = 64) -> None:
        super().__init__()
        if hidden_size < 1 or index_dim < 1:
            raise ValueError("hidden_size and index_dim must be positive")
        self.key_projection = nn.Linear(hidden_size, index_dim, bias=False)
        self.query_projection = nn.Linear(hidden_size, index_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(()))
        self.scale = index_dim**-0.5

    def forward(
        self,
        features: torch.Tensor,
        anchor: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or anchor.ndim != 2:
            raise ValueError("features must be [B,N,H] and anchor [B,H]")
        if features.shape[0] != anchor.shape[0] or features.shape[2] != anchor.shape[1]:
            raise ValueError("feature and anchor batch/hidden dimensions must match")
        query = self.query_projection(anchor)
        keys = self.key_projection(features)
        scores = torch.einsum("bd,bnd->bn", query, keys) * self.scale
        if positions is not None and positions.numel():
            if positions.shape != features.shape[:2]:
                raise ValueError("positions must match [B,N] feature dimensions")
            # A weak, learned relative-position term; positions are normalized
            # per example and depend only on already-observed context.
            relative = positions.to(dtype=scores.dtype)
            relative = relative / relative.amax(dim=1, keepdim=True).clamp_min(1)
            scores = scores + self.position_bias * relative
        return scores

    @torch.no_grad()
    def select(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        anchor: torch.Tensor,
        *,
        raw_budget: int,
        local_window: int,
    ) -> torch.Tensor:
        """Return sorted original positions, including the recent guard.

        The recent guard counts against ``raw_budget``.  Learned top-k only
        fills the remaining capacity, and stable ordering resolves score ties.
        """
        if raw_budget < 0 or local_window < 0:
            raise ValueError("raw_budget and local_window must be non-negative")
        if local_window > raw_budget:
            raise ValueError("local_window must fit inside raw_budget")
        if features.ndim != 3 or positions.shape != features.shape[:2]:
            raise ValueError("positions must match [B,N] feature dimensions")
        if anchor.shape != (features.shape[0], features.shape[2]):
            raise ValueError("anchor must have shape [B,H]")
        if positions.shape[1] > 1 and torch.any(positions[:, 1:] <= positions[:, :-1]):
            raise ValueError("positions must be strictly increasing")
        budget = min(raw_budget, features.shape[1])
        if budget == 0:
            return positions[:, :0]

        scores = self(features, anchor, positions)
        selected_rows: list[torch.Tensor] = []
        for batch_index in range(features.shape[0]):
            row_positions = positions[batch_index]
            recent_count = min(local_window, budget)
            local_indices = torch.arange(
                max(0, row_positions.numel() - recent_count),
                row_positions.numel(),
                device=positions.device,
            )
            capacity = budget - recent_count
            if capacity:
                eligible = torch.ones(
                    row_positions.numel(), dtype=torch.bool, device=positions.device
                )
                eligible[local_indices] = False
                candidates = eligible.nonzero(as_tuple=False).flatten()
                ranked = torch.argsort(
                    scores[batch_index, candidates], descending=True, stable=True
                )
                learned_indices = candidates[ranked[:capacity]]
                chosen = torch.cat((local_indices, learned_indices))
            else:
                chosen = local_indices
            chosen = torch.sort(chosen).values
            selected_rows.append(row_positions[chosen])
        return torch.stack(selected_rows, dim=0)


class ComplementaryCompressor(nn.Module):
    """Differentiable global slots with an equivalent streaming accumulator."""

    def __init__(self, hidden_size: int, num_slots: int = 128) -> None:
        super().__init__()
        if hidden_size < 1 or num_slots < 0:
            raise ValueError("hidden_size must be positive and num_slots non-negative")
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.slot_queries = nn.Parameter(torch.empty(num_slots, hidden_size))
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        if num_slots:
            nn.init.normal_(self.slot_queries, std=hidden_size**-0.5)
        self.scale = hidden_size**-0.5

    def _assign(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key_projection(features)
        scores = torch.einsum("bnh,sh->bns", keys, self.slot_queries) * self.scale
        assignment = torch.softmax(scores, dim=-1)
        values = self.value_projection(features)
        return assignment, values

    def empty_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CompressorState:
        shape = (batch_size, self.num_slots)
        return CompressorState(
            weighted_values=torch.zeros(
                (*shape, self.hidden_size), device=device, dtype=dtype
            ),
            weight_sum=torch.zeros(shape, device=device, dtype=dtype),
            weighted_positions=torch.zeros(shape, device=device, dtype=torch.float64),
            observed_min=torch.full(
                (batch_size,), torch.iinfo(torch.long).max, device=device, dtype=torch.long
            ),
            observed_max=torch.full(
                (batch_size,), -1, device=device, dtype=torch.long
            ),
        )

    @torch.no_grad()
    def update(
        self,
        state: CompressorState,
        features: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if features.ndim != 3 or positions.shape != features.shape[:2]:
            raise ValueError("features [B,N,H] and positions [B,N] are required")
        if features.shape[-1] != self.hidden_size:
            raise ValueError("feature hidden dimension does not match compressor")
        if state.weight_sum.shape[0] != features.shape[0]:
            raise ValueError("state batch dimension does not match features")
        if self.num_slots == 0 or features.shape[1] == 0:
            return
        assignment, values = self._assign(features)
        weights = assignment.transpose(1, 2)
        state.weighted_values.add_(torch.matmul(weights, values))
        state.weight_sum.add_(weights.sum(dim=-1))
        state.weighted_positions.add_(
            torch.matmul(weights.to(torch.float64), positions.to(torch.float64).unsqueeze(-1))
            .squeeze(-1)
        )
        state.observed_min.copy_(
            torch.minimum(state.observed_min, positions.min(dim=1).values)
        )
        state.observed_max.copy_(
            torch.maximum(state.observed_max, positions.max(dim=1).values)
        )

    def finalize(self, state: CompressorState) -> tuple[torch.Tensor, torch.Tensor]:
        valid = state.weight_sum > 0
        values = state.weighted_values / state.weight_sum.clamp_min(1e-12).unsqueeze(-1)
        positions = torch.where(
            valid,
            torch.round(
                state.weighted_positions / state.weight_sum.to(torch.float64).clamp_min(1e-12)
            ).to(torch.long),
            torch.zeros_like(state.weight_sum, dtype=torch.long),
        )
        if torch.any(state.observed_max >= 0):
            positions = torch.minimum(positions, state.observed_max.unsqueeze(1).clamp_min(0))
            positions = torch.maximum(positions, state.observed_min.unsqueeze(1).clamp_min(0))
        return values, positions

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> CompressedMemory:
        if features.ndim != 3 or positions.shape != features.shape[:2]:
            raise ValueError("features [B,N,H] and positions [B,N] are required")
        if features.shape[-1] != self.hidden_size:
            raise ValueError("feature hidden dimension does not match compressor")
        batch_size = features.shape[0]
        if self.num_slots == 0:
            return CompressedMemory(
                values=features.new_zeros((batch_size, 0, self.hidden_size)),
                positions=positions.new_zeros((batch_size, 0)),
                valid_mask=torch.zeros((batch_size, 0), dtype=torch.bool, device=features.device),
            )
        assignment, values = self._assign(features)
        weights = assignment.transpose(1, 2)
        weight_sum = weights.sum(dim=-1)
        slot_values = torch.matmul(weights, values) / weight_sum.clamp_min(1e-12).unsqueeze(-1)
        weighted_positions = torch.matmul(
            weights.to(torch.float64), positions.to(torch.float64).unsqueeze(-1)
        ).squeeze(-1)
        slot_positions = torch.round(
            weighted_positions / weight_sum.to(torch.float64).clamp_min(1e-12)
        ).to(torch.long)
        if positions.shape[1]:
            slot_positions = slot_positions.clamp(
                min=int(positions.min().item()), max=int(positions.max().item())
            )
        valid = weight_sum > 0
        slot_values = slot_values * valid.unsqueeze(-1).to(slot_values.dtype)
        return CompressedMemory(slot_values, slot_positions, valid)


def build_sparse_attention_mask(
    raw_positions: torch.Tensor,
    slot_positions: torch.Tensor,
    live_positions: torch.Tensor,
    *,
    query_length: int,
    slot_gate: torch.Tensor | float = -2.0,
    slot_valid_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a compact additive mask; input positions retain original RoPE ids.

    Key order is selected raw features, compressed slots, then the live block.
    All keys are visible to each DFlash query, matching the pretrained
    non-causal block semantics. Invalid slots are masked; valid slots receive
    the learned additive gate bias.
    """
    for name, value in (
        ("raw_positions", raw_positions),
        ("slot_positions", slot_positions),
        ("live_positions", live_positions),
    ):
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [B,L]")
    batch = raw_positions.shape[0]
    if slot_positions.shape[0] != batch or live_positions.shape[0] != batch:
        raise ValueError("raw, slot, and live positions must share a batch dimension")
    if query_length < 1 or live_positions.shape[1] < query_length:
        raise ValueError("query_length must be positive and fit in live positions")
    raw_count, slot_count = raw_positions.shape[1], slot_positions.shape[1]
    live_count = live_positions.shape[1]
    total_keys = raw_count + slot_count + live_count
    minimum = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, query_length, total_keys), device=raw_positions.device, dtype=dtype)
    if slot_count:
        if slot_valid_mask is None:
            slot_valid_mask = torch.ones_like(slot_positions, dtype=torch.bool)
        if slot_valid_mask.shape != slot_positions.shape:
            raise ValueError("slot_valid_mask must match slot_positions")
        gate = torch.as_tensor(slot_gate, device=mask.device, dtype=dtype)
        slot_bias = gate.expand(batch, slot_count)
        mask[:, :, :, raw_count : raw_count + slot_count] = slot_bias[:, None, None, :]
        invalid = ~slot_valid_mask.to(device=mask.device, dtype=torch.bool)
        if invalid.any():
            slot_mask = mask[:, :, :, raw_count : raw_count + slot_count]
            slot_mask.masked_fill_(invalid[:, None, None, :], minimum)
    return mask


def preference_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise logistic loss for accepted-prefix preferences."""
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative scores must have equal shapes")
    terms = F.softplus(negative_scores - positive_scores)
    if weights is None:
        return terms.mean() if terms.numel() else terms.sum()
    if weights.shape != terms.shape:
        raise ValueError("preference weights must match score shape")
    weights = weights.to(device=terms.device, dtype=terms.dtype)
    return (terms * weights).sum() / weights.sum().clamp_min(1e-8)


def alignment_kl_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    proposal_valid: torch.Tensor,
) -> torch.Tensor:
    """Candidate-prefix KL surrogate; target distribution is detached."""
    if draft_logits.shape != target_logits.shape or draft_logits.ndim != 3:
        raise ValueError("draft and target logits must have equal [B,P,V] shapes")
    if proposal_valid.shape != draft_logits.shape[:2]:
        raise ValueError("proposal_valid must match [B,P]")
    log_q = F.log_softmax(draft_logits.float(), dim=-1)
    p_target = F.softmax(target_logits.detach().float(), dim=-1)
    per_position = F.kl_div(log_q, p_target, reduction="none").sum(dim=-1)
    early = torch.arange(
        1, draft_logits.shape[1] + 1, device=draft_logits.device, dtype=per_position.dtype
    ).reciprocal()
    weights = proposal_valid.to(per_position.dtype) * early.unsqueeze(0)
    return (per_position * weights).sum() / weights.sum().clamp_min(1e-8)


def greedy_acceptance(proposals: torch.Tensor, target_choices: torch.Tensor) -> Acceptance:
    """Count equal proposals up to the first greedy-target mismatch."""
    if proposals.shape != target_choices.shape or proposals.ndim != 2:
        raise ValueError("proposal and target-choice tensors must have equal [B,P] shapes")
    if proposals.shape[0] != 1:
        raise ValueError("AMR-DFlash V0 acceptance accounting supports batch size 1")
    mismatches = proposals[0].ne(target_choices[0]).nonzero(as_tuple=False).flatten()
    if mismatches.numel():
        first = int(mismatches[0].item())
        return Acceptance(first, first)
    return Acceptance(proposals.shape[1], None)
