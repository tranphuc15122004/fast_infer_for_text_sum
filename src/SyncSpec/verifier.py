"""Exact greedy/stochastic verification and a small KV transaction helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass
class VerificationResult:
    committed_ids: torch.Tensor
    accepted_length: int
    correction_token_id: int | None = None
    residual_probs: torch.Tensor | None = None
    rejected_position: int | None = None
    # Optional opaque target transaction payload.  The Transformers adapter
    # fills these fields so an all-accepted block can commit the already
    # computed cache without running a duplicate target forward.  Pure
    # verifier callers and synthetic adapters leave them unset.
    transaction_cache: Any | None = None
    transaction_next_logits: torch.Tensor | None = None
    transaction_last_hidden: torch.Tensor | None = None
    transaction_hidden_states: torch.Tensor | None = None
    transaction_proposal_length: int | None = None
    transaction_base_length: int | None = None


def greedy_verify(proposals: torch.Tensor, target_logits: torch.Tensor) -> VerificationResult:
    proposals = proposals.to(torch.long).flatten()
    if target_logits.ndim != 2 or target_logits.shape[0] < proposals.numel():
        raise ValueError("target_logits must have one row per proposal")
    target_ids = target_logits[: proposals.numel()].argmax(dim=-1)
    matches = target_ids.eq(proposals)
    mismatch = (~matches).nonzero(as_tuple=False)
    if mismatch.numel() == 0:
        return VerificationResult(proposals.clone(), int(proposals.numel()))
    index = int(mismatch[0].item())
    correction = int(target_ids[index].item())
    committed = torch.cat([proposals[:index], target_ids[index : index + 1]])
    return VerificationResult(committed, index, correction, rejected_position=index)


def _normalise_distribution(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(0.0)
    denom = values.sum(dim=-1, keepdim=True)
    return values / denom.clamp_min(torch.finfo(values.dtype).tiny)


def stochastic_verify(
    proposals: torch.Tensor,
    target_probs: torch.Tensor,
    proposal_probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> VerificationResult:
    proposals = proposals.to(torch.long).flatten()
    p = _normalise_distribution(target_probs[: proposals.numel()])
    q = _normalise_distribution(proposal_probs[: proposals.numel()])
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError("target_probs and proposal_probs must both be [K,V]")
    committed: list[int] = []
    for j, token in enumerate(proposals.tolist()):
        acceptance = min(1.0, float(p[j, token].item()) / max(float(q[j, token].item()), 1e-12))
        draw = float(torch.rand((), device=p.device, generator=generator).item())
        if draw <= acceptance:
            committed.append(int(token))
            continue
        residual = (p[j] - q[j]).clamp_min(0.0)
        residual = _normalise_distribution(residual.unsqueeze(0))[0]
        correction = int(torch.multinomial(residual, 1, generator=generator).item())
        committed.append(correction)
        return VerificationResult(
            committed_ids=torch.tensor(committed, dtype=torch.long, device=proposals.device),
            accepted_length=j,
            correction_token_id=correction,
            residual_probs=residual,
            rejected_position=j,
        )
    return VerificationResult(
        committed_ids=torch.tensor(committed, dtype=torch.long, device=proposals.device),
        accepted_length=len(committed),
        residual_probs=None,
    )


class KVTransaction:
    """Reference transaction semantics; model adapters provide real cache ops."""

    def __init__(self, committed_ids: Iterable[int] = ()):
        self.committed_ids = [int(x) for x in committed_ids]
        self._uncommitted: list[int] = []

    def begin(self) -> int:
        self._uncommitted = []
        return len(self.committed_ids)

    def append_uncommitted(self, token_ids: Iterable[int]) -> None:
        self._uncommitted.extend(int(x) for x in token_ids)

    def rollback(self, snapshot: int) -> None:
        self.committed_ids = self.committed_ids[:snapshot]
        self._uncommitted = []

    def commit(self) -> None:
        self.committed_ids.extend(self._uncommitted)
        self._uncommitted = []
