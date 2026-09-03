"""Tiny deterministic adapters used by CPU and CUDA contract smoke tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .verifier import VerificationResult, greedy_verify, stochastic_verify


@dataclass
class SyntheticTargetState:
    source_ids: torch.Tensor
    generated: list[int] = field(default_factory=list)


class SyntheticTarget:
    def __init__(self, vocab_size: int = 64, eos_token_id: int | None = None, device: str = "cpu"):
        self.vocab_size = int(vocab_size)
        self.eos_token_id = eos_token_id
        self.device = torch.device(device)

    def prefill(self, source_ids: torch.Tensor) -> SyntheticTargetState:
        return SyntheticTargetState(source_ids.detach().to(self.device).flatten().clone())

    def _next_id(self, state: SyntheticTargetState) -> int:
        last = state.generated[-1] if state.generated else int(state.source_ids[-1].item())
        return (last + 1) % (self.vocab_size - 1 if self.eos_token_id is not None else self.vocab_size)

    def next_logits(self, state: SyntheticTargetState) -> torch.Tensor:
        logits = torch.full((self.vocab_size,), -20.0, device=self.device)
        logits[self._next_id(state)] = 20.0
        return logits

    def logits_for_proposals(self, state: SyntheticTargetState, proposals: torch.Tensor) -> torch.Tensor:
        temp = SyntheticTargetState(state.source_ids, list(state.generated))
        rows = []
        for proposal in proposals.flatten().tolist():
            rows.append(self.next_logits(temp))
            # Verification logits after position j are conditioned on the
            # actual proposed prefix, exactly like a causal target forward.
            temp.generated.append(int(proposal))
        return torch.stack(rows) if rows else torch.empty((0, self.vocab_size), device=self.device)

    def verify(self, state: SyntheticTargetState, proposals: torch.Tensor, **kwargs) -> VerificationResult:
        logits = self.logits_for_proposals(state, proposals)
        if kwargs.get("stochastic", False):
            probs = logits.softmax(-1)
            return stochastic_verify(proposals, probs, kwargs["proposal_probs"], kwargs.get("generator"))
        return greedy_verify(proposals, logits)

    def commit(self, state: SyntheticTargetState, result: VerificationResult) -> None:
        state.generated.extend(int(x) for x in result.committed_ids.tolist())

    def generate_greedy(self, source_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        state = self.prefill(source_ids)
        for _ in range(max_new_tokens):
            token = self.next_logits(state).argmax().reshape(1)
            state.generated.append(int(token.item()))
            if self.eos_token_id is not None and int(token.item()) == self.eos_token_id:
                break
        return torch.tensor(state.generated, dtype=torch.long, device=self.device)


@dataclass
class DraftOutput:
    candidate_ids: torch.Tensor
    candidate_logits: torch.Tensor
    hidden: torch.Tensor


class SyntheticDrafter:
    def __init__(self, target: SyntheticTarget, top_m: int = 4, hidden_size: int = 16):
        self.target = target
        self.top_m = int(top_m)
        self.hidden_size = int(hidden_size)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

    def draft(self, state: SyntheticTargetState, kd: int, **_) -> DraftOutput:
        rows: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        hidden: list[torch.Tensor] = []
        temp = SyntheticTargetState(state.source_ids, list(state.generated))
        for j in range(kd):
            correct = self.target._next_id(temp)
            candidates = torch.tensor(
                [correct] + [((correct + i + 1) % self.target.vocab_size) for i in range(self.top_m - 1)],
                dtype=torch.long,
                device=self.target.device,
            )
            row = torch.full((self.top_m,), -10.0, device=self.target.device)
            row[0] = 10.0
            rows.append(candidates)
            logits.append(row)
            vector = torch.zeros(self.hidden_size, device=self.target.device)
            vector[j % self.hidden_size] = 1.0
            hidden.append(vector)
            temp.generated.append(correct)
        return DraftOutput(torch.stack(rows), torch.stack(logits), torch.stack(hidden))
