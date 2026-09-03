"""Top-M source-coherent selector and normalized proposal distribution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .evidence import SourceNgramIndex


@dataclass
class SelectionOutput:
    token_ids: torch.Tensor
    q: torch.Tensor
    scores: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_logits: torch.Tensor
    gates: torch.Tensor
    ngram_features: torch.Tensor


class SourceCoherentSelector(nn.Module):
    """Small sequential lattice selector from the v1.1 contract."""

    def __init__(
        self, hidden_size: int, rank: int = 128, ngram_dim: int = 6,
        temperature: float = 1.0, vocab_size: int | None = None,
    ):
        super().__init__()
        if hidden_size <= 0 or rank <= 0:
            raise ValueError("hidden_size and rank must be positive")
        if vocab_size is None:
            # Callers serving a real target pass its exact vocabulary size.
            # The bounded default keeps standalone CPU construction cheap.
            vocab_size = 65536
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = int(vocab_size)
        self.left = nn.Linear(hidden_size, rank, bias=False)
        self.right = nn.Linear(hidden_size, rank, bias=False)
        self.predecessor_embedding = nn.Embedding(self.vocab_size, rank)
        self.successor_embedding = nn.Embedding(self.vocab_size, rank)
        self.ngram_score = nn.Linear(ngram_dim, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size + ngram_dim, max(16, rank // 4)),
            nn.SiLU(),
            nn.Linear(max(16, rank // 4), 1),
        )
        self.temperature = float(temperature)

    def _token_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(torch.long)
        if (token_ids < 0).any():
            raise ValueError("selector token IDs must be non-negative")
        # Exact target vocabularies use the identity mapping.  Modulo keeps
        # legacy/standalone callers bounded without ever changing the
        # candidate IDs returned to the verifier.
        return token_ids.remainder(self.vocab_size)

    def select(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_logits: torch.Tensor,
        history: list[int] | tuple[int, ...],
        source_index: SourceNgramIndex,
        stochastic: bool = False,
        generator: torch.Generator | None = None,
        target_ids: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
    ) -> SelectionOutput:
        if hidden.ndim != 2 or candidate_ids.ndim != 2 or candidate_logits.shape != candidate_ids.shape:
            raise ValueError("hidden=[K,D], candidate_ids/logits=[K,M] are required")
        selected_tokens: list[torch.Tensor] = []
        selected_hidden: list[torch.Tensor] = []
        all_q: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_gates: list[torch.Tensor] = []
        all_features: list[torch.Tensor] = []
        compute_dtype = self.left.weight.dtype
        prev = torch.zeros(hidden.shape[-1], dtype=compute_dtype, device=hidden.device)
        running_history = list(history)
        teacher_forcing = float(max(0.0, min(1.0, teacher_forcing)))
        if target_ids is not None:
            target_ids = target_ids.to(hidden.device).flatten().to(torch.long)
            if target_ids.numel() != hidden.shape[0]:
                raise ValueError("target_ids must have one token per candidate position")
        for j in range(hidden.shape[0]):
            h = hidden[j].to(compute_dtype)
            logits = candidate_logits[j].to(compute_dtype)
            feats = source_index.features(running_history, candidate_ids[j]).to(compute_dtype)
            coherence = torch.zeros_like(logits)
            predecessor = self.left(h)
            if j > 0:
                predecessor = predecessor + self.right(prev)
            if running_history:
                predecessor = predecessor + self.predecessor_embedding(
                    self._token_indices(torch.tensor(
                        [running_history[-1]], device=hidden.device,
                    ))
                )[0].to(compute_dtype)
            candidate_context = self.successor_embedding(
                self._token_indices(candidate_ids[j]),
            ).to(compute_dtype)
            coherence = (candidate_context * predecessor).sum(dim=-1)
            gate = torch.sigmoid(self.gate(torch.cat([h, feats.mean(dim=0)], dim=0))).reshape(())
            scores = logits + coherence + gate * self.ngram_score(feats).squeeze(-1)
            q = torch.softmax(scores / max(self.temperature, 1e-5), dim=-1)
            if stochastic:
                choice = torch.multinomial(q, 1, generator=generator).reshape(())
            else:
                choice = torch.argmax(q)
            token = candidate_ids[j, choice]
            selected_tokens.append(token)
            selected_hidden.append(h)
            all_q.append(q)
            all_scores.append(scores)
            all_gates.append(gate)
            all_features.append(feats)
            prev = h
            next_token = int(token.item())
            if target_ids is not None and j + 1 < hidden.shape[0]:
                use_teacher = teacher_forcing >= 1.0
                if 0.0 < teacher_forcing < 1.0:
                    random_value = torch.rand((), device=hidden.device, generator=generator)
                    use_teacher = bool(random_value.item() < teacher_forcing)
                if use_teacher:
                    next_token = int(target_ids[j].item())
            running_history.append(next_token)
        return SelectionOutput(
            token_ids=torch.stack(selected_tokens),
            q=torch.stack(all_q),
            scores=torch.stack(all_scores),
            candidate_ids=candidate_ids,
            candidate_logits=candidate_logits,
            gates=torch.stack(all_gates),
            ngram_features=torch.stack(all_features),
        )
