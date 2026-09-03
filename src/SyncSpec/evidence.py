"""Exact lexical source evidence and bounded source-memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Iterable

import torch


class SourceNgramIndex:
    """Source-only n-gram index used to rerank existing candidate IDs."""

    def __init__(self, source_ids: Iterable[int], min_n: int = 2, max_n: int = 6):
        self.source_ids = tuple(int(x) for x in source_ids)
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        if self.min_n < 1 or self.min_n > self.max_n:
            raise ValueError("invalid n-gram range")
        self.counts: dict[int, Counter[tuple[int, ...]]] = {}
        self.positions: dict[int, defaultdict[tuple[int, ...], list[int]]] = {}
        for n in range(self.min_n, self.max_n + 1):
            counter: Counter[tuple[int, ...]] = Counter()
            positions: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)
            for start in range(max(0, len(self.source_ids) - n + 1)):
                gram = self.source_ids[start : start + n]
                counter[gram] += 1
                positions[gram].append(start)
            self.counts[n] = counter
            self.positions[n] = positions

    def _longest(self, history: tuple[int, ...], candidate: int) -> tuple[int, tuple[int, ...]]:
        best_n = 0
        best_gram: tuple[int, ...] = ()
        for n in range(self.min_n, self.max_n + 1):
            if len(history) + 1 < n:
                continue
            gram = (history + (candidate,))[-n:]
            if gram in self.counts[n] and n > best_n:
                best_n, best_gram = n, gram
        return best_n, best_gram

    def features(self, history: Iterable[int], candidate_ids: torch.Tensor) -> torch.Tensor:
        """Return [candidate, 6] lexical evidence without changing IDs."""
        hist = tuple(int(x) for x in history)
        flat = candidate_ids.detach().reshape(-1).tolist()
        total = max(1, len(self.source_ids))
        rows: list[list[float]] = []
        for candidate in flat:
            n, gram = self._longest(hist, int(candidate))
            count = self.counts.get(n, Counter()).get(gram, 0) if n else 0
            positions = self.positions.get(n, {}).get(gram, []) if n else []
            continuity = 1.0 if positions else 0.0
            location = (min(positions) / total) if positions else 0.0
            token_count = self.source_ids.count(int(candidate))
            # The last two slots are cheap continuity/location proxies; they
            # remain bounded for stable calibration and no external NER is used.
            rows.append([
                float(n),
                float(count),
                continuity,
                float(count) / max(1.0, len(self.source_ids)),
                float(location),
                float(token_count > 0),
            ])
        return torch.tensor(rows, dtype=torch.float32, device=candidate_ids.device).reshape(
            *candidate_ids.shape, 6
        )

    def match_score(self, history: Iterable[int], candidate_id: int) -> float:
        return float(self.features(history, torch.tensor([candidate_id]))[0, 0].item())


@dataclass
class RetrievalResult:
    descriptors: torch.Tensor
    indices: torch.Tensor
    status: str


@dataclass
class SourceMemoryBank:
    descriptors: torch.Tensor
    chunk_offsets: tuple[tuple[int, int], ...]
    source_ids: torch.Tensor
    top_r: int = 8

    @classmethod
    def from_source(
        cls,
        source_ids: torch.Tensor,
        embeddings: torch.Tensor | None = None,
        chunk_size: int = 128,
        top_r: int = 8,
    ) -> "SourceMemoryBank":
        ids = source_ids.detach().to(dtype=torch.long).flatten()
        if ids.numel() == 0:
            descriptors = torch.empty((0, 0), dtype=torch.float32, device=ids.device)
            return cls(descriptors, (), ids, top_r)
        if embeddings is None:
            values = ids.to(torch.float32).unsqueeze(-1)
        else:
            values = embeddings.detach().to(torch.float32)
            if values.shape[0] != ids.numel():
                raise ValueError("embeddings and source_ids must have equal length")
        chunks: list[torch.Tensor] = []
        offsets: list[tuple[int, int]] = []
        for start in range(0, ids.numel(), chunk_size):
            end = min(ids.numel(), start + chunk_size)
            chunks.append(values[start:end].mean(dim=0))
            offsets.append((start, end))
        return cls(torch.stack(chunks), tuple(offsets), ids, top_r)

    def retrieve(self, query: torch.Tensor, top_r: int | None = None) -> RetrievalResult:
        q = query.detach().to(torch.float32).flatten()
        if self.descriptors.numel() == 0:
            return RetrievalResult(
                descriptors=torch.zeros((1, q.numel()), dtype=torch.float32, device=q.device),
                indices=torch.full((1,), -1, dtype=torch.long, device=q.device),
                status="fallback_anchor",
            )
        descriptors = self.descriptors.to(q.device)
        if descriptors.shape[-1] < q.numel():
            descriptors = torch.nn.functional.pad(descriptors, (0, q.numel() - descriptors.shape[-1]))
        elif descriptors.shape[-1] > q.numel():
            descriptors = descriptors[:, : q.numel()]
        scores = descriptors @ q
        count = min(int(top_r or self.top_r), descriptors.shape[0])
        indices = torch.argsort(scores, descending=True, stable=True)[:count]
        return RetrievalResult(descriptors[indices], indices, "ok")
