"""Serializable inference records for SyncSpec benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class InferenceResult:
    token_ids: torch.Tensor
    batch_size: int = 1
    status: str = "ok"
    rounds: int = 0
    committed_tokens: int = 0
    accepted_lengths: list[int] = field(default_factory=list)
    budgets: list[dict[str, int]] = field(default_factory=list)
    fallback_rounds: int = 0
    timing_ms: dict[str, float] = field(default_factory=dict)
    survival_features: list[list[float]] = field(default_factory=list)
    survival_labels: list[float] = field(default_factory=list)
    runtime_feedback: dict[str, float | int] = field(default_factory=dict)
    error: str | None = None

    def to_record(self, input_id: str = "0", method: str = "syncspec", **extra: Any) -> dict[str, Any]:
        record = {
            "method": method,
            "input_id": input_id,
            "batch_size": int(self.batch_size),
            "status": self.status,
            "token_ids": self.token_ids.detach().cpu().tolist(),
            "output_tokens": int(self.token_ids.numel()),
            "rounds": self.rounds,
            "committed_tokens": self.committed_tokens,
            "accepted_lengths": self.accepted_lengths,
            "budgets": self.budgets,
            "fallback_rounds": self.fallback_rounds,
            "timing_ms": self.timing_ms,
            "runtime_feedback": self.runtime_feedback,
            "error": self.error,
        }
        record.update(extra)
        return record
