"""Prefix survival hazard head and cumulative-survival calibration labels."""

from __future__ import annotations

import torch
from torch import nn


def survival_from_hazard(hazard: torch.Tensor) -> torch.Tensor:
    hazard = hazard.clamp(0.0, 1.0)
    return torch.cumprod(1.0 - hazard, dim=-1)


def survival_labels(accepted_length: int, length: int, soft: torch.Tensor | None = None) -> torch.Tensor:
    if accepted_length < 0 or length < 0:
        raise ValueError("accepted_length and length must be non-negative")
    if soft is not None:
        if soft.numel() != length:
            raise ValueError("soft survival length mismatch")
        return soft.to(torch.float32)
    return (torch.arange(length) < accepted_length).to(torch.float32)


class SurvivalHead(nn.Module):
    """Predict discrete hazards; :meth:`survival` returns cumulative prefix survival."""
    def __init__(self, feature_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features).squeeze(-1))

    def survival(self, features: torch.Tensor) -> torch.Tensor:
        return survival_from_hazard(self(features))
