"""Lightweight uncertainty summaries for DFlash candidate distributions."""

from __future__ import annotations

import math
from typing import Any

import torch


def summarize_draft_logits(logits: torch.Tensor) -> dict[str, list[float]]:
    """Return per-candidate entropy and top-two confidence summaries.

    `logits` must contain one block with shape [1, candidates, vocabulary].
    Calculations use FP32 regardless of the model's inference dtype. Only
    scalar summaries are copied to the host; vocabulary distributions are
    never retained.
    """

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [1, candidates, vocabulary]")
    if logits.shape[0] != 1 or logits.shape[1] < 1 or logits.shape[2] < 2:
        raise ValueError("logits must include one nonempty candidate block and at least two tokens")
    values = logits.detach().float()
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("logits contain non-finite values")

    log_probabilities = torch.log_softmax(values, dim=-1)
    probabilities = log_probabilities.exp()
    raw_entropy = -(probabilities * log_probabilities).sum(dim=-1)
    normalized_entropy = raw_entropy / math.log(int(values.shape[-1]))
    top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
    top_logits = values.gather(-1, top_indices)
    output = {
        "raw_entropy": raw_entropy[0].cpu().tolist(),
        "normalized_entropy": normalized_entropy[0].cpu().tolist(),
        "confidence": top_values[0, :, 0].cpu().tolist(),
        "probability_margin": (top_values[0, :, 0] - top_values[0, :, 1]).cpu().tolist(),
        "logit_margin": (top_logits[0, :, 0] - top_logits[0, :, 1]).cpu().tolist(),
    }
    return {key: [float(item) for item in row] for key, row in output.items()}
