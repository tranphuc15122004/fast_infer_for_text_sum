"""Lightweight uncertainty summaries for DFlash candidate distributions."""

from __future__ import annotations

import math
from typing import Any

import torch


def summarize_draft_logits(
    logits: torch.Tensor,
    *,
    include_entropy: bool = True,
    cuda_end_event: Any = None,
) -> dict[str, list[float]]:
    """Return per-candidate entropy and top-two confidence summaries.

    `logits` must contain one block with shape [1, candidates, vocabulary].
    Calculations use FP32 regardless of the model's inference dtype. Only
    scalar summaries are copied to the host; vocabulary distributions are
    never retained. When `include_entropy=False`, full-vocabulary entropy
    materialization is skipped and only top-2 margin/confidence summaries
    are computed via `topk` + `logsumexp` in a single host transfer.
    """

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [1, candidates, vocabulary]")
    if logits.shape[0] != 1 or logits.shape[1] < 1 or logits.shape[2] < 2:
        raise ValueError("logits must include one nonempty candidate block and at least two tokens")
    detached = logits.detach()
    top_raw, top_indices = torch.topk(detached, k=2, dim=-1)
    top_logits = top_raw.float()
    values = detached.float()
    finite_row = torch.isfinite(values).all(dim=-1).to(torch.float32)[0]

    if include_entropy:
        log_probabilities = torch.log_softmax(values, dim=-1)
        probabilities = log_probabilities.exp()
        raw_entropy = -(probabilities * log_probabilities).sum(dim=-1)
        normalized_entropy = raw_entropy / math.log(int(values.shape[-1]))
        top_values = log_probabilities.gather(-1, top_indices).exp()
    else:
        lse = torch.logsumexp(values, dim=-1, keepdim=True)
        top_values = (top_logits - lse).exp()
        zeros = torch.zeros_like(finite_row)
        raw_entropy = zeros.unsqueeze(0)
        normalized_entropy = zeros.unsqueeze(0)

    prob_margin = top_values[0, :, 0] - top_values[0, :, 1]
    logit_margin = top_logits[0, :, 0] - top_logits[0, :, 1]
    # Fuse all per-candidate [K] vectors plus the finiteness flag into a single
    # [6, K] host transfer so each call costs 1 device-to-host sync total.
    stacked = torch.stack(
        (
            raw_entropy[0],
            normalized_entropy[0],
            top_values[0, :, 0],
            prob_margin,
            logit_margin,
            finite_row,
        ),
        dim=0,
    )
    if cuda_end_event is not None:
        cuda_end_event.record()
    rows = stacked.cpu().tolist()
    if any(flag < 0.5 for flag in rows[5]):
        raise ValueError("logits contain non-finite values")
    keys = ("raw_entropy", "normalized_entropy", "confidence", "probability_margin", "logit_margin")
    output = {key: row for key, row in zip(keys, rows[:5])}
    return {key: [float(item) for item in row] for key, row in output.items()}

