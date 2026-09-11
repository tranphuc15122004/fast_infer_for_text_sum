"""Small numerical helpers for validating KV-cache hidden states."""

from __future__ import annotations

from typing import Any

import torch


def compare_hidden_states(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> dict[str, Any]:
    """Compare two hidden-state tensors and return a JSON-friendly report.

    The comparison is performed in float32 so that the tolerance describes
    the numerical difference between full and incremental execution rather
    than the storage dtype (usually BF16) of the captured states.
    """
    reference = torch.as_tensor(reference).detach().float()
    candidate = torch.as_tensor(candidate).detach().float()
    report: dict[str, Any] = {
        "shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "atol": float(atol),
        "rtol": float(rtol),
    }
    if reference.shape != candidate.shape:
        report.update(
            {
                "allclose": False,
                "numel": int(reference.numel()),
                "max_abs_error": None,
                "mean_abs_error": None,
                "rmse": None,
            }
        )
        return report

    finite = torch.isfinite(reference).all() and torch.isfinite(candidate).all()
    if not bool(finite):
        report.update(
            {
                "allclose": False,
                "numel": int(reference.numel()),
                "max_abs_error": float("inf"),
                "mean_abs_error": float("inf"),
                "rmse": float("inf"),
            }
        )
        return report

    difference = (reference - candidate).abs()
    report.update(
        {
            "allclose": bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
            "numel": int(reference.numel()),
            "max_abs_error": float(difference.max().item()) if difference.numel() else 0.0,
            "mean_abs_error": float(difference.mean().item()) if difference.numel() else 0.0,
            "rmse": float(torch.mean((reference - candidate) ** 2).sqrt().item())
            if difference.numel()
            else 0.0,
        }
    )
    return report
