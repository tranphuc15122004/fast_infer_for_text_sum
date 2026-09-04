"""P4 context×draft-depth interaction analysis."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .metrics import fit_context_depth_interaction


def analyze_interaction(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 500,
    seed: int = 42,
    min_documents: int = 5,
) -> dict[str, Any]:
    """Fit and gate the context-induced suffix-decay interaction."""

    return fit_context_depth_interaction(
        rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        min_documents=min_documents,
    )
