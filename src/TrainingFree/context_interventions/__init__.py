"""Exact, auditable replay primitives for DFlash context interventions."""

from .masks import (
    build_region_positions,
    exact_budget_positions,
    matched_deletion_positions,
    remove_attention_positions,
)

__all__ = [
    "build_region_positions",
    "exact_budget_positions",
    "matched_deletion_positions",
    "remove_attention_positions",
]
