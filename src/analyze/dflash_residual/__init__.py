"""Analysis tools for DFlash/DFlash2 residual candidate headroom."""

from .metrics import (
    fit_context_depth_interaction,
    oracle_prefix_length,
    prefix_match_length,
    recall_at_k,
    summarize_headroom,
    summarize_p1,
    survival,
)
from .schema import normalize_trace_row, validate_trace_row

__all__ = [
    "fit_context_depth_interaction",
    "normalize_trace_row",
    "oracle_prefix_length",
    "prefix_match_length",
    "recall_at_k",
    "summarize_headroom",
    "summarize_p1",
    "survival",
    "validate_trace_row",
]
