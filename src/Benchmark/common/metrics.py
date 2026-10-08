"""Expose the shared LongBench metric functions to Benchmark modules."""

from common.metrics import (
    add_semantic,
    aggregate_semantic,
    aggregate_speed,
    aggregate_speculative,
)

__all__ = [
    "add_semantic",
    "aggregate_semantic",
    "aggregate_speed",
    "aggregate_speculative",
]
