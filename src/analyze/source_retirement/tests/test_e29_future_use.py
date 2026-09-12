from __future__ import annotations

import pytest

from src.analyze.source_retirement.e29_future_use import (
    active_units,
    checkpoint_metrics,
    chunk_ranges,
    future_use_at,
    projected_work,
)


def test_chunk_ranges_cover_source_without_overlap() -> None:
    assert chunk_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_future_use_excludes_prefix_queries_and_computes_mean() -> None:
    trace = [[1.0, 0.0], [0.0, 2.0], [2.0, 0.0]]
    result = future_use_at(trace, 1)
    assert result["future_queries"] == 2
    assert result["cumulative"] == [2.0, 2.0]
    assert result["mean_per_query"] == [1.0, 1.0]


def test_projected_work_reports_hindsight_gain() -> None:
    trace = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    result = projected_work(trace, delta=0.5)
    assert result["active_work"] == 1.5
    assert result["retire_gain"] == 0.5


def test_metrics_validate_threshold_and_trace() -> None:
    assert active_units([0.1, 0.01], delta=0.05) == [0]
    rows = checkpoint_metrics([[1.0, 0.0], [0.0, 1.0]], checkpoints=(1,), deltas=(0.1,))
    assert rows[0]["active_units"] == 1
    with pytest.raises(ValueError):
        projected_work([[1.0], [float("nan")]], delta=0.1)
