from __future__ import annotations

import math

import pytest

from src.analyze.groundsync.core import (
    accepted_prefix_length,
    aggregate_source_mass,
    bootstrap_mean_ci,
    document_level_mean,
    finite_metric,
    grounding_horizon,
    js_divergence,
    lag_similarity,
    persistence_summary,
    policy_k,
    segment_lengths,
    stable_sigmoid,
)


def test_js_divergence_is_symmetric_and_bounded() -> None:
    p = [0.8, 0.2, 0.0]
    q = [0.1, 0.2, 0.7]
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))
    assert 0.0 <= js_divergence(p, q) <= 1.0
    assert js_divergence(p, p) == pytest.approx(0.0)


def test_js_divergence_rejects_invalid_distributions() -> None:
    with pytest.raises(ValueError, match="same length"):
        js_divergence([1.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        js_divergence([-1.0, 2.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="positive"):
        js_divergence([0.0, 0.0], [0.5, 0.5])


def test_aggregate_source_mass_drops_sink_and_normalizes_chunks() -> None:
    mass = [10.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    result = aggregate_source_mass(mass, chunk_size=2, skip_tokens=2)
    assert result == pytest.approx([5.0 / 14.0, 9.0 / 14.0])
    assert sum(result) == pytest.approx(1.0)


def test_aggregate_source_mass_uses_positional_prior() -> None:
    result = aggregate_source_mass(
        [2.0, 2.0], chunk_size=1, positional_prior=[2.0, 1.0]
    )
    assert result == pytest.approx([1.0 / 3.0, 2.0 / 3.0])


def test_lag_similarity_compares_adjacent_and_null_traces() -> None:
    trace = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    result = lag_similarity(trace, [1, 2])
    assert result["1"]["count"] == 2
    assert result["1"]["mean"] > result["2"]["mean"]


def test_segment_lengths_and_horizon_find_change_point() -> None:
    trace = [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.0, 1.0]]
    assert segment_lengths(trace, threshold=0.1) == [2, 2]
    assert grounding_horizon(trace, start=0, threshold=0.1) == 2
    assert grounding_horizon(trace, start=2, threshold=0.1) is None


def test_persistence_summary_reports_segments_and_adjacent_similarity() -> None:
    trace = [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.0, 1.0]]
    result = persistence_summary(trace, threshold=0.1, lags=[1, 2])
    assert result["segment_count"] == 2
    assert result["median_segment_length"] == pytest.approx(2.0)
    assert result["lag_similarity"]["1"]["count"] == 3


def test_accepted_prefix_length_handles_full_and_first_rejection() -> None:
    assert accepted_prefix_length([1, 2, 3], [1, 2, 3, 4]) == 3
    assert accepted_prefix_length([9, 2], [1, 2]) == 0
    assert accepted_prefix_length([1, 8, 3], [1, 2, 3]) == 1


def test_document_level_mean_does_not_weight_long_documents_twice() -> None:
    rows = [
        {"document_id": "a", "value": 1.0},
        {"document_id": "a", "value": 3.0},
        {"document_id": "b", "value": 10.0},
    ]
    assert document_level_mean(rows, "value") == pytest.approx(6.0)


def test_policy_k_is_clipped_and_never_zero() -> None:
    assert policy_k(0, max_k=8) == 1
    assert policy_k(20, max_k=8) == 8
    assert policy_k(None, max_k=8, fallback=4) == 4


def test_finite_metric_rejects_non_finite_values() -> None:
    assert finite_metric(1.25) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="finite"):
        finite_metric(math.inf)
    with pytest.raises(ValueError, match="finite"):
        finite_metric(float("nan"))


def test_stable_sigmoid_handles_large_values() -> None:
    assert stable_sigmoid(1000.0) == pytest.approx(1.0)
    assert stable_sigmoid(-1000.0) == pytest.approx(0.0)


def test_bootstrap_mean_ci_is_deterministic_and_finite() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], seed=7, samples=200)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], seed=7, samples=200)
    assert first == second
    assert first["mean"] == pytest.approx(2.5)
    assert first["low"] <= first["mean"] <= first["high"]
