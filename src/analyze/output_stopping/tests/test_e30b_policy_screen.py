from __future__ import annotations

from src.analyze.output_stopping.e30b_policy_screen import (
    _prefix_features,
    _source_prefix_features,
    fixed_index,
    oracle_index,
    selected_summary,
)


def _row(index: int, tokens: int, delta: float = 0.0, cost: float = 10.0) -> dict:
    return {
        "prefix_index": index,
        "prefix_token_count": tokens,
        "delta_rougeL": delta,
        "delta_bertscore_f1": delta,
        "projected_cost_ms": cost,
        "rougeL": 0.5,
        "bertscore_f1": 0.9,
    }


def test_fixed_and_oracle_indices() -> None:
    rows = [_row(0, 40, 0.10), _row(1, 90, 0.03), _row(2, 140, 0.0)]
    assert fixed_index(rows, "96") == 1
    assert fixed_index(rows, "64") == 0
    assert oracle_index(rows, 0.02) == 2


def test_selected_summary_counts_quality_risk() -> None:
    docs = {"a": [_row(0, 40, 0.0, 5.0)], "b": [_row(0, 40, 0.03, 5.0)]}
    result = selected_summary(docs, {"a": 0, "b": 0}, epsilon=0.02)
    assert result["documents"] == 2
    assert result["violating_documents"] == 1
    assert result["risk_rate"] == 0.5


def test_prefix_features_are_runtime_only() -> None:
    features = _prefix_features("First sentence about climate. Second sentence repeats climate.", 1)
    assert features["prefix_token_count"] > 0
    assert 0.0 <= features["self_redundancy"] <= 1.0
    assert 0.0 <= features["lexical_novelty"] <= 1.0


def test_source_prefix_features_have_coverage_and_gain() -> None:
    features = _source_prefix_features(
        "Climate change affects military bases. Procurement rules affect contracts.",
        ["Climate change affects bases.", "Procurement rules affect contracts."],
        1,
    )
    assert features["source_sentence_count"] == 2
    assert features["source_prefix_coverage"] >= 0.0
    assert features["source_coverage_gain"] >= 0.0
