from __future__ import annotations

import math

from src.analyze.output_stopping.e30a_output_prefix_oracle import (
    PrefixRecord,
    fixed_prefix_index,
    is_safe_quality,
    percentile,
    projected_cost,
    summarize_fixed,
    prefix_texts,
)


def _prefix(index: int, tokens: int, rouge: float, bert: float, cost: float) -> PrefixRecord:
    return PrefixRecord(
        index=index,
        text=f"sentence {index}",
        token_count=tokens,
        rouge_l=rouge,
        bertscore_f1=bert,
        projected_cost_ms=cost,
        is_full=index == 3,
    )


def test_fixed_prefix_uses_last_sentence_before_budget() -> None:
    prefixes = [_prefix(1, 40, .8, .9, 10), _prefix(2, 90, .9, .91, 20), _prefix(3, 150, .92, .92, 30)]
    assert fixed_prefix_index(prefixes, 100) == 1


def test_prefix_texts_are_cumulative() -> None:
    prefixes = prefix_texts("First sentence. Second sentence! Third?")
    assert prefixes == ["First sentence.", "First sentence. Second sentence!", "First sentence. Second sentence! Third?"]


def test_fixed_prefix_first_sentence_fallback_is_explicit() -> None:
    prefixes = [_prefix(1, 140, .8, .9, 10), _prefix(2, 180, .9, .91, 20)]
    assert fixed_prefix_index(prefixes, 100) == 0


def test_quality_safety_uses_any_metric_violation() -> None:
    full = {"rougeL": .50, "bertscore_f1": .90}
    good = {"rougeL": .49, "bertscore_f1": .89}
    bad = {"rougeL": .47, "bertscore_f1": .89}
    assert is_safe_quality(good, full, .02)
    assert not is_safe_quality(bad, full, .02)


def test_projected_cost_scales_only_decode() -> None:
    assert projected_cost(100.0, 900.0, 30, 90) == 400.0


def test_fixed_summary_reports_empirical_risk() -> None:
    docs = {
        "a": {"64": _prefix(1, 40, .50, .90, 10)},
        "b": {"64": _prefix(1, 40, .47, .90, 10)},
    }
    full = {"a": {"rougeL": .50, "bertscore_f1": .90}, "b": {"rougeL": .50, "bertscore_f1": .90}}
    result = summarize_fixed(docs, full, label="64", epsilon=.02)
    assert result["violating_documents"] == 1
    assert math.isclose(result["risk_rate"], .5)


def test_percentile_handles_singleton_and_interpolation() -> None:
    assert percentile([3], .5) == 3
    assert percentile([0, 10], .25) == 2.5
