from __future__ import annotations

import pytest

from src.TrainingFree.evaluation import (
    EvaluationConfig,
    aggregate_results,
    evaluate_trace,
    future_use_matrix,
    ranking_metrics,
)
from src.TrainingFree.tests.test_schema import _record


def test_future_use_excludes_the_current_segment() -> None:
    result = future_use_matrix([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])

    assert result[0] == pytest.approx([0.5, 0.5])
    assert result[1] == pytest.approx([0.8, 0.2])
    assert result[2] == pytest.approx([0.0, 0.0])


def test_ranking_metrics_are_better_for_the_correct_top_block() -> None:
    result = ranking_metrics([0.9, 0.1], [0.1, 0.9], k=1)

    assert result["recall_at_k"] == pytest.approx(0.1 / 0.9)
    assert result["ndcg_at_k"] == pytest.approx(0.1 / 0.9)


def test_evaluator_keeps_hindsight_oracle_out_of_policy_scores() -> None:
    result = evaluate_trace(_record(), EvaluationConfig(budgets=(1,)))

    assert result["status"] == "ok"
    assert result["steps"] == 1
    assert result["metrics"]["1"]["current"]["count"] == 1
    assert "future_use" not in result["metrics"]["1"]["recap"]
    assert "recap_attention" in result["metrics"]["1"]


def test_aggregate_marks_small_pilot_inconclusive() -> None:
    result = aggregate_results([{"status": "ok", "dataset": "gov_report"}])

    assert result["status"] == "INCONCLUSIVE"
    assert result["reason"] == "minimum_dataset_or_document_gate_not_met"
