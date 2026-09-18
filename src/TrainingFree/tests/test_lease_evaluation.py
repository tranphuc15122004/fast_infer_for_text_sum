from __future__ import annotations

from src.TrainingFree.lease_evaluation import (
    LeaseEvaluationConfig,
    aggregate_lease_results,
    evaluate_lease_trace,
)


def _trace(dataset: str = "gov_report", sample_id: str = "x") -> dict:
    return {
        "schema_version": "recap.lease.trace.v1",
        "status": "ok",
        "sample_id": sample_id,
        "dataset": dataset,
        "source_blocks": 10,
        "steps": [
            {
                "step": 0,
                "anchor_step": 0,
                "bound": 0.08,
                "actual_cold_mass": 0.03,
                "valid": True,
                "expired": False,
                "cold_fraction": 0.4,
                "full_source_score": True,
            },
            {
                "step": 1,
                "anchor_step": 0,
                "bound": 0.09,
                "actual_cold_mass": 0.04,
                "valid": True,
                "expired": False,
                "cold_fraction": 0.4,
                "full_source_score": False,
            },
            {
                "step": 2,
                "anchor_step": 0,
                "bound": 0.11,
                "actual_cold_mass": 0.05,
                "valid": False,
                "expired": True,
                "cold_fraction": 0.4,
                "full_source_score": False,
            },
        ],
    }


def test_evaluate_lease_trace_reports_violation_and_refresh_metrics() -> None:
    result = evaluate_lease_trace(_trace(), LeaseEvaluationConfig())

    assert result["status"] == "ok"
    assert result["metrics"]["lease_count"] == 1
    assert result["metrics"]["median_lease_length"] == 2.0
    assert result["metrics"]["refresh_rate"] == 2 / 3
    assert result["metrics"]["violation_rate"] == 0.0


def test_aggregate_requires_two_datasets_and_minimum_documents() -> None:
    config = LeaseEvaluationConfig(min_documents_per_dataset=1)
    result = aggregate_lease_results(
        [evaluate_lease_trace(_trace(), config)], config
    )

    assert result["status"] == "INCONCLUSIVE"


def test_aggregate_stops_when_headroom_gate_fails() -> None:
    config = LeaseEvaluationConfig(min_documents_per_dataset=1)
    rows = [
        evaluate_lease_trace(_trace("gov_report", "g"), config),
        evaluate_lease_trace(_trace("multi_news", "m"), config),
    ]

    result = aggregate_lease_results(rows, config)

    assert result["status"] == "STOP_BEFORE_PHYSICAL"
    assert result["gate"]["median_lease_length"] is False


def test_aggregate_can_pass_all_registered_gates() -> None:
    config = LeaseEvaluationConfig(
        min_documents_per_dataset=1,
        min_lease_length=1,
        min_cold_fraction=0.2,
    )
    rows = [
        evaluate_lease_trace(_trace("gov_report", "g"), config),
        evaluate_lease_trace(_trace("multi_news", "m"), config),
    ]

    result = aggregate_lease_results(rows, config)

    assert result["status"] == "GO_PHYSICAL_FOLLOWUP"
