from __future__ import annotations

from src.TrainingFree.hierarchy_evaluation import (
    HierarchyEvaluationConfig,
    aggregate_hierarchy_results,
    evaluate_hierarchy_trace,
)


def _trace(dataset: str, *, missed: float = 0.005, expansion: float = 0.20) -> dict:
    return {
        "schema_version": "recap.hierarchy.trace.v1",
        "status": "ok",
        "sample_id": f"{dataset}-0",
        "dataset": dataset,
        "source_tokens": 100,
        "steps": [
            {
                "step": 0,
                "missed_attention_mass": missed,
                "max_missed_attention_mass": missed,
                "exact_expansion_fraction": expansion,
                "upper_bound_violations": 0,
                "index_overhead": 0.05,
                "routing_fraction": 0.10,
                "active_qk_tokens": int(100 * expansion),
                "full_qk_tokens": 100,
                "routing_representatives": 10,
                "upper_missed_mass_bound": missed,
                "routing_time_ms": 1.0,
            }
        ],
    }


def test_hierarchy_evaluator_applies_phase_one_gate() -> None:
    row = evaluate_hierarchy_trace(_trace("gov_report"), HierarchyEvaluationConfig())

    assert row["status"] == "ok"
    assert row["metrics"]["missed_attention_mass"] == 0.005
    assert row["gate"]["missed_mass"] is True
    assert row["gate"]["exact_expansion"] is True
    assert row["gate"]["index_overhead"] is True


def test_hierarchy_evaluator_rejects_high_missed_mass() -> None:
    row = evaluate_hierarchy_trace(
        _trace("gov_report", missed=0.03, expansion=0.20),
        HierarchyEvaluationConfig(),
    )

    assert row["gate"]["missed_mass"] is False
    assert row["status"] == "gate_fail"


def test_aggregate_requires_two_datasets_and_reports_gate() -> None:
    config = HierarchyEvaluationConfig(min_documents_per_dataset=1)
    rows = [
        evaluate_hierarchy_trace(_trace("gov_report"), config),
        evaluate_hierarchy_trace(_trace("multi_news"), config),
    ]

    aggregate = aggregate_hierarchy_results(rows, config)

    assert aggregate["status"] == "GO_ROUTING_FOLLOWUP"
    assert aggregate["datasets"] == {"gov_report": 1, "multi_news": 1}
    assert aggregate["gate"]["upper_bound_sound"] is True
