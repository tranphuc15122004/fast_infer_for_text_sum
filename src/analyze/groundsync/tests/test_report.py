from __future__ import annotations

import pytest

from src.analyze.groundsync.report import (
    binary_prediction_metrics,
    build_horizon_rows,
    build_predictor_rows,
    decide_status,
    replay_policy,
    split_by_document,
    summarize_h3_predictor,
    summarize_h5_predictor,
    summarize_speculative_traces,
    summarize_target_traces,
)


def test_split_by_document_never_leaks_document_ids() -> None:
    rows = [
        {"document_id": "b", "label": 0},
        {"document_id": "a", "label": 1},
        {"document_id": "c", "label": 0},
        {"document_id": "d", "label": 1},
    ]
    train, dev, test = split_by_document(rows, train_fraction=0.5, dev_fraction=0.25)
    assert {row["document_id"] for row in train} == {"a", "b"}
    assert {row["document_id"] for row in dev} == {"c"}
    assert {row["document_id"] for row in test} == {"d"}


def test_binary_prediction_metrics_reports_auc_logloss_and_brier() -> None:
    result = binary_prediction_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["count"] == 4
    assert result["auroc"] == pytest.approx(1.0)
    assert result["log_loss"] < 0.3
    assert result["brier"] < 0.05


def test_binary_prediction_metrics_marks_single_class_auc_unavailable() -> None:
    result = binary_prediction_metrics([1, 1], [0.6, 0.7])
    assert result["auroc"] is None
    assert result["status"] == "insufficient_class_variation"


def test_decide_status_requires_coverage_and_threshold() -> None:
    assert decide_status(value=0.12, threshold=0.08, direction=">=", count=10, min_count=5) == "PASS"
    assert decide_status(value=0.04, threshold=0.08, direction=">=", count=10, min_count=5) == "FAIL"
    assert decide_status(value=0.12, threshold=0.08, direction=">=", count=2, min_count=5) == "INCONCLUSIVE"
    assert decide_status(value=None, threshold=0.08, direction=">=", count=10, min_count=5) == "UNAVAILABLE"


def _target_rows() -> list[dict]:
    trace_a = [
        {"raw": [0.9, 0.1], "nosink": [0.9, 0.1]},
        {"raw": [0.8, 0.2], "nosink": [0.8, 0.2]},
        {"raw": [0.1, 0.9], "nosink": [0.1, 0.9]},
        {"raw": [0.2, 0.8], "nosink": [0.2, 0.8]},
    ]
    return [
        {
            "status": "ok",
            "document_id": "doc-a",
            "output_tokens": 4,
            "target_entropy": [1.0, 0.9, 0.8, 0.7],
            "sentence_boundary": [0, 0, 1, 0],
            "copyability": [1, 1, 0, 0],
            "attention": trace_a,
        }
    ]


def test_target_summary_contains_persistence_and_null() -> None:
    result = summarize_target_traces(
        _target_rows(), threshold=0.2, min_documents=1, lags=(1, 2)
    )
    assert result["documents"] == 1
    assert result["adjacent_similarity"] is not None
    assert result["null_adjacent_similarity"] is not None
    assert result["decision"] in {"PASS", "FAIL"}


def test_spec_summary_and_predictor_join_are_document_aware() -> None:
    specs = [
        {
            "status": "ok", "document_id": "doc-a", "start_position": 0,
            "draft_confidence": [0.9, 0.8], "max_k": 2,
            "drift_at_start": 0.01, "fully_accepted": True, "accepted_len": 2,
        },
        {
            "status": "ok", "document_id": "doc-a", "start_position": 1,
            "draft_confidence": [0.4, 0.3], "max_k": 2,
            "drift_at_start": 0.5, "fully_accepted": False, "accepted_len": 0,
        },
    ]
    result = summarize_speculative_traces(specs, min_group_count=1)
    assert result["high_minus_low_rejection_rate"] == pytest.approx(1.0)
    joined = build_predictor_rows(
        _target_rows(), specs, horizon_threshold=0.2, max_horizon=2
    )
    assert len(joined) == 2
    assert joined[1]["label"] == 1
    assert joined[1]["grounding_horizon"] == 1


def test_policy_replay_is_explicitly_not_a_speed_claim_without_timing() -> None:
    rows = [{"status": "ok", "accepted_len": 2, "fully_accepted": True}]
    result = replay_policy(rows, policy="fixed", max_k=4, fixed_k=4)
    assert result["mean_committed_tokens"] == 3.0
    assert result["timing_basis"] == "acceptance_only_no_speed_claim"
    unavailable = replay_policy(
        rows, policy="fixed", max_k=4, fixed_k=4, require_timing=True
    )
    assert unavailable["status"] == "UNAVAILABLE"


def test_policy_replay_uses_timing_for_selected_k_only() -> None:
    rows = [{
        "status": "ok",
        "accepted_len": 4,
        "grounding_horizon": 2,
        "draft_time_by_k_ms": [1.0, 2.0, 3.0, 4.0],
        "verification_time_by_k_ms": [2.0, 3.0, 4.0, 5.0],
    }]
    result = replay_policy(rows, policy="oracle", max_k=4, require_timing=True)
    assert result["status"] == "ok"
    assert result["mean_cost_ms"] == pytest.approx(5.0)
    assert result["timing_basis"] == "measured_draft_plus_verification"


def test_horizon_builder_and_empty_predictor_are_conservative() -> None:
    horizons = build_horizon_rows(_target_rows(), threshold=0.2, max_horizon=2)
    assert horizons
    assert {row["label"] for row in horizons} == {0, 1}
    result = summarize_h5_predictor([])
    assert result["decision"] == "UNAVAILABLE"
    assert summarize_h3_predictor([])["decision"] == "UNAVAILABLE"


def test_h5_is_inconclusive_without_controlled_acceptance_data() -> None:
    from src.analyze.groundsync.report import build_hypothesis_report

    result = build_hypothesis_report(
        _target_rows(), [], threshold=0.2, horizon_threshold=0.2, max_horizon=2
    )
    assert result["hypotheses"]["H5"]["decision"] == "INCONCLUSIVE"
