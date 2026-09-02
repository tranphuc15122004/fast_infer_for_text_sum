from __future__ import annotations

import pytest

from src.analyze.groundsync.report import (
    build_predictor_rows,
    build_hypothesis_report,
    calibrated_trace,
    fit_relative_positional_prior,
    replay_policy_sweep,
    summarize_position_relocation,
    summarize_rejection_hazard,
)


def test_relative_positional_prior_is_learned_and_calibration_normalizes() -> None:
    rows = [
        {"status": "ok", "attention": [{"raw": [0.8, 0.2]}, {"raw": [0.7, 0.3]}]},
        {"status": "ok", "attention": [{"raw": [0.6, 0.4]}, {"raw": [0.5, 0.5]}]},
    ]
    prior = fit_relative_positional_prior(rows, variant="raw", bins=4)
    assert len(prior) == 4
    calibrated = calibrated_trace([[0.8, 0.2], [0.7, 0.3]], prior)
    assert len(calibrated) == 2
    assert all(sum(vector) == pytest.approx(1.0) for vector in calibrated)


def test_rejection_hazard_reports_risk_set_and_events_by_relative_position() -> None:
    rows = [
        {"status": "ok", "max_k": 3, "first_reject_rel": 1, "fully_accepted": False},
        {"status": "ok", "max_k": 3, "first_reject_rel": 2, "fully_accepted": False},
        {"status": "ok", "max_k": 3, "first_reject_rel": None, "fully_accepted": True},
    ]
    result = summarize_rejection_hazard(rows, max_k=3)
    assert result["by_relative_position"]["1"]["at_risk"] == 3
    assert result["by_relative_position"]["1"]["events"] == 1
    assert result["by_relative_position"]["2"]["at_risk"] == 2
    assert result["by_relative_position"]["2"]["events"] == 1


def test_rejection_hazard_includes_document_cluster_bootstrap_coefficient() -> None:
    rows = [
        {
            "status": "ok", "document_id": f"doc-{index}", "max_k": 3,
            "drift_at_start": 0.01 if index < 6 else 0.5,
            "first_reject_rel": 1 if index < 3 else None,
            "fully_accepted": index >= 3,
        }
        for index in range(12)
    ]
    result = summarize_rejection_hazard(rows, max_k=3)
    assert result["hazard_model"]["status"] == "ok"
    assert result["hazard_model"]["bootstrap_unit"] == "document"
    assert "drift_coefficient_ci" in result["hazard_model"]


def test_position_relocation_summarizes_raw_and_sink_controlled_mass() -> None:
    rows = []
    for case, start in (("begin", 0), ("middle", 16), ("end", 32)):
        rows.append({
            "status": "ok",
            "document_id": case,
            "relocation_case": case,
            "evidence_token_start": start,
            "evidence_token_end": start + 8,
            "evidence_skip_source_tokens": 8,
            "evidence_chunk_size": 16,
            "attention": [
                {
                    "raw_chunk_16": [0.7, 0.1, 0.1, 0.1],
                    "nosink_8_chunk_16": [0.6, 0.2, 0.1],
                }
            ],
        })
    result = summarize_position_relocation(rows)
    assert result["status"] == "ok"
    assert set(result["variants"]) == {"raw_chunk_16", "nosink_8_chunk_16"}
    assert set(result["variants"]["raw_chunk_16"]["cases"]) == {
        "begin", "middle", "end"
    }


def test_policy_sweep_contains_fixed_and_true_cost_oracle() -> None:
    rows = [{
        "status": "ok",
        "accepted_len": 2,
        "grounding_horizon": 2,
        "draft_time_by_k_ms": [1.0, 1.0, 10.0],
        "verification_time_by_k_ms": [1.0, 1.0, 10.0],
    }]
    result = replay_policy_sweep(rows, max_k=3, require_timing=True, fixed_ks=(1, 2, 3))
    assert set(result["policies"]) >= {"fixed_k1", "fixed_k2", "fixed_k3", "true_cost_oracle"}
    assert result["policies"]["fixed_k2"]["status"] == "ok"
    assert result["policies"]["true_cost_oracle"]["status"] == "ok"


def test_policy_sweep_marks_true_cost_unavailable_without_verifier_arrays() -> None:
    result = replay_policy_sweep(
        [{"status": "ok", "accepted_len": 0, "max_k": 8}],
        max_k=8,
        require_timing=True,
    )
    assert result["policies"]["true_cost_oracle"]["status"] == "UNAVAILABLE"


def test_hypothesis_report_exposes_calibration_hazard_and_policy_sweep() -> None:
    attention = [
        {"raw": [0.9, 0.1], "nosink": [0.9, 0.1]},
        {"raw": [0.8, 0.2], "nosink": [0.8, 0.2]},
        {"raw": [0.1, 0.9], "nosink": [0.1, 0.9]},
        {"raw": [0.2, 0.8], "nosink": [0.2, 0.8]},
    ]
    targets = [{
        "status": "ok", "document_id": f"doc-{index}",
        "output_tokens": 4, "target_entropy": [0.1] * 4,
        "sentence_boundary": [0] * 4, "copyability": [0] * 4,
        "attention": attention,
    } for index in range(6)]
    specs = [{
        "status": "ok", "document_id": f"doc-{index}",
        "start_position": 1, "max_k": 2, "accepted_len": index % 2,
        "first_reject_rel": 1 if index % 2 == 0 else None,
        "fully_accepted": index % 2 == 1,
        "drift_at_start": 0.1 + index / 10,
        "draft_confidence": [0.8, 0.7],
        "draft_time_by_k_ms": [1.0, 2.0],
        "verification_time_by_k_ms": [1.0, 2.0],
    } for index in range(6)]
    result = build_hypothesis_report(
        targets, specs, threshold=0.2, horizon_threshold=0.2,
        max_horizon=3, max_k=2,
    )
    assert "calibrated" in result["hypotheses"]["H1"]["controls"]
    assert "hazard" in result["hypotheses"]["H2"]
    assert "policy_sweep" in result["hypotheses"]["H4"]


def test_predictor_rows_expose_first_rejection_and_shifted_grounding_controls() -> None:
    target = [{
        "status": "ok", "document_id": "doc-a", "output_tokens": 20,
        "target_entropy": [0.1] * 20,
        "sentence_boundary": [0] * 20, "copyability": [0] * 20,
        "attention": [
            {"nosink": [0.8, 0.2], "raw": [0.8, 0.2]}
            for _ in range(20)
        ],
    }]
    specs = [{
        "status": "ok", "document_id": "doc-a", "start_position": 0,
        "max_k": 2, "accepted_len": 0, "fully_accepted": False,
        "first_reject_rel": 1, "drift_at_start": None,
        "draft_confidence": [0.4, 0.3],
    }]
    rows = build_predictor_rows(target, specs, horizon_threshold=0.2, max_horizon=16)
    assert len(rows) == 1
    assert rows[0]["first_token_rejected"] == 1
    assert rows[0]["source_concentration"] == pytest.approx(0.8)
    assert rows[0]["shift10_source_concentration"] == pytest.approx(0.8)
    assert rows[0]["lag_drift"] == pytest.approx(0.0)
    assert rows[0]["lag_drift_missing"] == pytest.approx(1.0)


def test_predictor_rows_compute_lag_drift_for_later_start() -> None:
    target = [{
        "status": "ok", "document_id": "doc-lag", "output_tokens": 3,
        "target_entropy": [0.1, 0.2, 0.3],
        "sentence_boundary": [0, 0, 0], "copyability": [0, 0, 0],
        "attention": [
            {"nosink": [0.9, 0.1], "raw": [0.9, 0.1]},
            {"nosink": [0.5, 0.5], "raw": [0.5, 0.5]},
            {"nosink": [0.1, 0.9], "raw": [0.1, 0.9]},
        ],
    }]
    specs = [{
        "status": "ok", "document_id": "doc-lag", "start_position": 2,
        "max_k": 2, "accepted_len": 1, "fully_accepted": True,
        "first_reject_rel": None, "drift_at_start": 0.3,
        "draft_confidence": [0.8, 0.7],
    }]
    rows = build_predictor_rows(target, specs, horizon_threshold=0.2)
    assert rows[0]["lag_drift"] > 0.0
    assert rows[0]["lag_drift_missing"] == pytest.approx(0.0)
