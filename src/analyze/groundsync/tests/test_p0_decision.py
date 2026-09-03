from __future__ import annotations

import pytest

from src.analyze.groundsync.p0_decision import (
    across_round_persistence,
    corrected_grounding_horizon,
    first_token_admission_oracle,
    replay_oracle_ladder,
    summarize_within_block_burstiness,
    transition_hazard_rows,
)


def _target(document_id: str = "doc-a") -> dict:
    return {
        "status": "ok",
        "document_id": document_id,
        "target_entropy": [0.1] * 8,
        "attention": [
            {"nosink": [0.9, 0.1]},
            {"nosink": [0.8, 0.2]},
            {"nosink": [0.1, 0.9]},
            {"nosink": [0.2, 0.8]},
            {"nosink": [0.2, 0.8]},
            {"nosink": [0.2, 0.8]},
            {"nosink": [0.2, 0.8]},
            {"nosink": [0.2, 0.8]},
        ],
    }


def _spec(
    document_id: str,
    start: int,
    accepted: int,
    first_reject: int | None,
    *,
    max_k: int = 4,
    with_timing: bool = False,
) -> dict:
    row = {
        "status": "ok",
        "document_id": document_id,
        "start_position": start,
        "max_k": max_k,
        "accepted_len": accepted,
        "first_reject_rel": first_reject,
        "fully_accepted": first_reject is None,
        "draft_confidence": [0.8, 0.7, 0.6, 0.5][:max_k],
    }
    if with_timing:
        row.update({
            "autoregressive_time_ms": 2.0,
            "draft_time_by_k_ms": [1.0, 1.1, 1.2, 1.3][:max_k],
            "verification_time_by_k_ms": [1.0, 1.1, 1.2, 1.3][:max_k],
        })
    return row


def test_corrected_horizon_maps_no_transition_to_kmax() -> None:
    trace = [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]
    assert corrected_grounding_horizon(trace, start=0, threshold=0.2, max_horizon=2) == 2


def test_transition_hazard_uses_within_block_drift_at_each_relative_position() -> None:
    rows = transition_hazard_rows(
        [_target()],
        [_spec("doc-a", 0, accepted=1, first_reject=2)],
        max_k=2,
    )
    assert [row["relative_position"] for row in rows] == [1, 2]
    assert rows[0]["event"] == 0
    assert rows[1]["event"] == 1
    assert rows[0]["d_transition"] != rows[1]["d_transition"]
    assert rows[1]["draft_confidence"] == pytest.approx(0.7)


def test_ladder_contains_k_zero_and_k16_and_uses_ar_cost() -> None:
    row = _spec("doc-a", 0, accepted=2, first_reject=3, max_k=4, with_timing=True)
    result = replay_oracle_ladder([row], candidate_ks=(0, 2, 4), require_timing=True)
    assert result["policies"]["fixed_k0"]["mean_cost_ms"] == pytest.approx(2.0)
    assert result["policies"]["fixed_k2"]["mean_committed_tokens"] == pytest.approx(2.0)
    assert "fixed_k4" in result["policies"]


def test_ladder_uses_common_timing_rows_for_every_policy() -> None:
    complete = _spec("doc-a", 0, accepted=1, first_reject=2, max_k=4, with_timing=True)
    incomplete = _spec("doc-b", 0, accepted=1, first_reject=2, max_k=2, with_timing=True)
    result = replay_oracle_ladder(
        [complete, incomplete], candidate_ks=(0, 2, 4), require_timing=True
    )
    assert result["common_timing_count"] == 1
    assert {result["policies"][f"fixed_k{k}"]["count"] for k in (0, 2, 4)} == {1}


def test_admission_oracle_only_uses_first_token_acceptance_bit() -> None:
    rows = [
        _spec("a", 0, accepted=0, first_reject=1, max_k=2, with_timing=True),
        _spec("b", 0, accepted=2, first_reject=None, max_k=2, with_timing=True),
    ]
    result = first_token_admission_oracle(rows, candidate_k=2, require_timing=True)
    assert result["count"] == 2
    assert result["admitted_speculation"] == 1
    assert result["policy_rows"][0]["selected_k"] == 0
    assert result["policy_rows"][1]["selected_k"] == 2


def test_burstiness_reports_within_hazard_and_across_round_persistence() -> None:
    rows = [
        _spec("doc-a", 1, accepted=0, first_reject=1, max_k=2),
        _spec("doc-a", 2, accepted=1, first_reject=None, max_k=2),
        _spec("doc-a", 3, accepted=1, first_reject=None, max_k=2),
        _spec("doc-a", 4, accepted=0, first_reject=1, max_k=2),
    ]
    within = summarize_within_block_burstiness(rows, max_k=2)
    assert within["by_relative_position"]["1"]["hazard"] == pytest.approx(0.5)
    assert within["h1_to_later_hazard_ratio"] > 0.0
    across = across_round_persistence(rows, deltas=(1, 2))
    assert across["by_delta"]["1"]["pair_count"] == 3
    assert across["by_delta"]["1"]["conditional_probability"] is not None


def test_burstiness_censors_early_ended_draft_after_observed_tokens() -> None:
    row = _spec("doc-a", 1, accepted=2, first_reject=None, max_k=4)
    row["proposal_token_ids"] = [10, 11]
    within = summarize_within_block_burstiness([row], max_k=4)
    assert within["by_relative_position"]["2"]["at_risk"] == 1
    assert within["by_relative_position"]["3"]["at_risk"] == 0
