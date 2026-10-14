from __future__ import annotations

from TrainingFree.context_interventions.execution import _action_row
from TrainingFree.context_interventions.replay import ReferencePrefix


def test_action_row_accepts_and_records_ranking_origin_round() -> None:
    prefix = ReferencePrefix(
        round_index=4,
        processed_output_tokens=48,
        pending_anchor=123,
        remaining_output_tokens=976,
        prefix_hash="prefix-hash",
        previous_dense_round_index=2,
    )

    row = _action_row(
        experiment="A2_selector",
        phase="pilot",
        sample={
            "id": "sample-1",
            "dataset": "gov_report",
            "_cad_source_group_id": "group-1",
        },
        prefix=prefix,
        label="previous_dflash_B1024",
        baseline_accepted=8,
        outcome={"status": "ok", "accepted_candidates": 7},
        nominal_budget=1024,
        ranking_origin_round=2,
    )

    assert row["ranking_origin_round"] == 2


def test_action_row_keeps_ranking_origin_from_outcome_when_not_overridden() -> None:
    prefix = ReferencePrefix(
        round_index=4,
        processed_output_tokens=48,
        pending_anchor=123,
        remaining_output_tokens=976,
        prefix_hash="prefix-hash",
        previous_dense_round_index=2,
    )

    row = _action_row(
        experiment="A2_selector",
        phase="pilot",
        sample={
            "id": "sample-1",
            "dataset": "gov_report",
            "_cad_source_group_id": "group-1",
        },
        prefix=prefix,
        label="previous_dflash_B1024",
        baseline_accepted=8,
        outcome={"status": "ok", "accepted_candidates": 7, "ranking_origin_round": 2},
        nominal_budget=1024,
    )

    assert row["ranking_origin_round"] == 2
