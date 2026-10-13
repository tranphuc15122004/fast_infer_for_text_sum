from __future__ import annotations

import pytest

from TrainingFree.speculation_experiments.analysis import analyze_draft_shape


def _row(state: str, group: str, gamma: int, repetition: int, ms: float, accepted: int) -> dict:
    return {
        "type": "draft_shape_action",
        "state_id": state,
        "dataset": "gov_report",
        "source_group_id": group,
        "sample_id": group,
        "gamma": gamma,
        "repetition": repetition,
        "repetition_split": "selection" if repetition < 2 else "evaluation",
        "round_ms": ms,
        "accepted_candidates": accepted,
        "proposed_token_ids": list(range(gamma)),
    }


def test_b1_analysis_separates_selection_oracle_from_heldout_timing() -> None:
    rows = []
    for state, group, best in (("s1", "d1", 3), ("s2", "d2", 15), ("s3", "d3", 15)):
        for gamma in (3, 7, 11, 15):
            accepted = min(gamma, 2) if best == 3 else min(gamma, 12)
            per_token_ms = 1.0 if gamma == best else 2.0
            rows.extend(
                _row(state, group, gamma, repetition, per_token_ms * (1 + accepted), accepted)
                for repetition in range(4)
            )

    result = analyze_draft_shape(rows, bootstrap_replicates=100, seed=7)

    assert result["status"] == "complete"
    assert result["state_count"] == 3
    assert result["oracle_gamma_histogram"] == {"3": 1, "15": 2}
    assert result["best_fixed_gamma"] == 15
    assert result["evaluation"]["oracle_gap_j"] == pytest.approx(0.25, abs=0.01)
    assert result["evaluation"]["oracle_gap_t"] > 0.0
    assert result["evaluation"]["oracle_gap_j_ci95"][0] <= result["evaluation"]["oracle_gap_j"]


def test_b1_analysis_rejects_unpaired_or_missing_repetitions() -> None:
    rows = [_row("s1", "d1", gamma, rep, 3.0, 1) for gamma in (3, 7, 11, 15) for rep in range(4)]
    rows = [row for row in rows if not (row["gamma"] == 11 and row["repetition"] == 3)]

    with pytest.raises(ValueError, match="paired"):
        analyze_draft_shape(rows, bootstrap_replicates=10)
