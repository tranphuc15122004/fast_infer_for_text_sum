from __future__ import annotations

from TrainingFree.speculation_experiments.report import make_decision_summary


def test_decision_summary_distinguishes_opportunity_predictability_and_next_step() -> None:
    b1 = {"evaluation": {"oracle_gap_j": 0.08, "oracle_gap_j_ci95": [0.03, 0.12]}}
    b2 = {
        "brier_skill_vs_R": {"R-DH": 0.08, "R-DC": 0.02},
        "brier_skill_ci95": {"R-DH": [0.02, 0.13], "R-DC": [-0.01, 0.04]},
        "brier_skill_vs_control": {"R-A-DH_vs_R-A": 0.04, "R-A-DC_vs_R-A": 0.01},
        "brier_skill_ci95_vs_control": {"R-A-DH_vs_R-A": [0.01, 0.07], "R-A-DC_vs_R-A": [-0.02, 0.03]},
    }

    result = make_decision_summary(b1, b2)

    assert result["b1_opportunity"] == "supported"
    assert result["b2_drafter_entropy"] == "predictive"
    assert result["b2_entropy_incremental_after_history"] == "supported"
    assert result["next_experiment"] == "B1-V"


def test_decision_summary_does_not_promote_brier_gain_without_ci_support() -> None:
    b1 = {"evaluation": {"oracle_gap_j": 0.03, "oracle_gap_j_ci95": [-0.01, 0.08]}}
    b2 = {
        "brier_skill_vs_R": {"R-DH": 0.03},
        "brier_skill_ci95": {"R-DH": [-0.04, 0.09]},
        "brier_skill_vs_control": {"R-A-DH_vs_R-A": 0.01},
        "brier_skill_ci95_vs_control": {"R-A-DH_vs_R-A": [-0.03, 0.05]},
    }

    result = make_decision_summary(b1, b2)

    assert result["b1_opportunity"] == "inconclusive"
    assert result["b2_drafter_entropy"] == "inconclusive"
    assert result["b2_entropy_incremental_after_history"] == "inconclusive"
    assert result["next_experiment"] == "collect_more_groups"
