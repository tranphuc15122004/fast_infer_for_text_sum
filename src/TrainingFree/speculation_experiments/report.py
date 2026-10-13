"""Preregistered screening decisions for B1/B2 results."""

from __future__ import annotations

from typing import Any, Mapping


def _interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def make_decision_summary(
    b1: Mapping[str, Any] | None,
    b2: Mapping[str, Any] | None,
    *,
    b1_positive_gap: float = 0.05,
    b1_practical_null: float = 0.02,
    b2_positive_skill: float = 0.05,
    b2_incremental_skill: float = 0.02,
) -> dict[str, Any]:
    """Translate effect sizes and document-level intervals into next steps."""

    b1_status = "pending"
    b1_gap = None
    b1_ci = None
    if b1 is not None:
        evaluation = b1.get("evaluation") or {}
        b1_gap = evaluation.get("oracle_gap_j")
        b1_ci = _interval(evaluation.get("oracle_gap_j_ci95"))
        if b1_gap is not None and b1_ci:
            if float(b1_gap) >= b1_positive_gap and b1_ci[0] > b1_practical_null:
                b1_status = "supported"
            elif b1_ci[1] <= b1_practical_null:
                b1_status = "low_opportunity"
            else:
                b1_status = "inconclusive"

    entropy_status = "pending"
    entropy_skill = entropy_ci = None
    if b2 is not None:
        entropy_skill = (b2.get("brier_skill_vs_R") or {}).get("R-DH")
        entropy_ci = _interval((b2.get("brier_skill_ci95") or {}).get("R-DH"))
        if entropy_skill is not None and entropy_ci:
            if float(entropy_skill) >= b2_positive_skill and entropy_ci[0] > 0.0:
                entropy_status = "predictive"
            elif entropy_ci[1] < b2_positive_skill:
                entropy_status = "no_practical_gain"
            else:
                entropy_status = "inconclusive"

    incremental_status = "pending"
    incremental_skill = incremental_ci = None
    if b2 is not None:
        incremental_skill = (b2.get("brier_skill_vs_control") or {}).get("R-A-DH_vs_R-A")
        incremental_ci = _interval((b2.get("brier_skill_ci95_vs_control") or {}).get("R-A-DH_vs_R-A"))
        if incremental_skill is not None and incremental_ci:
            if float(incremental_skill) >= b2_incremental_skill and incremental_ci[0] > 0.0:
                incremental_status = "supported"
            elif incremental_ci[1] < b2_incremental_skill:
                incremental_status = "no_incremental_value"
            else:
                incremental_status = "inconclusive"

    confidence_status = "pending"
    if b2 is not None:
        skill = (b2.get("brier_skill_vs_R") or {}).get("R-DC")
        ci = _interval((b2.get("brier_skill_ci95") or {}).get("R-DC"))
        if skill is not None and ci:
            if float(skill) >= b2_positive_skill and ci[0] > 0.0:
                confidence_status = "predictive"
            elif ci[1] < b2_positive_skill:
                confidence_status = "no_practical_gain"
            else:
                confidence_status = "inconclusive"

    if "inconclusive" in (b1_status, entropy_status, incremental_status, confidence_status):
        next_experiment = "collect_more_groups"
    elif b1_status == "supported" or entropy_status == "predictive" or confidence_status == "predictive":
        next_experiment = "B1-V"
    else:
        next_experiment = "keep_fixed_gamma_and_report_negative_result"
    return {
        "schema_version": "dflash-speculation-b.decision.v1",
        "b1_opportunity": b1_status,
        "b1_oracle_gap_j": b1_gap,
        "b1_oracle_gap_j_ci95": list(b1_ci) if b1_ci else None,
        "b2_drafter_entropy": entropy_status,
        "b2_entropy_skill_vs_R": entropy_skill,
        "b2_entropy_skill_ci95": list(entropy_ci) if entropy_ci else None,
        "b2_entropy_incremental_after_history": incremental_status,
        "b2_entropy_incremental_skill": incremental_skill,
        "b2_entropy_incremental_ci95": list(incremental_ci) if incremental_ci else None,
        "b2_drafter_confidence": confidence_status,
        "next_experiment": next_experiment,
        "thresholds": {
            "b1_positive_gap": b1_positive_gap,
            "b1_practical_null": b1_practical_null,
            "b2_positive_skill": b2_positive_skill,
            "b2_incremental_skill": b2_incremental_skill,
        },
    }
