from __future__ import annotations

from src.analyze.safe_budget.e27a_policy_screen import (
    ACTION_LABELS,
    build_source_features,
    choose_action_from_predictions,
    evaluate_policy_actions,
    quality_violation,
)


def _outcome(label: str, cost: float, rouge: float, bert: float) -> dict:
    return {
        "label": label,
        "cost_ms": cost,
        "rougeL": rouge,
        "bertscore_f1": bert,
    }


def test_source_features_are_deterministic_and_finite():
    text = "Heading\nAlpha beta gamma. Alpha beta gamma.\nSecond section."
    first = build_source_features(text, source_tokens=12)
    second = build_source_features(text, source_tokens=12)
    assert first == second
    assert first["source_tokens"] == 12.0
    assert first["sentence_count"] >= 2.0
    assert all(value == value for value in first.values())


def test_quality_violation_uses_both_metrics_with_or_semantics():
    full = _outcome("full", 100.0, 0.50, 0.80)
    rouge_bad = _outcome("ratio_0.25", 50.0, 0.47, 0.80)
    bert_bad = _outcome("ratio_0.4", 60.0, 0.50, 0.77)
    assert quality_violation(rouge_bad, full, epsilon=0.02) is True
    assert quality_violation(bert_bad, full, epsilon=0.02) is True


def test_action_selector_falls_back_to_full_when_no_prediction_is_safe():
    predicted_risk = {label: 0.9 for label in ACTION_LABELS}
    predicted_cost = {label: 1.0 for label in ACTION_LABELS}
    predicted_risk["full"] = 0.0
    assert choose_action_from_predictions(predicted_risk, predicted_cost, alpha=0.05) == "full"


def test_capture_is_zero_for_fixed_and_one_for_matching_oracle():
    outcomes = {
        "d1": {
            "full": _outcome("full", 100.0, 0.50, 0.80),
            "ratio_0.25": _outcome("ratio_0.25", 50.0, 0.50, 0.80),
        },
        "d2": {
            "full": _outcome("full", 100.0, 0.50, 0.80),
            "ratio_0.25": _outcome("ratio_0.25", 50.0, 0.40, 0.60),
        },
    }
    fixed = {"d1": "full", "d2": "full"}
    oracle = {"d1": "ratio_0.25", "d2": "full"}
    fixed_result = evaluate_policy_actions(fixed, outcomes, epsilon=0.02, alpha=0.0)
    oracle_result = evaluate_policy_actions(oracle, outcomes, epsilon=0.02, alpha=0.0)
    assert fixed_result["capture"] == 0.0
    assert oracle_result["capture"] == 1.0
