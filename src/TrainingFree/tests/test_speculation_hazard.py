from __future__ import annotations

import math
from types import SimpleNamespace
import pytest

from TrainingFree.speculation_experiments.survival import (
    HAZARD_VARIANTS,
    SURVIVAL_VARIANTS,
    cross_validated_survival,
    deserialize_survival_curve,
    fit_hazard_model,
    fit_survival_curve,
    predict_hazard_curve,
    predict_survival_curve,
    serialize_survival_curve,
)
from TrainingFree.speculation_experiments.b_controller import (
    VerificationPrefixController,
    choose_verification_prefix,
)


def _make_candidate_state(index: int, accepted: int, base_conf: float = 0.8) -> dict:
    confidences = [
        max(0.1, min(0.99, base_conf - 0.04 * pos)) if pos < accepted else max(0.05, 0.3 - 0.02 * pos)
        for pos in range(15)
    ]
    entropies = [1.0 - c * 0.8 for c in confidences]
    prob_margins = [max(0.01, 2.0 * c - 1.0) if c > 0.5 else 0.02 for c in confidences]
    logit_margins = [math.log(max(m, 1e-4)) + 2.0 for m in prob_margins]
    return {
        "type": "candidate_signals",
        "state_id": f"state-{index}",
        "dataset": "gov_report",
        "source_group_id": f"doc-{index % 4}",
        "sample_id": f"sample-{index}",
        "prompt_tokens": 4096 + index * 64,
        "processed_output_tokens": index * 16,
        "max_new_tokens": 1024,
        "accepted_candidates": accepted,
        "draft_uncertainty": {
            "normalized_entropy": entropies,
            "confidence": confidences,
            "probability_margin": prob_margins,
            "logit_margin": logit_margins,
        },
        "parent_signal": {"normalized_entropy": 0.2, "confidence": 0.8},
        "history_ema": 0.5,
    }


def test_hazard_variants_registered_and_uncertainty_flags():
    expected_variants = (
        "R-Hazard-Pos",
        "R-Hazard-Present",
        "R-Hazard-Prefix",
        "R-Hazard-Suffix-Margin",
        "R-Hazard-Suffix-Entropy",
    )
    for variant in expected_variants:
        assert variant in HAZARD_VARIANTS
        assert variant in SURVIVAL_VARIANTS

    # Position-only hazard does not require draft uncertainty
    controller_pos = VerificationPrefixController(
        model={"variant": "R-Hazard-Pos"},
        cost_profiles={},
        dataset="gov_report",
    )
    assert controller_pos.requires_uncertainty is False

    # Draft-aware hazard variants require draft uncertainty
    for variant in ("R-Hazard-Present", "R-Hazard-Prefix", "R-Hazard-Suffix-Margin", "R-Hazard-Suffix-Entropy"):
        controller = VerificationPrefixController(
            model={"variant": variant},
            cost_profiles={},
            dataset="gov_report",
        )
        assert controller.requires_uncertainty is True


def test_hazard_fit_and_strictly_decreasing_survival():
    rows = [_make_candidate_state(i, 10 if i % 2 == 0 else 3, base_conf=0.85 if i % 2 == 0 else 0.4) for i in range(12)]

    for variant in HAZARD_VARIANTS:
        model = fit_survival_curve(rows, variant=variant, max_candidates=15)
        assert model["model_type"] == "hazard"
        assert model["variant"] == variant
        assert len(model["beta"]) == len(model["features"]) + 1
        assert all(math.isfinite(b) for b in model["beta"])

        # Test prediction on a new row
        test_row = _make_candidate_state(99, 5, base_conf=0.7)
        probabilities, fallback = predict_survival_curve(model, test_row)
        assert len(probabilities) == 15
        assert all(fallback[i] == "hazard" for i in range(15))

        # Check monotonicity: S(k) <= S(k-1) naturally holds without any clipping/min repair
        for i in range(1, 15):
            assert probabilities[i] <= probabilities[i - 1], f"Variant {variant} not nonincreasing at index {i}"
            if probabilities[i - 1] > 1e-12:
                assert probabilities[i] < probabilities[i - 1]
            assert 0.0 <= probabilities[i] < 1.0


def test_hazard_right_censoring_in_risk_set():
    # If all 15 candidates are accepted, risk set has 15 positive transitions and no false failure
    full_accepted_row = _make_candidate_state(0, 15, base_conf=0.9)
    model_full = fit_hazard_model([full_accepted_row], variant="R-Hazard-Pos", max_candidates=15)
    assert model_full["train_risk_states"] == 15

    # If 4 candidates are accepted, risk set has 4 successes + 1 first failure = 5 transitions
    partial_row = _make_candidate_state(1, 4, base_conf=0.9)
    model_part = fit_hazard_model([partial_row], variant="R-Hazard-Pos", max_candidates=15)
    assert model_part["train_risk_states"] == 5

    # If 0 candidates are accepted, risk set has 0 successes + 1 first failure = 1 transition
    zero_row = _make_candidate_state(2, 0, base_conf=0.2)
    model_zero = fit_hazard_model([zero_row], variant="R-Hazard-Pos", max_candidates=15)
    assert model_zero["train_risk_states"] == 1


def test_hazard_serialization_roundtrip():
    rows = [_make_candidate_state(i, 8, base_conf=0.75) for i in range(8)]
    for variant in ("R-Hazard-Pos", "R-Hazard-Present", "R-Hazard-Prefix"):
        model = fit_survival_curve(rows, variant=variant)
        serialized = serialize_survival_curve(model)
        assert serialized["schema_version"] == "dflash-speculation-b.survival-model.v1"
        assert serialized["model_type"] == "hazard"

        deserialized = deserialize_survival_curve(serialized)
        assert deserialized["variant"] == variant
        assert deserialized["model_type"] == "hazard"
        assert deserialized["beta"] == model["beta"]

        test_row = _make_candidate_state(100, 6)
        p1, f1 = predict_survival_curve(model, test_row)
        p2, f2 = predict_survival_curve(deserialized, test_row)
        assert p1 == pytest.approx(p2)
        assert f1 == f2


def test_cross_validated_survival_with_hazard():
    rows = [_make_candidate_state(i, 11 if i % 2 == 0 else 4, base_conf=0.9 if i % 2 == 0 else 0.4) for i in range(16)]

    result = cross_validated_survival(
        rows,
        n_splits=4,
        variants=("R", "R-Hazard-Pos", "R-Hazard-Present", "R-Hazard-Prefix"),
        bootstrap_replicates=50,
        seed=42,
    )

    assert result["status"] == "complete"
    assert "R-Hazard-Pos" in result["brier_by_variant"]
    assert "R-Hazard-Present" in result["brier_by_variant"]
    assert "R-Hazard-Prefix" in result["brier_by_variant"]

    # Paired comparisons
    assert "R-Hazard-Present_vs_R-Hazard-Pos" in result["brier_skill_vs_control"]
    assert "R-Hazard-Prefix_vs_R-Hazard-Present" in result["brier_skill_vs_control"]

    for row in result["oof_predictions"]:
        for variant in ("R-Hazard-Pos", "R-Hazard-Present", "R-Hazard-Prefix"):
            probs = row["predictions"][variant]
            assert len(probs) == 15
            for i in range(1, 15):
                assert probs[i] <= probs[i - 1]


def test_cost_opportunity_gate_and_hysteresis():
    survival = [0.95, 0.85, 0.75, 0.60, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005]
    
    # Case 1: Low opportunity: C(15) = 10.0, C(3) = 9.5 -> Delta = 1 - 0.95 = 0.05 <= eta = 0.10
    decision_gate = choose_verification_prefix(
        survival_probabilities=survival,
        cost_by_prefix={3: 9.5, 7: 9.7, 11: 9.9, 15: 10.0},
        allowed_prefixes=(3, 7, 11, 15),
        fallback_prefix=15,
        opportunity_threshold_eta=0.10,
    )
    assert decision_gate["prefix"] == 15
    assert decision_gate["reason"] == "cost_opportunity_gate_below_threshold"
    assert decision_gate["opportunity_delta"] == pytest.approx(0.05)

    # Case 2: High opportunity Delta = 1 - 6.0/10.0 = 0.40 > eta = 0.10, but J(7) = 8.5/5.20 = 1.635 vs J(15) = 1.756
    # Ratio = 1.635 / 1.756 = 0.931 >= (1 - 0.10) = 0.90 -> Hysteresis retains 15!
    decision_hysteresis = choose_verification_prefix(
        survival_probabilities=survival,
        cost_by_prefix={3: 6.0, 7: 8.5, 11: 9.5, 15: 10.0},
        allowed_prefixes=(3, 7, 11, 15),
        fallback_prefix=15,
        opportunity_threshold_eta=0.10,
    )
    assert decision_hysteresis["prefix"] == 15
    assert decision_hysteresis["reason"] == "hysteresis_insufficient_gain"

    # Case 3: Genuine significant advantage: C(3)=2.0, commits(3)=3.5 -> J(3) = 2.0/3.5 = 0.5714 < 0.90 * 1.6667 = 1.50
    decision_win = choose_verification_prefix(
        survival_probabilities=survival,
        cost_by_prefix={3: 2.0, 7: 7.8, 11: 9.0, 15: 10.0},
        allowed_prefixes=(3, 7, 11, 15),
        fallback_prefix=15,
        opportunity_threshold_eta=0.10,
    )
    assert decision_win["prefix"] == 3
    assert decision_win["reason"] == "minimum_predicted_cost_per_commit"


def test_controller_bypass_signals_when_gate_triggers():
    model = fit_survival_curve(
        [_make_candidate_state(i, 8) for i in range(8)],
        variant="R-Hazard-Present",
    )
    cost_profiles = {
        "gov_report::context=1": {
            "cost_by_prefix": {"3": 9.6, "7": 9.8, "11": 9.9, "15": 10.0}
        }
    }
    controller = VerificationPrefixController(
        model=model,
        cost_profiles=cost_profiles,
        dataset="gov_report",
        fallback_prefix=15,
        opportunity_threshold_eta=0.10,
    )

    # Context 4096 is bucket 1
    assert controller.should_bypass_signals(logical_length=4096) is True

    state = SimpleNamespace(
        logical_length=4096,
        processed_output_tokens=0,
        parent_entropy=0.2,
        parent_confidence=0.8,
        history_acceptance=0.5,
    )
    # Calling controller should bypass without needing valid uncertainty vectors
    decision = controller(
        state=state,
        draft_uncertainty={},
        draft_gamma=15,
        available_prefixes=(3, 7, 11, 15),
    )
    assert decision["prefix"] == 15
    assert decision["reason"] == "cost_opportunity_gate_below_threshold"

    # Boundary round near max_new_tokens where draft_gamma = 7 < 15
    boundary_decision = controller(
        state=state,
        draft_uncertainty={},
        draft_gamma=7,
        available_prefixes=(3, 7),
    )
    assert boundary_decision["prefix"] == 7
    assert boundary_decision["reason"] == "cost_opportunity_gate_below_threshold"

