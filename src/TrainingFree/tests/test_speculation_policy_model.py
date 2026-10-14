from __future__ import annotations

import json

from TrainingFree.speculation_experiments.survival import (
    deserialize_survival_curve,
    fit_survival_curve,
    predict_survival_curve,
    serialize_survival_curve,
)


def _row(group: int, entropy: float, confidence: float, accepted: int) -> dict:
    return {
        "type": "candidate_signals",
        "state_id": f"state-{group}",
        "dataset": "gov_report",
        "sample_id": f"doc-{group}",
        "source_group_id": f"group-{group}",
        "round_index": 2,
        "prompt_tokens": 4096,
        "processed_output_tokens": 256,
        "context_tokens": 4352,
        "accepted_candidates": accepted,
        "history_ema": 0.5,
        "history_missing": False,
        "draft_uncertainty": {
            "normalized_entropy": [entropy] * 15,
            "confidence": [confidence] * 15,
        },
        "lagged_signals": {
            "source_state_id": f"previous-{group}",
            "draft_entropy_max": entropy,
            "draft_confidence_min": confidence,
        },
    }


def test_survival_policy_model_roundtrips_to_json_and_predicts_causally() -> None:
    rows = [_row(group, group / 20, 1.0 - group / 25, group % 12) for group in range(1, 13)]
    model = fit_survival_curve(
        rows,
        variant="R-A-DH",
        min_support_states=2,
        min_support_groups=2,
    )
    saved = json.loads(json.dumps(serialize_survival_curve(model)))
    restored = deserialize_survival_curve(saved)

    assert restored["variant"] == "R-A-DH"
    assert restored["training_groups"] == model["training_groups"]
    before = predict_survival_curve(model, rows[0])
    after = predict_survival_curve(restored, {**rows[0], "accepted_candidates": 0})
    assert after == before


def test_b4_lagged_variants_use_the_prior_round_signals() -> None:
    rows = [_row(group, group / 20, 1.0 - group / 25, group % 12) for group in range(1, 13)]
    model = fit_survival_curve(
        rows,
        variant="R-A-LDH",
        min_support_states=2,
        min_support_groups=2,
    )

    curve, fallback = predict_survival_curve(model, rows[-1])

    assert len(curve) == 15
    assert all(left >= right for left, right in zip(curve, curve[1:]))
    assert len(fallback) == 15
