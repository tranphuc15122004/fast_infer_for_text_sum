from __future__ import annotations

import pytest

from TrainingFree.speculation_experiments.survival import (
    _bootstrap_brier_skill,
    cross_validated_survival,
)


def _state(index: int, accepted: int, uncertainty: float) -> dict:
    vector = [uncertainty if position < accepted else min(1.0, uncertainty + 0.35) for position in range(15)]
    confidence = [1.0 - value * 0.8 for value in vector]
    return {
        "type": "candidate_signals",
        "state_id": f"state-{index}",
        "dataset": "gov_report",
        "source_group_id": f"doc-{index}",
        "sample_id": f"doc-{index}",
        "prompt_tokens": 4096 + index,
        "processed_output_tokens": index * 16,
        "max_new_tokens": 1024,
        "accepted_candidates": accepted,
        "draft_uncertainty": {
            "normalized_entropy": vector,
            "confidence": confidence,
            "probability_margin": [0.0] * 15,
            "logit_margin": [0.0] * 15,
        },
        "parent_signal": {"normalized_entropy": uncertainty, "confidence": 1 - uncertainty},
        "history_ema": uncertainty,
    }


def test_grouped_survival_is_out_of_fold_monotone_and_uses_training_groups_only() -> None:
    rows = [_state(index, 12 if index % 2 else 3, 0.1 if index % 2 else 0.8) for index in range(16)]

    result = cross_validated_survival(
        rows,
        n_splits=4,
        variants=("R", "R-DH", "R-DC", "R-A-DH"),
        min_support_states=3,
        min_support_groups=2,
        bootstrap_replicates=100,
        seed=9,
    )

    assert result["status"] == "complete"
    assert len(result["oof_predictions"]) == len(rows)
    for row in result["oof_predictions"]:
        for probabilities in row["predictions"].values():
            assert len(probabilities) == 15
            assert all(left >= right for left, right in zip(probabilities, probabilities[1:]))
        assert row["source_group_id"] not in row["training_groups"]
    assert result["brier_by_variant"]["R-DH"] < result["brier_by_variant"]["R"]
    assert result["brier_skill_vs_R"]["R-DH"] > 0
    assert result["descriptive_spearman_vs_accepted"]["draft_entropy_mean"]["pooled"] is not None


def test_survival_analysis_rejects_missing_or_malformed_signal_vectors() -> None:
    rows = [_state(index, index % 4, 0.2) for index in range(8)]
    rows[0]["draft_uncertainty"]["normalized_entropy"] = [0.1]

    with pytest.raises(ValueError, match="15 candidates"):
        cross_validated_survival(rows, n_splits=4, bootstrap_replicates=0)


def test_missing_history_is_encoded_and_confident_but_wrong_candidates_are_counted() -> None:
    rows = [_state(index, 2 if index % 2 else 11, 0.2) for index in range(12)]
    rows[0]["history_ema"] = None
    rows[0]["history_missing"] = True
    rows[0]["accepted_candidates"] = 2
    rows[0]["draft_uncertainty"]["confidence"] = [0.95] * 15

    result = cross_validated_survival(
        rows,
        n_splits=4,
        variants=("R", "R-A", "R-A-DH"),
        min_support_states=2,
        min_support_groups=2,
        bootstrap_replicates=0,
    )

    assert result["status"] == "complete"
    assert result["candidate_risk_diagnostics"][2]["confidence_ge_0_9_reached"] >= 1
    assert result["candidate_risk_diagnostics"][2]["confidence_ge_0_9_rejections"] >= 1


def test_brier_bootstrap_macro_averages_datasets_instead_of_pooling_groups() -> None:
    rows = []
    for dataset, group_count in (("small", 1), ("large", 9)):
        for index in range(group_count):
            signal_probability = 0.0 if dataset == "small" else 0.5
            rows.append({
                "dataset": dataset,
                "source_group_id": f"{dataset}-{index}",
                "labels": [0] * 15,
                "predictions": {
                    "R": [0.5] * 15,
                    "signal": [signal_probability] * 15,
                },
            })

    draws = _bootstrap_brier_skill(
        rows,
        variant="signal",
        baseline="R",
        replicates=50,
        seed=11,
    )

    # The small dataset contributes half the macro score despite having one
    # group; pooling all ten documents would report only 10% skill instead.
    assert draws == pytest.approx([0.5] * 50)
