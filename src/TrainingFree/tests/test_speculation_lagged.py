from __future__ import annotations

from types import SimpleNamespace

from TrainingFree.speculation_experiments.lagged import (
    LaggedDraftLengthController,
    cross_validated_draft_cost_policy,
    fit_draft_cost_policy,
    pair_adjacent_round_signals,
    predict_draft_costs,
)


def _signal(
    round_index: int, entropy: float, confidence: float, *, processed: int | None = None
) -> dict:
    return {
        "type": "candidate_signals",
        "state_id": f"s{round_index}",
        "dataset": "gov_report",
        "sample_id": "doc-1",
        "source_group_id": "group-1",
        "round_index": round_index,
        "processed_output_tokens": round_index * 5 if processed is None else processed,
        "prompt_tokens": 4096,
        "context_tokens": 4096 + round_index * 16,
        "accepted_candidates": 4,
        "history_ema": 0.5,
        "draft_uncertainty": {
            "normalized_entropy": [entropy] * 15,
            "confidence": [confidence] * 15,
        },
    }


def test_lagged_pairs_use_only_the_immediately_previous_round() -> None:
    rows = pair_adjacent_round_signals(
        [_signal(3, 0.9, 0.6), _signal(1, 0.2, 0.95), _signal(2, 0.4, 0.8)]
    )

    assert [row["round_index"] for row in rows] == [2, 3]
    assert rows[0]["lagged_source_state_id"] == "s1"
    assert rows[0]["lagged_draft_entropy_max"] == 0.2
    assert rows[0]["lagged_draft_confidence_min"] == 0.95
    assert rows[1]["lagged_source_state_id"] == "s2"
    assert rows[1]["lagged_draft_entropy_max"] == 0.4
    assert rows[1]["accepted_candidates"] == 4


def test_lagged_pairing_drops_round_gaps_and_never_crosses_documents() -> None:
    rows = pair_adjacent_round_signals(
        [
            _signal(0, 0.2, 0.9),
            _signal(2, 0.4, 0.8),
            {**_signal(1, 0.3, 0.85), "sample_id": "doc-2", "source_group_id": "group-2"},
        ]
    )

    assert rows == []


def test_lagged_pairing_drops_adjacent_rounds_with_noncausal_output_progress() -> None:
    rows = pair_adjacent_round_signals(
        [_signal(0, 0.2, 0.9, processed=0), _signal(1, 0.3, 0.85, processed=16)]
    )

    assert rows == []


def test_b4_cost_model_selects_shape_from_previous_round_features() -> None:
    rows = []
    for group in range(1, 17):
        entropy = group / 18
        short = entropy < 0.5
        rows.append({
            "state_id": f"s{group}",
            "dataset": "gov_report",
            "sample_id": f"doc-{group}",
            "source_group_id": f"group-{group}",
            "context_tokens": 4096,
            "processed_output_tokens": 256,
            "history_ema": 0.5,
            "lagged_signals": {
                "draft_entropy_max": entropy,
                "draft_confidence_min": 1.0 - entropy,
            },
            "selection_costs_ms_per_commit_by_gamma": {
                "3": 1.0 if short else 4.0,
                "7": 2.0 if short else 3.0,
                "11": 3.0 if short else 2.0,
                "15": 4.0 if short else 1.0,
            },
            "evaluation_costs_ms_per_commit_by_gamma": {
                "3": 1.1 if short else 4.1,
                "7": 2.1 if short else 3.1,
                "11": 3.1 if short else 2.1,
                "15": 4.1 if short else 1.1,
            },
        })

    model = fit_draft_cost_policy(
        rows,
        variant="R-A-LDHC",
        min_support_states=2,
        min_support_groups=2,
    )
    costs, level = predict_draft_costs(model, rows[0])
    policy = LaggedDraftLengthController(model)
    short_decision = policy(
        state=SimpleNamespace(logical_length=4096, processed_output_tokens=256, history_acceptance=0.5, round_index=2),
        previous_draft_uncertainty={"normalized_entropy": [0.2] * 15, "confidence": [0.9] * 15},
        previous_draft_gamma=15,
        feasible_gammas=(3, 7, 11, 15),
        fallback_gamma=15,
    )
    unsupported = policy(
        state=SimpleNamespace(logical_length=4096, processed_output_tokens=256, history_acceptance=0.5, round_index=2),
        previous_draft_uncertainty={"normalized_entropy": [0.2] * 7, "confidence": [0.9] * 7},
        previous_draft_gamma=7,
        feasible_gammas=(3, 7, 11, 15),
        fallback_gamma=15,
    )

    assert costs
    assert level == "signal"
    assert short_decision["gamma"] == 3
    assert unsupported["reason"] == "fallback_unsupported_previous_gamma"
    assert unsupported["gamma"] == 15


def test_b4_grouped_evaluation_keeps_documents_held_out() -> None:
    rows = []
    for group in range(1, 13):
        entropy = group / 14
        base = 1.0 + (group % 3) / 10
        rows.append({
            "state_id": f"s{group}", "dataset": "gov_report", "sample_id": f"doc-{group}",
            "source_group_id": f"group-{group}", "context_tokens": 4096,
            "processed_output_tokens": group * 20, "history_ema": 0.5,
            "lagged_signals": {"draft_entropy_max": entropy, "draft_confidence_min": 1.0 - entropy},
            "selection_costs_ms_per_commit_by_gamma": {
                str(gamma): base + gamma / 100 for gamma in (3, 7, 11, 15)
            },
            "evaluation_costs_ms_per_commit_by_gamma": {
                str(gamma): base + gamma / 100 for gamma in (3, 7, 11, 15)
            },
        })

    report = cross_validated_draft_cost_policy(
        rows, variants=("R", "R-A", "R-A-LDHC"), n_splits=3, bootstrap_replicates=0
    )

    assert report["status"] == "complete"
    assert report["state_count"] == 12
    assert all(len(row["training_groups"]) < 12 for row in report["oof_predictions"])
