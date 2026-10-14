from __future__ import annotations

import pytest
from scripts.run_dflash_speculation_B import (
    _b3_overhead_adjusted_opportunity,
    _locked_benchmark_signature,
    _locked_e2e_protocol,
    _parser,
    _validate_policy_lock,
)

from TrainingFree.speculation_experiments.b_controller import (
    FixedVerificationPrefixController,
    choose_verification_prefix,
    resolve_verification_prefix,
)


def test_verification_policy_uses_expected_commits_and_all_measured_costs() -> None:
    decision = choose_verification_prefix(
        survival_probabilities=[0.95, 0.85, 0.70, 0.45, 0.20, 0.05],
        cost_by_prefix={3: 12.0, 6: 11.0},
        allowed_prefixes=(3, 6, 9),
        overhead_ms=2.0,
        fallback_prefix=3,
    )

    assert decision["prefix"] == 6
    assert decision["reason"] == "minimum_predicted_cost_per_commit"
    assert decision["predicted_cost_per_commit"] == pytest.approx(13.0 / (1 + sum([0.95, 0.85, 0.70, 0.45, 0.20, 0.05])))
    assert decision["unsupported_prefixes"] == [9]


def test_verification_policy_falls_back_when_cost_or_survival_support_is_missing() -> None:
    decision = choose_verification_prefix(
        survival_probabilities=[0.8, 0.5],
        cost_by_prefix={9: 10.0},
        allowed_prefixes=(3, 9),
        fallback_prefix=3,
    )

    assert decision["prefix"] == 3
    assert decision["reason"] == "fallback_no_supported_action"
    assert decision["predicted_cost_per_commit"] is None


def test_verification_policy_rejects_invalid_probability_or_cost_values() -> None:
    with pytest.raises(ValueError, match="nonincreasing"):
        choose_verification_prefix(
            survival_probabilities=[0.4, 0.7],
            cost_by_prefix={2: 3.0},
            allowed_prefixes=(2,),
            fallback_prefix=2,
        )
    with pytest.raises(ValueError, match="finite positive"):
        choose_verification_prefix(
            survival_probabilities=[0.4, 0.2],
            cost_by_prefix={2: float("nan")},
            allowed_prefixes=(2,),
            fallback_prefix=2,
        )


def test_verifier_rejects_a_policy_prefix_larger_than_its_draft_block() -> None:
    assert resolve_verification_prefix(
        {"prefix": 7, "reason": "fixed_control"}, maximum_prefix=15
    ) == (7, {"prefix": 7, "reason": "fixed_control"})

    with pytest.raises(ValueError, match="outside the drafted block"):
        resolve_verification_prefix({"prefix": 11}, maximum_prefix=7)


def test_fixed_verification_control_uses_supported_tail_prefix_without_signal_collection() -> None:
    controller = FixedVerificationPrefixController(15)
    decision = controller(
        state=None,
        draft_uncertainty={},
        draft_gamma=15,
        available_prefixes=(3, 7, 11),
    )

    assert controller.requires_uncertainty is False
    assert decision["prefix"] == 11
    assert decision["reason"] == "fixed_prefix_tail_fallback"


def test_b3_opportunity_is_reselected_and_scored_after_overhead() -> None:
    rows = []
    for dataset_index, dataset in enumerate(("gov_report", "multi_news", "qmsum")):
        for group_index in range(4):
            state_id = f"{dataset}-{group_index}"
            best_prefix = (3, 7, 11, 15)[group_index]
            for split in ("selection", "evaluation"):
                for prefix in (3, 7, 11, 15):
                    rows.append({
                        "state_id": state_id,
                        "dataset": dataset,
                        "source_group_id": f"{dataset}-{group_index}",
                        "k": prefix,
                        "repetition_split": split,
                        "acceptance_identity_valid": True,
                        "accepted_candidates": 0,
                        "total_round_ms": 1.0 if prefix == best_prefix else 10.0,
                    })

    report = _b3_overhead_adjusted_opportunity(
        rows, overhead_ms=0.0, bootstrap_replicates=100
    )

    assert report["status"] == "supported"
    assert report["state_count"] == 12
    assert report["oracle_gap_j"] > 0.5


def test_frozen_policy_lock_rejects_protocol_or_benchmark_drift() -> None:
    args = _parser().parse_args(["--mode", "b3-confirm", "--run-id", "confirm"])
    campaign = {
        "corpus": [{"path": "/data/a.jsonl", "sha256": "abc"}],
        "exposure_sha256": "exposure-hash",
    }
    lock = {
        "protocol_parameters": _locked_e2e_protocol(args),
        "benchmark_signature": _locked_benchmark_signature(campaign),
    }

    _validate_policy_lock(lock, args=args, campaign=campaign, phase="B3")

    args.max_new_tokens += 1
    with pytest.raises(ValueError, match="protocol parameters"):
        _validate_policy_lock(lock, args=args, campaign=campaign, phase="B3")

    args.max_new_tokens -= 1
    changed_campaign = {**campaign, "exposure_sha256": "different-exposure"}
    with pytest.raises(ValueError, match="corpus/exposure"):
        _validate_policy_lock(lock, args=args, campaign=changed_campaign, phase="B3")
