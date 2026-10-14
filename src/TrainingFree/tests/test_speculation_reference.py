from __future__ import annotations

from argparse import Namespace

from TrainingFree.context_adaptive.schema import atomic_json
from TrainingFree.speculation_experiments.reference import (
    reference_history,
    reference_trajectory_ids,
    select_reference_prefixes,
)
import scripts.run_dflash_speculation_B as b_runner
from scripts.run_dflash_speculation_B import (
    _trace_signature_matches,
)


def test_reference_trajectory_uses_dflash_ids_when_ar_is_diagnostic_only() -> None:
    trace = {
        "ar_ids": [11, 12, 13],
        "native_ids": [11, 21, 31],
        "adapter_ids": [11, 21, 31],
    }

    assert reference_trajectory_ids(trace) == [11, 21, 31]


def test_reference_trajectory_rejects_native_adapter_disagreement() -> None:
    trace = {"native_ids": [1, 2], "adapter_ids": [1, 3], "ar_ids": [1, 2]}

    try:
        reference_trajectory_ids(trace)
    except ValueError as exc:
        assert "native full-DFlash and adapter" in str(exc)
    else:
        raise AssertionError("native/adapter divergence must not define a B1 reference trajectory")


def test_resume_does_not_accept_trace_from_a_different_code_signature() -> None:
    assert _trace_signature_matches({"trace_signature": "same"}, "same")
    assert not _trace_signature_matches({"trace_signature": "old"}, "new")


def test_resume_loads_saved_campaign_without_requiring_exposure_manifest(tmp_path, monkeypatch) -> None:
    run_id = "B_main_02"
    output_root = tmp_path / "outputs"
    run_root = output_root / run_id
    base_sample = {"dataset": "qmsum", "id": "sample-1", "text": "input"}
    cohort = {"groups": [{
        "dataset": "qmsum", "sample_id": "sample-1",
        "source_group_id": "group-1", "split": "confirm",
    }]}
    split = {"assignments": {"group-1": 0}}
    corpus_signature = [{"path": "/data/longbench.jsonl", "sha256": "corpus-hash"}]
    campaign = {
        "run_id": run_id,
        "cohort_sha256": b_runner._hash_json(cohort),
        "split_manifest_sha256": b_runner._hash_json(split),
        "corpus": corpus_signature,
        "selected_sample_keys": ["qmsum::sample-1"],
        "cohort_phase": "confirm",
        "seed": 7,
        "gammas": [3, 7, 11, 15],
        "max_new_tokens": 1024,
        "validation_tokens": 128,
        "max_prefixes": 15,
        "warmups": 3,
        "repetitions": 6,
        "selection_repetitions": 3,
        "n_splits": 4,
        "bootstrap_replicates": 2000,
        "attention_backend": "flash_attention_4",
    }
    atomic_json(run_root / "campaign.json", campaign)
    atomic_json(run_root / "cohort_manifest.json", cohort)
    atomic_json(run_root / "split_manifest.json", split)
    monkeypatch.setattr(b_runner, "_load_samples", lambda args: ([base_sample], ["data.jsonl"]))
    monkeypatch.setattr(b_runner, "_corpus_signature", lambda paths: corpus_signature)
    args = Namespace(
        run_id=run_id,
        output_root=str(output_root),
        allow_partial_cohort=False,
    )

    selected, saved_campaign, loaded_root = b_runner._load_resume_campaign(args)

    assert loaded_root == run_root
    assert saved_campaign == campaign
    assert selected[0]["_cad_source_group_id"] == "group-1"
    assert selected[0]["_cad_split"] == "confirm"
    assert args.gammas == "3,7,11,15"
    assert args.cohort_phase == "confirm"


def test_prefix_selection_covers_beginning_middle_end_and_respects_regular_horizon() -> None:
    output = list(range(200))
    rounds = [
        {"type": "round", "round_index": index, "gamma_executed": 15, "processed_output_before": index * 4,
         "accepted_candidates": 3}
        for index in range(40)
    ]

    selected = select_reference_prefixes(
        rounds,
        output,
        prompt_length=100,
        max_new_tokens=200,
        eos_ids=set(),
        max_prefixes=15,
    )

    assert 1 <= len(selected) <= 15
    assert [row.round_index for row in selected] == sorted(row.round_index for row in selected)
    assert all(row.remaining_output_tokens >= 16 for row in selected)
    thirds = [sum(1 for row in selected if row.processed_output_tokens < 67),
              sum(1 for row in selected if 67 <= row.processed_output_tokens < 134),
              sum(1 for row in selected if row.processed_output_tokens >= 134)]
    assert all(count >= 1 for count in thirds)


def test_reference_history_only_uses_prior_rounds_and_marks_missing_first() -> None:
    rounds = [
        {"type": "round", "round_index": 0, "accepted_candidates": 4, "gamma_executed": 15},
        {"type": "round", "round_index": 1, "accepted_candidates": 10, "gamma_executed": 15},
        {"type": "round", "round_index": 2, "accepted_candidates": 1, "gamma_executed": 15},
    ]

    first = reference_history(rounds, current_round=0)
    third = reference_history(rounds, current_round=2)

    assert first["missing"] is True
    assert third["previous_accepted_candidates"] == 10
    assert third["ema_acceptance"] == 0.1 * (10 / 15) + 0.9 * (4 / 15)
