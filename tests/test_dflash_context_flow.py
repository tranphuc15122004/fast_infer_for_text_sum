"""Exercise real campaign orchestration/artifacts with GPU forwards substituted."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_dflash_context_experiments as campaign


class RoutingTokenizer:
    chat_template = None

    def __call__(self, text, **kwargs):
        step = max(len(text) // 32, 1)
        offsets = [(start, min(start + step, len(text))) for start in range(0, len(text), step)]
        return {"input_ids": torch.arange(len(offsets)).reshape(1, -1),
                "offset_mapping": torch.tensor(offsets).reshape(1, -1, 2)}


def test_real_phase_flow_locks_confirms_reports_and_resumes_without_recomputing(tmp_path, monkeypatch):
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.context_adaptive.schema import read_jsonl
    from TrainingFree.context_interventions import execution, environment, report

    tokenizer = RoutingTokenizer()
    runtime = {
        "target": SimpleNamespace(device=torch.device("cpu")),
        "draft": SimpleNamespace(layers=[object()] * 5), "tokenizer": tokenizer,
        "runtime_signature": {"signature": "cpu-substitute-for-routing-only"},
        "target_signature": "target", "draft_signature": "draft", "tokenizer_signature": "tokenizer",
        "target_asset": {}, "draft_asset": {}, "attention_backend": "flash_attention_4",
    }
    monkeypatch.setattr(campaign, "_runtime", lambda args: runtime)
    monkeypatch.setattr(environment, "run_preflight", lambda **kwargs: {"status": "ready"})
    # Keep the real statistic/pairing/decision code; reduce bootstrap resamples
    # in this orchestration test. Full bootstrap behavior has dedicated tests.
    bootstrap = report.paired_cluster_bootstrap
    monkeypatch.setattr(report, "paired_cluster_bootstrap", lambda *args, **kwargs: bootstrap(
        *args, **{**kwargs, "resamples": 32},
    ))
    reference_calls = []
    replay_calls = []
    rollout_calls = []

    def reference(_runtime, sample, **kwargs):
        ids, layout = prepare_prompt(sample, tokenizer, chunk_size=128)
        reference_calls.append((sample["id"], kwargs["max_new_tokens"]))
        output = list(range(100, 132))
        return {
            "input_ids": ids, "layout": layout, "ar_ids": output, "native_ids": output,
            "adapter_ids": output, "eos_ids": set(), "adapter_rounds": [{
                "type": "round", "round_index": 0, "gamma_executed": 15,
                "processed_output_before": 0, "logical_length_before": ids.shape[1], "accepted_candidates": 3,
            }], "parity": {
                "status": "pass", "native_equals_adapter": True, "native_equals_ar": True,
                "adapter_equals_ar": True, "ar_output_tokens": len(output),
            },
        }

    def replay(args, _runtime, sample, ids, layout, prefix, output, **kwargs):
        replay_calls.append((kwargs["phase"], sample["id"], prefix.round_index))

        def row(experiment, label, accepted, budget=2048, **extra):
            return execution._action_row(
                experiment=experiment, phase=kwargs["phase"], sample=sample, prefix=prefix,
                label=label, baseline_accepted=3, outcome={"status": "ok", "accepted_candidates": accepted},
                nominal_budget=budget, **extra,
            )

        return {
            "status": "ok", "a1": [
                row("A1_attention", "target_mean5_high_D2048", 1,
                    attention_scoring="target_mean5", deletion_side="high", deletion_count=2048),
                row("A1_attention", "target_mean5_low_D2048", 3,
                    attention_scoring="target_mean5", deletion_side="low", deletion_count=2048),
            ], "a2": [row("A2_selector", label, 3) for label in (
                "previous_dflash_B2048", "recency_B2048", "target_parent_B2048",
                "random_B2048_seed17", "random_B2048_seed42", "random_B2048_seed123",
            )], "a3_replay": [row("A3_budget_replay", "previous_dflash_age1_B2048", 3)],
        }

    def rollout(_runtime, sample, **kwargs):
        rollout_calls.append((sample["id"], kwargs["max_new_tokens"], kwargs["refresh_period"], kwargs["repetition"]))
        selector = kwargs["selector"]
        return [{
            "variant": variant, "status": "ok", "wall_e2e_ms": 50.0 if variant == selector else 100.0,
            "counters": {"mean_accepted_candidates": 3.0}, "output_ids": [100, 101],
            "matches_native_full_dflash": True, "matches_ar": True,
            "reported_timings": {}, "rounds": [], "stopped_by": "max_new_tokens",
        } for variant in ("ar", "native_full_dflash", "adapter_full", selector)]

    monkeypatch.setattr(execution, "run_reference_pair", reference)
    monkeypatch.setattr(campaign, "_compute_replay_prefix", replay)
    monkeypatch.setattr(execution, "run_a3_comparison", rollout)
    common = [
        "--run-id", "flow-test", "--output-root", str(tmp_path),
        "--data-dir", str(ROOT / "data/longbench_100_14k"),
        "--exposure-manifest", str(ROOT / "data/dflash_context_A_audit/combined_exposure_manifest.json"),
    ]
    phases = ("preflight", "prepare", "validate", "pilot", "dev", "lock", "confirm", "report")
    previous_counts = None
    for resumed in (False, True):
        for phase in phases:
            argv = ["campaign", *common, "--mode", phase]
            if resumed:
                argv.append("--resume")
            if phase == "confirm":
                argv.append("--allow-partial-confirm")
            if phase == "report":
                argv.extend(["--report-phase", "confirm"])
            monkeypatch.setattr(sys, "argv", argv)
            assert campaign.main() == 0, phase
        counts = (len(reference_calls), len(replay_calls), len(rollout_calls))
        if resumed:
            assert counts == previous_counts
        previous_counts = counts
    run_root = tmp_path / "flow-test"
    locked = json.loads((run_root / "locked_config.json").read_text())
    assert locked["status"] == "locked_for_heldout"
    assert locked["dev_source_groups_by_dataset"] == {"gov_report": 10, "multi_news": 10, "qmsum": 4}
    confirmation = json.loads((run_root / "reports/confirm_coverage.json").read_text())
    assert confirmation["partial_confirmation"] is True
    assert confirmation["cohort_summary"]["actual_groups_by_dataset"] == {"gov_report": 18, "multi_news": 18, "qmsum": 3}
    assert (run_root / "reports/confirm_decision.md").is_file()
    assert confirmation["rollout_metrics"]["previous_dflash"]["B2048_period4"]["paired_repetitions"] == 39 * 3
    for phase, planned in (("pilot", 6 * 4), ("dev", 24 * 4 * 3 * 4), ("confirm", 39 * 3 * 4)):
        rows = read_jsonl(run_root / phase / "a3_rollout.jsonl")
        assert sum(row.get("type") == "rollout" for row in rows) == planned
        assert rows[-1]["status"] == "complete"
