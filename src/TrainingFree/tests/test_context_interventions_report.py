from __future__ import annotations

import json

from TrainingFree.context_interventions.report import summarize_phase
from TrainingFree.context_interventions.report import build_report


def test_a2_report_pairs_all_random_seeds_with_previous_by_prefix() -> None:
    rows = []
    for dataset in ("gov_report", "multi_news", "qmsum"):
        for group in (f"{dataset}-0", f"{dataset}-1"):
            common = {"experiment": "A2_selector", "phase": "dev", "status": "ok",
                      "dataset": dataset, "source_group_id": group, "prefix_round_index": 3,
                      "nominal_budget": 2048}
            for seed in (17, 42, 123):
                rows.append({**common, "selector_id": f"random_B2048_seed{seed}", "L_action": 4})
            for selector, value in (("previous_dflash_B2048", 6), ("recency_B2048", 5), ("target_parent_B2048", 5)):
                rows.append({**common, "selector_id": selector, "L_action": value})
    report = summarize_phase(rows, phase="dev")
    random_ci = report["family_metrics"]["A2"]["comparisons"]["previous_vs_random"]
    assert random_ci is not None
    assert random_ci["estimate"] == 2.0
    assert random_ci["n_source_groups"] == 6


def test_empty_campaign_report_marks_missing_results_inconclusive() -> None:
    report = summarize_phase([], phase="pilot")
    assert report["family_metrics"]["A1"]["decision"] == "inconclusive"
    assert report["family_metrics"]["A2"]["decision"] == "inconclusive_missing_pairs"


def test_report_marks_replay_families_inconclusive_when_prefix_baseline_fails(tmp_path) -> None:
    phase_root = tmp_path / "pilot"
    phase_root.mkdir()
    (phase_root / "prefix_diagnostics.jsonl").write_text(
        json.dumps({
            "type": "prefix_diagnostic",
            "experiment": "full_dflash_replay_baseline",
            "phase": "pilot",
            "sample_id": "sample-1",
            "dataset": "gov_report",
            "prefix_round_index": 129,
            "status": "baseline_mismatch",
            "reason": "replay=0, reference=1",
        }) + "\n",
        encoding="utf-8",
    )

    report = build_report(tmp_path, "pilot")

    assert report["replay_diagnostics"]["failed_prefix_count"] == 1
    assert report["family_metrics"]["A1"]["decision"] == "inconclusive_incomplete_replay_evidence"
    assert report["family_metrics"]["A2"]["decision"] == "inconclusive_incomplete_replay_evidence"


def test_report_marks_rollout_metrics_inconclusive_when_rollout_is_partial(tmp_path) -> None:
    phase_root = tmp_path / "confirm"
    phase_root.mkdir()
    row = {
        "type": "rollout", "status": "ok", "variant": "previous_dflash",
        "selector": "previous_dflash", "budget": 2048, "refresh_period": 4,
        "repetition": 0, "dataset": "gov_report", "source_group_id": "group-1",
        "wall_e2e_ms": 10.0, "matches_native_full_dflash": True,
        "counters": {"mean_accepted_candidates": 4.0},
    }
    summary = {
        "type": "summary", "experiment": "A3_rollout", "phase": "confirm",
        "status": "partial_with_errors", "rows": 3, "planned_rows": 4,
        "status_counts": {"ok": 3, "error": 1},
    }
    (phase_root / "a3_rollout.jsonl").write_text(
        "\n".join(json.dumps(item) for item in (row, summary)) + "\n",
        encoding="utf-8",
    )

    report = build_report(tmp_path, "confirm")

    assert report["rollout_completeness"]["rows"] == 3
    assert report["rollout_metrics"]["previous_dflash"]["B2048_period4"]["decision"] == (
        "inconclusive_incomplete_rollout"
    )


def test_report_does_not_claim_replay_support_with_error_cells(tmp_path) -> None:
    phase_root = tmp_path / "dev"
    phase_root.mkdir()
    (phase_root / "a2.jsonl").write_text(
        json.dumps({
            "type": "replay", "experiment": "A2_selector", "phase": "dev",
            "sample_id": "sample-1", "dataset": "gov_report", "source_group_id": "group-1",
            "prefix_round_index": 129, "selector_id": "previous_dflash_B2048",
            "status": "error", "reason": "CUDA OOM",
        }) + "\n",
        encoding="utf-8",
    )

    report = build_report(tmp_path, "dev")

    assert report["replay_diagnostics"]["failed_cell_count"] == 1
    assert report["family_metrics"]["A2"]["decision"] == "inconclusive_incomplete_replay_evidence"


def test_optional_grid_error_does_not_block_complete_selected_configuration(tmp_path) -> None:
    phase_root = tmp_path / "dev"
    phase_root.mkdir()
    rows = []
    for period in (4, 8):
        for variant in ("ar", "native_full_dflash", "adapter_full", "previous_dflash"):
            rows.append({
                "type": "rollout", "variant": variant, "selector": "previous_dflash",
                "budget": 2048, "refresh_period": period, "repetition": 0,
                "dataset": "gov_report", "source_group_id": "group-1",
                "status": "ok" if period == 4 else "error",
                "wall_e2e_ms": 10.0 if variant == "previous_dflash" else 20.0,
                "matches_native_full_dflash": True, "matches_ar": True,
                "counters": {"mean_accepted_candidates": 4.0},
            })
    rows.append({
        "type": "summary", "experiment": "A3_rollout", "status": "partial_with_errors",
        "rows": 8, "planned_rows": 8, "status_counts": {"ok": 4, "error": 4},
        "configurations": [
            {"selector": "previous_dflash", "budget": 2048, "refresh_period": period,
             "status": "complete" if period == 4 else "partial_with_errors"}
            for period in (4, 8)
        ],
    })
    (phase_root / "a3_rollout.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )
    report = build_report(tmp_path, "dev")
    metrics = report["rollout_metrics"]["previous_dflash"]
    assert metrics["B2048_period4"]["decision"] == "acceptance_and_speedup_gate_pass"
    assert metrics["B2048_period8"]["decision"] == "inconclusive_incomplete_rollout"


def test_missing_reference_is_reported_as_incomplete_evidence(tmp_path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    (evidence_root / "parity_pilot.json").write_text(json.dumps({
        "status": "partial_with_errors", "cases": [
            {"status": "error", "sample_id": "sample-1", "dataset": "gov_report", "reason": "OOM"},
        ],
    }), encoding="utf-8")
    report = build_report(tmp_path, "pilot")
    assert report["replay_diagnostics"]["failed_reference_count"] == 1
    assert report["family_metrics"]["A1"]["decision"] == "inconclusive_incomplete_replay_evidence"
