from __future__ import annotations

import json

from src.analyze.groundsync.run_experiment import (
    build_parser,
    run_analysis,
    run,
    run_synthetic_fixture,
)


def test_synthetic_fixture_runs_through_report(tmp_path) -> None:
    run_synthetic_fixture(tmp_path)
    report = run_analysis(
        tmp_path,
        threshold=0.2,
        horizon_threshold=0.2,
        max_horizon=4,
    )
    assert report["mode"] == "synthetic_fixture"
    assert set(report["hypotheses"]) == {"H1", "H2", "H3", "H4", "H5"}
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "hypothesis_report.md").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "policy_utility.png").exists()
    assert (tmp_path / "horizon_labels.png").exists()
    loaded = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert loaded["hypotheses"]["H1"]["decision"] in {
        "PASS", "FAIL", "INCONCLUSIVE", "UNAVAILABLE"
    }


def test_unavailable_trace_files_are_reported_without_crashing(tmp_path) -> None:
    (tmp_path / "target_traces.jsonl").write_text(
        json.dumps({"schema_version": "groundsync.target.v1", "status": "unavailable"})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "speculative_traces.jsonl").write_text("", encoding="utf-8")
    report = run_analysis(tmp_path, threshold=0.2, horizon_threshold=0.2)
    assert report["hypotheses"]["H1"]["decision"] == "UNAVAILABLE"
    assert report["hypotheses"]["H2"]["decision"] == "UNAVAILABLE"


def test_analyze_run_dir_does_not_replace_existing_manifest(tmp_path) -> None:
    run_synthetic_fixture(tmp_path)
    manifest = (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    args = build_parser().parse_args([
        "--phase", "analyze", "--run-dir", str(tmp_path)
    ])
    assert run(args) == 0
    assert (tmp_path / "run_manifest.json").read_text(encoding="utf-8") == manifest


def test_runner_parser_accepts_protocol_sensitivity_lists() -> None:
    args = build_parser().parse_args([
        "--phase", "target", "--input", "input.jsonl",
        "--sensitivity-chunk-sizes", "64,128,256",
        "--sink-sizes", "4,8,16",
    ])
    assert args.sensitivity_chunk_sizes == "64,128,256"
    assert args.sink_sizes == "4,8,16"
