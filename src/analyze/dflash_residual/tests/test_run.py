from __future__ import annotations

import json

from src.analyze.dflash_residual.run import build_parser, run_synthetic, run


def test_synthetic_all_writes_phase_bundle_and_keeps_p3_unavailable(tmp_path) -> None:
    input_path = tmp_path / "synthetic.jsonl"
    run_synthetic(input_path, documents=6)
    output_dir = tmp_path / "result"
    args = build_parser().parse_args([
        "--phase", "all",
        "--trace", str(input_path),
        "--output", str(output_dir),
        "--bootstrap-samples", "20",
        "--min-documents", "5",
    ])
    assert run(args) == 0
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["phases"]) == {"p1", "p2", "p3", "p4"}
    assert metrics["phases"]["p3"]["status"] == "unavailable"
    assert (output_dir / "p2" / "coverage.csv").is_file()
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "run_manifest.json").is_file()


def test_p0_cli_uses_legacy_official_and_custom_outputs(tmp_path) -> None:
    official = tmp_path / "official.jsonl"
    custom = tmp_path / "custom.jsonl"
    rows = {"id": "s1", "acceptance_lengths": [2, 2, 1, 2, 2]}
    official.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    custom.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    args = build_parser().parse_args([
        "--phase", "p0",
        "--official", str(official),
        "--custom", str(custom),
        "--output", str(tmp_path / "p0"),
        "--min-blocks", "5",
    ])
    assert run(args) == 0
    metrics = json.loads((tmp_path / "p0" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "PASS"


def test_parser_requires_phase_and_supports_selection_join(tmp_path) -> None:
    args = build_parser().parse_args([
        "--phase", "p3",
        "--trace", "trace.jsonl",
        "--dflash2-selection", "selection.jsonl",
        "--output", str(tmp_path),
    ])
    assert args.dflash2_selection == "selection.jsonl"


def test_p0_parser_accepts_protocol_manifests() -> None:
    args = build_parser().parse_args([
        "--phase", "p0", "--official", "a.jsonl", "--custom", "b.jsonl",
        "--official-manifest", "a.json", "--custom-manifest", "b.json",
        "--output", "/tmp/out",
    ])
    assert args.official_manifest == "a.json"


def test_missing_trace_writes_unavailable_report_for_every_requested_phase(tmp_path) -> None:
    output_dir = tmp_path / "blocked"
    args = build_parser().parse_args([
        "--phase", "all",
        "--output", str(output_dir),
    ])

    assert run(args) == 2
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["phases"]) == {"p1", "p2", "p3", "p4"}
    assert all(item["status"] == "unavailable" for item in metrics["phases"].values())
    assert "--trace is required" in metrics["phases"]["p1"]["reason"]
