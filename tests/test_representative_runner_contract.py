import json
import subprocess
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_metrics  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


ALL_INFERENCE_BASELINES = {
    "eagle3",
    "dflash",
    "llmlingua",
    "fastkv",
    "rocketkv",
    "gemfilter",
    "specprefill",
    "minference",
    "magicdec",
    "longspec",
    "specextend",
    "higoe",
    "semantic_selection",
}


def test_dispatcher_covers_all_inference_baselines():
    dispatcher = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
    for baseline in ALL_INFERENCE_BASELINES:
        assert f"{baseline})" in dispatcher


def test_representative_runner_defaults_cover_all_inference_baselines():
    runner = (ROOT / "scripts/run_representative_100.sh").read_text(
        encoding="utf-8"
    )
    for baseline in ALL_INFERENCE_BASELINES:
        assert baseline in runner
    assert 'BASELINES="$REPRESENTATIVE_BASELINES"' in runner
    assert "UNSUPPORTED_BASELINES" in runner


def test_representative_runner_defaults_to_full_and_strict_collection():
    runner = (ROOT / "scripts/run_representative_100.sh").read_text(
        encoding="utf-8"
    )
    config = (ROOT / "config/representative_100.env").read_text(encoding="utf-8")
    assert 'MODE="full"' in config
    assert "--strict" in runner
    assert "--expected-baselines" in runner
    assert "--expected-datasets" in runner
    assert "--expected-samples" in runner


def test_representative_runner_has_canonical_model_overrides():
    runner = (ROOT / "scripts/run_representative_100.sh").read_text(
        encoding="utf-8"
    )
    for marker in (
        "REP_TARGET_MODEL",
        "REP_SPEC_MODEL",
        "REP_EAGLE_MODEL",
        "REP_DFLASH_MODEL",
        "REP_VICUNA_MODEL",
        "REP_SPECEXTEND_DRAFT_MODEL",
    ):
        assert marker in runner


def test_representative_runner_selects_t4_configs_in_smoke_mode():
    runner = (ROOT / "scripts/run_representative_100.sh").read_text(
        encoding="utf-8"
    )
    for marker in (
        "fastkv:smoke",
        "gemfilter:smoke",
        "specprefill:smoke",
        "specextend:smoke",
    ):
        assert marker in runner


def test_semantic_selection_adapter_files_exist():
    assert (ROOT / "scripts/run_semantic_selection.sh").is_file()
    assert (ROOT / "config/semantic_selection.env").is_file()


def test_dflash_and_longspec_have_representative_adapters():
    assert (ROOT / "scripts/infer_dflash.py").is_file()
    assert (ROOT / "scripts/run_dflash.sh").is_file()
    assert (ROOT / "config/dflash.env").is_file()
    assert (ROOT / "scripts/infer_longspec.py").is_file()
    assert (ROOT / "scripts/run_longspec.sh").is_file()

    runner = (ROOT / "scripts/run_representative_100.sh").read_text(
        encoding="utf-8"
    )
    assert 'REPRESENTATIVE_BASELINES="llmlingua fastkv gemfilter specprefill minference specextend eagle3 semantic_selection dflash longspec"' in runner
    assert "dflash longspec" in runner
    assert "dflash|longspec" in runner


def test_representative_runner_dflash_longspec_dry_run(tmp_path):
    output_dir = tmp_path / "representative"
    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_representative_100.sh"),
            "--mode",
            "full",
            "--baselines",
            "dflash longspec",
            "--datasets",
            "xsum",
            "--max-samples",
            "1",
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--skip-collect",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    dflash_config = (output_dir / "configs/dflash_xsum.env").read_text()
    assert "LLaMA3.1-8B-Instruct-DFlash-UltraChat" in dflash_config
    assert "data/representative_100/xsum_representative.jsonl" in dflash_config

    longspec_config = (output_dir / "configs/longspec_xsum.env").read_text()
    assert "MODEL_NAME='vicuna7b'" in longspec_config
    assert "DATA_FILE='data/representative_100/xsum_representative.jsonl'" in longspec_config


def test_collector_normalizes_semantic_selection_schema():
    row = collect_metrics.normalize_record(
        {
            "example_id": "xsum_1",
            "selector": "lead",
            "original_tokens": 100,
            "selected_tokens": 50,
            "selection_total_wall_ms": 12.5,
            "pipeline_ttft_ms": 20.0,
            "pipeline_e2e_ms": 80.0,
            "output_tokens": 8,
            "output_tokens_per_second": 100.0,
        },
        "semantic_selection",
    )
    assert collect_metrics.record_id(row) == "xsum_1"
    assert row["method"] == "semantic_selection_lead"
    assert row["input_tokens"] == 100
    assert row["retained_tokens"] == 50
    assert row["selector_latency_ms"] == 12.5
    assert row["ttft_ms"] == 20.0
    assert row["e2e_ms"] == 80.0


def test_collector_strict_completeness_accepts_repeated_selector_records(tmp_path):
    output = tmp_path / "semantic_selection_xsum.jsonl"
    rows = [
        {"example_id": "xsum-1", "selector": "lead", "text": "a"},
        {"example_id": "xsum-1", "selector": "tfidf", "text": "b"},
        {"example_id": "xsum-2", "selector": "lead", "text": "c"},
        {"type": "summary", "num_samples": 3},
    ]
    output.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    errors = collect_metrics.validate_completeness(
        tmp_path,
        expected_baselines=["semantic_selection"],
        expected_datasets=["xsum"],
        expected_samples=2,
    )
    assert errors == []


def test_collector_strict_completeness_rejects_missing_sample(tmp_path):
    output = tmp_path / "llmlingua_xsum.jsonl"
    rows = [
        {"doc_id": "xsum-1", "summary": "a"},
        {"type": "summary", "num_samples": 1},
    ]
    output.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    errors = collect_metrics.validate_completeness(
        tmp_path,
        expected_baselines=["llmlingua"],
        expected_datasets=["xsum"],
        expected_samples=2,
    )
    assert any("1/2" in error for error in errors)
