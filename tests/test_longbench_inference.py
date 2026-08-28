import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_longbench_loader_maps_master_values(tmp_path):
    master = tmp_path / "master.env"
    master.write_text(
        'LONG_BENCH_DATA_DIR="data/fixture"\n'
        'LONG_BENCH_OUTPUT_DIR="outputs/fixture"\n'
        'LONG_BENCH_MODEL="/models/llama"\n'
        'LONG_BENCH_BASELINES="vanilla_hf vanilla_fa"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/common/config.sh; "
            "fast_infer_load_config longbench; "
            "printf '%s\\n' \"$LONG_BENCH_MODEL\" \"$LONG_BENCH_BASELINES\"",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "FAST_INFER_MASTER_CONFIG": str(master),
            "FAST_INFER_PYTHON": sys.executable,
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "/models/llama" in result.stdout
    assert "vanilla_hf vanilla_fa" in result.stdout


def test_longbench_launcher_uses_master_and_shared_runtime():
    text = (ROOT / "scripts/run_longbench_200.sh").read_text(encoding="utf-8")
    assert "fast_infer_load_config longbench" in text
    assert "scripts/common/runtime.sh" in text
    assert "run_longbench_200.py" in text


def test_measure_call_returns_elapsed_and_output():
    from common.benchmark_runtime import measure_call

    value, timing = measure_call(lambda: "ok", device=torch.device("cpu"))

    assert value == "ok"
    assert timing["e2e_ms"] >= 0
    assert timing["device"] == "cpu"


def test_build_status_record_never_invents_performance_metrics():
    from common.benchmark_runtime import build_status_record

    row = build_status_record(
        method="vanilla_fa",
        dataset="lcc",
        sample_id="x",
        status="unsupported_cpu",
        reason="CUDA unavailable",
    )

    assert row["status"] == "unsupported_cpu"
    assert row["e2e_ms"] is None
    assert row["throughput_tok_s"] is None


def test_vanilla_parser_exposes_distinct_attention_defaults():
    from infer_vanilla_fa import build_parser as fa_parser
    from infer_vanilla_hf import build_parser as hf_parser

    assert hf_parser().parse_args(["--output", "x"]).attention_backend == "eager"
    assert (
        fa_parser().parse_args(["--output", "x"]).attention_backend
        == "flash_attention_2"
    )


def test_vanilla_record_contains_shared_timing_fields():
    from common.benchmark_runtime import build_sample_record

    record = build_sample_record(
        method="vanilla_hf",
        dataset="lcc",
        sample_id="id",
        model="m",
        input_tokens=10,
        output_tokens=2,
        timing={"e2e_ms": 4.0, "prefill_ms": 1.0, "decode_ms": 3.0},
        config={"attention_backend": "eager"},
        text="x",
        reference_output="y",
    )

    assert record["throughput_tok_s"] == 500.0
    assert record["attention_backend"] == "eager"


def test_registry_contains_exactly_requested_baselines():
    from common.longbench_adapter import BASELINES

    assert BASELINES == (
        "vanilla_hf",
        "vanilla_fa",
        "magicdec",
        "longspec",
        "eagle3",
        "dflash",
        "specextend",
        "sssd",
        "fafo",
    )


def test_eagle_converter_preserves_canonical_id_and_reference(tmp_path):
    from common.longbench_adapter import convert_records_for_baseline

    output = tmp_path / "eagle.jsonl"
    convert_records_for_baseline(
        "eagle3",
        [{"id": "lcc_1", "prompt": "code", "reference": "next"}],
        output,
    )

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["question_id"] == "lcc_1"
    assert row["turns"] == ["code"]
    assert row["answer"] == "next"


def test_orchestrator_smoke_preflight_writes_manifest_without_loading_model(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_longbench_200.py",
            "--mode",
            "smoke",
            "--preflight-only",
            "--baselines",
            "vanilla_hf",
            "--datasets",
            "lcc",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifests = list(tmp_path.glob("*/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["mode"] == "smoke"
    assert manifest["preflight_only"] is True


def test_full_profile_requires_cuda_unless_unsupported_is_allowed():
    from run_longbench_200 import resolve_profile

    with pytest.raises(SystemExit):
        resolve_profile(mode="full", cuda_available=False, allow_unsupported=False)


def test_collector_ignores_preflight_records_for_speed_aggregates(tmp_path):
    from collect_metrics import load_run_records

    path = tmp_path / "vanilla_fa" / "lcc.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "status": "unsupported_cpu",
            "method": "vanilla_fa",
            "dataset": "lcc",
            "e2e_ms": None,
        },
        {"type": "summary", "status": "preflight_only"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = load_run_records(tmp_path)

    assert result["coverage"]["success"] == 0


def test_code_completion_aggregate_excludes_rouge_keys():
    from collect_metrics import aggregate_run_group

    result = aggregate_run_group(
        [
            {
                "status": "success",
                "task_type": "code_completion",
                "text": "return x",
                "reference_output": "return x",
            }
        ]
    )

    assert "rouge1_f" not in result["quality"]
    assert result["quality"]["code_exact_match"] == 1.0
