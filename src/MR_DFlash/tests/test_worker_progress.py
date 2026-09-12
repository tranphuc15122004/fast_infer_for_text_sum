"""Tests cho heartbeat/progress artifact của worker GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_worker_progress_is_atomic_and_contains_gpu_context(tmp_path: Path, monkeypatch) -> None:
    from progress import read_progress, write_progress

    monkeypatch.setenv("MR_DFLASH_GPU_ID", "1")
    path = tmp_path / "progress.json"
    write_progress(
        path,
        phase="generating",
        sample_id="sample-7",
        row_index=7,
        generated_tokens=32,
        generation_budget=2048,
    )

    payload = read_progress(path)
    assert payload["schema_version"] == "mr_dflash_worker_progress_v1"
    assert payload["gpu_id"] == 1
    assert payload["phase"] == "generating"
    assert payload["sample_id"] == "sample-7"
    assert payload["generated_tokens"] == 32
    assert not path.with_name(".progress.json.tmp").exists()
    json.dumps(payload)


def test_progress_estimate_reports_eta_from_run_tokens() -> None:
    from progress import estimate_fields

    estimate = estimate_fields(
        total_samples=100,
        total_tokens=1_000,
        completed_samples=25,
        completed_tokens=250,
        run_tokens=250,
        active_elapsed_seconds=10.0,
    )

    assert estimate["remaining_samples"] == 75
    assert estimate["remaining_tokens"] == 750
    assert estimate["throughput_tokens_per_second"] == 25.0
    assert estimate["eta_seconds"] == 30.0
    assert estimate["eta_human"] == "30.0s"


def test_progress_reporter_preserves_worker_context(tmp_path: Path) -> None:
    from progress import ProgressReporter, read_progress

    reporter = ProgressReporter(tmp_path / "progress.json")
    reporter.set_context(total_samples=17)
    reporter.update("starting", completed_samples=0)
    reporter.update("reading_sample", completed_samples=3, sample_id="s3")

    payload = read_progress(tmp_path / "progress.json")
    assert payload["total_samples"] == 17
    assert payload["completed_samples"] == 3


def test_progress_reporter_uses_pipeline_environment_path(tmp_path: Path, monkeypatch) -> None:
    from progress import ProgressReporter, read_progress

    progress_path = tmp_path / "pipeline.progress.json"
    monkeypatch.setenv("MR_DFLASH_PROGRESS_PATH", str(progress_path))
    monkeypatch.setenv("MR_DFLASH_PROGRESS_TOTAL_SAMPLES", "9")

    reporter = ProgressReporter(None)
    reporter.update("processing", completed_samples=4)

    payload = read_progress(progress_path)
    assert payload["phase"] == "processing"
    assert payload["total_samples"] == 9
    assert payload["completed_samples"] == 4


def test_token_heartbeat_preserves_completed_sample_count(tmp_path: Path) -> None:
    from progress import ProgressReporter, read_progress

    reporter = ProgressReporter(tmp_path / "progress.json", interval_tokens=1)
    reporter.update("sample_done", completed_samples=7, sample_id="s7")
    reporter.maybe_tokens(16, generation_budget=32)

    payload = read_progress(tmp_path / "progress.json")
    assert payload["phase"] == "generating"
    assert payload["generated_tokens"] == 16
    assert payload["completed_samples"] == 7


def test_regenerate_progress_counts_rows_assigned_to_worker(tmp_path: Path) -> None:
    from regenerate_pilot import _count_shard_rows

    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "".join(json.dumps({"id": f"s{i}"}) + "\n" for i in range(10)),
        encoding="utf-8",
    )

    assert _count_shard_rows(input_path, shard_index=0, num_shards=3) == 4
    assert _count_shard_rows(input_path, shard_index=1, num_shards=3) == 3
    assert _count_shard_rows(input_path, shard_index=2, num_shards=3) == 3


def test_parallel_aggregate_uses_slowest_worker_eta() -> None:
    from parallel_stage import aggregate_progress

    aggregate = aggregate_progress(
        [
            {"progress": {"total_samples": 50, "completed_samples": 25, "total_tokens": 500, "completed_tokens": 250, "remaining_samples": 25, "remaining_tokens": 250, "eta_seconds": 10.0, "throughput_tokens_per_second": 25.0}},
            {"progress": {"total_samples": 50, "completed_samples": 10, "total_tokens": 500, "completed_tokens": 100, "remaining_samples": 40, "remaining_tokens": 400, "eta_seconds": 20.0, "throughput_tokens_per_second": 20.0}},
        ]
    )

    assert aggregate["total_samples"] == 100
    assert aggregate["completed_samples"] == 35
    assert aggregate["total_tokens"] == 1_000
    assert aggregate["throughput_tokens_per_second"] == 45.0
    assert aggregate["eta_seconds"] == 20.0
    assert aggregate["eta_human"] == "20.0s"


def test_parallel_aggregate_keeps_global_total_before_all_gpu_heartbeats() -> None:
    from parallel_stage import aggregate_progress

    aggregate = aggregate_progress(
        [
            {"rank": 0, "progress": {"total_samples": 25, "completed_samples": 7}},
            {"rank": 1, "progress": {}},
            {"rank": 2, "progress": {"total_samples": 25, "completed_samples": 3}},
            {"rank": 3, "progress": {}},
        ],
        total_samples=100,
    )

    assert aggregate["total_samples"] == 100
    assert aggregate["completed_samples"] == 10
    assert aggregate["remaining_samples"] == 90


def test_watch_formatter_reports_aggregate_eta() -> None:
    from watch_parallel_stage import format_status

    output = format_status(
        {
            "status": "running",
            "aggregate": {
                "completed_samples": 3,
                "total_samples": 10,
                "completed_tokens": 300,
                "total_tokens": 1000,
                "throughput_tokens_per_second": 30.0,
                "eta_human": "23.3s",
            },
            "workers": [],
        }
    )

    assert "samples=3/10" in output
    assert "tokens=300/1000" in output
    assert "eta=23.3s" in output


def test_parallel_status_includes_worker_progress(tmp_path: Path) -> None:
    from progress import write_progress
    from parallel_stage import build_worker_status

    root = tmp_path / "rank_00"
    root.mkdir(parents=True)
    write_progress(root / "progress.json", phase="generating", sample_id="s0", generated_tokens=8)

    status = build_worker_status(
        rank=0,
        gpu_id=1,
        pid=123,
        return_code=None,
        worker_root=root,
    )

    assert status["gpu_id"] == 1
    assert status["pid"] == 123
    assert status["progress"]["phase"] == "generating"
    assert status["progress"]["generated_tokens"] == 8


def test_watch_formatter_reports_each_gpu_and_staleness() -> None:
    from watch_parallel_stage import format_status

    output = format_status(
        {
            "status": "running",
            "workers": [
                {
                    "rank": 0,
                    "gpu_id": 1,
                    "pid": 123,
                    "return_code": None,
                    "progress_age_seconds": 2.0,
                    "progress": {
                        "phase": "generating",
                        "sample_id": "s0",
                        "generated_tokens": 64,
                        "generation_budget": 2048,
                        "completed_samples": 7,
                    },
                }
            ],
        }
    )
    assert "gpu=1" in output
    assert "phase=generating" in output
    assert "sample=s0" in output
    assert "tokens=64/2048" in output


def test_generation_heartbeat_processor_does_not_change_scores(tmp_path: Path) -> None:
    from progress import ProgressReporter, read_progress
    from regenerate_pilot import _ProgressLogitsProcessor

    reporter = ProgressReporter(tmp_path / "progress.json", interval_tokens=1)
    processor = _ProgressLogitsProcessor(
        reporter,
        prompt_tokens=4,
        sample_id="s0",
        budget=8,
    )
    scores = torch.randn(1, 32)
    result = processor(torch.ones((1, 5), dtype=torch.long), scores)
    assert result is scores
    assert torch.equal(result, scores)
    assert read_progress(tmp_path / "progress.json")["generated_tokens"] == 1


def test_feature_cache_audit_checks_all_offsets(tmp_path: Path) -> None:
    from verify_feature_cache import validate_feature_cache

    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"id": "s0"}) + "\n", encoding="utf-8")
    cache = tmp_path / "cache"
    from MR_DFlash.offline_features import write_feature_shard, write_sharded_feature_manifest

    write_feature_shard(
        cache / "shard_00000.pt",
        [
            {
                "id": "s0",
                "input_ids": [1, 2, 3],
                "loss_mask": [0, 1, 1],
                "hidden_states": torch.ones(3, 4),
            }
        ],
    )
    write_sharded_feature_manifest(
        cache,
        target_model_path="target",
        feature_layer_ids=[1, 2],
        hidden_size=2,
        feature_width=4,
        max_length=8,
        requested_torch_dtype="bfloat16",
        shards=[{"path": "shard_00000.pt", "count": 1, "ids": ["s0"], "lengths": [3]}],
        sample_ids=["s0"],
    )
    report = validate_feature_cache(data_path=data, cache_path=cache)
    assert report["valid"] is True
    assert report["num_samples"] == 1
    assert report["checked_offsets"] == 3


def test_observability_controls_do_not_change_pipeline_data_hash(tmp_path: Path) -> None:
    from dataclasses import replace

    from run_preprocess_pipeline import PipelineOptions, pipeline_config_hash

    base = PipelineOptions(
        repo_root=tmp_path,
        data_root=tmp_path / "data",
        target_model_path="target",
    )
    changed = replace(
        base,
        progress_interval_tokens=64,
        regenerate_output_batch_size=8,
        worker_stall_timeout_seconds=900.0,
        worker_stop_file="/tmp/stop",
    )
    assert pipeline_config_hash(base) == pipeline_config_hash(changed)


def test_pipeline_parallel_progress_uses_global_input_sample_count(tmp_path: Path) -> None:
    from run_preprocess_pipeline import Stage, parallel_progress_spec

    input_path = tmp_path / "train_prompts.jsonl"
    input_path.write_text(
        "\n".join(json.dumps({"id": f"s{i}"}) for i in range(5)) + "\n\n",
        encoding="utf-8",
    )
    work_root = tmp_path / "parallel_regenerate_train"
    stage = Stage(
        name="regenerate_full_train",
        command=[
            "python3",
            "scripts/mr_dflash/parallel_stage.py",
            "--mode",
            "regenerate",
            "--input",
            str(input_path),
            "--work-root",
            str(work_root),
        ],
        artifacts=(),
    )

    spec = parallel_progress_spec(stage)

    assert spec is not None
    assert spec["mode"] == "regenerate"
    assert spec["total_samples"] == 5
    assert spec["status_path"] == work_root / "status.json"


def test_pipeline_parallel_cache_progress_uses_tokenized_sample_count(tmp_path: Path) -> None:
    from run_preprocess_pipeline import Stage, parallel_progress_spec

    input_path = tmp_path / "regenerated_train.jsonl"
    input_path.write_text(
        "\n".join(json.dumps({"id": f"s{i}"}) for i in range(4)) + "\n",
        encoding="utf-8",
    )
    tokenized_path = tmp_path / "tokenized" / "train"
    tokenized_path.mkdir(parents=True)
    (tokenized_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "mr_dflash_tokenized_v1",
                "num_samples": 5,
                "shards": [{"path": "shard_00000.pt", "count": 5}],
            }
        ),
        encoding="utf-8",
    )
    stage = Stage(
        name="cache_8k_train",
        command=[
            "python3",
            "scripts/mr_dflash/parallel_stage.py",
            "--mode",
            "cache",
            "--input",
            str(input_path),
            "--tokenized-path",
            str(tokenized_path),
            "--work-root",
            str(tmp_path / "parallel_cache_train"),
        ],
        artifacts=(),
    )

    spec = parallel_progress_spec(stage)

    assert spec is not None
    assert spec["total_samples"] == 5


def test_pipeline_parallel_progress_description_is_concise() -> None:
    from run_preprocess_pipeline import parallel_progress_description

    assert parallel_progress_description("regenerate", "regenerate_full_train") == "regenerate train"
    assert "parallel" not in parallel_progress_description("regenerate", "regenerate_full_train")


def test_pipeline_progress_snapshot_reads_parallel_aggregate(tmp_path: Path) -> None:
    from run_preprocess_pipeline import _stage_progress_snapshot

    path = tmp_path / "status.json"
    path.write_text(
        json.dumps(
            {
                "status": "running",
                "aggregate": {"total_samples": 10, "completed_samples": 4},
            }
        ),
        encoding="utf-8",
    )

    total, completed, _payload = _stage_progress_snapshot(path, 99)

    assert (total, completed) == (10, 4)


def test_pipeline_progress_snapshot_keeps_parallel_total_fixed(tmp_path: Path) -> None:
    from run_preprocess_pipeline import _stage_progress_snapshot

    path = tmp_path / "status.json"
    path.write_text(
        json.dumps(
            {
                "status": "running",
                "aggregate": {"total_samples": 25, "completed_samples": 4},
            }
        ),
        encoding="utf-8",
    )

    total, completed, _payload = _stage_progress_snapshot(
        path, 100, fixed_total=True
    )

    assert (total, completed) == (100, 4)
