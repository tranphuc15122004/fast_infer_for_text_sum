"""Tests cho auto-batch profiling và tích hợp vào cache pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


def test_default_cache_buckets_and_schedule_lookup() -> None:
    from MR_DFlash.cache_batching import CacheBatchSchedule, default_bucket_boundaries

    assert default_bucket_boundaries(32768, 8192) == [8192, 16384, 24576, 32768]
    schedule = CacheBatchSchedule.from_payload(
        {
            "schema_version": "mr_dflash_cache_batch_profile_v1",
            "target_model_path": "tiny-target",
            "feature_layer_ids": [0, 1],
            "max_length": 64,
            "requested_torch_dtype": "bfloat16",
            "attention_backend": "sdpa",
            "buckets": [
                {"min_length": 1, "max_length": 8, "selected_batch_size": 4},
                {"min_length": 9, "max_length": 64, "selected_batch_size": 1},
            ],
        }
    )
    assert schedule.batch_size_for_length(1) == 4
    assert schedule.batch_size_for_length(8) == 4
    assert schedule.batch_size_for_length(9) == 1
    assert schedule.batch_size_for_length(64) == 1
    with pytest.raises(ValueError, match="vượt max_length"):
        schedule.batch_size_for_length(65)


def test_choose_largest_safe_batch_size_uses_reserved_memory() -> None:
    from MR_DFlash.cache_batching import choose_largest_safe_batch_size

    results = [
        {"batch_size": 1, "status": "pass", "peak_memory_reserved_bytes": 90},
        {"batch_size": 2, "status": "pass", "peak_memory_reserved_bytes": 150},
        {"batch_size": 4, "status": "oom", "peak_memory_reserved_bytes": None},
        {"batch_size": 8, "status": "not_run", "peak_memory_reserved_bytes": None},
    ]
    assert choose_largest_safe_batch_size(results, vram_limit_bytes=160) == 2

    with pytest.raises(RuntimeError, match="batch size 1"):
        choose_largest_safe_batch_size(
            [{"batch_size": 1, "status": "oom"}],
            vram_limit_bytes=160,
        )


def test_cache_uses_profiled_batch_size_per_length_bucket(tmp_path: Path, monkeypatch) -> None:
    script_dir = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import cache_target_features
    from MR_DFlash.tokenized_data import write_tokenized_manifest

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 2

    class FakeCapturer:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.tokenizer = TinyTokenizer()
            self.device = torch.device("cpu")
            self.layer_ids = [0, 1]
            self.context_feature_dim = 4
            self.model = type("Model", (), {"config": type("Config", (), {"hidden_size": 2})()})()
            self.batch_sizes = []
            self.__class__.instances.append(self)

        def capture_batch(self, input_ids, attention_mask):
            self.batch_sizes.append(int(input_ids.shape[0]))
            return [
                torch.ones(int(length), 4, dtype=torch.bfloat16)
                for length in attention_mask.sum(dim=-1).tolist()
            ]

        def close(self):
            return None

    monkeypatch.setattr(cache_target_features, "HFTargetCapture", FakeCapturer)
    tokenized = tmp_path / "tokenized"
    tokenized.mkdir()
    samples = [
        {"id": "a", "input_ids": torch.arange(6), "loss_mask": torch.ones(6), "length": 6},
        {"id": "b", "input_ids": torch.arange(5), "loss_mask": torch.ones(5), "length": 5},
        {"id": "c", "input_ids": torch.arange(4), "loss_mask": torch.ones(4), "length": 4},
    ]
    torch.save({"samples": samples}, tokenized / "shard_00000.pt")
    write_tokenized_manifest(
        tokenized,
        shards=[{"path": "shard_00000.pt", "count": 3}],
        num_samples=3,
        target_model="tiny-target",
        feature_layer_ids=[0, 1],
        chat_template="tiny",
        max_length=64,
        supervision_mode="last_assistant",
    )
    profile = tmp_path / "batch_profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "mr_dflash_cache_batch_profile_v1",
                "target_model_path": "tiny-target",
                "feature_layer_ids": [0, 1],
                "max_length": 64,
                "requested_torch_dtype": "bfloat16",
                "attention_backend": "sdpa",
                "buckets": [
                    {"min_length": 1, "max_length": 8, "selected_batch_size": 2},
                    {"min_length": 9, "max_length": 64, "selected_batch_size": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    cache_target_features.cache_dataset(
        target_model_path="tiny-target",
        tokenized_path=str(tokenized),
        output_path=str(tmp_path / "cache"),
        max_length=64,
        batch_size=1,
        bucket_buffer_size=2,
        shard_size=4,
        layer_ids=[0, 1],
        device="cpu",
        batch_profile=str(profile),
    )
    assert FakeCapturer.instances[0].batch_sizes == [2, 1]
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert manifest["cache_batch_profile"] == str(profile)
    assert len(manifest["cache_batch_profile_sha256"]) == 64


def test_pipeline_auto_batch_adds_profile_stage_and_profile_argument(tmp_path: Path) -> None:
    script_dir = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from run_preprocess_pipeline import PipelineOptions, build_stage_plan

    options = PipelineOptions(
        repo_root=tmp_path,
        data_root=tmp_path / "pilot",
        target_model_path="/models/Qwen3-4B",
        full_context=True,
        full_context_length=32768,
        cache_auto_batch=True,
        cache_profile_gpu_id=0,
        cache_profile_max_batch_size=8,
        cache_profile_vram_limit_gb=170.0,
        parallel_gpu_ids=(0, 1, 2, 3),
    )
    plan = build_stage_plan(options)
    names = [stage.name for stage in plan]
    profile_name = "profile_cache_batch_full"
    assert profile_name in names
    assert names.index(profile_name) < names.index("cache_full_train")
    profile = next(stage for stage in plan if stage.name == profile_name)
    assert profile.command[1].endswith("profile_cache_batches.py")
    assert "--vram-limit-gb" in profile.command
    cache = next(stage for stage in plan if stage.name == "cache_full_train")
    assert "--batch-profile" in cache.command
    assert str(tmp_path / "pilot" / "manifests" / "cache_batch_profile_full.json") in cache.command


def test_pipeline_auto_batch_adds_profile_for_cache_only_selection(tmp_path: Path) -> None:
    script_dir = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from run_preprocess_pipeline import PipelineOptions, _add_auto_batch_dependencies, _select_stages, build_stage_plan

    options = PipelineOptions(
        repo_root=tmp_path,
        data_root=tmp_path / "pilot",
        target_model_path="/models/Qwen3-4B",
        full_context=True,
        full_context_length=32768,
        cache_auto_batch=True,
    )
    plan = build_stage_plan(options)
    selected = _select_stages(plan, only=["cache_full_train"], from_stage=None, stop_after=None)
    selected = _add_auto_batch_dependencies(
        plan, selected, enabled=True, has_explicit_profile=False
    )
    assert [stage.name for stage in selected] == [
        "profile_cache_batch_full",
        "cache_full_train",
    ]


def test_parallel_cache_worker_receives_batch_profile(tmp_path: Path) -> None:
    script_dir = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from parallel_stage import _build_worker_command, parse_args

    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    args = parse_args(
        [
            "--mode", "cache",
            "--gpu-ids", "0", "1",
            "--input", str(tmp_path / "input.jsonl"),
            "--tokenized-path", str(tmp_path / "tokenized"),
            "--output", str(tmp_path / "cache"),
            "--manifest", str(tmp_path / "manifest.json"),
            "--target-model-path", "tiny-target",
            "--max-length", "64",
            "--target-layer-ids", "0", "1",
            "--batch-profile", str(profile),
        ]
    )
    command = _build_worker_command(args, tmp_path / "rank_00", 0, 2)
    assert command[command.index("--batch-profile") + 1] == str(profile)


def test_pipeline_allow_short_is_explicitly_forwarded_to_prepare(tmp_path: Path) -> None:
    script_dir = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from run_preprocess_pipeline import PipelineOptions, build_stage_plan, parse_args

    assert parse_args([]).allow_short is False
    args = parse_args(["--allow-short"])
    assert args.allow_short is True
    options = PipelineOptions(
        repo_root=tmp_path,
        data_root=tmp_path / "pilot",
        target_model_path="/models/Qwen3-4B",
        allow_short=True,
    )
    prepare = next(stage for stage in build_stage_plan(options) if stage.name == "prepare")
    assert "--allow-short" in prepare.command
