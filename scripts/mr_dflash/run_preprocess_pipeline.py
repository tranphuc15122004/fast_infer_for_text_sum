"""Chạy toàn bộ pipeline chuẩn bị dữ liệu và cache target theo từng stage.

Pipeline này cố ý là một orchestrator mỏng: mỗi stage vẫn gọi script độc lập
để có thể chạy/debug riêng, nhưng wrapper lưu lại command, stdout/stderr,
artifact và trạng thái thành công/thất bại. Vì vậy trên server B200 chỉ cần
chạy một lệnh, sau đó có thể tiếp tục bằng ``--resume`` hoặc chạy lại một
stage bằng ``--only-stage``.

Pipeline không train model. Train là consumer của feature cache và có thể
chạy nhiều lần cho DFlash-2L, MR-DFlash-2S và DFlash-5L mà không chạy lại
target. Đây là ranh giới quan trọng để tránh vô tình tốn GPU/ghi đè cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - server production là Linux
    fcntl = None

from _common import (
    REPO_ROOT as DEFAULT_REPO_ROOT,
    resolve_parallel_work_root,
    write_json,
)
from prepare_server_data import (
    DEFAULT_ARXIV_SOURCE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SHAREGPT_SOURCE,
)


DEFAULT_TARGET_MODEL = "/workspace/storage-shared/models/Qwen3-4B"
DEFAULT_FEATURE_LAYER_IDS = (1, 9, 17, 25, 33)


@dataclass(frozen=True)
class PipelineOptions:
    """Các tham số ảnh hưởng đến artifact của pipeline."""

    repo_root: Path
    data_root: Path
    target_model_path: str
    sharegpt_source: str = DEFAULT_SHAREGPT_SOURCE
    arxiv_source: str = DEFAULT_ARXIV_SOURCE
    sharegpt_count: int = 50_000
    arxiv_count: int = 50_000
    # Mặc định strict để không vô tình chạy benchmark thiếu dữ liệu. Khi bật,
    # prepare/build dùng toàn bộ số mẫu hợp lệ thực tế nếu source ngắn hơn yêu
    # cầu và ghi số lượng đó vào manifest (không nhân bản sample).
    allow_short: bool = False
    max_lengths: tuple[int, ...] = (3072, 8192)
    full_context: bool = False
    full_context_length: int = 32768
    cache_splits: tuple[str, ...] = ("train", "val")
    target_layer_ids: tuple[int, ...] = DEFAULT_FEATURE_LAYER_IDS
    supervision_mode: str = "last_assistant"
    seed: int = 42
    max_new_tokens: int = 768
    overflow_policy: str = "error"
    sample_error_policy: str = "error"
    temperature: float = 0.0
    analysis_limit: int = 1000
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    target_revision: Optional[str] = None
    cache_batch_size_3k: int = 2
    # Cấu hình áp dụng cho mọi regime dài hơn 3K (8K, 16K, 32K, ...).
    # Tên không gắn với một độ dài cụ thể để tránh hiểu nhầm rằng đây là
    # batch size của đúng 8.192 token.
    cache_batch_size_long_context: int = 1
    cache_bucket_buffer_3k: int = 16
    cache_bucket_buffer_long_context: int = 8
    cache_shard_size_3k: int = 64
    cache_shard_size_long_context: int = 32
    cache_attention_backend: str = "sdpa"
    cache_io_threads: int = 2
    cache_io_queue_size: int = 4
    # Auto-batch profile được tạo trước cache trên một GPU rồi dùng cố định
    # cho mọi worker theo bucket độ dài. ``cache_batch_profile`` cho phép
    # truyền profile đã tạo từ trước và bỏ qua stage profile.
    cache_auto_batch: bool = False
    cache_batch_profile: Optional[str] = None
    cache_profile_gpu_id: int = 0
    cache_profile_max_batch_size: int = 8
    cache_profile_vram_limit_gb: Optional[float] = None
    cache_profile_headroom_fraction: float = 0.10
    cache_profile_bucket_step: int = 8192
    parallel_gpu_ids: tuple[int, ...] = ()
    progress_interval_tokens: int = 256
    regenerate_output_batch_size: int = 1
    worker_stall_timeout_seconds: float = 0.0
    worker_stop_file: Optional[str] = None
    local_files_only: bool = True
    resume: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class Stage:
    """Một command độc lập và các file phải có sau khi command thành công."""

    name: str
    command: list[str]
    artifacts: Sequence[Path]
    kind: str = "files"


class _PipelineLock:
    """Khóa OS-level để hai process không cùng ghi một data-root.

    ``flock`` tự nhả khóa khi process chết, nên một lần kill không tạo stale
    lock khiến lần ``--resume`` sau bị kẹt. File vẫn giữ PID/command để chẩn
    đoán khi có job khác đang chạy.
    """

    def __init__(self, data_root: Path, config_hash: str) -> None:
        self.path = data_root / "pipeline_state" / ".pipeline.lock"
        self.config_hash = config_hash
        self._handle = None

    def acquire(self) -> None:
        if fcntl is None:
            raise RuntimeError("pipeline lock cần fcntl; chỉ hỗ trợ server Linux")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                f"data-root đang được một pipeline khác sử dụng: {self.path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "config_hash": self.config_hash,
                    "started_at": _now(),
                    "command": sys.argv,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_script(options: PipelineOptions, name: str) -> str:
    return str(options.repo_root / "scripts" / "mr_dflash" / name)


def _resume_args(options: PipelineOptions) -> list[str]:
    return ["--resume"] if options.resume else []


def _local_files_args(options: PipelineOptions) -> list[str]:
    return ["--local-files-only"] if options.local_files_only else []


def _regime_name(max_length: int) -> str:
    if max_length == 3072:
        return "3k"
    if max_length == 8192:
        return "8k"
    return str(max_length)


def _regenerated_root(data_root: Path, regime: str, max_length: int) -> Path:
    suffix = "" if max_length == 8192 else f"_{regime}"
    return data_root / f"regenerated{suffix}"


def _tokenized_root(data_root: Path, regime: str, max_length: int) -> Path:
    suffix = "" if max_length == 8192 else f"_{regime}"
    return data_root / f"tokenized{suffix}"


def _feature_root(data_root: Path, regime: str) -> Path:
    return data_root / f"target_features_qwen3_4b_{regime}"


def _cache_sizes(options: PipelineOptions, regime: str) -> tuple[int, int, int]:
    if regime == "3k":
        return (
            options.cache_batch_size_3k,
            options.cache_bucket_buffer_3k,
            options.cache_shard_size_3k,
        )
    return (
        options.cache_batch_size_long_context,
        options.cache_bucket_buffer_long_context,
        options.cache_shard_size_long_context,
    )


def build_stage_plan(options: PipelineOptions) -> list[Stage]:
    """Dựng plan deterministic; không chạy command và không đọc dữ liệu."""

    python = sys.executable
    root = options.data_root
    normalized = root / "normalized"
    manifests = root / "manifests"
    plan: list[Stage] = []

    prepare_command = [
        python,
        _stage_script(options, "prepare_server_data.py"),
        "--sharegpt-source",
        options.sharegpt_source,
        "--arxiv-source",
        options.arxiv_source,
        "--output-root",
        str(root),
        "--sharegpt-count",
        str(options.sharegpt_count),
        "--arxiv-count",
        str(options.arxiv_count),
        "--seed",
        str(options.seed),
        *(["--allow-short"] if options.allow_short else []),
        "--tokenizer",
        options.target_model_path,
        *(_resume_args(options)),
    ]
    plan.append(
        Stage(
            name="prepare",
            command=prepare_command,
            artifacts=(
                normalized / "sharegpt_prompts.jsonl",
                normalized / "arxiv_prompts.jsonl",
                normalized / "pilot_prompts.jsonl",
                normalized / "train_prompts.jsonl",
                normalized / "val_prompts.jsonl",
                normalized / "test_prompts.jsonl",
                manifests / "source_manifest.json",
                manifests / "split_manifest.json",
            ),
        )
    )

    plan.append(
        Stage(
            name="analyze",
            command=[
                python,
                _stage_script(options, "analyze_pilot_data.py"),
                "--input",
                str(normalized / "pilot_prompts.jsonl"),
                "--output",
                str(manifests / "analysis.json"),
                "--limit",
                str(options.analysis_limit),
            ],
            artifacts=(manifests / "analysis.json",),
        )
    )

    # Tách các stage theo regime/split. Nếu 3K đã xong nhưng 8K lỗi, resume
    # chỉ phải tiếp tục 8K; các artifact 3K vẫn dùng chung cho matrix train.
    configured_lengths = (
        (int(options.full_context_length),)
        if options.full_context
        else tuple(int(value) for value in options.max_lengths)
    )
    for max_length in configured_lengths:
        max_length = int(max_length)
        regime = "full" if options.full_context else _regime_name(max_length)
        regenerated = _regenerated_root(root, regime, max_length)
        tokenized = _tokenized_root(root, regime, max_length)
        feature_root = _feature_root(root, regime)

        for split in ("train", "val", "test"):
            output = regenerated / f"{split}.jsonl"
            skipped_report = regenerated / f"{split}.skipped.jsonl"
            manifest = manifests / f"regeneration_{regime}_{split}.json"
            command = [
                python,
                _stage_script(options, "regenerate_pilot.py"),
                "--input",
                str(normalized / f"{split}_prompts.jsonl"),
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--target-model-path",
                options.target_model_path,
                "--max-length",
                str(max_length),
                "--max-new-tokens",
                str(options.max_new_tokens),
                "--temperature",
                str(options.temperature),
                "--seed",
                str(options.seed),
                "--device",
                options.device,
                "--torch-dtype",
                options.torch_dtype,
                *(["--preserve-full-input"] if options.full_context else []),
                "--overflow-policy",
                options.overflow_policy,
                "--sample-error-policy",
                options.sample_error_policy,
                "--skipped-report",
                str(skipped_report),
                *(["--target-revision", options.target_revision] if options.target_revision else []),
                *(_local_files_args(options)),
                *(_resume_args(options)),
            ]
            if options.parallel_gpu_ids:
                command = [
                    python,
                    _stage_script(options, "parallel_stage.py"),
                    "--mode",
                    "regenerate",
                    "--gpu-ids",
                    *(str(value) for value in options.parallel_gpu_ids),
                    "--input",
                    str(normalized / f"{split}_prompts.jsonl"),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--work-root",
                    str(resolve_parallel_work_root(regenerated / f"{split}.jsonl", "regenerate")),
                    "--target-model-path",
                    options.target_model_path,
                    "--max-length",
                    str(max_length),
                    "--max-new-tokens",
                    str(options.max_new_tokens),
                    "--temperature",
                    str(options.temperature),
                    "--seed",
                    str(options.seed),
                    "--torch-dtype",
                    options.torch_dtype,
                    "--overflow-policy",
                    options.overflow_policy,
                    "--sample-error-policy",
                    options.sample_error_policy,
                    "--supervision-mode",
                    options.supervision_mode,
                    "--progress-interval-tokens",
                    str(options.progress_interval_tokens),
                    "--output-batch-size",
                    str(options.regenerate_output_batch_size),
                    "--stall-timeout-seconds",
                    str(options.worker_stall_timeout_seconds),
                    *(["--stop-file", options.worker_stop_file] if options.worker_stop_file else []),
                    *(["--preserve-full-input"] if options.full_context else []),
                    *(["--target-revision", options.target_revision] if options.target_revision else []),
                    *(_local_files_args(options)),
                    *(_resume_args(options)),
                ]
            plan.append(
                Stage(
                    name=f"regenerate_{regime}_{split}",
                    command=command,
                    artifacts=(output, skipped_report, manifest),
                )
            )

        for split in ("train", "val", "test"):
            regenerated_file = regenerated / f"{split}.jsonl"
            report = manifests / f"validation_{regime}_{split}.json"
            plan.append(
                Stage(
                    name=f"validate_{regime}_{split}",
                    command=[
                        python,
                        _stage_script(options, "validate_pilot_dataset.py"),
                        "--input",
                        str(regenerated_file),
                        "--tokenizer",
                        options.target_model_path,
                        "--max-length",
                        str(max_length),
                        "--expected-target-model",
                        options.target_model_path,
                        "--require-generated",
                        "--report",
                        str(report),
                        *(_local_files_args(options)),
                    ],
                    artifacts=(regenerated_file, report),
                )
            )

        for split in ("train", "val", "test"):
            output_dir = tokenized / split
            output_manifest = output_dir / "manifest.json"
            provenance = manifests / f"tokenization_{regime}_{split}.json"
            plan.append(
                Stage(
                    name=f"tokenize_{regime}_{split}",
                    command=[
                        python,
                        _stage_script(options, "tokenize_dataset.py"),
                        "--input",
                        str(regenerated / f"{split}.jsonl"),
                        "--output",
                        str(output_dir),
                        "--provenance-manifest",
                        str(provenance),
                        "--target-model-path",
                        options.target_model_path,
                        "--max-length",
                        str(max_length),
                        "--supervision-mode",
                        options.supervision_mode,
                        "--feature-layer-ids",
                        *(str(layer) for layer in options.target_layer_ids),
                        *(_local_files_args(options)),
                        *(_resume_args(options)),
                    ],
                    artifacts=(output_manifest, provenance),
                    kind="tokenized",
                )
            )

        profile_path = (
            Path(options.cache_batch_profile)
            if options.cache_batch_profile
            else manifests / f"cache_batch_profile_{regime}.json"
        )
        if options.cache_auto_batch and not options.cache_batch_profile:
            profile_device = options.device
            if profile_device in {"cuda", "auto"}:
                profile_device = f"cuda:{options.cache_profile_gpu_id}"
            plan.append(
                Stage(
                    name=f"profile_cache_batch_{regime}",
                    command=[
                        python,
                        _stage_script(options, "profile_cache_batches.py"),
                        "--target-model-path",
                        options.target_model_path,
                        "--tokenized-path",
                        str(tokenized / "train"),
                        "--output",
                        str(profile_path),
                        "--max-length",
                        str(max_length),
                        "--target-layer-ids",
                        *(str(layer) for layer in options.target_layer_ids),
                        "--max-batch-size",
                        str(options.cache_profile_max_batch_size),
                        "--bucket-step",
                        str(options.cache_profile_bucket_step),
                        "--vram-headroom-fraction",
                        str(options.cache_profile_headroom_fraction),
                        "--attention-backend",
                        options.cache_attention_backend,
                        "--device",
                        profile_device,
                        "--torch-dtype",
                        options.torch_dtype,
                        *(
                            ["--vram-limit-gb", str(options.cache_profile_vram_limit_gb)]
                            if options.cache_profile_vram_limit_gb is not None
                            else []
                        ),
                        *(
                            ["--target-revision", options.target_revision]
                            if options.target_revision
                            else []
                        ),
                        *(_local_files_args(options)),
                    ],
                    artifacts=(profile_path,),
                    kind="profile",
                )
            )

        batch_size, bucket_buffer, shard_size = _cache_sizes(options, regime)
        use_batch_profile = bool(options.cache_auto_batch or options.cache_batch_profile)
        for split in options.cache_splits:
            output_dir = feature_root / split
            manifest = output_dir / "manifest.json"
            cache_command = [
                        python,
                        _stage_script(options, "cache_target_features.py"),
                        "--target-model-path",
                        options.target_model_path,
                        "--data-path",
                        str(regenerated / f"{split}.jsonl"),
                        "--tokenized-path",
                        str(tokenized / split),
                        "--output-path",
                        str(output_dir),
                        "--max-length",
                        str(max_length),
                        "--target-layer-ids",
                        *(str(layer) for layer in options.target_layer_ids),
                        "--batch-size",
                        str(batch_size),
                        "--bucket-buffer-size",
                        str(bucket_buffer),
                        "--shard-size",
                        str(shard_size),
                        "--attention-backend",
                        options.cache_attention_backend,
                        "--io-threads",
                        str(options.cache_io_threads),
                        "--io-queue-size",
                        str(options.cache_io_queue_size),
                        *(
                            ["--batch-profile", str(profile_path)]
                            if use_batch_profile
                            else []
                        ),
                        "--device",
                        options.device,
                        "--torch-dtype",
                        options.torch_dtype,
                        "--supervision-mode",
                        options.supervision_mode,
                        *(["--target-revision", options.target_revision] if options.target_revision else []),
                        *(_local_files_args(options)),
                        *(_resume_args(options)),
                    ]
            if options.parallel_gpu_ids:
                cache_command = [
                    python,
                    _stage_script(options, "parallel_stage.py"),
                    "--mode",
                    "cache",
                    "--gpu-ids",
                    *(str(value) for value in options.parallel_gpu_ids),
                    "--input",
                    str(regenerated / f"{split}.jsonl"),
                    "--tokenized-path",
                    str(tokenized / split),
                    "--output",
                    str(output_dir),
                    "--manifest",
                    str(manifest),
                    "--work-root",
                    str(resolve_parallel_work_root(output_dir, "cache")),
                    "--target-model-path",
                    options.target_model_path,
                    "--max-length",
                    str(max_length),
                    "--target-layer-ids",
                    *(str(layer) for layer in options.target_layer_ids),
                    "--batch-size",
                    str(batch_size),
                    "--bucket-buffer-size",
                    str(bucket_buffer),
                    "--shard-size",
                    str(shard_size),
                    "--attention-backend",
                    options.cache_attention_backend,
                    "--io-threads",
                    str(options.cache_io_threads),
                    "--io-queue-size",
                    str(options.cache_io_queue_size),
                    *(
                        ["--batch-profile", str(profile_path)]
                        if use_batch_profile
                        else []
                    ),
                    "--torch-dtype",
                    options.torch_dtype,
                    "--supervision-mode",
                    options.supervision_mode,
                    "--stall-timeout-seconds",
                    str(options.worker_stall_timeout_seconds),
                    *(["--stop-file", options.worker_stop_file] if options.worker_stop_file else []),
                    *(["--target-revision", options.target_revision] if options.target_revision else []),
                    *(_local_files_args(options)),
                    *(_resume_args(options)),
                ]
            plan.append(
                Stage(
                    name=f"cache_{regime}_{split}",
                    command=cache_command,
                    artifacts=(manifest,),
                    kind="cache",
                )
            )
    return plan


def _options_payload(options: PipelineOptions) -> dict[str, Any]:
    payload = asdict(options)
    # ``write_json`` intentionally stays strict so that accidental non-JSON
    # values are visible at call sites. PipelineOptions dùng Path để thao tác
    # filesystem, do đó normalize hai field này trước khi ghi plan/summary.
    payload["repo_root"] = str(options.repo_root)
    payload["data_root"] = str(options.data_root)
    return payload


def pipeline_config_hash(options: PipelineOptions) -> str:
    payload_options = _options_payload(options)
    # Đây là các control-plane knobs: chỉ ảnh hưởng quan sát/cleanup và cách
    # flush JSONL, không thay đổi nội dung target trajectory/cache. Loại khỏi
    # artifact hash để cập nhật cơ chế heartbeat vẫn resume được data-root cũ.
    for key in (
        "progress_interval_tokens",
        "regenerate_output_batch_size",
        "worker_stall_timeout_seconds",
        "worker_stop_file",
    ):
        payload_options.pop(key, None)
    payload = json.dumps(
        payload_options,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _marker_path(options: PipelineOptions, stage: Stage, status: str) -> Path:
    return options.data_root / "pipeline_state" / f"{stage.name}.{status}.json"


def _artifact_paths(stage: Stage) -> list[str]:
    return [str(Path(path)) for path in stage.artifacts]


def _manifest_for_stage(stage: Stage) -> Optional[Path]:
    for path in stage.artifacts:
        candidate = Path(path)
        if candidate.name == "manifest.json":
            return candidate
    return None


def _check_artifacts(stage: Stage) -> None:
    missing = [str(path) for path in stage.artifacts if not Path(path).is_file()]
    if missing:
        raise RuntimeError(
            f"stage {stage.name} không tạo đủ artifact: "
            + ", ".join(missing[:5])
        )
    if stage.kind not in {"cache", "tokenized"}:
        return
    manifest_path = _manifest_for_stage(stage)
    if manifest_path is None:
        raise RuntimeError(f"stage {stage.name} thiếu manifest artifact")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"stage {stage.name} có manifest không đọc được: {manifest_path}") from exc
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError(f"stage {stage.name} tạo manifest nhưng không có shard")
    if int(payload.get("num_samples", 0)) <= 0:
        raise RuntimeError(f"stage {stage.name} tạo cache/tokenized rỗng")
    root = manifest_path.parent
    missing_shards = [
        str(root / str(item.get("path")))
        for item in shards
        if not isinstance(item, dict) or not (root / str(item.get("path"))).is_file()
    ]
    if missing_shards:
        raise RuntimeError(
            f"stage {stage.name} thiếu shard: " + ", ".join(missing_shards[:5])
        )


def _state_payload(
    stage: Stage,
    *,
    status: str,
    config_hash: str,
    log_path: Path,
    return_code: Optional[int] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "mr_dflash_pipeline_stage_v1",
        "stage": stage.name,
        "status": status,
        "config_hash": config_hash,
        "command": stage.command,
        "command_shell": shlex.join(stage.command),
        "artifacts": _artifact_paths(stage),
        "log": str(log_path),
        "timestamp": _now(),
    }
    if return_code is not None:
        payload["return_code"] = int(return_code)
    if error:
        payload["error"] = error
    return payload


def _reusable_success(stage: Stage, options: PipelineOptions, config_hash: str) -> bool:
    marker = _marker_path(options, stage, "success")
    if not options.resume or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    old_hash = payload.get("config_hash")
    if old_hash != config_hash:
        raise RuntimeError(
            f"stage {stage.name} đã có success marker với config_hash khác; "
            "dùng --no-resume hoặc data-root mới để tránh trộn artifact"
        )
    try:
        _check_artifacts(stage)
    except RuntimeError:
        return False
    print(f"[pipeline] SKIP {stage.name}: artifact đã hoàn tất (resume)")
    return True


def run_stage(stage: Stage, options: PipelineOptions, config_hash: str) -> dict[str, Any]:
    """Chạy một stage, stream log và ghi marker thành công/thất bại."""

    if options.dry_run:
        print(f"[pipeline][dry-run] {stage.name}: {shlex.join(stage.command)}")
        return {"stage": stage.name, "status": "dry-run"}
    if _reusable_success(stage, options, config_hash):
        return {"stage": stage.name, "status": "skipped"}

    state_dir = options.data_root / "pipeline_state"
    log_dir = options.data_root / "pipeline_logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.name}.log"
    success_marker = _marker_path(options, stage, "success")
    failure_marker = _marker_path(options, stage, "failed")
    running_marker = _marker_path(options, stage, "running")
    # Khi chạy lại stage, success cũ không còn đại diện cho lần chạy hiện tại.
    success_marker.unlink(missing_ok=True)
    running_marker.unlink(missing_ok=True)
    write_json(
        running_marker,
        _state_payload(
            stage,
            status="running",
            config_hash=config_hash,
            log_path=log_path,
        ),
    )

    print(f"[pipeline] START {stage.name}")
    print(f"[pipeline] command: {shlex.join(stage.command)}")
    environment = os.environ.copy()
    source_path = str(options.repo_root / "src")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_path if not existing_pythonpath else source_path + os.pathsep + existing_pythonpath
    )
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return_code: Optional[int] = None
    process: Optional[subprocess.Popen[str]] = None

    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        # run_stage có session riêng cho stage. Forward signal xuống parallel
        # stage để nó tiếp tục forward tới từng worker/model process.
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except (ProcessLookupError, PermissionError):
                pass
        raise KeyboardInterrupt(f"stage interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)

    def restore_signal_handlers() -> None:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    try:
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n=== attempt { _now() } ===\n")
            log_handle.flush()
            process = subprocess.Popen(
                stage.command,
                cwd=str(options.repo_root),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                print(f"[{stage.name}] {line.rstrip()}", flush=True)
            return_code = process.wait()
        restore_signal_handlers()
        if return_code != 0:
            raise RuntimeError(f"command trả về exit code {return_code}")
        _check_artifacts(stage)
    except BaseException as exc:
        restore_signal_handlers()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                process.wait()
        failure_payload = _state_payload(
            stage,
            status="failed",
            config_hash=config_hash,
            log_path=log_path,
            return_code=return_code,
            error=str(exc),
        )
        write_json(failure_marker, failure_payload)
        running_marker.unlink(missing_ok=True)
        print(
            f"[pipeline] FAILED {stage.name}; xem {log_path} và {failure_marker}",
            file=sys.stderr,
        )
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeError(f"pipeline dừng tại stage {stage.name}: {exc}") from exc

    success_payload = _state_payload(
        stage,
        status="success",
        config_hash=config_hash,
        log_path=log_path,
        return_code=0,
    )
    write_json(success_marker, success_payload)
    failure_marker.unlink(missing_ok=True)
    running_marker.unlink(missing_ok=True)
    print(f"[pipeline] DONE {stage.name}")
    return success_payload


def _select_stages(
    plan: Sequence[Stage],
    *,
    only: Sequence[str],
    from_stage: Optional[str],
    stop_after: Optional[str],
) -> list[Stage]:
    names = [stage.name for stage in plan]
    unknown = [name for name in (*only, *(x for x in (from_stage, stop_after) if x)) if name not in names]
    if unknown:
        raise ValueError(f"stage không tồn tại: {', '.join(unknown)}")
    if only:
        return [stage for stage in plan if stage.name in set(only)]
    start = names.index(from_stage) if from_stage else 0
    end = names.index(stop_after) + 1 if stop_after else len(plan)
    if end < start:
        raise ValueError("--stop-after phải đứng sau hoặc bằng --from-stage")
    return list(plan[start:end])


def _add_auto_batch_dependencies(
    plan: Sequence[Stage],
    selected: Sequence[Stage],
    *,
    enabled: bool,
    has_explicit_profile: bool,
) -> list[Stage]:
    """Tự thêm stage profile khi user chọn chạy riêng một cache stage.

    Nếu chạy toàn pipeline thì profile đã có sẵn đúng vị trí trong plan. Hàm
    này xử lý các lệnh tiện dụng như ``--only-stage cache_full_train`` hoặc
    ``--from-stage cache_full_train`` để không vô tình chạy cache khi profile
    còn thiếu.
    """
    if not enabled or has_explicit_profile:
        return list(selected)
    selected_names = {stage.name for stage in selected}
    required_profiles: set[str] = set()
    for stage in selected:
        if stage.kind != "cache" or not stage.name.startswith("cache_"):
            continue
        parts = stage.name.split("_")
        if len(parts) < 3:
            continue
        required_profiles.add(f"profile_cache_batch_{parts[1]}")
    if not required_profiles:
        return list(selected)
    selected_names.update(required_profiles)
    return [stage for stage in plan if stage.name in selected_names]


def _write_pipeline_summary(
    options: PipelineOptions,
    *,
    config_hash: str,
    selected: Sequence[Stage],
    status: str,
    completed: Iterable[str] = (),
    failed_stage: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "mr_dflash_pipeline_v1",
        "status": status,
        "config_hash": config_hash,
        "data_root": str(options.data_root),
        "target_model_path": options.target_model_path,
        "selected_stages": [stage.name for stage in selected],
        "completed_stages": list(completed),
        "timestamp": _now(),
    }
    if failed_stage:
        payload["failed_stage"] = failed_stage
    if error:
        payload["error"] = error
    write_json(options.data_root / "pipeline_summary.json", payload)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MR-DFlash: prepare -> regenerate -> validate -> tokenize -> cache"
    )
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--data-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--sharegpt-source", default=DEFAULT_SHAREGPT_SOURCE)
    parser.add_argument("--arxiv-source", default=DEFAULT_ARXIV_SOURCE)
    parser.add_argument("--sharegpt-count", type=int, default=50_000)
    parser.add_argument("--arxiv-count", type=int, default=50_000)
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help=(
            "cho phép dùng ít hơn số lượng yêu cầu nếu source không đủ; "
            "manifest ghi số lượng thực tế, không nhân bản dữ liệu"
        ),
    )
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[3072, 8192])
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="chạy một regime full-context, không cắt prompt trước khi target generate",
    )
    parser.add_argument(
        "--full-context-length",
        type=int,
        default=32768,
        help="giới hạn tổng token của target context trong full-context mode",
    )
    parser.add_argument("--cache-splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=list(DEFAULT_FEATURE_LAYER_IDS))
    parser.add_argument("--supervision-mode", choices=["all_assistant", "last_assistant"], default="last_assistant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--overflow-policy",
        choices=["error", "skip"],
        default="error",
        help="sample có prompt vượt context: dừng hoặc ghi skipped report rồi tiếp tục",
    )
    parser.add_argument(
        "--sample-error-policy",
        choices=["error", "skip"],
        default="error",
        help="lỗi từng sample: dừng hoặc ghi skipped report rồi tiếp tục",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--analysis-limit", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--cache-batch-size-3k", type=int, default=2)
    parser.add_argument(
        "--cache-batch-size-long-context",
        "--cache-batch-size-8k",
        dest="cache_batch_size_long_context",
        type=int,
        default=1,
        help=(
            "số sample cache đồng thời trên mỗi GPU cho mọi context >3K "
            "(8K/16K/32K/...); --cache-batch-size-8k là alias cũ"
        ),
    )
    parser.add_argument("--cache-bucket-buffer-3k", type=int, default=16)
    parser.add_argument(
        "--cache-bucket-buffer-long-context",
        "--cache-bucket-buffer-8k",
        dest="cache_bucket_buffer_long_context",
        type=int,
        default=8,
        help="buffer phân bucket cho mọi context >3K; --cache-bucket-buffer-8k là alias cũ",
    )
    parser.add_argument("--cache-shard-size-3k", type=int, default=64)
    parser.add_argument(
        "--cache-shard-size-long-context",
        "--cache-shard-size-8k",
        dest="cache_shard_size_long_context",
        type=int,
        default=32,
        help="số sample mỗi shard cho mọi context >3K; --cache-shard-size-8k là alias cũ",
    )
    parser.add_argument(
        "--cache-attention-backend",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="backend attention của backbone khi cache target; sdpa là mặc định an toàn",
    )
    parser.add_argument(
        "--cache-io-threads",
        type=int,
        default=2,
        help="số thread ghi shard trên mỗi cache worker; 0 = ghi đồng bộ",
    )
    parser.add_argument(
        "--cache-io-queue-size",
        type=int,
        default=4,
        help="số shard tối đa chờ ghi trên mỗi cache worker",
    )
    parser.add_argument(
        "--cache-auto-batch",
        action="store_true",
        help=(
            "profile batch size trên một GPU trước cache, sau đó chọn batch "
            "cố định theo bucket độ dài"
        ),
    )
    parser.add_argument(
        "--cache-batch-profile",
        default=None,
        help="dùng profile JSON đã có; không tạo lại stage profile",
    )
    parser.add_argument(
        "--cache-profile-gpu-id",
        type=int,
        default=0,
        help="GPU dùng để profile auto-batch (ID logical theo CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--cache-profile-max-batch-size",
        type=int,
        default=8,
        help="candidate batch lớn nhất khi profile; mặc định 1,2,4,8",
    )
    parser.add_argument(
        "--cache-profile-vram-limit-gb",
        type=float,
        default=None,
        help="hard limit VRAM cho profile, ví dụ 170 trên B200 180GB",
    )
    parser.add_argument(
        "--cache-profile-headroom-fraction",
        type=float,
        default=0.10,
        help="headroom nếu không truyền hard VRAM limit; mặc định 10%%",
    )
    parser.add_argument(
        "--cache-profile-bucket-step",
        type=int,
        default=8192,
        help="độ rộng bucket độ dài; 8192 tạo bucket 8K/16K/24K/32K",
    )
    parser.add_argument(
        "--parallel-gpu-ids",
        type=int,
        nargs="+",
        default=[],
        help="data-parallel GPU vật lý; ví dụ 1 2 3 (không dùng GPU 0 nếu không truyền)",
    )
    parser.add_argument(
        "--progress-interval-tokens",
        type=int,
        default=256,
        help="số token giữa hai heartbeat của mỗi worker generate",
    )
    parser.add_argument(
        "--regenerate-output-batch-size",
        type=int,
        default=1,
        help="số sample gom trước khi worker flush output regenerate; mặc định 1",
    )
    parser.add_argument(
        "--worker-stall-timeout-seconds",
        type=float,
        default=0.0,
        help="timeout heartbeat worker; 0 tắt để không ngắt generation dài hợp lệ",
    )
    parser.add_argument(
        "--worker-stop-file",
        default=None,
        help="file điều khiển dừng sạch; worker bị dừng ở vòng poll kế tiếp",
    )
    parser.add_argument("--only-stage", action="append", default=[], help="chỉ chạy stage này; có thể lặp flag")
    parser.add_argument("--from-stage", default=None)
    parser.add_argument("--stop-after", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.sharegpt_count < 1 or args.arxiv_count < 1:
        raise ValueError("sharegpt-count và arxiv-count phải dương")
    if not args.max_lengths:
        raise ValueError("cần ít nhất một --max-lengths")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens phải >= 1")
    configured_lengths = [int(args.full_context_length)] if args.full_context else [int(x) for x in args.max_lengths]
    if any(length < 1 for length in configured_lengths):
        raise ValueError("mọi context length phải >= 1")
    if int(args.max_new_tokens) > min(configured_lengths):
        raise ValueError(
            "max-new-tokens không được lớn hơn context length nhỏ nhất; "
            "dùng budget nhỏ hơn hoặc chỉ chạy full-context"
        )
    if any(int(value) < 0 for value in args.parallel_gpu_ids):
        raise ValueError("parallel-gpu-ids không được âm")
    if len(args.parallel_gpu_ids) != len(set(args.parallel_gpu_ids)):
        raise ValueError("parallel-gpu-ids không được trùng")
    if args.progress_interval_tokens < 1 or args.regenerate_output_batch_size < 1:
        raise ValueError("progress-interval-tokens và regenerate-output-batch-size phải >= 1")
    if args.cache_io_threads < 0 or args.cache_io_queue_size < 0:
        raise ValueError("cache-io-threads và cache-io-queue-size không được âm")
    if args.cache_profile_gpu_id < 0:
        raise ValueError("cache-profile-gpu-id không được âm")
    if args.cache_profile_max_batch_size < 1 or args.cache_profile_bucket_step < 1:
        raise ValueError("cache-profile-max-batch-size và bucket-step phải >= 1")
    if not 0.0 <= float(args.cache_profile_headroom_fraction) < 1.0:
        raise ValueError("cache-profile-headroom-fraction phải thuộc [0, 1)")
    if args.cache_profile_vram_limit_gb is not None and args.cache_profile_vram_limit_gb <= 0:
        raise ValueError("cache-profile-vram-limit-gb phải > 0")
    if args.worker_stall_timeout_seconds < 0:
        raise ValueError("worker-stall-timeout-seconds không được âm")
    options = PipelineOptions(
        repo_root=Path(args.repo_root).resolve(),
        data_root=Path(args.data_root),
        target_model_path=str(args.target_model_path),
        sharegpt_source=str(args.sharegpt_source),
        arxiv_source=str(args.arxiv_source),
        sharegpt_count=int(args.sharegpt_count),
        arxiv_count=int(args.arxiv_count),
        allow_short=bool(args.allow_short),
        max_lengths=tuple(int(value) for value in args.max_lengths),
        full_context=bool(args.full_context),
        full_context_length=int(args.full_context_length),
        cache_splits=tuple(args.cache_splits),
        target_layer_ids=tuple(int(value) for value in args.target_layer_ids),
        supervision_mode=str(args.supervision_mode),
        seed=int(args.seed),
        max_new_tokens=int(args.max_new_tokens),
        overflow_policy=str(args.overflow_policy),
        sample_error_policy=str(args.sample_error_policy),
        temperature=float(args.temperature),
        analysis_limit=int(args.analysis_limit),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        target_revision=args.target_revision,
        cache_batch_size_3k=int(args.cache_batch_size_3k),
        cache_batch_size_long_context=int(args.cache_batch_size_long_context),
        cache_bucket_buffer_3k=int(args.cache_bucket_buffer_3k),
        cache_bucket_buffer_long_context=int(args.cache_bucket_buffer_long_context),
        cache_shard_size_3k=int(args.cache_shard_size_3k),
        cache_shard_size_long_context=int(args.cache_shard_size_long_context),
        cache_attention_backend=str(args.cache_attention_backend),
        cache_io_threads=int(args.cache_io_threads),
        cache_io_queue_size=int(args.cache_io_queue_size),
        cache_auto_batch=bool(args.cache_auto_batch),
        cache_batch_profile=str(args.cache_batch_profile) if args.cache_batch_profile else None,
        cache_profile_gpu_id=int(args.cache_profile_gpu_id),
        cache_profile_max_batch_size=int(args.cache_profile_max_batch_size),
        cache_profile_vram_limit_gb=(
            float(args.cache_profile_vram_limit_gb)
            if args.cache_profile_vram_limit_gb is not None
            else None
        ),
        cache_profile_headroom_fraction=float(args.cache_profile_headroom_fraction),
        cache_profile_bucket_step=int(args.cache_profile_bucket_step),
        parallel_gpu_ids=tuple(int(value) for value in args.parallel_gpu_ids),
        progress_interval_tokens=int(args.progress_interval_tokens),
        regenerate_output_batch_size=int(args.regenerate_output_batch_size),
        worker_stall_timeout_seconds=float(args.worker_stall_timeout_seconds),
        worker_stop_file=str(args.worker_stop_file) if args.worker_stop_file else None,
        local_files_only=bool(args.local_files_only),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
    )
    plan = build_stage_plan(options)
    selected = _select_stages(
        plan,
        only=args.only_stage,
        from_stage=args.from_stage,
        stop_after=args.stop_after,
    )
    selected = _add_auto_batch_dependencies(
        plan,
        selected,
        enabled=bool(args.cache_auto_batch),
        has_explicit_profile=bool(args.cache_batch_profile),
    )
    config_hash = pipeline_config_hash(options)
    print(f"[pipeline] data_root={options.data_root}")
    print(f"[pipeline] target={options.target_model_path}")
    print(f"[pipeline] stages={', '.join(stage.name for stage in selected)}")
    lock: Optional[_PipelineLock] = None
    if not options.dry_run:
        options.data_root.mkdir(parents=True, exist_ok=True)
        lock = _PipelineLock(options.data_root, config_hash)
        try:
            lock.acquire()
        except RuntimeError as exc:
            print(f"[pipeline] LOCKED: {exc}", file=sys.stderr)
            return 2

    try:
        if not options.dry_run:
            write_json(
                options.data_root / "pipeline_plan.json",
                {
                    "schema_version": "mr_dflash_pipeline_plan_v1",
                    "config_hash": config_hash,
                    "options": _options_payload(options),
                    "stages": [
                        {
                            "name": stage.name,
                            "command": stage.command,
                            "command_shell": shlex.join(stage.command),
                            "artifacts": _artifact_paths(stage),
                            "kind": stage.kind,
                        }
                        for stage in selected
                    ],
                },
            )

        completed: list[str] = []
        for stage in selected:
            try:
                result = run_stage(stage, options, config_hash)
                if result.get("status") in {"success", "skipped"}:
                    completed.append(stage.name)
            except KeyboardInterrupt:
                if not options.dry_run:
                    _write_pipeline_summary(
                        options,
                        config_hash=config_hash,
                        selected=selected,
                        status="interrupted",
                        completed=completed,
                        failed_stage=stage.name,
                        error="người dùng hoặc scheduler ngắt tiến trình; chạy lại với --resume",
                    )
                print(
                    f"[pipeline] INTERRUPTED tại {stage.name}; chạy lại với --resume",
                    file=sys.stderr,
                )
                return 130
            except Exception as exc:
                if not options.dry_run:
                    _write_pipeline_summary(
                        options,
                        config_hash=config_hash,
                        selected=selected,
                        status="failed",
                        completed=completed,
                        failed_stage=stage.name,
                        error=str(exc),
                    )
                print(
                    f"[pipeline] FAILED at {stage.name}. "
                    f"State/log: {options.data_root / 'pipeline_state'} / "
                    f"{options.data_root / 'pipeline_logs'}",
                    file=sys.stderr,
                )
                return 1
        if not options.dry_run:
            _write_pipeline_summary(
                options,
                config_hash=config_hash,
                selected=selected,
                status="success",
                completed=completed,
            )
        print("[pipeline] DONE toàn bộ stage đã chọn")
        return 0
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
