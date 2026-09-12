"""Chạy phase 1 MR-DFlash trên một subset nhỏ từ regenerate đến cache.

Script này nhận canonical prompt-only JSONL đã chuẩn hóa, lấy một subset
deterministic (mặc định 20 mẫu), rồi chạy đúng chuỗi artifact của pipeline
chính:

    subset -> regenerate -> validate -> tokenize -> cache

Mỗi stage dùng cùng runner/marker/log semantics với
``run_preprocess_pipeline.py``. Source không bị sửa; output luôn nằm trong
``--output-root`` riêng để dễ audit và xóa thủ công sau khi kiểm tra.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from _common import REPO_ROOT, read_jsonl, write_json, write_jsonl
from run_preprocess_pipeline import Stage, _PipelineLock, run_stage


DEFAULT_TARGET_MODEL = "/workspace/storage-shared/models/Qwen3-4B"
DEFAULT_LAYER_IDS = (1, 9, 17, 25, 33)


def select_subset(
    rows: Iterable[dict[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    """Reservoir-sample các row có ID duy nhất, không load toàn bộ source."""
    if limit < 1:
        raise ValueError("num-samples phải >= 1")
    import random

    rng = random.Random(seed)
    reservoir: list[tuple[int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    eligible = 0
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError(f"source row {row_index} thiếu id")
        if sample_id in seen_ids:
            raise ValueError(f"source có id trùng: {sample_id!r}")
        seen_ids.add(sample_id)
        eligible += 1
        item = (row_index, row)
        if len(reservoir) < limit:
            reservoir.append(item)
            continue
        replacement = rng.randrange(eligible)
        if replacement < limit:
            reservoir[replacement] = item
    reservoir.sort(key=lambda item: item[0])
    return [row for _index, row in reservoir]


@dataclass(frozen=True)
class SmokeOptions:
    input_path: Path
    output_root: Path
    target_model_path: str = DEFAULT_TARGET_MODEL
    num_samples: int = 20
    seed: int = 42
    max_length: int = 8192
    max_new_tokens: int = 768
    temperature: float = 0.0
    target_layer_ids: tuple[int, ...] = DEFAULT_LAYER_IDS
    supervision_mode: str = "last_assistant"
    overflow_policy: str = "skip"
    sample_error_policy: str = "skip"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    target_revision: str | None = None
    cache_batch_size: int = 1
    cache_bucket_buffer: int = 8
    cache_shard_size: int = 32
    cache_attention_backend: str = "sdpa"
    cache_io_threads: int = 2
    cache_io_queue_size: int = 4
    cache_batch_profile: str | None = None
    progress_interval_tokens: int = 256
    local_files_only: bool = True
    resume: bool = True
    dry_run: bool = False
    repo_root: Path = REPO_ROOT

    @property
    def data_root(self) -> Path:
        """Tên tương thích với runner stage dùng bởi pipeline chính."""
        return self.output_root


def _subset_paths(options: SmokeOptions) -> tuple[Path, Path]:
    normalized = options.output_root / "normalized"
    manifests = options.output_root / "manifests"
    return normalized / "smoke_prompts.jsonl", manifests / "smoke_subset.json"


def _phase_paths(options: SmokeOptions) -> dict[str, Path]:
    root = options.output_root
    return {
        "subset": _subset_paths(options)[0],
        "subset_manifest": _subset_paths(options)[1],
        "regenerated": root / "regenerated_smoke" / "train.jsonl",
        "regenerated_skipped": root / "regenerated_smoke" / "train.skipped.jsonl",
        "regeneration_manifest": root / "manifests" / "regeneration_smoke_train.json",
        "validation_manifest": root / "manifests" / "validation_smoke_train.json",
        "tokenized": root / "tokenized_smoke" / "train",
        "tokenization_manifest": root / "manifests" / "tokenization_smoke_train.json",
        "features": root / "target_features_smoke" / "train",
        "cache_skipped": root / "target_features_smoke" / "train" / "skipped.jsonl",
    }


def _local_files_args(options: SmokeOptions) -> list[str]:
    return ["--local-files-only"] if options.local_files_only else []


def _resume_args(options: SmokeOptions) -> list[str]:
    return ["--resume"] if options.resume else []


def build_stage_plan(options: SmokeOptions) -> list[Stage]:
    """Tạo lệnh của bốn stage GPU/data, chưa thực thi command nào."""
    paths = _phase_paths(options)
    python = sys.executable
    scripts = options.repo_root / "scripts" / "mr_dflash"
    common_model = ["--target-model-path", options.target_model_path]
    common_length = ["--max-length", str(options.max_length)]
    layer_ids = ["--target-layer-ids", *(str(value) for value in options.target_layer_ids)]

    regenerate = [
        python,
        str(scripts / "regenerate_pilot.py"),
        "--input",
        str(paths["subset"]),
        "--output",
        str(paths["regenerated"]),
        "--manifest",
        str(paths["regeneration_manifest"]),
        *common_model,
        *common_length,
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
        "--overflow-policy",
        options.overflow_policy,
        "--sample-error-policy",
        options.sample_error_policy,
        "--skipped-report",
        str(paths["regenerated_skipped"]),
        "--output-batch-size",
        "1",
        "--progress-interval-tokens",
        str(options.progress_interval_tokens),
        *(["--target-revision", options.target_revision] if options.target_revision else []),
        *_local_files_args(options),
        *_resume_args(options),
    ]

    validate = [
        python,
        str(scripts / "validate_pilot_dataset.py"),
        "--input",
        str(paths["regenerated"]),
        "--tokenizer",
        options.target_model_path,
        *common_length,
        "--expected-target-model",
        options.target_model_path,
        "--require-generated",
        "--report",
        str(paths["validation_manifest"]),
        *_local_files_args(options),
    ]

    tokenize = [
        python,
        str(scripts / "tokenize_dataset.py"),
        "--input",
        str(paths["regenerated"]),
        "--output",
        str(paths["tokenized"]),
        "--provenance-manifest",
        str(paths["tokenization_manifest"]),
        *common_model,
        *common_length,
        "--supervision-mode",
        options.supervision_mode,
        "--feature-layer-ids",
        *(str(value) for value in options.target_layer_ids),
        "--shard-size",
        str(options.cache_shard_size),
        *_local_files_args(options),
        *_resume_args(options),
    ]

    cache = [
        python,
        str(scripts / "cache_target_features.py"),
        *common_model,
        "--data-path",
        str(paths["regenerated"]),
        "--tokenized-path",
        str(paths["tokenized"]),
        "--output-path",
        str(paths["features"]),
        *common_length,
        *layer_ids,
        "--batch-size",
        str(options.cache_batch_size),
        "--bucket-buffer-size",
        str(options.cache_bucket_buffer),
        "--shard-size",
        str(options.cache_shard_size),
        "--attention-backend",
        options.cache_attention_backend,
        "--io-threads",
        str(options.cache_io_threads),
        "--io-queue-size",
        str(options.cache_io_queue_size),
        "--device",
        options.device,
        "--torch-dtype",
        options.torch_dtype,
        "--supervision-mode",
        options.supervision_mode,
        "--skipped-report",
        str(paths["cache_skipped"]),
        "--progress-path",
        str(options.output_root / "pipeline_state" / "cache_progress.json"),
        *(["--batch-profile", options.cache_batch_profile] if options.cache_batch_profile else []),
        *(["--target-revision", options.target_revision] if options.target_revision else []),
        *_local_files_args(options),
        *_resume_args(options),
    ]

    return [
        Stage(
            name="regenerate_smoke_train",
            command=regenerate,
            artifacts=(paths["regenerated"], paths["regenerated_skipped"], paths["regeneration_manifest"]),
        ),
        Stage(
            name="validate_smoke_train",
            command=validate,
            artifacts=(paths["regenerated"], paths["validation_manifest"]),
        ),
        Stage(
            name="tokenize_smoke_train",
            command=tokenize,
            artifacts=(paths["tokenized"] / "manifest.json", paths["tokenization_manifest"]),
            kind="tokenized",
        ),
        Stage(
            name="cache_smoke_train",
            command=cache,
            artifacts=(paths["features"] / "manifest.json",),
            kind="cache",
        ),
    ]


def materialize_subset(options: SmokeOptions) -> dict[str, Any]:
    """Ghi subset và manifest, hoặc kiểm tra lại subset khi resume."""
    if not options.input_path.is_file():
        raise FileNotFoundError(f"không tìm thấy canonical input: {options.input_path}")
    subset_path, manifest_path = _subset_paths(options)
    source_stat = options.input_path.stat()
    source_descriptor = {
        "path": str(options.input_path.resolve()),
        "size": int(source_stat.st_size),
        "mtime_ns": int(source_stat.st_mtime_ns),
    }
    if options.resume and subset_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "input": source_descriptor,
            "num_requested": int(options.num_samples),
            "seed": int(options.seed),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"subset hiện có không khớp input/seed/num_samples: {manifest_path}; "
                "dùng output-root mới hoặc --no-resume"
            )
        selected = list(read_jsonl(subset_path))
        if len(selected) != int(manifest.get("num_selected", -1)):
            raise RuntimeError(f"subset manifest không khớp số dòng: {subset_path}")
        return manifest

    source_total = sum(1 for _ in read_jsonl(options.input_path))
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm có trong requirements server
        tqdm = lambda iterator, **_kwargs: iterator
    selected = select_subset(
        tqdm(
            read_jsonl(options.input_path),
            total=source_total,
            desc="Phase1 subset",
            unit="sample",
        ),
        limit=int(options.num_samples),
        seed=int(options.seed),
    )
    write_jsonl(subset_path, selected)
    ids = [str(row["id"]) for row in selected]
    source_counts = Counter(str(row.get("source", "unknown")) for row in selected)
    lengths = []
    for row in selected:
        metadata = row.get("metadata") or {}
        value = metadata.get("source_token_length", metadata.get("source_length"))
        if isinstance(value, (int, float)) and value > 0:
            lengths.append(int(value))
    manifest = {
        "schema_version": "mr_dflash_phase1_smoke_subset_v1",
        "input": source_descriptor,
        "num_requested": int(options.num_samples),
        "num_selected": len(selected),
        "seed": int(options.seed),
        "selection": "reservoir_sample_sorted_by_source_index",
        "source_counts": dict(sorted(source_counts.items())),
        "length_summary": (
            {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "count": len(lengths),
            }
            if lengths
            else {"count": 0}
        ),
        "ids": ids,
        "id_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
    }
    write_json(manifest_path, manifest)
    return manifest


def _config_hash(options: SmokeOptions, subset_manifest: dict[str, Any]) -> str:
    payload = asdict(options)
    payload["input_path"] = str(options.input_path.resolve())
    payload["output_root"] = str(options.output_root.resolve())
    payload["repo_root"] = str(options.repo_root.resolve())
    payload["subset_id_sha256"] = subset_manifest.get("id_sha256")
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _write_plan(options: SmokeOptions, plan: Sequence[Stage], config_hash: str, subset_manifest: dict[str, Any]) -> None:
    write_json(
        options.output_root / "pipeline_plan.json",
        {
            "schema_version": "mr_dflash_phase1_smoke_plan_v1",
            "config_hash": config_hash,
            "options": {
                **{key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(options).items()},
                "target_layer_ids": list(options.target_layer_ids),
            },
            "subset_manifest": subset_manifest,
            "stages": [
                {
                    "name": stage.name,
                    "command": stage.command,
                    "command_shell": shlex.join(stage.command),
                    "artifacts": [str(path) for path in stage.artifacts],
                    "kind": stage.kind,
                }
                for stage in plan
            ],
        },
    )


def _write_summary(
    options: SmokeOptions,
    *,
    config_hash: str,
    plan: Sequence[Stage],
    status: str,
    completed: Sequence[str],
    failed_stage: str | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "mr_dflash_phase1_smoke_summary_v1",
        "status": status,
        "config_hash": config_hash,
        "data_root": str(options.output_root),
        "target_model_path": options.target_model_path,
        "selected_stages": [stage.name for stage in plan],
        "completed_stages": list(completed),
    }
    if failed_stage:
        payload["failed_stage"] = failed_stage
    if error:
        payload["error"] = error
    write_json(options.output_root / "pipeline_summary.json", payload)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MR-DFlash phase 1 smoke: subset -> regenerate -> validate -> tokenize -> cache"
    )
    parser.add_argument("--input", required=True, help="canonical prompt-only JSONL, ví dụ normalized/pilot_prompts.jsonl")
    parser.add_argument("--output-root", required=True, help="thư mục output mới cho smoke pipeline")
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=list(DEFAULT_LAYER_IDS))
    parser.add_argument("--supervision-mode", choices=["all_assistant", "last_assistant"], default="last_assistant")
    parser.add_argument("--overflow-policy", choices=["error", "skip"], default="skip")
    parser.add_argument("--sample-error-policy", choices=["error", "skip"], default="skip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--cache-batch-size", type=int, default=1)
    parser.add_argument("--cache-bucket-buffer", type=int, default=8)
    parser.add_argument("--cache-shard-size", type=int, default=32)
    parser.add_argument("--cache-attention-backend", choices=["auto", "eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--cache-io-threads", type=int, default=2)
    parser.add_argument("--cache-io-queue-size", type=int, default=4)
    parser.add_argument("--cache-batch-profile", default=None)
    parser.add_argument("--progress-interval-tokens", type=int, default=256)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.num_samples < 1 or args.max_length < 1 or args.max_new_tokens < 1:
        raise ValueError("num-samples, max-length và max-new-tokens phải >= 1")
    if args.max_new_tokens > args.max_length:
        raise ValueError("max-new-tokens không được lớn hơn max-length")
    if args.cache_batch_size < 1 or args.cache_bucket_buffer < args.cache_batch_size:
        raise ValueError("cache-bucket-buffer phải >= cache-batch-size")
    if args.cache_shard_size < 1 or args.cache_io_threads < 0 or args.cache_io_queue_size < 0:
        raise ValueError("cache shard/io config không hợp lệ")
    if args.progress_interval_tokens < 1:
        raise ValueError("progress-interval-tokens phải >= 1")

    options = SmokeOptions(
        input_path=Path(args.input).resolve(),
        output_root=Path(args.output_root).resolve(),
        target_model_path=str(args.target_model_path),
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        target_layer_ids=tuple(int(value) for value in args.target_layer_ids),
        supervision_mode=str(args.supervision_mode),
        overflow_policy=str(args.overflow_policy),
        sample_error_policy=str(args.sample_error_policy),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        target_revision=args.target_revision,
        cache_batch_size=int(args.cache_batch_size),
        cache_bucket_buffer=int(args.cache_bucket_buffer),
        cache_shard_size=int(args.cache_shard_size),
        cache_attention_backend=str(args.cache_attention_backend),
        cache_io_threads=int(args.cache_io_threads),
        cache_io_queue_size=int(args.cache_io_queue_size),
        cache_batch_profile=str(args.cache_batch_profile) if args.cache_batch_profile else None,
        progress_interval_tokens=int(args.progress_interval_tokens),
        local_files_only=bool(args.local_files_only),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        repo_root=Path(args.repo_root).resolve(),
    )
    plan = build_stage_plan(options)
    print(f"[phase1-smoke] input={options.input_path}")
    print(f"[phase1-smoke] output_root={options.output_root}")
    print(f"[phase1-smoke] stages={', '.join(stage.name for stage in plan)}")
    if options.dry_run:
        print("[phase1-smoke] dry-run; chưa đọc source và chưa chạy model")
        for stage in plan:
            print(f"[phase1-smoke][dry-run] {stage.name}: {shlex.join(stage.command)}")
        return 0

    if not options.resume and options.output_root.exists() and any(options.output_root.iterdir()):
        raise FileExistsError(
            f"output-root đã tồn tại và không rỗng: {options.output_root}; "
            "dùng thư mục mới hoặc --resume"
        )
    options.output_root.mkdir(parents=True, exist_ok=True)
    subset_manifest = materialize_subset(options)
    config_hash = _config_hash(options, subset_manifest)
    lock = _PipelineLock(options.output_root, config_hash)
    try:
        lock.acquire()
    except RuntimeError as exc:
        print(f"[phase1-smoke] LOCKED: {exc}", file=sys.stderr)
        return 2

    completed: list[str] = []
    current_stage: str | None = None
    try:
        _write_plan(options, plan, config_hash, subset_manifest)
        for stage in plan:
            current_stage = stage.name
            result = run_stage(stage, options, config_hash)
            if result.get("status") in {"success", "skipped"}:
                completed.append(stage.name)
        _write_summary(
            options,
            config_hash=config_hash,
            plan=plan,
            status="success",
            completed=completed,
        )
        print(f"[phase1-smoke] DONE output_root={options.output_root}")
        return 0
    except KeyboardInterrupt:
        _write_summary(
            options,
            config_hash=config_hash,
            plan=plan,
            status="interrupted",
            completed=completed,
            failed_stage=current_stage,
            error="interrupted; chạy lại với --resume",
        )
        print(f"[phase1-smoke] INTERRUPTED tại {current_stage}; chạy lại với --resume", file=sys.stderr)
        return 130
    except Exception as exc:
        _write_summary(
            options,
            config_hash=config_hash,
            plan=plan,
            status="failed",
            completed=completed,
            failed_stage=current_stage,
            error=str(exc),
        )
        print(f"[phase1-smoke] FAILED tại {current_stage}: {exc}", file=sys.stderr)
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
