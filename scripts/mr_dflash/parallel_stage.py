"""Chạy song song target regeneration/cache trên các GPU độc lập.

Đây là data-parallel launcher cho các phase inference-only. Mỗi worker chỉ
nhìn thấy một GPU vật lý qua ``CUDA_VISIBLE_DEVICES`` và xử lý một shard dòng
``row_index % num_shards``. Worker ghi artifact riêng; parent chỉ publish
artifact cuối sau khi *tất cả* worker thành công và kiểm tra đủ sample ID.

Không dùng NCCL ở đây: một target Qwen3-4B riêng trên mỗi GPU phù hợp hơn cho
generation/cache và không làm các worker tranh chấp cùng một output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from _common import REPO_ROOT, read_jsonl, resolve_parallel_work_root, write_json, write_jsonl
from progress import format_duration, read_progress
from shared_scheduler import SharedLeaseQueue, WorkItem


def _worker_root(work_root: Path, rank: int) -> Path:
    return work_root / f"rank_{rank:02d}"


def build_worker_status(
    *,
    rank: int,
    gpu_id: int,
    pid: int,
    return_code: int | None,
    worker_root: Path,
) -> Dict[str, Any]:
    """Tạo snapshot chẩn đoán cho một worker và GPU vật lý tương ứng."""
    progress = read_progress(worker_root / "progress.json")
    updated_at = progress.get("updated_at_unix")
    age = None
    if isinstance(updated_at, (int, float)):
        age = max(0.0, time.time() - float(updated_at))
    return {
        "rank": int(rank),
        "host": socket.gethostname(),
        "gpu_id": int(gpu_id),
        "pid": int(pid),
        "return_code": return_code,
        "alive": return_code is None,
        "progress_age_seconds": age,
        "progress": progress,
        "log": str(worker_root / "worker.log"),
    }


def aggregate_progress(
    worker_statuses: Sequence[Dict[str, Any]],
    *,
    total_samples: int | None = None,
) -> Dict[str, Any]:
    """Gom workload/ETA của toàn bộ worker để ghi ở ``status.json``.

    Các worker có thể có sample dài ngắn khác nhau; ETA của stage là ETA lớn
    nhất còn lại của một worker, còn throughput là tổng throughput hiện tại.
    """
    progress_values = [
        worker.get("progress") or {}
        for worker in worker_statuses
        if isinstance(worker, dict)
    ]

    def total(name: str) -> int:
        return sum(int(value.get(name, 0) or 0) for value in progress_values)

    def completed_total() -> int:
        completed = 0
        for value in progress_values:
            worker_total = max(0, int(value.get("total_samples", 0) or 0))
            worker_completed = max(0, int(value.get("completed_samples", 0) or 0))
            completed += min(worker_completed, worker_total) if worker_total else worker_completed
        return completed

    global_total = (
        max(0, int(total_samples))
        if total_samples is not None
        else total("total_samples")
    )
    global_completed = min(completed_total(), global_total)

    etas = [
        float(value["eta_seconds"])
        for value in progress_values
        if isinstance(value.get("eta_seconds"), (int, float))
    ]
    rates = [
        float(value["throughput_tokens_per_second"])
        for value in progress_values
        if isinstance(value.get("throughput_tokens_per_second"), (int, float))
    ]
    eta = max(etas) if etas else None
    throughput = sum(rates) if rates else None
    result: Dict[str, Any] = {
        "total_samples": global_total,
        "completed_samples": global_completed,
        "remaining_samples": max(0, global_total - global_completed),
        "total_tokens": total("total_tokens"),
        "completed_tokens": total("completed_tokens"),
        "remaining_tokens": total("remaining_tokens"),
        "throughput_tokens_per_second": round(throughput, 3) if throughput is not None else None,
        "eta_seconds": round(eta, 3) if eta is not None else None,
        "eta_human": format_duration(eta),
    }
    return result


def _active_worker_pids(work_root: Path) -> list[int]:
    """Tìm worker local còn sống để không launch trùng target.

    ``status.json`` nằm trên shared filesystem và có thể được resume từ host
    khác.  PID chỉ có ý nghĩa trong namespace của host đã tạo nó; không được
    gọi ``kill(pid, 0)`` trên host mới vì có thể trùng với một process không
    liên quan.
    """
    status_path = work_root / "status.json"
    if not status_path.is_file():
        return []
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict) or payload.get("status") not in {"running", "joining"}:
        return []
    active: list[int] = []
    local_host = socket.gethostname()
    for worker in payload.get("workers", []):
        if not isinstance(worker, dict) or not worker.get("alive"):
            continue
        recorded_host = worker.get("host")
        if recorded_host and str(recorded_host) != local_host:
            continue
        try:
            pid = int(worker["pid"])
            os.kill(pid, 0)
        except (KeyError, ValueError, ProcessLookupError, PermissionError, OSError):
            continue
        active.append(pid)
    return active


def _validate_gpu_ids(gpu_ids: Sequence[int]) -> list[int]:
    values = [int(value) for value in gpu_ids]
    if not values:
        raise ValueError("cần ít nhất một --gpu-ids")
    if any(value < 0 for value in values):
        raise ValueError("GPU id không được âm")
    if len(values) != len(set(values)):
        raise ValueError("--gpu-ids không được trùng")
    return values


def build_shared_queue_items(
    mode: str,
    input_path: str | Path,
    *,
    tokenized_path: str | Path | None = None,
) -> list[WorkItem]:
    """Build queue metadata without loading model weights.

    Regeneration uses the prepared source token length when available. Cache
    uses exact tokenized length, which is the quantity that controls padding
    and static-pool pressure.
    """
    if mode == "cache" and tokenized_path is not None:
        from MR_DFlash.tokenized_data import TokenizedDFlashDataset

        dataset = TokenizedDFlashDataset(str(tokenized_path))
        return [
            WorkItem(
                sample_id=str(item["id"]),
                index=index,
                length=len(item["input_ids"]),
            )
            for index in range(len(dataset))
            for item in (dataset[index],)
        ]
    rows = list(read_jsonl(input_path))
    items: list[WorkItem] = []
    for index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        length = metadata.get("source_token_length", metadata.get("source_length"))
        if length is None and isinstance(row.get("input_ids"), list):
            length = len(row["input_ids"])
        if length is None:
            user_text = "\n".join(
                str(message.get("content", ""))
                for message in row.get("conversations", [])
                if isinstance(message, dict) and str(message.get("role", "")).lower() == "user"
            )
            length = len(user_text)
        items.append(WorkItem(sample_id=str(row.get("id", f"row_{index}")), index=index, length=max(0, int(length))))
    return items


def _shared_queue_config_hash(args: argparse.Namespace) -> str:
    payload = {
        "mode": args.mode,
        "input": str(Path(args.input).resolve()),
        "tokenized_path": str(Path(args.tokenized_path).resolve()) if args.tokenized_path else None,
        "target_model_path": str(args.target_model_path),
        "max_length": int(args.max_length),
        "max_new_tokens": int(args.max_new_tokens),
        "target_layer_ids": [int(value) for value in (args.target_layer_ids or [])],
        "cache_backend": str(args.cache_backend),
        "attention_backend": str(args.attention_backend),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def prepare_shared_queue(args: argparse.Namespace, work_root: Path) -> SharedLeaseQueue:
    queue_root = Path(args.queue_root) if args.queue_root else work_root / "shared_queue"
    queue = SharedLeaseQueue(
        queue_root,
        stage=f"{args.mode}:{Path(args.output).name}",
        config_hash=_shared_queue_config_hash(args),
        lease_ttl_seconds=float(args.queue_lease_ttl_seconds),
        lock_ttl_seconds=float(args.queue_lock_ttl_seconds),
        max_attempts=int(args.queue_max_attempts),
    )
    queue.initialize(
        build_shared_queue_items(
            args.mode,
            args.input,
            tokenized_path=args.tokenized_path,
        ),
        resume=bool(args.resume),
    )
    # Migrate durable outputs produced by the old modulo-shard launcher. This
    # makes the first shared-lease resume safe even when the previous host was
    # killed before it wrote a queue event.
    for worker_root in sorted(work_root.glob("rank_*")):
        if args.mode == "regenerate":
            for artifact_path in (worker_root / "output.jsonl", worker_root / "skipped.jsonl"):
                if not artifact_path.is_file():
                    continue
                ids = [str(row.get("id", "")) for row in read_jsonl(artifact_path) if row.get("id")]
                queue.seed_completed(ids, artifact=str(artifact_path))
        else:
            artifact_path = worker_root / "manifest.json"
            if artifact_path.is_file():
                try:
                    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
                ids = payload.get("sample_ids", []) if isinstance(payload, dict) else []
                queue.seed_completed(ids, artifact=str(artifact_path))
    return queue


def assign_shared_leases(
    queue: SharedLeaseQueue,
    args: argparse.Namespace,
    work_root: Path,
) -> dict[int, Any]:
    """Claim one sorted work slice per available GPU and publish ID files."""
    snapshot = queue.snapshot()
    if snapshot["pending"] == 0 and snapshot["leased"] > 0:
        raise RuntimeError(
            "shared queue đang có lease chưa hết hạn; dừng host cũ hoặc chờ lease TTL "
            "trước khi chuyển host"
        )
    pending = int(snapshot["pending"])
    chunk_size = max(1, math.ceil(pending / max(1, len(args.gpu_ids)))) if pending else 1
    sample_ids_files: dict[int, Path] = {}
    leases: dict[int, Any] = {}
    for rank, _gpu_id in enumerate(args.gpu_ids):
        lease = queue.claim(
            worker_id=f"{socket.gethostname()}/rank-{rank}/pid-{os.getpid()}",
            max_items=chunk_size,
        )
        ids_file = _worker_root(work_root, rank) / "leased_sample_ids.jsonl"
        ids_file.parent.mkdir(parents=True, exist_ok=True)
        rows = [] if lease is None else [{"sample_id": item.sample_id} for item in lease.items]
        write_jsonl(ids_file, rows)
        sample_ids_files[rank] = ids_file
        if lease is not None:
            leases[rank] = lease
    args._shared_sample_ids_files = sample_ids_files
    return leases


def durable_worker_ids(args: argparse.Namespace, work_root: Path, rank: int) -> set[str]:
    """Read IDs already durable in a worker root after a partial failure."""
    root = _worker_root(work_root, rank)
    if args.mode == "regenerate":
        ids: set[str] = set()
        for path in (root / "output.jsonl", root / "skipped.jsonl"):
            if path.is_file():
                ids.update(str(row.get("id", "")) for row in read_jsonl(path) if row.get("id"))
        return ids
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return set()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(value) for value in payload.get("sample_ids", [])} if isinstance(payload, dict) else set()


def _read_object_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"worker thiếu artifact: {path}")
    return list(read_jsonl(path))


def _index_rows(rows: Iterable[Dict[str, Any]], *, path: Path) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id:
            raise ValueError(f"artifact {path} có row thiếu id")
        if sample_id in indexed:
            raise ValueError(f"artifact {path} có id trùng: {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def merge_regenerated_outputs(
    *,
    input_path: Path,
    worker_roots: Sequence[Path],
    output_path: Path,
    skipped_path: Path,
    manifest_path: Path,
    gpu_ids: Sequence[int],
) -> Dict[str, Any]:
    """Merge trajectory workers theo thứ tự input và kiểm tra coverage 100%."""
    input_rows = list(read_jsonl(input_path))
    input_ids = [str(row.get("id", "")) for row in input_rows]
    if any(not sample_id for sample_id in input_ids):
        raise ValueError(f"input {input_path} có sample thiếu id")
    if len(input_ids) != len(set(input_ids)):
        raise ValueError(f"input {input_path} có sample id trùng")

    generated: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    worker_stats: list[Dict[str, Any]] = []
    for rank, root in enumerate(worker_roots):
        output_rows = _read_object_rows(root / "output.jsonl")
        skipped_rows = _read_object_rows(root / "skipped.jsonl")
        for sample_id, row in _index_rows(output_rows, path=root / "output.jsonl").items():
            if sample_id in generated:
                duplicates += 1
                continue
            # A replay can leave a stale skipped record beside a later
            # successful output. Prefer the successful trajectory.
            skipped.pop(sample_id, None)
            generated[sample_id] = row
        for sample_id, row in _index_rows(skipped_rows, path=root / "skipped.jsonl").items():
            if sample_id in generated or sample_id in skipped:
                duplicates += 1
                continue
            skipped[sample_id] = row
        worker_manifest = root / "manifest.json"
        worker_payload = {}
        if worker_manifest.is_file():
            worker_payload = json.loads(worker_manifest.read_text(encoding="utf-8"))
        worker_stats.append({"rank": rank, "gpu_id": int(gpu_ids[rank]), "stats": worker_payload.get("stats", {})})

    expected = set(input_ids)
    covered = set(generated) | set(skipped)
    missing = sorted(expected - covered)
    extra = sorted(covered - expected)
    if missing or extra:
        raise RuntimeError(
            "parallel regenerate coverage không đầy đủ: "
            f"missing={missing[:5]} ({len(missing)}), "
            f"extra={extra[:5]} ({len(extra)})"
        )

    merged_output = [generated[sample_id] for sample_id in input_ids if sample_id in generated]
    merged_skipped = [skipped[sample_id] for sample_id in input_ids if sample_id in skipped]
    write_jsonl(output_path, merged_output)
    write_jsonl(skipped_path, merged_skipped)
    payload = {
        "schema_version": "mr_dflash_parallel_regeneration_v1",
        "input": str(input_path),
        "output": str(output_path),
        "skipped_report": str(skipped_path),
        "gpu_ids": [int(value) for value in gpu_ids],
        "worker_count": len(worker_roots),
        "stats": {
            "input_rows": len(input_ids),
            "written": len(merged_output),
            "skipped": len(merged_skipped),
            "missing": len(missing),
            "duplicates": int(duplicates),
        },
        "workers": worker_stats,
    }
    write_json(manifest_path, payload)
    return payload


def _publish_shard(source: Path, destination: Path) -> None:
    """Hard-link shard nếu cùng filesystem, fallback copy atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
    os.replace(temporary, destination)


def merge_feature_caches(
    *,
    input_path: Path,
    tokenized_path: Path | None = None,
    worker_roots: Sequence[Path],
    output_path: Path,
    manifest_path: Path,
    gpu_ids: Sequence[int],
    quarantined_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    """Merge worker feature stores mà không reserialize hidden tensor lớn."""
    if tokenized_path is not None:
        from MR_DFlash.tokenized_data import TokenizedDFlashDataset

        tokenized_dataset = TokenizedDFlashDataset(str(tokenized_path))
        expected_ids = [
            str(tokenized_dataset[index]["id"])
            for index in range(len(tokenized_dataset))
        ]
    else:
        input_rows = list(read_jsonl(input_path))
        expected_ids = [str(row.get("id", "")) for row in input_rows]
    if any(not sample_id for sample_id in expected_ids) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("cache input có id rỗng hoặc trùng")

    common_keys = (
        "target_model_path",
        "feature_layer_ids",
        "hidden_size",
        "feature_width",
        "max_length",
        "requested_torch_dtype",
        "source_tokenized_path",
        "target_revision",
        "capture_backend",
        "attention_backend",
        "cache_batch_profile",
        "cache_batch_profile_sha256",
    )
    base: Dict[str, Any] | None = None
    profile_history: list[Dict[str, Any]] = []
    all_ids: list[str] = []
    final_shards: list[Dict[str, Any]] = []
    for rank, root in enumerate(worker_roots):
        manifest_path_worker = root / "manifest.json"
        if not manifest_path_worker.is_file():
            assigned_path = root / "leased_sample_ids.jsonl"
            assigned_ids = {
                str(row.get("sample_id", row.get("id", "")))
                for row in read_jsonl(assigned_path)
                if row.get("sample_id", row.get("id", ""))
            } if assigned_path.is_file() else set()
            if not assigned_ids or assigned_ids.issubset(set(str(value) for value in quarantined_ids)):
                # A worker may be killed before ShardedFeatureWriter emits a
                # manifest. It is safe to omit the root only when it has no
                # live assignment left (or every assigned sample is already
                # quarantined); any other missing manifest means coverage is
                # ambiguous and must fail loudly.
                continue
            raise FileNotFoundError(f"worker cache thiếu manifest: {manifest_path_worker}")
        payload = json.loads(manifest_path_worker.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "mr_dflash_feature_sharded_v1":
            raise ValueError(f"worker cache schema không hợp lệ: {manifest_path_worker}")
        if base is None:
            base = payload
        else:
            for key in common_keys:
                if payload.get(key) != base.get(key):
                    raise ValueError(f"worker cache metadata lệch ở {key}: rank {rank}")
        for sample_id in payload.get("sample_ids", []):
            sample_id = str(sample_id)
            if sample_id in all_ids:
                raise ValueError(f"worker cache trùng sample id: {sample_id!r}")
            all_ids.append(sample_id)
        for history_item in payload.get("cache_profile_history", []):
            if isinstance(history_item, dict) and history_item not in profile_history:
                profile_history.append(dict(history_item))
        for local_index, descriptor in enumerate(payload.get("shards", [])):
            if not isinstance(descriptor, dict) or "path" not in descriptor:
                raise ValueError(f"worker cache shard descriptor lỗi: {manifest_path_worker}")
            source = root / str(descriptor["path"])
            if not source.is_file():
                raise FileNotFoundError(f"worker cache thiếu shard: {source}")
            name = f"shard_r{rank:02d}_{local_index:05d}.pt"
            destination = output_path / name
            _publish_shard(source, destination)
            merged_descriptor = dict(descriptor)
            merged_descriptor["path"] = name
            final_shards.append(merged_descriptor)

    expected = set(expected_ids)
    actual = set(all_ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    quarantined = set(str(value) for value in quarantined_ids)
    unresolved_missing = sorted(set(missing) - quarantined)
    if unresolved_missing or extra:
        raise RuntimeError(
            "parallel cache coverage không đầy đủ: "
            f"missing={unresolved_missing[:5]} ({len(unresolved_missing)}), "
            f"extra={extra[:5]} ({len(extra)})"
        )
    if base is None:
        raise ValueError("không có worker cache")

    from MR_DFlash.offline_features import write_sharded_feature_manifest

    output_manifest = write_sharded_feature_manifest(
        output_path,
        target_model_path=str(base["target_model_path"]),
        feature_layer_ids=base["feature_layer_ids"],
        hidden_size=int(base["hidden_size"]),
        feature_width=int(base["feature_width"]),
        max_length=int(base["max_length"]),
        requested_torch_dtype=str(base["requested_torch_dtype"]),
        shards=final_shards,
        sample_ids=all_ids,
        source_data_path=str(input_path),
        source_tokenized_path=base.get("source_tokenized_path"),
        target_revision=base.get("target_revision"),
        stored_feature_dtype=base.get("stored_feature_dtype"),
        capture_backend=str(base.get("capture_backend", "hf_backbone")),
        attention_backend=base.get("attention_backend"),
        cache_batch_profile=base.get("cache_batch_profile"),
        cache_batch_profile_sha256=base.get("cache_batch_profile_sha256"),
        cache_profile_history=profile_history,
        stats={
            "worker_count": len(worker_roots),
            "gpu_ids": [int(value) for value in gpu_ids],
            "num_samples": len(all_ids),
            "quarantined": len(quarantined & set(missing)),
            "missing_sample_ids": sorted(quarantined & set(missing)),
        },
    )
    write_json(manifest_path, {**output_manifest, "parallel_gpu_ids": [int(value) for value in gpu_ids]})
    return output_manifest


def _build_worker_command(
    args: argparse.Namespace,
    root: Path,
    rank: int,
    num_shards: int,
    *,
    sample_ids_file: Path | None = None,
) -> list[str]:
    python = sys.executable
    worker_device = "cuda:0"
    worker_shard_index = 0 if sample_ids_file is not None else rank
    worker_num_shards = 1 if sample_ids_file is not None else num_shards
    if args.mode == "regenerate":
        command = [
            python,
            str(Path(__file__).with_name("regenerate_pilot.py")),
            "--input", args.input,
            "--output", str(root / "output.jsonl"),
            "--manifest", str(root / "manifest.json"),
            "--target-model-path", args.target_model_path,
            "--max-length", str(args.max_length),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--seed", str(args.seed),
            "--device", worker_device,
            "--generation-batch-size", str(args.generation_batch_size),
            "--torch-dtype", args.torch_dtype,
            "--shard-index", str(worker_shard_index),
            "--num-shards", str(worker_num_shards),
            "--overflow-policy", args.overflow_policy,
            "--sample-error-policy", args.sample_error_policy,
            "--skipped-report", str(root / "skipped.jsonl"),
            "--progress-path", str(root / "progress.json"),
            "--progress-interval-tokens", str(args.progress_interval_tokens),
            "--output-batch-size", str(args.output_batch_size),
        ]
        if args.auto_batch:
            command.extend(
                [
                    "--auto-batch",
                    "--auto-batch-target-vram-gb", str(args.auto_batch_target_vram_gb),
                    "--auto-batch-hard-vram-gb", str(args.auto_batch_hard_vram_gb),
                    "--auto-batch-start-size", str(args.auto_batch_start_size),
                    "--auto-batch-max-size", str(args.auto_batch_max_size),
                    "--auto-batch-growth-factor", str(args.auto_batch_growth_factor),
                ]
            )
        if args.preserve_full_input:
            command.append("--preserve-full-input")
    else:
        command = [
            python,
            str(Path(__file__).with_name("cache_target_features.py")),
            "--target-model-path", args.target_model_path,
            "--data-path", args.input,
            "--output-path", str(root),
            "--max-length", str(args.max_length),
            *(
                ["--target-layer-ids", *(str(value) for value in args.target_layer_ids)]
                if args.target_layer_ids
                else []
            ),
            "--batch-size", str(args.batch_size),
            "--bucket-buffer-size", str(args.bucket_buffer_size),
            "--shard-size", str(args.shard_size),
            "--attention-backend", args.attention_backend,
            "--cache-backend", args.cache_backend,
            "--cache-memory-fraction", str(args.cache_memory_fraction),
            "--io-threads", str(args.io_threads),
            "--io-queue-size", str(args.io_queue_size),
            "--device", worker_device,
            "--torch-dtype", args.torch_dtype,
            "--supervision-mode", args.supervision_mode,
            "--shard-index", str(worker_shard_index),
            "--num-shards", str(worker_num_shards),
            "--skipped-report", str(root / "skipped.jsonl"),
            "--progress-path", str(root / "progress.json"),
        ]
        if args.tokenized_path:
            command.extend(["--tokenized-path", args.tokenized_path])
        if args.batch_profile:
            command.extend(["--batch-profile", args.batch_profile])
        if args.throughput_profile:
            command.extend(["--throughput-profile", args.throughput_profile])
        if args.cache_concurrency is not None:
            command.extend(["--cache-concurrency", str(args.cache_concurrency)])
        if args.cache_max_total_tokens is not None:
            command.extend(["--cache-max-total-tokens", str(args.cache_max_total_tokens)])
        if args.auto_batch:
            command.extend(
                [
                    "--auto-batch",
                    "--auto-batch-target-vram-gb", str(args.auto_batch_target_vram_gb),
                    "--auto-batch-hard-vram-gb", str(args.auto_batch_hard_vram_gb),
                    "--auto-batch-start-size", str(args.auto_batch_start_size),
                    "--auto-batch-safety-fraction", str(args.cache_auto_batch_safety_fraction),
                    "--auto-batch-max-size", str(args.auto_batch_max_size),
                    "--auto-batch-growth-factor", str(args.auto_batch_growth_factor),
                ]
            )
    if args.target_revision:
        command.extend(["--target-revision", args.target_revision])
    if args.local_files_only:
        command.append("--local-files-only")
    if args.resume:
        command.append("--resume")
    if sample_ids_file is not None:
        command.extend(["--sample-ids-file", str(sample_ids_file)])
    return command


def _terminate_workers(processes: Sequence[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
    deadline = time.monotonic() + 10
    for process in processes:
        if process.poll() is not None:
            continue
        remaining = max(0.1, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait()


def _write_parallel_retry_hint(
    work_root: Path,
    *,
    args: argparse.Namespace,
    failed: Sequence[tuple[int, int]],
) -> None:
    """Persist retry guidance when the OS kills a GPU worker (usually OOM)."""
    killed = [int(rank) for rank, code in failed if int(code) == -9]
    oom_ranks: list[int] = []
    if args.mode == "cache":
        for rank, _code in failed:
            log_path = _worker_root(work_root, int(rank)) / "worker.log"
            try:
                tail = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-32_000:]
            except OSError:
                tail = ""
            lowered = tail.lower()
            if "out of memory" in lowered or "cuda oom" in lowered:
                oom_ranks.append(int(rank))
    retry_ranks = sorted(set(killed) | set(oom_ranks))
    if not retry_ranks:
        return
    reduced_profile_path = None
    if args.throughput_profile:
        try:
            profile_payload = json.loads(
                Path(args.throughput_profile).read_text(encoding="utf-8")
            )
            if isinstance(profile_payload, dict) and isinstance(
                profile_payload.get("buckets"), list
            ):
                reduced_profile = dict(profile_payload)
                reduced_profile["buckets"] = [
                    {
                        **bucket,
                        "batch_size": max(1, int(bucket["batch_size"]) // 2),
                        "token_budget": max(1, int(bucket["token_budget"]) // 2),
                    }
                    for bucket in profile_payload["buckets"]
                    if isinstance(bucket, dict)
                ]
                reduced_profile_path = work_root / "retry_throughput_profile.json"
                write_json(reduced_profile_path, reduced_profile)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            reduced_profile_path = None
    payload = {
        "schema_version": "mr_dflash_parallel_retry_hint_v1",
        "reason": "worker_sigkill_or_oom",
        "killed_ranks": killed,
        "oom_ranks": oom_ranks,
        "failed_ranks": [int(rank) for rank, _code in failed],
        "resume": True,
        "suggested_batch_size": max(1, int(args.batch_size) // 2),
        "suggested_cache_concurrency": (
            None
            if args.cache_concurrency is None
            else max(1, int(args.cache_concurrency) // 2)
        ),
        "suggested_cache_max_total_tokens": (
            None
            if args.cache_max_total_tokens is None
            else max(1, int(args.cache_max_total_tokens) // 2)
        ),
        "suggested_cache_memory_fraction": min(
            0.98, max(0.80, float(args.cache_memory_fraction) - 0.02)
        ),
        "suggested_throughput_profile_path": (
            None if reduced_profile_path is None else str(reduced_profile_path)
        ),
        "note": (
            "Các shard đã durable được giữ nguyên. Chạy lại cùng command với "
            "--resume và các giá trị suggested; profile là performance-only."
        ),
    }
    write_json(work_root / "retry_hint.json", payload)


def _run_workers(
    args: argparse.Namespace,
    work_root: Path,
    gpu_ids: Sequence[int],
    *,
    total_samples: int,
) -> list[Path]:
    worker_roots = [_worker_root(work_root, rank) for rank in range(len(gpu_ids))]
    work_root.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[Any]] = []
    logs = []
    started_at: dict[int, float] = {}
    previous_handlers: dict[int, Any] = {}
    dist_port_base = 29600 + (os.getpid() % 1000) * 4
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm có trong runtime server
        aggregate_bar = None
    else:
        aggregate_bar = tqdm(
            total=max(0, int(total_samples)),
            desc=f"MR-DFlash parallel {args.mode}",
            unit="sample",
            dynamic_ncols=True,
            mininterval=1.0,
        )
    last_aggregate_completed = 0

    def handle_parent_signal(signum: int, _frame: Any) -> None:
        # SIGTERM/SIGINT của parent phải truyền xuống process group. Nếu chỉ
        # dừng parent, target model trong worker session độc lập sẽ còn treo
        # GPU như trường hợp người dùng gặp phải.
        print(f"[parallel] nhận signal={signum}; đang dừng toàn bộ worker", flush=True)
        _terminate_workers(processes)
        raise KeyboardInterrupt(f"parallel stage interrupted by signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_parent_signal)
        for rank, gpu_id in enumerate(gpu_ids):
            root = worker_roots[rank]
            root.mkdir(parents=True, exist_ok=True)
            sample_ids_file = getattr(args, "_shared_sample_ids_files", {}).get(rank)
            command = _build_worker_command(
                args,
                root,
                rank,
                len(gpu_ids),
                sample_ids_file=sample_ids_file,
            )
            log_path = root / "worker.log"
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(f"\n=== command gpu={gpu_id} rank={rank} ===\n")
            log_handle.flush()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            environment["MR_DFLASH_GPU_ID"] = str(gpu_id)
            # Mỗi worker là một process-group SGLang độc lập trên một GPU;
            # dùng port riêng để bốn worker không join nhầm cùng rendezvous.
            environment["MR_DFLASH_WORKER_RANK"] = str(rank)
            # Adapter cộng ``rank`` vào base này để mỗi child có rendezvous
            # riêng; giữ cùng base giúp trực tiếp chạy một worker cũng ổn.
            environment["MR_DFLASH_DIST_PORT"] = str(dist_port_base)
            environment.setdefault("PYTHONUNBUFFERED", "1")
            source_path = str(REPO_ROOT / "src")
            environment["PYTHONPATH"] = source_path + (
                os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
            )
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            started_at[rank] = time.time()
            logs.append((log_handle, log_path))
            print(f"[parallel] rank={rank} gpu={gpu_id} pid={process.pid} log={log_path}", flush=True)
            if args.mode == "cache" and args.cache_startup_stagger_seconds > 0 and rank + 1 < len(gpu_ids):
                time.sleep(args.cache_startup_stagger_seconds)
        while True:
            statuses = [process.poll() for process in processes]
            worker_statuses = [
                build_worker_status(
                    rank=rank,
                    gpu_id=gpu_ids[rank],
                    pid=processes[rank].pid,
                    return_code=statuses[rank],
                    worker_root=worker_roots[rank],
                )
                for rank in range(len(processes))
            ]
            now = time.time()
            heartbeat = getattr(args, "_shared_queue_heartbeat", None)
            if callable(heartbeat):
                heartbeat()
            aggregate = aggregate_progress(worker_statuses, total_samples=total_samples)
            if aggregate_bar is not None:
                completed = int(aggregate.get("completed_samples", 0) or 0)
                delta = completed - last_aggregate_completed
                if delta > 0:
                    aggregate_bar.update(delta)
                    last_aggregate_completed = completed
                aggregate_bar.set_postfix(
                    tokens=(
                        f"{int(aggregate.get('completed_tokens', 0) or 0)}"
                        f"/{int(aggregate.get('total_tokens', 0) or 0)}"
                    ),
                    rate=(
                        "-"
                        if aggregate.get("throughput_tokens_per_second") is None
                        else f"{float(aggregate['throughput_tokens_per_second']):.1f} tok/s"
                    ),
                    eta=aggregate.get("eta_human", "unknown"),
                )
                aggregate_bar.refresh()
            write_json(
                work_root / "status.json",
                {
                    "schema_version": "mr_dflash_parallel_status_v2",
                    "status": "running" if not all(value is not None for value in statuses) else "joining",
                    "gpu_ids": [int(value) for value in gpu_ids],
                    "updated_at_unix": now,
                    "workers": worker_statuses,
                    "aggregate": aggregate,
                },
            )
            if args.stop_file and Path(args.stop_file).exists():
                _terminate_workers(processes)
                raise RuntimeError(f"stop file yêu cầu dừng: {args.stop_file}")
            if args.stall_timeout_seconds > 0:
                stalled = []
                for rank, worker in enumerate(worker_statuses):
                    if not worker["alive"]:
                        continue
                    age = worker.get("progress_age_seconds")
                    if age is None:
                        age = now - started_at.get(rank, now)
                    if age > args.stall_timeout_seconds:
                        stalled.append((rank, age, worker["progress"].get("phase")))
                if stalled:
                    _terminate_workers(processes)
                    detail = ", ".join(
                        f"rank={rank} age={age:.1f}s phase={phase}"
                        for rank, age, phase in stalled
                    )
                    raise RuntimeError(
                        "worker không cập nhật progress trong thời gian cho phép: "
                        f"{detail}; xem status.json và worker.log"
                    )
            failed = [(rank, status) for rank, status in enumerate(statuses) if status not in (None, 0)]
            if failed:
                _terminate_workers(processes)
                _write_parallel_retry_hint(work_root, args=args, failed=failed)
                details = ", ".join(
                    f"rank={rank} code={status} log={worker_roots[rank] / 'worker.log'}"
                    for rank, status in failed
                )
                raise RuntimeError(f"parallel worker thất bại: {details}")
            if all(value is not None for value in statuses):
                break
            time.sleep(1.0)
        return worker_roots
    except BaseException:
        _terminate_workers(processes)
        raise
    finally:
        if aggregate_bar is not None:
            aggregate_bar.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for log_handle, _log_path in logs:
            log_handle.close()


def _count_parallel_samples(args: argparse.Namespace) -> int:
    """Đếm một lần nguồn sample mà các worker thực sự phân shard.

    Cache có thể lặp trên tokenized manifest thay vì JSONL ``--input``; khi
    đó manifest là nguồn chuẩn để mẫu số không lệch với workload của worker.
    """
    if args.mode == "cache" and args.tokenized_path:
        manifest_path = Path(args.tokenized_path) / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = payload.get("shards", [])
        if not isinstance(shards, list):
            raise ValueError(f"tokenized manifest không có shards hợp lệ: {manifest_path}")
        return sum(max(0, int(shard.get("count", 0) or 0)) for shard in shards)
    return sum(1 for _ in read_jsonl(Path(args.input)))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MR-DFlash data-parallel generate/cache")
    parser.add_argument("--mode", choices=["regenerate", "cache"], required=True)
    parser.add_argument(
        "--scheduler",
        choices=["static", "shared_lease"],
        default="static",
        help="static modulo shards hoặc queue lease resumable trên shared filesystem",
    )
    parser.add_argument(
        "--queue-root",
        default=None,
        help="thư mục queue chung; mặc định là work-root/shared_queue",
    )
    parser.add_argument("--queue-lease-ttl-seconds", type=float, default=300.0)
    parser.add_argument("--queue-lock-ttl-seconds", type=float, default=600.0)
    parser.add_argument("--queue-max-attempts", type=int, default=3)
    parser.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--tokenized-path",
        default=None,
        help="tokenized shard dùng trực tiếp cho cache; input vẫn là nguồn provenance/coverage",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        default=None,
        help="các layer target cần cache; bắt buộc truyền rõ khi chạy MR-DFlash",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--preserve-full-input", action="store_true")
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
        help="batch inference thật trong model.generate; khác output-batch-size",
    )
    parser.add_argument(
        "--auto-batch",
        action="store_true",
        help="bật adaptive batch cho regenerate hoặc SpecForge cache",
    )
    parser.add_argument(
        "--auto-batch-target-vram-gb",
        type=float,
        default=160.0,
        help="soft target VRAM mỗi GPU; mặc định 160 GiB",
    )
    parser.add_argument(
        "--auto-batch-hard-vram-gb",
        type=float,
        default=None,
        help="hard cap VRAM mỗi GPU; mặc định bằng soft target nếu không truyền",
    )
    parser.add_argument(
        "--auto-batch-start-size",
        type=int,
        default=1,
        help="batch khởi đầu cho adaptive mode; mặc định 1, sau đó tăng dần",
    )
    parser.add_argument(
        "--cache-auto-batch-safety-fraction",
        type=float,
        default=0.95,
        help="tỷ lệ token-pool cache an toàn trước hard capacity; mặc định 0.95",
    )
    parser.add_argument(
        "--auto-batch-max-size",
        type=int,
        default=128,
        help="batch tối đa trên mỗi length/budget bucket",
    )
    parser.add_argument(
        "--auto-batch-growth-factor",
        type=float,
        default=2.0,
        help="hệ số tăng batch sau mỗi batch thành công",
    )
    parser.add_argument("--overflow-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--sample-error-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bucket-buffer-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument(
        "--batch-profile",
        default=None,
        help="profile JSON dùng chung cho mọi worker cache; chọn batch theo length bucket",
    )
    parser.add_argument(
        "--throughput-profile",
        "--cache-throughput-profile",
        "--cache-profile",
        dest="throughput_profile",
        default=None,
        help="profile token-budget SpecForge dùng chung cho mọi worker cache",
    )
    parser.add_argument(
        "--cache-backend",
        choices=["hf", "specforge_sglang"],
        default="hf",
        help="backend cache target; specforge_sglang dùng offline SGLang",
    )
    parser.add_argument(
        "--cache-concurrency",
        type=int,
        default=None,
        help="max running requests trên mỗi GPU cho SGLang cache",
    )
    parser.add_argument(
        "--cache-max-total-tokens",
        type=int,
        default=None,
        help="max total tokens static pool trên mỗi GPU cho SGLang cache",
    )
    parser.add_argument(
        "--cache-memory-fraction",
        type=float,
        default=0.99,
        help="VRAM static pool fraction trên mỗi GPU (mặc định 0.99)",
    )
    parser.add_argument(
        "--attention-backend",
        choices=[
            "auto",
            "eager",
            "sdpa",
            "flash_attention_2",
            "flashinfer",
            "triton",
            "torch_native",
        ],
        default="sdpa",
        help="attention implementation cho HF backbone khi cache target",
    )
    parser.add_argument(
        "--io-threads",
        type=int,
        default=2,
        help="số thread ghi shard trên mỗi worker cache; 0 = ghi đồng bộ",
    )
    parser.add_argument(
        "--io-queue-size",
        type=int,
        default=4,
        help="số shard tối đa đang chờ ghi trên mỗi worker",
    )
    parser.add_argument("--supervision-mode", choices=["all_assistant", "last_assistant"], default="last_assistant")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--progress-interval-tokens",
        type=int,
        default=256,
        help="số token giữa hai heartbeat khi target đang decode",
    )
    parser.add_argument(
        "--output-batch-size",
        type=int,
        default=1,
        help="số sample gom trước khi flush output regenerate; mặc định 1 để dễ resume/debug",
    )
    parser.add_argument(
        "--stall-timeout-seconds",
        type=float,
        default=0.0,
        help="dừng worker nếu heartbeat quá cũ; 0 tắt kiểm tra timeout",
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="khi file tồn tại, parent dừng sạch toàn bộ worker ở vòng poll kế tiếp",
    )
    parser.add_argument(
        "--cache-startup-stagger-seconds",
        type=float,
        default=3.0,
        help="khoảng cách khởi động worker cache để tránh peak host/PCIe (giây)",
    )
    args = parser.parse_args(argv)
    if args.auto_batch_hard_vram_gb is None:
        args.auto_batch_hard_vram_gb = float(args.auto_batch_target_vram_gb)
    if args.queue_lease_ttl_seconds <= 0 or args.queue_lock_ttl_seconds <= 0:
        raise ValueError("queue TTL phải > 0")
    if args.queue_max_attempts < 1:
        raise ValueError("queue-max-attempts phải >= 1")
    args.gpu_ids = _validate_gpu_ids(args.gpu_ids)
    if args.mode == "cache" and not args.target_layer_ids:
        raise ValueError("parallel cache phải truyền rõ --target-layer-ids")
    if args.target_layer_ids and any(int(value) < 0 for value in args.target_layer_ids):
        raise ValueError("target-layer-ids không được âm")
    if args.batch_size < 1 or args.bucket_buffer_size < args.batch_size or args.shard_size < 1:
        raise ValueError("batch/bucket-buffer/shard-size không hợp lệ")
    if args.generation_batch_size < 1:
        raise ValueError("generation-batch-size phải >= 1")
    if args.auto_batch_max_size < args.generation_batch_size:
        raise ValueError("auto-batch-max-size phải >= generation-batch-size")
    if args.auto_batch_start_size < 1:
        raise ValueError("auto-batch-start-size phải >= 1")
    if args.auto_batch_start_size > args.auto_batch_max_size:
        raise ValueError("auto-batch-start-size phải <= auto-batch-max-size")
    if not 0.0 < args.cache_auto_batch_safety_fraction <= 1.0:
        raise ValueError("cache-auto-batch-safety-fraction phải thuộc (0, 1]")
    if args.auto_batch_growth_factor <= 1.0:
        raise ValueError("auto-batch-growth-factor phải > 1")
    if args.auto_batch_target_vram_gb <= 0.0:
        raise ValueError("auto-batch-target-vram-gb phải > 0")
    if args.auto_batch_hard_vram_gb < args.auto_batch_target_vram_gb:
        raise ValueError("hard VRAM phải >= soft target")
    if args.io_threads < 0 or args.io_queue_size < 0:
        raise ValueError("io-threads và io-queue-size không được âm")
    if args.cache_concurrency is not None and args.cache_concurrency < 1:
        raise ValueError("cache-concurrency phải >= 1")
    if args.cache_max_total_tokens is not None and args.cache_max_total_tokens < 1:
        raise ValueError("cache-max-total-tokens phải >= 1")
    if not 0.0 < args.cache_memory_fraction <= 1.0:
        raise ValueError("cache-memory-fraction phải thuộc (0, 1]")
    if args.cache_startup_stagger_seconds < 0:
        raise ValueError("cache-startup-stagger-seconds không được âm")
    if args.progress_interval_tokens < 1 or args.output_batch_size < 1:
        raise ValueError("progress-interval-tokens và output-batch-size phải >= 1")
    if args.stall_timeout_seconds < 0:
        raise ValueError("stall-timeout-seconds không được âm")
    if args.batch_profile and not Path(args.batch_profile).is_file():
        raise FileNotFoundError(f"không tìm thấy cache batch profile: {args.batch_profile}")
    if args.throughput_profile and not Path(args.throughput_profile).is_file():
        raise FileNotFoundError(
            f"không tìm thấy cache throughput profile: {args.throughput_profile}"
        )
    if args.batch_profile and args.throughput_profile:
        raise ValueError("chỉ được dùng một trong batch-profile hoặc throughput-profile")
    if args.work_root is None:
        output = Path(args.output)
        args.work_root = str(resolve_parallel_work_root(output, args.mode))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    work_root = Path(args.work_root)
    total_samples = _count_parallel_samples(args)
    if args.stop_file and Path(args.stop_file).exists():
        raise RuntimeError(f"stop file đã tồn tại trước khi khởi động: {args.stop_file}")
    active_pids = _active_worker_pids(work_root)
    if active_pids:
        raise RuntimeError(
            "work-root đang có worker còn sống; không launch trùng target. "
            f"pids={active_pids}; dùng watcher hoặc stop file trước khi resume"
        )
    output_exists = output_path.is_file() or manifest_path.is_file()
    if args.mode == "cache":
        output_exists = output_exists or (output_path / "manifest.json").is_file()
    if output_exists and not args.resume:
        raise FileExistsError(
            f"parallel output đã tồn tại ({output_path}); dùng --resume hoặc output/work-root mới"
        )
    if args.mode == "cache" and args.batch_profile and not Path(args.batch_profile).is_file():
        raise FileNotFoundError(f"không tìm thấy cache batch profile: {args.batch_profile}")
    plan_path = work_root / "parallel_plan.json"
    plan = {
        "schema_version": "mr_dflash_parallel_plan_v1",
        "mode": args.mode,
        "scheduler": args.scheduler,
        "gpu_ids": args.gpu_ids,
        "queue_root": str(Path(args.queue_root)) if args.queue_root else str(work_root / "shared_queue"),
        "input": str(Path(args.input)),
        "tokenized_path": str(Path(args.tokenized_path)) if args.tokenized_path else None,
        "output": str(output_path),
        "manifest": str(manifest_path),
        "target_model_path": args.target_model_path,
        "max_length": int(args.max_length),
        "target_layer_ids": (
            [int(value) for value in args.target_layer_ids]
            if args.target_layer_ids
            else None
        ),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "seed": int(args.seed),
        "torch_dtype": args.torch_dtype,
        "preserve_full_input": bool(args.preserve_full_input),
        "overflow_policy": args.overflow_policy,
        "sample_error_policy": args.sample_error_policy,
        "batch_size": int(args.batch_size),
        "bucket_buffer_size": int(args.bucket_buffer_size),
        "shard_size": int(args.shard_size),
        "batch_profile": str(args.batch_profile) if args.batch_profile else None,
        "batch_profile_sha256": (
            hashlib.sha256(Path(args.batch_profile).read_bytes()).hexdigest()
            if args.batch_profile
            else None
        ),
        "throughput_profile": str(args.throughput_profile) if args.throughput_profile else None,
        "throughput_profile_sha256": (
            hashlib.sha256(Path(args.throughput_profile).read_bytes()).hexdigest()
            if args.throughput_profile
            else None
        ),
        "cache_backend": args.cache_backend,
        "cache_concurrency": args.cache_concurrency,
        "cache_max_total_tokens": args.cache_max_total_tokens,
        "cache_memory_fraction": float(args.cache_memory_fraction),
        "cache_startup_stagger_seconds": float(args.cache_startup_stagger_seconds),
        "attention_backend": args.attention_backend,
        "io_threads": int(args.io_threads),
        "io_queue_size": int(args.io_queue_size),
        "supervision_mode": args.supervision_mode,
        "progress_interval_tokens": int(args.progress_interval_tokens),
        "output_batch_size": int(args.output_batch_size),
        "generation_batch_size": int(args.generation_batch_size),
        "auto_batch": bool(args.auto_batch),
        "auto_batch_target_vram_gb": float(args.auto_batch_target_vram_gb),
        "auto_batch_hard_vram_gb": float(args.auto_batch_hard_vram_gb),
        "auto_batch_start_size": int(args.auto_batch_start_size),
        "cache_auto_batch_safety_fraction": float(args.cache_auto_batch_safety_fraction),
        "auto_batch_max_size": int(args.auto_batch_max_size),
        "auto_batch_growth_factor": float(args.auto_batch_growth_factor),
        "stall_timeout_seconds": float(args.stall_timeout_seconds),
        "stop_file": str(args.stop_file) if args.stop_file else None,
        "num_shards": len(args.gpu_ids),
        "total_samples": total_samples,
    }
    if args.resume and plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        immutable = (
            "mode",
            "input",
            "tokenized_path",
            "output",
            "manifest",
            "target_model_path",
            "max_length",
            "max_new_tokens",
            "generation_batch_size",
            "target_layer_ids",
            "attention_backend",
            "io_threads",
            "io_queue_size",
            "total_samples",
            "cache_backend",
        )
        # New optimization fields were added after older work-roots may have
        # been created. Missing legacy keys are compatible; an explicitly
        # recorded value that changes is still rejected.
        mismatches = [
            key for key in immutable
            if key in old and old.get(key) != plan.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                f"parallel plan không khớp ở {', '.join(mismatches)}; dùng work-root mới"
            )
    elif work_root.exists() and any(work_root.iterdir()) and not args.resume:
        raise FileExistsError(f"parallel work-root đã tồn tại: {work_root}; dùng --resume hoặc work-root mới")
    write_json(plan_path, plan)
    queue: SharedLeaseQueue | None = None
    queue_leases: dict[int, Any] = {}
    if args.scheduler == "shared_lease":
        queue = prepare_shared_queue(args, work_root)
        queue_leases = assign_shared_leases(queue, args, work_root)

        def heartbeat_shared_queue() -> None:
            for lease in list(queue_leases.values()):
                queue.heartbeat(lease.lease_id)

        args._shared_queue_heartbeat = heartbeat_shared_queue
    try:
        if queue is None:
            worker_roots = _run_workers(
                args,
                work_root,
                args.gpu_ids,
                total_samples=total_samples,
            )
        else:
            worker_roots = []
            round_index = 0
            while True:
                try:
                    worker_roots = _run_workers(
                        args,
                        work_root,
                        args.gpu_ids,
                        total_samples=total_samples,
                    )
                    incomplete = False
                    for rank, lease in list(queue_leases.items()):
                        assigned_ids = {item.sample_id for item in lease.items}
                        done_ids = durable_worker_ids(args, work_root, rank) & assigned_ids
                        unresolved_ids = assigned_ids - done_ids
                        if not unresolved_ids:
                            continue
                        if done_ids:
                            queue.complete(
                                lease.lease_id,
                                sample_ids=done_ids,
                                artifacts={sample_id: str(_worker_root(work_root, rank)) for sample_id in done_ids},
                            )
                        retry_result = queue.retry(
                            lease.lease_id,
                            error=(
                                "worker completed without durable artifact for "
                                f"{len(unresolved_ids)} sample(s)"
                            ),
                        )
                        if retry_result == "quarantined":
                            queue_leases.pop(rank, None)
                        incomplete = True
                    if incomplete:
                        round_index += 1
                        if queue.snapshot()["pending"] == 0:
                            worker_roots = sorted(work_root.glob("rank_*"))
                            break
                        if round_index >= int(args.queue_max_attempts):
                            raise RuntimeError(
                                "shared queue còn sample không có artifact sau "
                                f"{args.queue_max_attempts} lần thử"
                            )
                        queue_leases = assign_shared_leases(queue, args, work_root)
                        continue
                    break
                except BaseException as worker_error:
                    # Commit completed rows from a worker that died after a
                    # durable flush, then retry only the unresolved IDs.
                    for rank, lease in list(queue_leases.items()):
                        done_ids = durable_worker_ids(args, work_root, rank)
                        done_ids &= {item.sample_id for item in lease.items}
                        if done_ids:
                            queue.complete(
                                lease.lease_id,
                                sample_ids=done_ids,
                                artifacts={sample_id: str(_worker_root(work_root, rank)) for sample_id in done_ids},
                            )
                        try:
                            retry_result = queue.retry(lease.lease_id, error=repr(worker_error))
                            if retry_result == "quarantined":
                                # The lease no longer exists after the final
                                # retry; do not call complete() on it in the
                                # post-loop artifact publication step.
                                queue_leases.pop(rank, None)
                        except RuntimeError:
                            pass
                    round_index += 1
                    if queue.snapshot()["pending"] == 0:
                        worker_roots = sorted(work_root.glob("rank_*"))
                        break
                    if round_index >= int(args.queue_max_attempts):
                        raise
                    queue_leases = assign_shared_leases(queue, args, work_root)
        merge_worker_roots = sorted(work_root.glob("rank_*")) if queue is not None else worker_roots
        if not merge_worker_roots:
            merge_worker_roots = worker_roots
        if queue is not None:
            for rank, lease in queue_leases.items():
                root = _worker_root(work_root, rank)
                artifact = root / ("output.jsonl" if args.mode == "regenerate" else "manifest.json")
                queue.complete(
                    lease.lease_id,
                    artifacts={item.sample_id: str(artifact) for item in lease.items},
                )
        quarantine_records = queue.quarantine_records() if queue is not None else []
        quarantined_ids = {str(record["sample_id"]) for record in quarantine_records}
        if quarantine_records:
            write_jsonl(work_root / "quarantine.jsonl", quarantine_records)
        if quarantine_records and args.mode == "regenerate":
            quarantine_root = _worker_root(work_root, 0)
            quarantine_path = quarantine_root / "skipped.jsonl"
            existing_quarantine = list(read_jsonl(quarantine_path)) if quarantine_path.is_file() else []
            existing_ids = {str(row.get("id", "")) for row in existing_quarantine}
            quarantine_rows = [
                {
                    "id": str(record["sample_id"]),
                    "kind": "quarantine",
                    "error": str(record.get("error", "")),
                    "retry_attempts": int(record.get("attempts", 0)),
                }
                for record in quarantine_records
                if str(record["sample_id"]) not in existing_ids
            ]
            if quarantine_rows:
                write_jsonl(quarantine_path, [*existing_quarantine, *quarantine_rows])
            quarantine_root.mkdir(parents=True, exist_ok=True)
            if quarantine_root not in merge_worker_roots:
                merge_worker_roots = [*merge_worker_roots, quarantine_root]
        if args.mode == "regenerate":
            # A worker can be killed by CUDA OOM before its first flush. Keep
            # empty artifacts for that rank so the canonical merge can still
            # represent samples that were quarantined after the final retry.
            for root in merge_worker_roots:
                (root / "output.jsonl").parent.mkdir(parents=True, exist_ok=True)
                for artifact_path in (root / "output.jsonl", root / "skipped.jsonl"):
                    artifact_path.touch(exist_ok=True)
        merge_gpu_ids = list(args.gpu_ids)
        if len(merge_worker_roots) > len(merge_gpu_ids):
            merge_gpu_ids.extend([-1] * (len(merge_worker_roots) - len(merge_gpu_ids)))
        if args.mode == "regenerate":
            payload = merge_regenerated_outputs(
                input_path=Path(args.input),
                worker_roots=merge_worker_roots,
                output_path=output_path,
                skipped_path=output_path.with_name(output_path.stem + ".skipped.jsonl"),
                manifest_path=manifest_path,
                gpu_ids=merge_gpu_ids,
            )
        else:
            payload = merge_feature_caches(
                input_path=Path(args.input),
                tokenized_path=Path(args.tokenized_path) if args.tokenized_path else None,
                worker_roots=merge_worker_roots,
                output_path=output_path,
                manifest_path=manifest_path,
                gpu_ids=merge_gpu_ids,
                quarantined_ids=sorted(quarantined_ids),
            )
        last_status: Dict[str, Any] = {}
        status_path = work_root / "status.json"
        if status_path.is_file():
            try:
                value = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    last_status = value
            except (OSError, json.JSONDecodeError):
                last_status = {}
        write_json(
            status_path,
            {
                "schema_version": "mr_dflash_parallel_status_v2",
                "status": "success_with_quarantine" if quarantine_records else "success",
                "payload": payload,
                "gpu_ids": [int(value) for value in args.gpu_ids],
                "workers": last_status.get("workers", []),
                "aggregate": aggregate_progress(
                    last_status.get("workers", []), total_samples=total_samples
                ),
                "updated_at_unix": time.time(),
            },
        )
        print(f"[parallel] DONE mode={args.mode} gpu_ids={args.gpu_ids}", flush=True)
        return 0
    except BaseException as exc:
        if queue is not None:
            for lease in queue_leases.values():
                try:
                    queue.retry(lease.lease_id, error=repr(exc))
                except RuntimeError:
                    pass
        status_path = work_root / "status.json"
        last_status: Dict[str, Any] = {}
        if status_path.is_file():
            try:
                value = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    last_status = value
            except (OSError, json.JSONDecodeError):
                last_status = {}
        write_json(
            status_path,
            {
                "schema_version": "mr_dflash_parallel_status_v2",
                "status": "failed",
                "error": repr(exc),
                "gpu_ids": [int(value) for value in args.gpu_ids],
                "workers": last_status.get("workers", []),
                "aggregate": aggregate_progress(
                    last_status.get("workers", []), total_samples=total_samples
                ),
                "updated_at_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
