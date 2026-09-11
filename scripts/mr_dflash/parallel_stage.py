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
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from _common import REPO_ROOT, read_jsonl, resolve_parallel_work_root, write_json, write_jsonl
from progress import format_duration, read_progress


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
        "gpu_id": int(gpu_id),
        "pid": int(pid),
        "return_code": return_code,
        "alive": return_code is None,
        "progress_age_seconds": age,
        "progress": progress,
        "log": str(worker_root / "worker.log"),
    }


def aggregate_progress(worker_statuses: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
        "total_samples": total("total_samples"),
        "completed_samples": total("completed_samples"),
        "remaining_samples": total("remaining_samples"),
        "total_tokens": total("total_tokens"),
        "completed_tokens": total("completed_tokens"),
        "remaining_tokens": total("remaining_tokens"),
        "throughput_tokens_per_second": round(throughput, 3) if throughput is not None else None,
        "eta_seconds": round(eta, 3) if eta is not None else None,
        "eta_human": format_duration(eta),
    }
    return result


def _active_worker_pids(work_root: Path) -> list[int]:
    """Tìm worker cũ còn sống để không launch trùng target trên cùng GPU."""
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
    for worker in payload.get("workers", []):
        if not isinstance(worker, dict) or not worker.get("alive"):
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
    worker_stats: list[Dict[str, Any]] = []
    for rank, root in enumerate(worker_roots):
        output_rows = _read_object_rows(root / "output.jsonl")
        skipped_rows = _read_object_rows(root / "skipped.jsonl")
        for sample_id, row in _index_rows(output_rows, path=root / "output.jsonl").items():
            if sample_id in generated or sample_id in skipped:
                raise ValueError(f"worker outputs trùng sample id: {sample_id!r}")
            generated[sample_id] = row
        for sample_id, row in _index_rows(skipped_rows, path=root / "skipped.jsonl").items():
            if sample_id in generated or sample_id in skipped:
                raise ValueError(f"worker reports trùng sample id: {sample_id!r}")
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
    )
    base: Dict[str, Any] | None = None
    all_ids: list[str] = []
    final_shards: list[Dict[str, Any]] = []
    for rank, root in enumerate(worker_roots):
        manifest_path_worker = root / "manifest.json"
        if not manifest_path_worker.is_file():
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
    if missing or extra:
        raise RuntimeError(
            "parallel cache coverage không đầy đủ: "
            f"missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})"
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
        stats={
            "worker_count": len(worker_roots),
            "gpu_ids": [int(value) for value in gpu_ids],
            "num_samples": len(all_ids),
        },
    )
    write_json(manifest_path, {**output_manifest, "parallel_gpu_ids": [int(value) for value in gpu_ids]})
    return output_manifest


def _build_worker_command(args: argparse.Namespace, root: Path, rank: int, num_shards: int) -> list[str]:
    python = sys.executable
    worker_device = "cuda:0"
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
            "--torch-dtype", args.torch_dtype,
            "--shard-index", str(rank),
            "--num-shards", str(num_shards),
            "--overflow-policy", args.overflow_policy,
            "--sample-error-policy", args.sample_error_policy,
            "--skipped-report", str(root / "skipped.jsonl"),
            "--progress-path", str(root / "progress.json"),
            "--progress-interval-tokens", str(args.progress_interval_tokens),
            "--output-batch-size", str(args.output_batch_size),
        ]
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
            "--io-threads", str(args.io_threads),
            "--io-queue-size", str(args.io_queue_size),
            "--device", worker_device,
            "--torch-dtype", args.torch_dtype,
            "--supervision-mode", args.supervision_mode,
            "--shard-index", str(rank),
            "--num-shards", str(num_shards),
            "--skipped-report", str(root / "skipped.jsonl"),
            "--progress-path", str(root / "progress.json"),
        ]
        if args.tokenized_path:
            command.extend(["--tokenized-path", args.tokenized_path])
    if args.target_revision:
        command.extend(["--target-revision", args.target_revision])
    if args.local_files_only:
        command.append("--local-files-only")
    if args.resume:
        command.append("--resume")
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


def _run_workers(args: argparse.Namespace, work_root: Path, gpu_ids: Sequence[int]) -> list[Path]:
    worker_roots = [_worker_root(work_root, rank) for rank in range(len(gpu_ids))]
    work_root.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[Any]] = []
    logs = []
    started_at: dict[int, float] = {}
    previous_handlers: dict[int, Any] = {}
    last_progress_log_at = 0.0

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
            command = _build_worker_command(args, root, rank, len(gpu_ids))
            log_path = root / "worker.log"
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(f"\n=== command gpu={gpu_id} rank={rank} ===\n")
            log_handle.flush()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            environment["MR_DFLASH_GPU_ID"] = str(gpu_id)
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
            aggregate = aggregate_progress(worker_statuses)
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
            if now - last_progress_log_at >= 30.0:
                print(
                    "[parallel] progress "
                    f"mode={args.mode} "
                    f"completed={aggregate['completed_samples']}/{aggregate['total_samples']} "
                    f"tokens={aggregate['completed_tokens']}/{aggregate['total_tokens']} "
                    f"rate={aggregate['throughput_tokens_per_second'] or 0:.1f} tok/s "
                    f"eta={aggregate['eta_human']}",
                    flush=True,
                )
                last_progress_log_at = now
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
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for log_handle, _log_path in logs:
            log_handle.close()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MR-DFlash data-parallel generate/cache")
    parser.add_argument("--mode", choices=["regenerate", "cache"], required=True)
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
    parser.add_argument("--overflow-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--sample-error-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bucket-buffer-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
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
    args = parser.parse_args(argv)
    args.gpu_ids = _validate_gpu_ids(args.gpu_ids)
    if args.mode == "cache" and not args.target_layer_ids:
        raise ValueError("parallel cache phải truyền rõ --target-layer-ids")
    if args.target_layer_ids and any(int(value) < 0 for value in args.target_layer_ids):
        raise ValueError("target-layer-ids không được âm")
    if args.batch_size < 1 or args.bucket_buffer_size < args.batch_size or args.shard_size < 1:
        raise ValueError("batch/bucket-buffer/shard-size không hợp lệ")
    if args.io_threads < 0 or args.io_queue_size < 0:
        raise ValueError("io-threads và io-queue-size không được âm")
    if args.progress_interval_tokens < 1 or args.output_batch_size < 1:
        raise ValueError("progress-interval-tokens và output-batch-size phải >= 1")
    if args.stall_timeout_seconds < 0:
        raise ValueError("stall-timeout-seconds không được âm")
    if args.work_root is None:
        output = Path(args.output)
        args.work_root = str(resolve_parallel_work_root(output, args.mode))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    work_root = Path(args.work_root)
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
    plan_path = work_root / "parallel_plan.json"
    plan = {
        "schema_version": "mr_dflash_parallel_plan_v1",
        "mode": args.mode,
        "gpu_ids": args.gpu_ids,
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
        "attention_backend": args.attention_backend,
        "io_threads": int(args.io_threads),
        "io_queue_size": int(args.io_queue_size),
        "supervision_mode": args.supervision_mode,
        "progress_interval_tokens": int(args.progress_interval_tokens),
        "output_batch_size": int(args.output_batch_size),
        "stall_timeout_seconds": float(args.stall_timeout_seconds),
        "stop_file": str(args.stop_file) if args.stop_file else None,
        "num_shards": len(args.gpu_ids),
    }
    if args.resume and plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        immutable = (
            "mode",
            "gpu_ids",
            "input",
            "tokenized_path",
            "output",
            "manifest",
            "target_model_path",
            "max_length",
            "max_new_tokens",
            "target_layer_ids",
            "attention_backend",
            "io_threads",
            "io_queue_size",
            "num_shards",
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
    try:
        worker_roots = _run_workers(args, work_root, args.gpu_ids)
        if args.mode == "regenerate":
            payload = merge_regenerated_outputs(
                input_path=Path(args.input),
                worker_roots=worker_roots,
                output_path=output_path,
                skipped_path=output_path.with_name(output_path.stem + ".skipped.jsonl"),
                manifest_path=manifest_path,
                gpu_ids=args.gpu_ids,
            )
        else:
            payload = merge_feature_caches(
                input_path=Path(args.input),
                tokenized_path=Path(args.tokenized_path) if args.tokenized_path else None,
                worker_roots=worker_roots,
                output_path=output_path,
                manifest_path=manifest_path,
                gpu_ids=args.gpu_ids,
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
                "status": "success",
                "payload": payload,
                "gpu_ids": [int(value) for value in args.gpu_ids],
                "workers": last_status.get("workers", []),
                "aggregate": aggregate_progress(last_status.get("workers", [])),
                "updated_at_unix": time.time(),
            },
        )
        print(f"[parallel] DONE mode={args.mode} gpu_ids={args.gpu_ids}", flush=True)
        return 0
    except BaseException as exc:
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
                "aggregate": aggregate_progress(last_status.get("workers", [])),
                "updated_at_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
