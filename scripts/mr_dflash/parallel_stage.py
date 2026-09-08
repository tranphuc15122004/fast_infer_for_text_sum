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

from _common import REPO_ROOT, read_jsonl, write_json, write_jsonl


def _worker_root(work_root: Path, rank: int) -> Path:
    return work_root / f"rank_{rank:02d}"


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
    worker_roots: Sequence[Path],
    output_path: Path,
    manifest_path: Path,
    gpu_ids: Sequence[int],
) -> Dict[str, Any]:
    """Merge worker feature stores mà không reserialize hidden tensor lớn."""
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
        "target_revision",
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
        target_revision=base.get("target_revision"),
        stored_feature_dtype=base.get("stored_feature_dtype"),
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
            "--batch-size", str(args.batch_size),
            "--bucket-buffer-size", str(args.bucket_buffer_size),
            "--shard-size", str(args.shard_size),
            "--device", worker_device,
            "--torch-dtype", args.torch_dtype,
            "--supervision-mode", args.supervision_mode,
            "--shard-index", str(rank),
            "--num-shards", str(num_shards),
            "--skipped-report", str(root / "skipped.jsonl"),
        ]
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
    try:
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
            logs.append((log_handle, log_path))
            print(f"[parallel] rank={rank} gpu={gpu_id} pid={process.pid} log={log_path}", flush=True)
        while True:
            statuses = [process.poll() for process in processes]
            write_json(
                work_root / "status.json",
                {
                    "status": "running" if not all(value is not None for value in statuses) else "joining",
                    "gpu_ids": [int(value) for value in gpu_ids],
                    "workers": [
                        {"rank": rank, "gpu_id": int(gpu_ids[rank]), "return_code": status}
                        for rank, status in enumerate(statuses)
                    ],
                },
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
        for log_handle, _log_path in logs:
            log_handle.close()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MR-DFlash data-parallel generate/cache")
    parser.add_argument("--mode", choices=["regenerate", "cache"], required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--max-length", type=int, required=True)
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
    parser.add_argument("--supervision-mode", choices=["all_assistant", "last_assistant"], default="last_assistant")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.gpu_ids = _validate_gpu_ids(args.gpu_ids)
    if args.batch_size < 1 or args.bucket_buffer_size < args.batch_size or args.shard_size < 1:
        raise ValueError("batch/bucket-buffer/shard-size không hợp lệ")
    if args.work_root is None:
        output = Path(args.output)
        args.work_root = str(output.parent / f".parallel_{args.mode}_{output.stem}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    work_root = Path(args.work_root)
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
        "output": str(output_path),
        "manifest": str(manifest_path),
        "target_model_path": args.target_model_path,
        "max_length": int(args.max_length),
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
        "supervision_mode": args.supervision_mode,
        "num_shards": len(args.gpu_ids),
    }
    if args.resume and plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        immutable = ("mode", "gpu_ids", "input", "output", "manifest", "target_model_path", "max_length", "max_new_tokens", "num_shards")
        mismatches = [key for key in immutable if old.get(key) != plan.get(key)]
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
                worker_roots=worker_roots,
                output_path=output_path,
                manifest_path=manifest_path,
                gpu_ids=args.gpu_ids,
            )
        write_json(work_root / "status.json", {"status": "success", "payload": payload})
        print(f"[parallel] DONE mode={args.mode} gpu_ids={args.gpu_ids}", flush=True)
        return 0
    except BaseException as exc:
        write_json(work_root / "status.json", {"status": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
