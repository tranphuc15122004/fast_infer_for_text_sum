"""Profile batch size target capture trước khi bắt đầu cache.

Profiler này chạy một lần trên GPU được chỉ định, dùng sample thật gần cận
trên của từng bucket độ dài và padding tới cận đó. Nó chỉ đọc tokenized
dataset, không ghi feature shard. Kết quả là một schedule JSON để mọi cache
worker dùng cùng một quyết định batch theo độ dài.

Ví dụ:

``python scripts/mr_dflash/profile_cache_batches.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --tokenized-path /path/to/tokenized_full/train \
  --output /path/to/manifests/cache_batch_profile_full.json \
  --max-length 32768 --target-layer-ids 1 9 17 25 33 \
  --max-batch-size 8 --vram-limit-gb 170 \
  --attention-backend sdpa --device cuda:0 --local-files-only``
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from _common import write_json
from progress import ProgressReporter, install_exception_hook
from MR_DFlash.cache_batching import (
    CACHE_BATCH_PROFILE_SCHEMA_VERSION,
    CacheBatchSchedule,
    candidate_batch_sizes,
    choose_largest_safe_batch_size,
    make_bucket_specs,
)
from MR_DFlash.capture import HFTargetCapture
from MR_DFlash.tokenized_data import TokenizedDFlashDataset


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_bytes(device: torch.device) -> tuple[int, int]:
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    return allocated, reserved


def _representative_index(dataset: TokenizedDFlashDataset, lower: int, upper: int) -> int:
    candidates = [
        (index, int(length))
        for index, length in enumerate(dataset.lengths)
        if lower <= int(length) <= upper
    ]
    if not candidates:
        candidates = [
            (index, int(length))
            for index, length in enumerate(dataset.lengths)
            if 1 <= int(length) <= upper
        ]
    if not candidates:
        raise RuntimeError(f"dataset không có sample hợp lệ cho bucket {lower}-{upper}")
    # Ưu tiên sample dài nhất trong bucket: profile gần workload tệ nhất.
    return max(candidates, key=lambda item: item[1])[0]


def _make_probe_batch(
    sample: dict[str, Any],
    *,
    batch_size: int,
    probe_length: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(sample["input_ids"], dtype=torch.long).flatten()
    if values.numel() < 1:
        raise ValueError(
            f"sample {sample.get('id')!r} không có token hợp lệ để profile"
        )
    # Một bucket có thể chưa có sample đúng khoảng độ dài (ví dụ dataset pilot
    # chưa sinh được mẫu 24--32K). Khi đó dùng prefix của sample gần nhất để
    # vẫn đo được cận memory của bucket, thay vì làm toàn bộ pipeline dừng.
    values = values[:probe_length]
    input_ids = torch.full(
        (int(batch_size), int(probe_length)),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (int(batch_size), int(probe_length)), dtype=torch.long, device=device
    )
    input_ids[:, : values.numel()] = values.to(device=device)
    attention_mask[:, : values.numel()] = 1
    return input_ids, attention_mask


def _profile_candidate(
    capturer: HFTargetCapture,
    sample: dict[str, Any],
    *,
    batch_size: int,
    probe_length: int,
    pad_token_id: int,
    device: torch.device,
    vram_limit_bytes: int,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    input_ids, attention_mask = _make_probe_batch(
        sample,
        batch_size=batch_size,
        probe_length=probe_length,
        pad_token_id=pad_token_id,
        device=device,
    )
    started = time.monotonic()
    try:
        capturer.capture_batch(input_ids, attention_mask)
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        allocated, reserved = _memory_bytes(device)
        status = "pass" if reserved <= int(vram_limit_bytes) else "over_limit"
        return {
            "batch_size": int(batch_size),
            "status": status,
            "elapsed_seconds": round(elapsed, 4),
            "peak_memory_allocated_bytes": allocated,
            "peak_memory_reserved_bytes": reserved,
            "peak_memory_allocated_gb": round(allocated / 1024**3, 3),
            "peak_memory_reserved_gb": round(reserved / 1024**3, 3),
            "vram_limit_bytes": int(vram_limit_bytes),
        }
    except Exception as exc:
        allocated, reserved = _memory_bytes(device)
        if _is_cuda_oom(exc):
            status = "oom"
        else:
            status = "error"
        result = {
            "batch_size": int(batch_size),
            "status": status,
            "error": repr(exc),
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "peak_memory_allocated_bytes": allocated,
            "peak_memory_reserved_bytes": reserved,
            "peak_memory_allocated_gb": round(allocated / 1024**3, 3),
            "peak_memory_reserved_gb": round(reserved / 1024**3, 3),
            "vram_limit_bytes": int(vram_limit_bytes),
        }
        if status == "error":
            raise RuntimeError(
                f"profile batch={batch_size} probe_length={probe_length} lỗi không phải OOM: {exc!r}"
            ) from exc
        return result
    finally:
        del input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()


def profile_cache_batches(
    *,
    target_model_path: str,
    tokenized_path: str,
    output: str,
    max_length: int,
    target_layer_ids: Sequence[int],
    max_batch_size: int = 8,
    requested_batch_sizes: Sequence[int] | None = None,
    bucket_step: int = 8192,
    vram_limit_gb: float | None = None,
    vram_headroom_fraction: float = 0.10,
    cache_dir: str = "./cache",
    trust_remote_code: bool = False,
    torch_dtype: str = "bfloat16",
    device: str = "cuda:0",
    local_files_only: bool | None = None,
    target_revision: str | None = None,
    attention_backend: str = "sdpa",
    warmup_runs: int = 1,
) -> dict[str, Any]:
    """Profile và ghi schedule; không tạo feature cache."""
    reporter = ProgressReporter(None)
    previous_hook = install_exception_hook(reporter)
    if not torch.cuda.is_available():
        raise RuntimeError("auto-batch profile cần CUDA khả dụng trên GPU profile")
    if max_length < 1 or bucket_step < 1:
        raise ValueError("max_length và bucket_step phải >= 1")
    if not 0.0 <= float(vram_headroom_fraction) < 1.0:
        raise ValueError("vram_headroom_fraction phải thuộc [0, 1)")
    if warmup_runs < 0:
        raise ValueError("warmup_runs không được âm")
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        raise ValueError("profile auto-batch phải dùng device CUDA")
    if device_obj.index is None:
        device_obj = torch.device("cuda:0")
    torch.cuda.set_device(device_obj)
    total_vram_bytes = int(torch.cuda.get_device_properties(device_obj).total_memory)
    if vram_limit_gb is None:
        vram_limit_bytes = int(total_vram_bytes * (1.0 - float(vram_headroom_fraction)))
    else:
        if float(vram_limit_gb) <= 0:
            raise ValueError("vram_limit_gb phải > 0")
        vram_limit_bytes = int(float(vram_limit_gb) * 1024**3)
    if vram_limit_bytes >= total_vram_bytes:
        raise ValueError(
            f"vram limit {vram_limit_bytes / 1024**3:.2f} GiB phải nhỏ hơn VRAM vật lý "
            f"{total_vram_bytes / 1024**3:.2f} GiB"
        )

    dataset = TokenizedDFlashDataset(
        tokenized_path,
        expected_target_model=target_model_path,
        expected_feature_layer_ids=[int(value) for value in target_layer_ids],
        expected_max_length=max_length,
    )
    if len(dataset) < 1:
        raise RuntimeError(f"tokenized dataset rỗng: {tokenized_path}")
    candidates = candidate_batch_sizes(max_batch_size, requested_batch_sizes)
    bucket_specs = make_bucket_specs(max_length, bucket_step)
    reporter.set_context(total_samples=len(bucket_specs))
    reporter.update(
        "starting",
        input=str(tokenized_path),
        output=str(output),
        completed_samples=reporter.completed_samples(),
    )
    print(
        f"[profile] gpu={device_obj.index} name={torch.cuda.get_device_name(device_obj)} "
        f"total_vram={total_vram_bytes / 1024**3:.2f}GiB "
        f"limit={vram_limit_bytes / 1024**3:.2f}GiB candidates={candidates}",
        flush=True,
    )
    capturer = HFTargetCapture(
        target_model_path,
        [int(value) for value in target_layer_ids],
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device=str(device_obj),
        local_files_only=local_files_only,
        target_revision=target_revision,
        attention_backend=attention_backend,
    )
    pad_token_id = getattr(capturer.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(capturer.tokenizer, "eos_token_id", 0) or 0
    try:
        if [int(value) for value in capturer.layer_ids] != [int(value) for value in target_layer_ids]:
            raise RuntimeError(
                "target layer ids sau khi load không khớp profile request: "
                f"{capturer.layer_ids} != {list(target_layer_ids)}"
            )
        bucket_payloads: list[dict[str, Any]] = []
        for bucket_index, bucket in enumerate(bucket_specs, 1):
            lower = int(bucket["min_length"])
            upper = int(bucket["max_length"])
            sample_index = _representative_index(dataset, lower, upper)
            sample = dataset[sample_index]
            print(
                f"[profile] bucket={lower}-{upper} sample={sample['id']} "
                f"sample_length={sample['length']} probe_length={upper}",
                flush=True,
            )
            for _ in range(int(warmup_runs)):
                warmup_ids, warmup_mask = _make_probe_batch(
                    sample,
                    batch_size=1,
                    probe_length=upper,
                    pad_token_id=int(pad_token_id),
                    device=device_obj,
                )
                capturer.capture_batch(warmup_ids, warmup_mask)
                torch.cuda.synchronize(device_obj)
                del warmup_ids, warmup_mask
                torch.cuda.empty_cache()
            results: list[dict[str, Any]] = []
            for batch_size in candidates:
                result = _profile_candidate(
                    capturer,
                    sample,
                    batch_size=batch_size,
                    probe_length=upper,
                    pad_token_id=int(pad_token_id),
                    device=device_obj,
                    vram_limit_bytes=vram_limit_bytes,
                )
                results.append(result)
                print(
                    f"[profile] bucket={lower}-{upper} batch={batch_size} "
                    f"status={result['status']} "
                    f"reserved={result['peak_memory_reserved_gb']:.2f}GiB",
                    flush=True,
                )
                # OOM thường đơn điệu theo batch. Dừng probe lớn hơn để tránh
                # làm CUDA context bất ổn; batch an toàn đã được ghi đầy đủ.
                if result["status"] in {"oom", "over_limit"}:
                    break
            selected = choose_largest_safe_batch_size(
                results, vram_limit_bytes=vram_limit_bytes
            )
            bucket_payloads.append(
                {
                    **bucket,
                    "probe_length": upper,
                    "representative_sample_id": str(sample["id"]),
                    "representative_sample_length": int(sample["length"]),
                    "selected_batch_size": int(selected),
                    "results": results,
                }
            )
            reporter.update(
                "sample_done",
                completed_samples=int(bucket_index),
                sample_id=str(sample["id"]),
                bucket=f"{lower}-{upper}",
            )
            print(
                f"[profile] selected bucket={lower}-{upper}: batch_size={selected}",
                flush=True,
            )
        payload: dict[str, Any] = {
            "schema_version": CACHE_BATCH_PROFILE_SCHEMA_VERSION,
            "created_at": _now(),
            "target_model_path": str(target_model_path),
            "target_revision": target_revision,
            "feature_layer_ids": [int(value) for value in capturer.layer_ids],
            "max_length": int(max_length),
            "requested_torch_dtype": str(torch_dtype),
            "attention_backend": str(attention_backend),
            "device": str(device_obj),
            "gpu_name": torch.cuda.get_device_name(device_obj),
            "gpu_id": int(device_obj.index),
            "vram_total_bytes": total_vram_bytes,
            "vram_total_gb": round(total_vram_bytes / 1024**3, 3),
            "vram_limit_bytes": vram_limit_bytes,
            "vram_limit_gb": round(vram_limit_bytes / 1024**3, 3),
            "vram_headroom_fraction": float(vram_headroom_fraction),
            "candidate_batch_sizes": candidates,
            "bucket_step": int(bucket_step),
            "tokenized_path": str(tokenized_path),
            "buckets": bucket_payloads,
        }
        # Parse trước khi publish để không bao giờ ghi một profile không dùng
        # được bởi cache worker.
        CacheBatchSchedule.from_payload(payload, profile_path=str(output))
        write_json(output, payload)
        print(f"[profile] DONE output={output}", flush=True)
        reporter.update("done", completed_samples=len(bucket_specs), buckets=len(bucket_specs))
        sys.excepthook = previous_hook
        return payload
    finally:
        capturer.close()
        del capturer
        gc.collect()
        torch.cuda.empty_cache()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile auto-batch cho target feature cache")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--tokenized-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", required=True)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--candidate-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--bucket-step", type=int, default=8192)
    parser.add_argument("--vram-limit-gb", type=float, default=None)
    parser.add_argument("--vram-headroom-fraction", type=float, default=0.10)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    profile_cache_batches(
        target_model_path=args.target_model_path,
        tokenized_path=args.tokenized_path,
        output=args.output,
        max_length=args.max_length,
        target_layer_ids=args.target_layer_ids,
        max_batch_size=args.max_batch_size,
        requested_batch_sizes=args.candidate_batch_sizes,
        bucket_step=args.bucket_step,
        vram_limit_gb=args.vram_limit_gb,
        vram_headroom_fraction=args.vram_headroom_fraction,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        device=args.device,
        local_files_only=args.local_files_only,
        target_revision=args.target_revision,
        attention_backend=args.attention_backend,
        warmup_runs=args.warmup_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
