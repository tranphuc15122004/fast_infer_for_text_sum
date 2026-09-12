"""Cache target hidden states theo shard để tái sử dụng cho nhiều lần train.

Input chính là tokenized shard đã tạo từ canonical regenerated JSONL (đã chứa
assistant response do target model sinh). Có thể truyền thêm JSONL qua
``--data-path`` để giữ provenance; script không generate lại response. Khi có
``--tokenized-path``, script không render/tokenize lại JSONL mà chạy target
teacher-forcing và lưu hidden states ở toàn bộ offset hợp lệ của mỗi sample.

Ví dụ trên server B200:

``python scripts/mr_dflash/cache_target_features.py \
  --target-model-path Qwen/Qwen3-4B \
  --data-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated/train.jsonl \
  --tokenized-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/tokenized/train \
  --output-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/target_features_qwen3_4b_8k/train \
  --max-length 8192 --batch-size 2 --shard-size 64 \
  --attention-backend sdpa --io-threads 2 --io-queue-size 4 --resume \
  --local-files-only``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from _common import append_jsonl_durable, read_jsonl
from MR_DFlash.cache_batching import CacheBatchSchedule
from MR_DFlash.cache_throughput import CacheThroughputBucket, CacheThroughputProfile
from MR_DFlash.capture import HFTargetCapture
from progress import ProgressReporter, estimate_fields, install_exception_hook


def _is_cuda_oom(exc: BaseException) -> bool:
    """OOM không thể coi là lỗi của một sample rồi tiếp tục cache."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _effective_attention_backend(cache_backend: str, attention_backend: str) -> str:
    """Map the legacy HF default to a backend registered by SpecForge SGLang."""
    backend = str(cache_backend).lower()
    attention = str(attention_backend).lower()
    if backend != "specforge_sglang":
        return attention
    if attention in {"auto", "sdpa"}:
        # sdpa is a Transformers name. The vendored SGLang registry uses
        # flashinfer for the B200 path and does not register sdpa.
        return "flashinfer"
    if attention not in {"flashinfer", "triton", "torch_native"}:
        raise ValueError(
            "SpecForge SGLang chỉ hỗ trợ attention backend "
            "flashinfer, triton hoặc torch_native; "
            f"got {attention_backend!r}"
        )
    return attention


def _default_specforge_profile(max_length: int) -> CacheThroughputProfile:
    """Aggressive B200 profile cho dữ liệu chủ yếu <=8K.

    Token budget giữ đủ headroom để batch 64×4K, 32×8K và 16×16K chạy
    chung một static pool; bucket 32K giảm còn 4 sample vì KV/activation tăng
    nhanh. Profile chỉ là performance hint, hidden-state contract không đổi.
    """
    requested = int(max_length)
    if requested < 1:
        raise ValueError("max_length phải >= 1")
    candidates = (
        (1, 4096, 64, 262144),
        (4097, 8192, 32, 262144),
        (8193, 16384, 16, 262144),
        (16385, 32768, 4, 131072),
    )
    buckets: list[CacheThroughputBucket] = []
    for minimum, maximum, batch, budget in candidates:
        if minimum > requested:
            break
        buckets.append(
            CacheThroughputBucket(
                min_length=minimum,
                max_length=min(maximum, requested),
                batch_size=batch,
                token_budget=budget,
            )
        )
    return CacheThroughputProfile(buckets=tuple(buckets))


def _write_retry_hint(
    output_path: str | Path,
    *,
    error: BaseException,
    batch_size: int,
    max_total_tokens: Optional[int],
    throughput_profile: Optional[CacheThroughputProfile],
) -> None:
    """Ghi hướng dẫn retry sau OOM mà không đụng các shard đã durable."""
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    reduced_profile = None
    reduced_profile_path = None
    if throughput_profile is not None:
        reduced_profile = throughput_profile.to_payload()
        reduced_profile["buckets"] = [
            {
                **bucket,
                "batch_size": max(1, int(bucket["batch_size"]) // 2),
                "token_budget": max(1, int(bucket["token_budget"]) // 2),
            }
            for bucket in reduced_profile["buckets"]
        ]
        reduced_profile_path = root / "retry_throughput_profile.json"
        temporary_profile = reduced_profile_path.with_name(
            f".{reduced_profile_path.name}.tmp"
        )
        temporary_profile.write_text(
            json.dumps(reduced_profile, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary_profile.replace(reduced_profile_path)
    payload = {
        "schema_version": "mr_dflash_cache_retry_hint_v1",
        "reason": "cuda_oom_or_worker_killed",
        "error": repr(error),
        "suggested_batch_size": max(1, int(batch_size) // 2),
        "suggested_max_total_tokens": (
            None if max_total_tokens is None else max(1, int(max_total_tokens) // 2)
        ),
        "suggested_throughput_profile": reduced_profile,
        "suggested_throughput_profile_path": (
            None if reduced_profile_path is None else str(reduced_profile_path)
        ),
        "resume": True,
        "note": "Giữ shard đã ghi; chạy lại với budget thấp hơn và --resume.",
    }
    temporary = root / ".retry_hint.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(root / "retry_hint.json")


def _preflight_storage(
    output_path: str | Path,
    *,
    remaining_tokens: int,
    feature_width: int,
    torch_dtype: str,
) -> dict[str, int]:
    """Ước lượng hidden-state footprint trước khi chạy forward dài."""
    bytes_per_value = {
        "float32": 4,
        "bfloat16": 2,
        "float16": 2,
    }.get(str(torch_dtype), 2)
    estimated = int(max(0, remaining_tokens)) * int(feature_width) * bytes_per_value
    # Input/loss/shard metadata và serialization tạm thời cần thêm headroom.
    required = int(estimated * 1.15) + 256 * 1024 * 1024
    usage = shutil.disk_usage(Path(output_path))
    available = int(usage.free)
    if required > available:
        raise RuntimeError(
            "storage không đủ cho feature cache: "
            f"cần khoảng {required / 1024**4:.2f} TiB, "
            f"còn {available / 1024**4:.2f} TiB tại {output_path}; "
            "chọn staging/local NVMe hoặc giảm dataset"
        )
    return {
        "estimated_feature_bytes": int(estimated),
        "estimated_required_bytes": int(required),
        "available_bytes": available,
    }


def _build_capturer(
    *,
    target_model_path: str,
    layer_ids: List[int],
    cache_backend: str,
    cache_dir: str,
    trust_remote_code: bool,
    torch_dtype: str,
    device: str,
    local_files_only: Optional[bool],
    target_revision: Optional[str],
    attention_backend: str,
    max_length: int,
    batch_size: int,
    cache_concurrency: Optional[int],
    cache_max_total_tokens: Optional[int],
    cache_memory_fraction: float,
    throughput_profile: Optional[CacheThroughputProfile],
) -> Any:
    """Build HF fallback hoặc target SGLang capture độc lập trên một GPU."""
    backend = str(cache_backend).lower()
    if backend in {"hf", "hf_backbone", "legacy"}:
        return HFTargetCapture(
            target_model_path,
            layer_ids,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device=device,
            local_files_only=local_files_only,
            target_revision=target_revision,
            attention_backend=attention_backend,
        )
    if backend != "specforge_sglang":
        raise ValueError(
            "cache_backend phải là 'hf' hoặc 'specforge_sglang', "
            f"got {cache_backend!r}"
        )
    from specforge_capture import SpecForgeTargetCapture

    profile_batch = (
        throughput_profile.max_batch_size if throughput_profile is not None else batch_size
    )
    concurrency = int(cache_concurrency or profile_batch)
    if concurrency < 1:
        raise ValueError("cache_concurrency phải >= 1")
    profile_budget = (
        max(int(bucket.token_budget) for bucket in throughput_profile.buckets)
        if throughput_profile is not None
        else int(max_length) * concurrency
    )
    total_tokens = int(cache_max_total_tokens or profile_budget)
    if total_tokens < 1:
        raise ValueError("cache_max_total_tokens phải >= 1")
    return SpecForgeTargetCapture.from_pretrained(
        target_model_path,
        layer_ids,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attention_backend=attention_backend,
        mem_fraction_static=float(cache_memory_fraction),
        context_length=int(max_length),
        max_running_requests=concurrency,
        max_total_tokens=total_tokens,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        target_revision=target_revision,
    )


def _cache_progress_kwargs(workload: Dict[str, Any]) -> dict[str, Any]:
    """Cấu hình thanh tiến trình theo sample hợp lệ của worker hiện tại."""
    total = max(0, int(workload.get("valid_samples", 0)))
    initial = min(total, max(0, int(workload.get("existing_samples", 0))))
    return {
        "total": total,
        "initial": initial,
        "desc": "Cache target features",
        "unit": "sample",
        "dynamic_ncols": True,
        "mininterval": 1.0,
    }


def _pad_samples(
    samples: List[Dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(sample["input_ids"]) for sample in samples)
    input_ids = torch.full(
        (len(samples), max_len), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(samples), max_len), dtype=torch.long, device=device
    )
    for row, sample in enumerate(samples):
        length = len(sample["input_ids"])
        input_ids[row, :length] = torch.as_tensor(
            sample["input_ids"], dtype=torch.long, device=device
        )
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def _workload_signature(
    data_path: Optional[str],
    *,
    tokenized_path: Optional[str],
    max_length: int,
    supervision_mode: str,
    shard_index: int,
    num_shards: int,
    num_samples: Optional[int],
) -> dict[str, Any]:
    """Signature của preflight token-count để resume không dùng nhầm estimate."""
    sources: dict[str, Any] = {}
    for name, value in (("data_path", data_path), ("tokenized_path", tokenized_path)):
        if value is None:
            sources[name] = None
            continue
        source = Path(value)
        stat = source.stat()
        sources[name] = {
            "path": str(source.resolve()),
            "size": int(stat.st_size) if source.is_file() else None,
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return {
        "sources": sources,
        "max_length": int(max_length),
        "supervision_mode": str(supervision_mode),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "num_samples": None if num_samples is None else int(num_samples),
    }


def _estimate_workload(
    *,
    data_path: Optional[str],
    tokenized_path: Optional[str] = None,
    tokenizer: Any,
    max_length: int,
    supervision_mode: str,
    shard_index: int,
    num_shards: int,
    num_samples: Optional[int],
    existing_ids: set[str],
    report_path: Path,
    reporter: ProgressReporter,
) -> dict[str, int]:
    """Đếm chính xác sample/token mà cache worker sẽ xử lý.

    Đây là một pass tokenizer nhẹ, không chạy target model. Kết quả được lưu
    lại để các lần ``--resume`` sau không phải scan/tokenize lần nữa. Main
    capture vẫn render sample lần nữa để không giữ toàn bộ dataset trong RAM.
    """
    signature = _workload_signature(
        data_path,
        tokenized_path=tokenized_path,
        max_length=max_length,
        supervision_mode=supervision_mode,
        shard_index=shard_index,
        num_shards=num_shards,
        num_samples=num_samples,
    )
    if report_path.is_file():
        try:
            cached = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and cached.get("signature") == signature:
            workload = cached.get("workload")
            lengths_by_id = cached.get("valid_sample_lengths")
            if isinstance(workload, dict) and isinstance(lengths_by_id, dict):
                valid_lengths = {
                    str(sample_id): int(length)
                    for sample_id, length in lengths_by_id.items()
                }
                existing_lengths = [
                    length for sample_id, length in valid_lengths.items()
                    if sample_id in existing_ids
                ]
                return {
                    "input_rows": int(workload.get("input_rows", 0)),
                    "valid_samples": int(workload.get("valid_samples", 0)),
                    "valid_tokens": int(workload.get("valid_tokens", 0)),
                    "existing_samples": len(existing_lengths),
                    "existing_tokens": sum(existing_lengths),
                }

    from MR_DFlash.data import build_sample

    def iter_rows():
        if tokenized_path is not None:
            from MR_DFlash.tokenized_data import TokenizedDFlashDataset

            dataset = TokenizedDFlashDataset(tokenized_path)
            for index in range(len(dataset)):
                item = dataset[index]
                yield {
                    "id": str(item["id"]),
                    "input_ids": item["input_ids"],
                    "loss_mask": item["loss_mask"],
                }
            return
        if data_path is None:
            raise ValueError("cache cần --data-path hoặc --tokenized-path")
        yield from read_jsonl(data_path)

    result = {
        "input_rows": 0,
        "valid_samples": 0,
        "valid_tokens": 0,
        "existing_samples": 0,
        "existing_tokens": 0,
    }
    valid_sample_lengths: dict[str, int] = {}
    reporter.update("estimating_workload", **result)
    for row_index, row in enumerate(iter_rows()):
        if row_index % num_shards != shard_index:
            continue
        result["input_rows"] += 1
        sample = (
            row
            if tokenized_path is not None
            else build_sample(
                row,
                tokenizer,
                max_length,
                supervision_mode=supervision_mode,
            )
        )
        if sample is None:
            continue
        length = len(sample["input_ids"])
        result["valid_samples"] += 1
        result["valid_tokens"] += length
        valid_sample_lengths[str(sample["id"])] = length
        if str(sample["id"]) in existing_ids:
            result["existing_samples"] += 1
            result["existing_tokens"] += length
        if num_samples is not None and result["valid_samples"] >= int(num_samples):
            break
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "mr_dflash_cache_workload_v1",
                "signature": signature,
                "workload": result,
                "valid_sample_lengths": valid_sample_lengths,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return result


def cache_dataset(
    *,
    target_model_path: str,
    data_path: Optional[str] = None,
    tokenized_path: Optional[str] = None,
    output_path: str,
    max_length: int,
    layer_ids: Optional[List[int]] = None,
    num_samples: Optional[int] = None,
    batch_size: int = 1,
    bucket_buffer_size: Optional[int] = None,
    shard_size: int = 64,
    cache_dir: str = "./cache",
    trust_remote_code: bool = False,
    torch_dtype: str = "bfloat16",
    device: str = "auto",
    local_files_only: Optional[bool] = None,
    supervision_mode: str = "last_assistant",
    target_revision: Optional[str] = None,
    attention_backend: str = "sdpa",
    io_threads: int = 2,
    io_queue_size: int = 4,
    resume: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    skipped_report: Optional[str] = None,
    progress_path: Optional[str] = None,
    batch_profile: Optional[str] = None,
    throughput_profile: Optional[str] = None,
    cache_backend: str = "hf_backbone",
    cache_concurrency: Optional[int] = None,
    cache_max_total_tokens: Optional[int] = None,
    cache_memory_fraction: float = 0.99,
) -> Dict[str, int]:
    """Chạy target capture theo batch và ghi cache sharded resumable."""
    if data_path is None and tokenized_path is None:
        raise ValueError("cache cần data_path hoặc tokenized_path")
    if str(cache_backend).lower() == "specforge_sglang" and tokenized_path is None:
        raise ValueError(
            "cache backend specforge_sglang cần --tokenized-path để giữ đúng "
            "tokenization và tránh tokenize lại trong worker"
        )
    attention_backend = _effective_attention_backend(cache_backend, attention_backend)
    if batch_size < 1:
        raise ValueError("batch_size phải >= 1")
    if io_threads < 0 or io_queue_size < 0:
        raise ValueError("io_threads và io_queue_size không được âm")
    if not 0.0 < float(cache_memory_fraction) <= 1.0:
        raise ValueError("cache_memory_fraction phải thuộc (0, 1]")
    if cache_concurrency is not None and int(cache_concurrency) < 1:
        raise ValueError("cache_concurrency phải >= 1")
    if cache_max_total_tokens is not None and int(cache_max_total_tokens) < 1:
        raise ValueError("cache_max_total_tokens phải >= 1")
    if batch_profile and throughput_profile:
        raise ValueError("chỉ được dùng một trong batch_profile hoặc throughput_profile")
    batch_schedule = CacheBatchSchedule.from_path(batch_profile) if batch_profile else None
    throughput_schedule = (
        CacheThroughputProfile.from_path(throughput_profile)
        if throughput_profile
        else None
    )
    throughput_profile_label = throughput_profile
    if str(cache_backend).lower() == "specforge_sglang" and throughput_schedule is None:
        throughput_schedule = _default_specforge_profile(max_length)
        throughput_profile_label = "builtin:specforge-aggressive-v1"
    if throughput_schedule is not None:
        # Trong throughput mode, profile là giới hạn batch chính thức. Giá trị
        # ``batch_size`` cũ chỉ điều khiển HF mode khi không có profile.
        throughput_schedule.validate(
            max_length=max_length,
            batch_limit=throughput_schedule.max_batch_size,
        )
    profile_max_batch = max(
        batch_size,
        batch_schedule.max_batch_size if batch_schedule is not None else 0,
        throughput_schedule.max_batch_size if throughput_schedule is not None else 0,
    )
    if bucket_buffer_size is None:
        bucket_buffer_size = max(
            batch_size,
            profile_max_batch
            if (batch_schedule or throughput_schedule)
            else batch_size * 8,
        )
    if bucket_buffer_size < batch_size:
        raise ValueError("bucket_buffer_size phải >= batch_size")
    if batch_schedule is not None and bucket_buffer_size < batch_schedule.max_batch_size:
        # Buffer là RAM phía producer, không phải GPU batch. Tự mở rộng nó
        # để schedule auto-batch không bị giảm hiệu quả chỉ vì người dùng
        # còn giữ giá trị buffer cố định cũ.
        bucket_buffer_size = batch_schedule.max_batch_size
    if (
        throughput_schedule is not None
        and bucket_buffer_size < throughput_schedule.max_batch_size
    ):
        # Giữ đủ candidate trong buffer để scheduler token-budget có thể
        # chọn batch lớn nhất an toàn theo padded length thực tế.
        bucket_buffer_size = throughput_schedule.max_batch_size
    if num_shards < 1 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard-index phải nằm trong [0, num-shards)")

    from MR_DFlash.data import build_sample
    from MR_DFlash.offline_features import ShardedFeatureWriter

    reporter = ProgressReporter(progress_path)
    previous_hook = install_exception_hook(reporter)
    reporter.update(
        "starting",
        input=str(data_path),
        tokenized_input=str(tokenized_path) if tokenized_path else None,
        output=str(output_path),
        shard_index=int(shard_index),
        num_shards=int(num_shards),
        completed_samples=0,
    )

    reporter.update(
        "loading_model",
        target_model_path=str(target_model_path),
        cache_backend=str(cache_backend),
        attention_backend=str(attention_backend),
    )
    try:
        capturer = _build_capturer(
            target_model_path=target_model_path,
            layer_ids=layer_ids or [],
            cache_backend=cache_backend,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device=device,
            local_files_only=local_files_only,
            target_revision=target_revision,
            attention_backend=attention_backend,
            max_length=max_length,
            batch_size=batch_size,
            cache_concurrency=cache_concurrency,
            cache_max_total_tokens=cache_max_total_tokens,
            cache_memory_fraction=cache_memory_fraction,
            throughput_profile=throughput_schedule,
        )
    except Exception as exc:
        # Static-pool allocation happens before the writer exists. Keep the
        # same retry contract as capture-time OOM for this failure point.
        if _is_cuda_oom(exc):
            _write_retry_hint(
                output_path,
                error=exc,
                batch_size=max(batch_size, int(cache_concurrency or batch_size)),
                max_total_tokens=cache_max_total_tokens,
                throughput_profile=throughput_schedule,
            )
            reporter.update(
                "oom",
                error=repr(exc),
                retry_hint=str(Path(output_path) / "retry_hint.json"),
            )
        raise
    reporter.update(
        "model_ready",
        target_model_path=str(target_model_path),
        device=str(capturer.device),
        dtype=torch_dtype,
        feature_layer_ids=[int(value) for value in capturer.layer_ids],
        attention_backend=str(attention_backend),
        cache_backend=str(cache_backend),
        cache_concurrency=(
            int(cache_concurrency)
            if cache_concurrency is not None
            else int(throughput_schedule.max_batch_size)
            if throughput_schedule is not None
            else int(batch_size)
        ),
        cache_max_total_tokens=(
            None if cache_max_total_tokens is None else int(cache_max_total_tokens)
        ),
        cache_memory_fraction=float(cache_memory_fraction),
        io_threads=int(io_threads),
        io_queue_size=int(io_queue_size),
    )
    if batch_schedule is not None:
        batch_schedule.validate(
            target_model_path=target_model_path,
            feature_layer_ids=capturer.layer_ids,
            max_length=max_length,
            requested_torch_dtype=torch_dtype,
            attention_backend=attention_backend,
            target_revision=target_revision,
        )
        reporter.update(
            "batch_profile_ready",
            batch_profile=str(batch_profile),
            batch_schedule=[
                {
                    "min_length": int(bucket["min_length"]),
                    "max_length": int(bucket["max_length"]),
                    "batch_size": int(bucket["selected_batch_size"]),
                }
                for bucket in batch_schedule.buckets
            ],
        )
    if throughput_schedule is not None:
        reporter.update(
            "throughput_profile_ready",
            throughput_profile=str(throughput_profile_label),
            throughput_profile_sha256=throughput_schedule.profile_sha256,
            throughput_buckets=[
                {
                    "min_length": int(bucket.min_length),
                    "max_length": int(bucket.max_length),
                    "batch_size": int(bucket.batch_size),
                    "token_budget": int(bucket.token_budget),
                }
                for bucket in throughput_schedule.buckets
            ],
        )
    pad_token_id = getattr(capturer.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(capturer.tokenizer, "eos_token_id", 0) or 0
    writer = ShardedFeatureWriter(
        output_path,
        shard_size=shard_size,
        target_model_path=target_model_path,
        feature_layer_ids=capturer.layer_ids,
        hidden_size=int(capturer.model.config.hidden_size),
        feature_width=int(capturer.context_feature_dim),
        max_length=max_length,
        requested_torch_dtype=torch_dtype,
        source_data_path=data_path or tokenized_path,
        source_tokenized_path=tokenized_path,
        target_revision=target_revision,
        resume=resume,
        io_threads=io_threads,
        io_queue_size=io_queue_size,
        capture_backend=(
            "specforge_sglang" if str(cache_backend).lower() == "specforge_sglang"
            else "hf_backbone"
        ),
        attention_backend=attention_backend,
        cache_batch_profile=throughput_profile_label or batch_profile,
        cache_batch_profile_sha256=(
            throughput_schedule.profile_sha256
            if throughput_schedule is not None
            else batch_schedule.profile_sha256
            if batch_schedule is not None
            else None
        ),
    )
    stats = {
        "seen": 0,
        "valid": 0,
        "captured": 0,
        "resumed": len(writer.existing_ids),
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "capture_errors": 0,
    }
    skipped_report_path = Path(skipped_report) if skipped_report else Path(output_path) / "skipped.jsonl"
    workload = _estimate_workload(
        data_path=data_path,
        tokenized_path=tokenized_path,
        tokenizer=capturer.tokenizer,
        max_length=max_length,
        supervision_mode=supervision_mode,
        shard_index=shard_index,
        num_shards=num_shards,
        num_samples=num_samples,
        existing_ids=writer.existing_ids,
        report_path=Path(output_path) / "workload.json",
        reporter=reporter,
    )
    total_samples = int(workload["valid_samples"])
    total_tokens = int(workload["valid_tokens"])
    completed_tokens = int(workload["existing_tokens"])
    captured_tokens = 0
    capture_started_at: Optional[float] = None

    def estimate_progress() -> dict[str, Any]:
        elapsed = 0.0 if capture_started_at is None else time.monotonic() - capture_started_at
        return estimate_fields(
            total_samples=total_samples,
            total_tokens=total_tokens,
            completed_samples=int(writer.total_samples),
            completed_tokens=completed_tokens,
            run_tokens=captured_tokens,
            active_elapsed_seconds=elapsed,
        )

    reporter.update(
        "workload_ready",
        input_rows=int(workload["input_rows"]),
        valid_samples=total_samples,
        valid_tokens=total_tokens,
        **estimate_progress(),
    )
    storage_estimate = _preflight_storage(
        output_path,
        remaining_tokens=max(0, total_tokens - completed_tokens),
        feature_width=int(capturer.context_feature_dim),
        torch_dtype=torch_dtype,
    )
    reporter.update("storage_ready", **storage_estimate)

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm có trong requirements server
        cache_bar = None
    else:
        cache_bar = tqdm(**_cache_progress_kwargs(workload))
    last_bar_captured = int(stats["captured"])

    def refresh_cache_bar(sample_id: Optional[str] = None) -> None:
        """Cập nhật theo sample đã ghi, không cập nhật theo từng token."""
        nonlocal last_bar_captured
        if cache_bar is None:
            return
        captured_now = int(stats["captured"])
        delta = captured_now - last_bar_captured
        if delta > 0:
            cache_bar.update(delta)
            last_bar_captured = captured_now
        progress = estimate_progress()
        rate = progress.get("throughput_tokens_per_second")
        rate_text = "-" if rate is None else f"{float(rate):.1f} tok/s"
        cache_bar.set_postfix(
            sample=sample_id or "-",
            tokens=f"{int(progress['completed_tokens'])}/{total_tokens}",
            rate=rate_text,
            eta=progress.get("eta_human", "unknown"),
        )
        cache_bar.refresh()

    def record_skip(sample_id: str, *, kind: str, error: str) -> None:
        append_row = {
            "id": str(sample_id),
            "kind": kind,
            "error": error,
        }
        append_jsonl_durable(skipped_report_path, [append_row])
        reporter.update(
            "sample_skipped",
            sample_id=str(sample_id),
            skip_kind=kind,
            error=str(error),
            completed_samples=int(writer.total_samples),
        )

    def handle_oom(error: BaseException, batch_size_for_retry: int) -> None:
        """Flush durable CPU shards before propagating OOM to the launcher."""
        _write_retry_hint(
            output_path,
            error=error,
            batch_size=batch_size_for_retry,
            max_total_tokens=cache_max_total_tokens,
            throughput_profile=throughput_schedule,
        )
        try:
            writer.close(stats={**stats, "status": "oom", "error": repr(error)})
        except Exception:
            writer.abort()
        try:
            capturer.close()
        except Exception:
            pass

    buffer: List[Dict[str, Any]] = []

    def process(batch: List[Dict[str, Any]]) -> None:
        nonlocal capture_started_at, completed_tokens, captured_tokens
        if not batch:
            return
        if capture_started_at is None:
            capture_started_at = time.monotonic()
        reporter.update(
            "capturing",
            batch_size=len(batch),
            batch_sample_ids=[str(sample["id"]) for sample in batch],
            **estimate_progress(),
        )
        try:
            input_ids, attention_mask = _pad_samples(
                batch,
                pad_token_id=int(pad_token_id),
                device=capturer.device,
            )
            captured = capturer.capture_batch(input_ids, attention_mask)
            if len(captured) != len(batch):
                raise RuntimeError(
                    f"capture trả {len(captured)} sample thay vì {len(batch)}"
                )
        except Exception as exc:
            # Một sample lỗi không làm mất cả shard/batch. Fallback tuần tự
            # cũng giúp chẩn đoán rõ sample id trên model/driver bất ổn.
            if _is_cuda_oom(exc):
                handle_oom(exc, len(batch))
                raise
            print(f"[cache] batch capture lỗi, fallback từng mẫu: {exc!r}")
            for sample in batch:
                try:
                    single_ids, single_mask = _pad_samples(
                        [sample],
                        pad_token_id=int(pad_token_id),
                        device=capturer.device,
                    )
                    feature = capturer.capture_batch(single_ids, single_mask)[0]
                    added = writer.add({**sample, "hidden_states": feature})
                    if added:
                        stats["captured"] += 1
                        length = len(sample["input_ids"])
                        completed_tokens += length
                        captured_tokens += length
                    else:
                        stats["skipped_existing"] += 1
                except Exception as sample_exc:
                    if _is_cuda_oom(sample_exc):
                        handle_oom(sample_exc, 1)
                        raise
                    print(f"[cache] skip {sample['id']!r}: {sample_exc!r}")
                    record_skip(sample["id"], kind="capture_error", error=repr(sample_exc))
                    stats["capture_errors"] += 1
            reporter.update(
                "batch_done",
                batch_size=len(batch),
                last_sample_id=str(batch[-1]["id"]),
                captured=int(stats["captured"]),
                durable_samples=len(writer.sample_ids),
                **estimate_progress(),
            )
            refresh_cache_bar(str(batch[-1]["id"]))
            return
        for sample, feature in zip(batch, captured):
            try:
                added = writer.add({**sample, "hidden_states": feature})
                if added:
                    stats["captured"] += 1
                    length = len(sample["input_ids"])
                    completed_tokens += length
                    captured_tokens += length
                else:
                    stats["skipped_existing"] += 1
            except Exception as exc:
                if _is_cuda_oom(exc):
                    raise
                print(f"[cache] skip {sample['id']!r}: {exc!r}")
                record_skip(sample["id"], kind="write_error", error=repr(exc))
                stats["capture_errors"] += 1
        reporter.update(
            "batch_done",
            batch_size=len(batch),
            last_sample_id=str(batch[-1]["id"]),
            captured=int(stats["captured"]),
            durable_samples=len(writer.sample_ids),
            **estimate_progress(),
        )
        refresh_cache_bar(str(batch[-1]["id"]))

    if tokenized_path is not None:
        from MR_DFlash.tokenized_data import TokenizedDFlashDataset

        tokenized_dataset = TokenizedDFlashDataset(tokenized_path)

        def iter_cache_rows():
            for index in range(len(tokenized_dataset)):
                item = tokenized_dataset[index]
                yield {
                    "id": str(item["id"]),
                    "input_ids": item["input_ids"],
                    "loss_mask": item["loss_mask"],
                }
    else:
        if data_path is None:  # guarded above; keep the error local and clear
            raise ValueError("cache cần --data-path hoặc --tokenized-path")

        def iter_cache_rows():
            yield from read_jsonl(data_path)

    rows = iter_cache_rows()

    def next_batch_size(sample_length: Optional[int] = None) -> int:
        if throughput_schedule is not None:
            if sample_length is None:
                raise ValueError("throughput profile cần sample_length để chọn bucket")
            requested = throughput_schedule.batch_for_length(sample_length)
            # Buffer đã được sort giảm dần theo length trước khi gọi helper,
            # vì vậy length của sample đầu là padded length thật của batch.
            selected = throughput_schedule.cap_batch_size(
                padded_length=sample_length,
                requested_batch_size=requested,
            )
            if cache_max_total_tokens is not None:
                token_cap = int(cache_max_total_tokens) // int(sample_length)
                if token_cap < 1:
                    raise ValueError(
                        "cache-max-total-tokens nhỏ hơn một sample: "
                        f"{cache_max_total_tokens} < {sample_length}"
                    )
                selected = min(selected, token_cap)
        elif batch_schedule is not None:
            if sample_length is None:
                raise ValueError("auto-batch cần sample_length để chọn bucket")
            selected = batch_schedule.batch_size_for_length(sample_length)
        else:
            selected = batch_size
        if num_samples is None:
            return selected
        remaining = int(num_samples) - writer.total_samples
        return max(0, min(selected, remaining))

    for row_index, row in enumerate(rows):
        stats["seen"] += 1
        if row_index % num_shards != shard_index:
            continue
        if num_samples is not None and writer.total_samples >= int(num_samples):
            break
        sample_id = str(row.get("id", row_index))
        if stats["seen"] == 1 or stats["seen"] % max(1, bucket_buffer_size) == 0:
            reporter.update(
                "reading_sample",
                row_index=int(row_index),
                sample_id=sample_id,
                completed_samples=int(writer.total_samples),
                captured=int(stats["captured"]),
            )
        if sample_id in writer.existing_ids:
            stats["skipped_existing"] += 1
            continue
        sample = (
            row
            if tokenized_path is not None
            else build_sample(
                row,
                capturer.tokenizer,
                max_length,
                supervision_mode=supervision_mode,
            )
        )
        if sample is None:
            record_skip(sample_id, kind="invalid", error="sample không render được hoặc thiếu supervised tokens")
            stats["skipped_invalid"] += 1
            continue
        stats["valid"] += 1
        buffer.append(sample)
        if len(buffer) >= bucket_buffer_size:
            # Stable sort giữ kết quả deterministic trong cùng độ dài; batch
            # gần độ dài nhau để giảm padding trên target forward.
            buffer.sort(key=lambda item: -len(item["input_ids"]))
            while buffer:
                current_batch_size = next_batch_size(len(buffer[0]["input_ids"]))
                if current_batch_size == 0:
                    buffer.clear()
                    break
                # Với profile hợp lệ, bucket_buffer_size đã >= batch lớn nhất.
                # Điều kiện này vẫn bảo vệ trường hợp schedule đổi giữa lúc
                # debug hoặc số sample còn lại quá ít.
                if len(buffer) < current_batch_size:
                    break
                process(buffer[:current_batch_size])
                del buffer[:current_batch_size]
                if num_samples is not None and writer.total_samples >= int(num_samples):
                    buffer.clear()
                    break
    buffer.sort(key=lambda item: -len(item["input_ids"]))
    while buffer and (num_samples is None or writer.total_samples < int(num_samples)):
        current_batch_size = next_batch_size(len(buffer[0]["input_ids"]))
        if current_batch_size == 0:
            break
        # Batch cuối có thể nhỏ hơn schedule khi shard kết thúc; đây là batch
        # an toàn vì nó chỉ giảm memory, không làm tăng padding length.
        current_batch_size = min(current_batch_size, len(buffer))
        process(buffer[:current_batch_size])
        del buffer[:current_batch_size]

    stats["captured_total"] = writer.total_samples
    stats["workload_input_rows"] = int(workload["input_rows"])
    stats["workload_valid_samples"] = total_samples
    stats["workload_valid_tokens"] = total_tokens
    stats["workload_existing_samples"] = int(workload["existing_samples"])
    stats["workload_existing_tokens"] = int(workload["existing_tokens"])
    stats.update({f"storage_{key}": int(value) for key, value in storage_estimate.items()})
    final_estimate = estimate_progress()
    stats["cache_elapsed_seconds"] = int(round(final_estimate["active_elapsed_seconds"]))
    stats["throughput_tokens_per_second"] = int(round(final_estimate["throughput_tokens_per_second"] or 0))
    writer.close(stats=stats)
    capturer.close()
    skipped_report_path.parent.mkdir(parents=True, exist_ok=True)
    if not skipped_report_path.exists():
        skipped_report_path.touch()
    print(
        f"[cache] xong: captured_now={stats['captured']} "
        f"total={writer.total_samples} shards={len(writer.shards)} "
        f"output={output_path}"
    )
    reporter.update(
        "done",
        durable_samples=len(writer.sample_ids),
        captured=int(stats["captured"]),
        skipped=int(stats["skipped_invalid"] + stats["capture_errors"]),
        seen=int(stats["seen"]),
        **estimate_progress(),
    )
    if cache_bar is not None:
        refresh_cache_bar()
        cache_bar.close()
    sys.excepthook = previous_hook
    return {key: int(value) for key, value in stats.items() if isinstance(value, int)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache target hidden states theo shard")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--tokenized-path",
        default=None,
        help="tokenized shard đã tạo trước; bỏ qua render/tokenize JSONL khi có",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bucket-buffer-size", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--target-revision", default=None)
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
        help=(
            "attention implementation cho backbone capture; sdpa dùng kernel "
            "PyTorch tối ưu và không bắt buộc flash-attn"
        ),
    )
    parser.add_argument(
        "--cache-backend",
        choices=["hf", "specforge_sglang"],
        default="hf",
        help="backend capture; specforge_sglang dùng offline SGLang của SpecForge",
    )
    parser.add_argument(
        "--cache-concurrency",
        type=int,
        default=None,
        help="max running requests của offline SGLang; mặc định lấy batch lớn nhất profile",
    )
    parser.add_argument(
        "--cache-max-total-tokens",
        type=int,
        default=None,
        help="trần token KV/static của offline SGLang; ưu tiên dùng gần đầy VRAM",
    )
    parser.add_argument(
        "--cache-memory-fraction",
        type=float,
        default=0.99,
        help="tỷ lệ VRAM static pool cho SpecForge SGLang (mặc định 0.99)",
    )
    parser.add_argument(
        "--io-threads",
        type=int,
        default=2,
        help="số thread ghi shard song song với target capture; 0 = ghi đồng bộ",
    )
    parser.add_argument(
        "--io-queue-size",
        type=int,
        default=4,
        help="số shard tối đa đang chờ ghi; giới hạn RAM của writer bất đồng bộ",
    )
    parser.add_argument(
        "--supervision-mode",
        choices=["all_assistant", "last_assistant"],
        default="last_assistant",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skipped-report", default=None)
    parser.add_argument("--progress-path", default=None)
    parser.add_argument(
        "--batch-profile",
        default=None,
        help="profile JSON do profile_cache_batches.py tạo; chọn batch theo length bucket",
    )
    parser.add_argument(
        "--throughput-profile",
        "--cache-throughput-profile",
        "--cache-profile",
        dest="throughput_profile",
        default=None,
        help=(
            "profile token-budget JSON; batch được cap theo padded token count "
            "của batch thực tế"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.data_path is None and args.tokenized_path is None:
        raise SystemExit("cần truyền --data-path hoặc --tokenized-path")
    cache_dataset(
        target_model_path=args.target_model_path,
        data_path=args.data_path,
        tokenized_path=args.tokenized_path,
        output_path=args.output_path,
        max_length=args.max_length,
        layer_ids=args.target_layer_ids,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        bucket_buffer_size=args.bucket_buffer_size,
        shard_size=args.shard_size,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        device=args.device,
        local_files_only=args.local_files_only,
        supervision_mode=args.supervision_mode,
        target_revision=args.target_revision,
        attention_backend=args.attention_backend,
        io_threads=args.io_threads,
        io_queue_size=args.io_queue_size,
        resume=args.resume,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        skipped_report=args.skipped_report,
        progress_path=args.progress_path,
        batch_profile=args.batch_profile,
        throughput_profile=args.throughput_profile,
        cache_backend=args.cache_backend,
        cache_concurrency=args.cache_concurrency,
        cache_max_total_tokens=args.cache_max_total_tokens,
        cache_memory_fraction=args.cache_memory_fraction,
    )


if __name__ == "__main__":
    main()
