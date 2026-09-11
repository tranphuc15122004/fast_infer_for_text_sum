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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from _common import append_jsonl_durable, read_jsonl
from MR_DFlash.capture import HFTargetCapture
from progress import ProgressReporter, estimate_fields, install_exception_hook


def _is_cuda_oom(exc: BaseException) -> bool:
    """OOM không thể coi là lỗi của một sample rồi tiếp tục cache."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


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
) -> Dict[str, int]:
    """Chạy target capture theo batch và ghi cache sharded resumable."""
    if data_path is None and tokenized_path is None:
        raise ValueError("cache cần data_path hoặc tokenized_path")
    if batch_size < 1:
        raise ValueError("batch_size phải >= 1")
    if io_threads < 0 or io_queue_size < 0:
        raise ValueError("io_threads và io_queue_size không được âm")
    if bucket_buffer_size is None:
        bucket_buffer_size = max(batch_size, batch_size * 8)
    if bucket_buffer_size < batch_size:
        raise ValueError("bucket_buffer_size phải >= batch_size")
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

    reporter.update("loading_model", target_model_path=str(target_model_path))
    capturer = HFTargetCapture(
        target_model_path,
        layer_ids or [],
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device=device,
        local_files_only=local_files_only,
        target_revision=target_revision,
        attention_backend=attention_backend,
    )
    reporter.update(
        "model_ready",
        target_model_path=str(target_model_path),
        device=str(capturer.device),
        dtype=torch_dtype,
        feature_layer_ids=[int(value) for value in capturer.layer_ids],
        attention_backend=str(attention_backend),
        io_threads=int(io_threads),
        io_queue_size=int(io_queue_size),
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
        capture_backend="hf_backbone",
        attention_backend=attention_backend,
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
    buffer: List[Dict[str, Any]] = []

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda iterator, **_kwargs: iterator

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
        input_ids, attention_mask = _pad_samples(
            batch,
            pad_token_id=int(pad_token_id),
            device=capturer.device,
        )
        try:
            captured = capturer.capture_batch(input_ids, attention_mask)
            if len(captured) != len(batch):
                raise RuntimeError(
                    f"capture trả {len(captured)} sample thay vì {len(batch)}"
                )
        except Exception as exc:
            # Một sample lỗi không làm mất cả shard/batch. Fallback tuần tự
            # cũng giúp chẩn đoán rõ sample id trên model/driver bất ổn.
            if _is_cuda_oom(exc):
                raise
            print(f"[cache] batch capture lỗi, fallback từng mẫu: {exc!r}")
            for sample in batch:
                try:
                    feature = capturer.capture_one(sample["input_ids"])[0]
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

    rows = tqdm(iter_cache_rows(), desc="Cache target features", unit="row")

    def next_batch_size() -> int:
        if num_samples is None:
            return batch_size
        remaining = int(num_samples) - writer.total_samples
        return max(0, min(batch_size, remaining))

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
            while len(buffer) >= batch_size:
                current_batch_size = next_batch_size()
                if current_batch_size == 0:
                    buffer.clear()
                    break
                process(buffer[:current_batch_size])
                del buffer[:current_batch_size]
                if num_samples is not None and writer.total_samples >= int(num_samples):
                    buffer.clear()
                    break
    buffer.sort(key=lambda item: -len(item["input_ids"]))
    while buffer and (num_samples is None or writer.total_samples < int(num_samples)):
        current_batch_size = next_batch_size()
        process(buffer[:current_batch_size])
        del buffer[:current_batch_size]

    stats["captured_total"] = writer.total_samples
    stats["workload_input_rows"] = int(workload["input_rows"])
    stats["workload_valid_samples"] = total_samples
    stats["workload_valid_tokens"] = total_tokens
    stats["workload_existing_samples"] = int(workload["existing_samples"])
    stats["workload_existing_tokens"] = int(workload["existing_tokens"])
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
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help=(
            "attention implementation cho backbone capture; sdpa dùng kernel "
            "PyTorch tối ưu và không bắt buộc flash-attn"
        ),
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
    )


if __name__ == "__main__":
    main()
