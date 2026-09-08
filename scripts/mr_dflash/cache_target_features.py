"""Cache target hidden states theo shard để tái sử dụng cho nhiều lần train.

Input là canonical regenerated JSONL (đã chứa assistant response do target
model sinh). Script này không generate lại response. Response được giữ trong
JSONL; script chỉ render cùng chat template, chạy target teacher-forcing và
lưu hidden states ở toàn bộ offset hợp lệ của mỗi sample.

Ví dụ trên server B200:

``python scripts/mr_dflash/cache_target_features.py \
  --target-model-path Qwen/Qwen3-4B \
  --data-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated/train.jsonl \
  --output-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/target_features_qwen3_4b_8k/train \
  --max-length 8192 --batch-size 2 --shard-size 64 --resume \
  --local-files-only``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from _common import append_jsonl_durable, read_jsonl
from MR_DFlash.capture import HFTargetCapture


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
        input_ids[row, :length] = torch.tensor(
            sample["input_ids"], dtype=torch.long, device=device
        )
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def cache_dataset(
    *,
    target_model_path: str,
    data_path: str,
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
    resume: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    skipped_report: Optional[str] = None,
) -> Dict[str, int]:
    """Chạy target capture theo batch và ghi cache sharded resumable."""
    if batch_size < 1:
        raise ValueError("batch_size phải >= 1")
    if bucket_buffer_size is None:
        bucket_buffer_size = max(batch_size, batch_size * 8)
    if bucket_buffer_size < batch_size:
        raise ValueError("bucket_buffer_size phải >= batch_size")
    if num_shards < 1 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard-index phải nằm trong [0, num-shards)")

    from MR_DFlash.data import build_sample
    from MR_DFlash.offline_features import ShardedFeatureWriter

    capturer = HFTargetCapture(
        target_model_path,
        layer_ids or [],
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device=device,
        local_files_only=local_files_only,
        target_revision=target_revision,
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
        source_data_path=data_path,
        target_revision=target_revision,
        resume=resume,
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

    def record_skip(sample_id: str, *, kind: str, error: str) -> None:
        append_row = {
            "id": str(sample_id),
            "kind": kind,
            "error": error,
        }
        append_jsonl_durable(skipped_report_path, [append_row])
    buffer: List[Dict[str, Any]] = []

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda iterator, **_kwargs: iterator

    def process(batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
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
                    else:
                        stats["skipped_existing"] += 1
                except Exception as sample_exc:
                    if _is_cuda_oom(sample_exc):
                        raise
                    print(f"[cache] skip {sample['id']!r}: {sample_exc!r}")
                    record_skip(sample["id"], kind="capture_error", error=repr(sample_exc))
                    stats["capture_errors"] += 1
            return
        for sample, feature in zip(batch, captured):
            try:
                added = writer.add({**sample, "hidden_states": feature})
                if added:
                    stats["captured"] += 1
                else:
                    stats["skipped_existing"] += 1
            except Exception as exc:
                if _is_cuda_oom(exc):
                    raise
                print(f"[cache] skip {sample['id']!r}: {exc!r}")
                record_skip(sample["id"], kind="write_error", error=repr(exc))
                stats["capture_errors"] += 1

    rows = tqdm(read_jsonl(data_path), desc="Cache target features", unit="row")

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
        if sample_id in writer.existing_ids:
            stats["skipped_existing"] += 1
            continue
        sample = build_sample(
            row,
            capturer.tokenizer,
            max_length,
            supervision_mode=supervision_mode,
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
    return {key: int(value) for key, value in stats.items() if isinstance(value, int)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache target hidden states theo shard")
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--data-path", required=True)
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
        "--supervision-mode",
        choices=["all_assistant", "last_assistant"],
        default="last_assistant",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skipped-report", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cache_dataset(
        target_model_path=args.target_model_path,
        data_path=args.data_path,
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
        resume=args.resume,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        skipped_report=args.skipped_report,
    )


if __name__ == "__main__":
    main()
