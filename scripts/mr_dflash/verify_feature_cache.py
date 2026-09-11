"""Audit feature cache target trước khi đưa vào nhiều run train.

Audit này không chạy target model. Nó kiểm tra artifact đã ghi đủ và khớp
sample ID; nếu truyền ``--tokenizer`` thì render lại regenerated JSONL để đối
chiếu ``input_ids``/``loss_mask`` tại mọi offset với cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from _common import read_jsonl, write_json
from MR_DFlash.offline_features import SHARDED_FEATURE_SCHEMA_VERSION


def _load_manifest(cache_path: Path) -> tuple[Path, Dict[str, Any]]:
    manifest_path = cache_path if cache_path.is_file() else cache_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache thiếu manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SHARDED_FEATURE_SCHEMA_VERSION:
        raise ValueError(f"manifest cache không hợp lệ: {manifest_path}")
    return manifest_path.parent, payload


def validate_feature_cache(
    *,
    data_path: str | Path,
    cache_path: str | Path,
    tokenizer: Any = None,
    max_length: Optional[int] = None,
    supervision_mode: str = "last_assistant",
    expected_target_model: Optional[str] = None,
    expected_feature_layer_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Kiểm tra coverage, tensor shape/finite và optional token parity."""
    from MR_DFlash.data import build_sample

    cache_root, manifest = _load_manifest(Path(cache_path))
    input_rows = list(read_jsonl(data_path))
    expected_rows: Dict[str, Dict[str, Any]] = {}
    for row in input_rows:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in expected_rows:
            raise ValueError(f"input có id rỗng/trùng: {sample_id!r}")
        expected_rows[sample_id] = row
    expected_ids = set(expected_rows)
    manifest_ids = [str(value) for value in manifest.get("sample_ids", [])]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("manifest có sample ID trùng")
    if int(manifest.get("num_samples", -1)) != len(manifest_ids):
        raise ValueError("manifest num_samples không khớp sample_ids")
    if expected_target_model is not None and manifest.get("target_model_path") != expected_target_model:
        raise ValueError("cache target_model_path không khớp expected-target-model")
    if expected_feature_layer_ids is not None:
        actual_layers = [int(value) for value in manifest.get("feature_layer_ids", [])]
        if actual_layers != [int(value) for value in expected_feature_layer_ids]:
            raise ValueError(f"cache feature_layer_ids lệch: {actual_layers}")
    missing = sorted(expected_ids - set(manifest_ids))
    extra = sorted(set(manifest_ids) - expected_ids)
    if missing or extra:
        raise ValueError(f"cache coverage lệch: missing={missing[:5]}, extra={extra[:5]}")

    checked_ids: list[str] = []
    checked_offsets = 0
    checked_shards = 0
    manifest_shards = manifest.get("shards")
    if not isinstance(manifest_shards, list) or not manifest_shards:
        raise ValueError("manifest không có shards")
    limit = int(max_length if max_length is not None else manifest.get("max_length", 0))
    for descriptor in manifest_shards:
        if not isinstance(descriptor, dict) or "path" not in descriptor:
            raise ValueError("descriptor shard không hợp lệ")
        shard_path = cache_root / str(descriptor["path"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"cache thiếu shard: {shard_path}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        samples = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise ValueError(f"shard không có list samples: {shard_path}")
        descriptor_ids = [str(value) for value in descriptor.get("ids", [])]
        descriptor_lengths = [int(value) for value in descriptor.get("lengths", [])]
        if int(descriptor.get("count", -1)) != len(samples):
            raise ValueError(f"count shard lệch: {shard_path}")
        if descriptor_ids and descriptor_ids != [str(sample.get("id", "")) for sample in samples]:
            raise ValueError(f"descriptor ids lệch payload: {shard_path}")
        if descriptor_lengths and descriptor_lengths != [int(sample.get("length", -1)) for sample in samples]:
            raise ValueError(f"descriptor lengths lệch payload: {shard_path}")
        for sample in samples:
            sample_id = str(sample.get("id", ""))
            if not sample_id or sample_id in checked_ids:
                raise ValueError(f"shard có id rỗng/trùng: {sample_id!r}")
            if sample_id not in expected_rows:
                raise ValueError(f"shard có id ngoài input: {sample_id!r}")
            input_ids = torch.as_tensor(sample.get("input_ids"), dtype=torch.long).flatten()
            loss_mask = torch.as_tensor(sample.get("loss_mask"), dtype=torch.float32).flatten()
            hidden = torch.as_tensor(sample.get("hidden_states"))
            length = int(sample.get("length", input_ids.numel()))
            if length < 1 or (limit > 0 and length > limit):
                raise ValueError(f"sample {sample_id!r} có length không hợp lệ: {length}")
            if input_ids.numel() != length or loss_mask.numel() != length:
                raise ValueError(f"sample {sample_id!r} lệch input/loss length")
            if hidden.ndim != 2 or hidden.shape[0] != length:
                raise ValueError(f"sample {sample_id!r} hidden shape không khớp: {tuple(hidden.shape)}")
            if int(hidden.shape[1]) != int(manifest.get("feature_width", -1)):
                raise ValueError(f"sample {sample_id!r} hidden width không khớp manifest")
            if not torch.isfinite(hidden.float()).all().item():
                raise ValueError(f"sample {sample_id!r} hidden có NaN/Inf")
            if tokenizer is not None:
                expected = build_sample(
                    expected_rows[sample_id],
                    tokenizer,
                    limit,
                    supervision_mode=supervision_mode,
                )
                if expected is None:
                    raise ValueError(f"sample {sample_id!r} không render được khi audit")
                expected_ids_tensor = torch.as_tensor(expected["input_ids"], dtype=torch.long)
                expected_mask_tensor = torch.as_tensor(expected["loss_mask"], dtype=torch.float32)
                if not torch.equal(input_ids, expected_ids_tensor) or not torch.equal(loss_mask, expected_mask_tensor):
                    raise ValueError(f"sample {sample_id!r} input_ids/loss_mask lệch regenerated JSONL")
            checked_ids.append(sample_id)
            checked_offsets += length
        checked_shards += 1
    if set(checked_ids) != expected_ids or len(checked_ids) != len(manifest_ids):
        raise ValueError("coverage thực tế của shard không khớp manifest/input")
    return {
        "schema_version": "mr_dflash_feature_cache_audit_v1",
        "valid": True,
        "data_path": str(data_path),
        "cache_path": str(cache_path),
        "num_samples": len(checked_ids),
        "num_shards": checked_shards,
        "checked_offsets": checked_offsets,
        "token_parity_checked": tokenizer is not None,
        "feature_layer_ids": [int(value) for value in manifest.get("feature_layer_ids", [])],
        "feature_width": int(manifest.get("feature_width", 0)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit sharded MR-DFlash target feature cache")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--supervision-mode", choices=["all_assistant", "last_assistant"], default="last_assistant")
    parser.add_argument("--expected-target-model", default=None)
    parser.add_argument("--expected-feature-layer-ids", type=int, nargs="+", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=bool(args.local_files_only))
    report = validate_feature_cache(
        data_path=args.data_path,
        cache_path=args.cache_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        supervision_mode=args.supervision_mode,
        expected_target_model=args.expected_target_model,
        expected_feature_layer_ids=args.expected_feature_layer_ids,
    )
    if args.report:
        write_json(args.report, report)
    print(f"[verify_feature_cache] valid samples={report['num_samples']} offsets={report['checked_offsets']} shards={report['num_shards']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

