"""Tokenize canonical regenerated JSONL thành shard cho online training."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from _common import read_jsonl


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Tokenize MR-DFlash pilot split")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance-manifest", default=None)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--supervision-mode", choices=["all_assistant", "last_assistant"], default="last_assistant")
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument("--feature-layer-ids", type=int, nargs="+", default=[1, 9, 17, 25, 33])
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="giữ các shard đã ghi và bỏ qua sample id đã tokenize",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.shard_size < 1:
        raise ValueError("--shard-size phải >= 1")
    from transformers import AutoTokenizer
    load_kwargs = {"cache_dir": args.cache_dir, "local_files_only": args.local_files_only}
    if args.target_revision:
        load_kwargs["revision"] = args.target_revision
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, **load_kwargs)
    from MR_DFlash.data import has_consecutive_supervised_tokens, render_conversation
    from MR_DFlash.tokenized_data import write_tokenized_manifest

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []
    shards: List[Dict[str, Any]] = []
    existing_ids = set()
    total = 0
    existing_shards = sorted(root.glob("shard_*.pt"))
    manifest_path = root / "manifest.json"
    if not args.resume and (existing_shards or manifest_path.exists()):
        raise FileExistsError(
            f"tokenized output đã tồn tại: {root}; dùng --resume hoặc thư mục mới"
        )
    if args.resume:
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"tokenized manifest không đọc được: {manifest_path}") from exc
            if manifest.get("schema_version") != "mr_dflash_tokenized_v1":
                raise ValueError(f"tokenized manifest không tương thích: {manifest_path}")
            manifest_shards = manifest.get("shards", [])
            if not isinstance(manifest_shards, list):
                raise ValueError(f"tokenized manifest shards không phải list: {manifest_path}")
            missing = [
                str(root / str(item.get("path")))
                for item in manifest_shards
                if not isinstance(item, dict) or not (root / str(item.get("path"))).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "tokenized output thiếu shard đã ghi trong manifest: "
                    + ", ".join(missing[:3])
                )
        for shard_path in sorted(root.glob("shard_*.pt")):
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            shard_samples = payload.get("samples") if isinstance(payload, dict) else payload
            if not isinstance(shard_samples, list):
                raise ValueError(f"shard không có list samples: {shard_path}")
            shard_ids = {str(item.get("id", "")) for item in shard_samples if isinstance(item, dict)}
            existing_ids.update(shard_ids)
            shards.append({"path": shard_path.name, "count": len(shard_samples)})
            total += len(shard_samples)

    def flush() -> None:
        nonlocal samples
        if not samples:
            return
        name = f"shard_{len(shards):05d}.pt"
        target = root / name
        temporary = root / f".{name}.tmp"
        with temporary.open("wb") as handle:
            torch.save({"samples": samples}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        shards.append({"path": name, "count": len(samples)})
        samples = []

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm có trong requirements server
        tqdm = lambda iterator, **_kwargs: iterator
    rows = tqdm(read_jsonl(args.input), desc="Tokenize MR-DFlash", unit="row")
    for row_index, row in enumerate(rows):
        if args.limit is not None and total >= args.limit:
            break
        sample_id = str(row.get("id", row_index))
        if sample_id in existing_ids:
            continue
        conversations = row.get("conversations") or []
        input_ids, loss_mask = render_conversation(
            conversations,
            tokenizer,
            10**9,
            supervision_mode=args.supervision_mode,
        )
        if len(input_ids) > args.max_length:
            raise ValueError(
                f"sample {row.get('id', row_index)!r} dài {len(input_ids)} > "
                f"max_length={args.max_length}; regenerate với prompt budget nhỏ hơn"
            )
        if len(input_ids) < 3 or not has_consecutive_supervised_tokens(loss_mask):
            continue
        ids = torch.tensor(input_ids, dtype=torch.long)
        mask = torch.tensor(loss_mask, dtype=torch.float32)
        samples.append({
            "id": sample_id,
            "input_ids": ids,
            "loss_mask": mask,
            "length": int(ids.numel()),
        })
        existing_ids.add(sample_id)
        total += 1
        if len(samples) >= args.shard_size:
            flush()
    flush()
    manifest_payload = write_tokenized_manifest(
        root,
        shards=shards,
        num_samples=total,
        target_model=args.target_model_path,
        feature_layer_ids=args.feature_layer_ids,
        chat_template=args.chat_template,
        max_length=args.max_length,
        supervision_mode=args.supervision_mode,
        tokenizer_name=args.target_model_path,
        target_revision=args.target_revision,
    )
    if args.provenance_manifest:
        target = Path(args.provenance_manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    print(f"[tokenize_dataset] samples={total} shards={len(shards)} output={root}")


if __name__ == "__main__":
    main()
