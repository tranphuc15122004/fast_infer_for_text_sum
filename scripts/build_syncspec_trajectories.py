#!/usr/bin/env python3
"""Build offline target-generated Stage-0 SyncSpec trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.synthetic import SyntheticTarget  # noqa: E402
from SyncSpec.trajectory import TargetTrajectoryBuilder  # noqa: E402
from SyncSpec.training import (  # noqa: E402
    TrajectoryCache,
    artifact_fingerprint,
    cache_fingerprint,
)
from SyncSpec.transformers_adapter import TransformersTargetAdapter  # noqa: E402
from SyncSpec.prompt import encode_record  # noqa: E402


def _read_samples(path: Path | None, tokenizer=None, count: int = 4, max_input_tokens: int = 0):
    if int(count) < 0:
        raise ValueError("sample count must be non-negative")
    if int(count) == 0:
        return []
    if path is None:
        return [(f"synthetic-{i}", torch.tensor([2 + i, 3 + i, 4 + i])) for i in range(count)]
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        sample_id = raw.get("id", raw.get("sample_id", len(samples)))
        if "source_ids" in raw:
            ids = torch.tensor(raw["source_ids"], dtype=torch.long)
        else:
            ids = encode_record(raw, tokenizer, max_input_tokens=max_input_tokens)
        samples.append((str(sample_id), ids))
        if len(samples) >= int(count):
            break
    return samples


def _tokenizer_fingerprint(tokenizer) -> str:
    if tokenizer is None:
        return "synthetic"
    template = str(getattr(tokenizer, "chat_template", ""))
    name = str(getattr(tokenizer, "name_or_path", ""))
    vocab_size = str(getattr(tokenizer, "vocab_size", ""))
    return cache_fingerprint(name, vocab_size, template)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("synthetic", "transformers"), default="synthetic")
    parser.add_argument("--target-model")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--source-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--include-logits", action="store_true")
    parser.add_argument("--include-target-features", action="store_true")
    parser.add_argument(
        "--include-source-memory", action="store_true",
        help="cache target final-hidden source chunk descriptors for training",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fingerprint")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip sample IDs already present in the fingerprinted output cache",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.source_chunk_size <= 0:
        raise SystemExit("--source-chunk-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but unavailable; run trajectory smoke with --device cpu "
            "or execute training on the canonical CUDA server"
        )

    tokenizer = None
    if args.backend == "synthetic":
        target = SyntheticTarget(vocab_size=256, device=args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    else:
        if not args.target_model:
            raise SystemExit("--target-model is required for --backend transformers")
        target = TransformersTargetAdapter.from_pretrained(
            args.target_model, device=args.device, dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
        tokenizer = target.tokenizer
    samples = _read_samples(args.input, tokenizer, args.samples, args.max_input_tokens)
    tokenizer_fingerprint = _tokenizer_fingerprint(tokenizer)
    target_artifact_fingerprint = artifact_fingerprint(
        args.target_model or "synthetic"
    )
    fingerprint = args.fingerprint or cache_fingerprint(
        args.backend, str(args.target_model or "synthetic"), str(args.dtype),
        str(args.max_new_tokens), str(args.max_input_tokens),
        str(args.include_logits), str(args.include_target_features),
        str(args.include_source_memory), str(args.source_chunk_size),
        str(args.seed), tokenizer_fingerprint, target_artifact_fingerprint,
        "summarize-v1",
    )
    cache = TrajectoryCache(args.output, fingerprint)
    if args.resume and args.output.exists():
        existing_ids = {record.sample_id for record in cache.read()}
        samples = [sample for sample in samples if str(sample[0]) not in existing_ids]
    records = TargetTrajectoryBuilder(
        target, seed=args.seed, source_chunk_size=args.source_chunk_size, metadata={
            "target_model": str(args.target_model or "synthetic"),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "target_artifact_fingerprint": target_artifact_fingerprint,
            "dtype": str(args.dtype),
            "source_chunk_size": int(args.source_chunk_size),
        },
    ).build_records(
        samples, max_new_tokens=args.max_new_tokens, include_logits=args.include_logits,
        include_target_features=args.include_target_features,
        include_source_memory=args.include_source_memory,
    )
    cache.write(records, append=args.resume)
    print(json.dumps({"status": "ok", "records": len(records), "fingerprint": fingerprint, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
