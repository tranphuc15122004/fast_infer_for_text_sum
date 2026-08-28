#!/usr/bin/env python3
"""Build the fixed LongBench 5-task test set from a local JSONL mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.benchmark_data import (  # noqa: E402
    DATASETS,
    EXPECTED_SOURCE_COUNTS,
    PROMPT_CONFIG,
    canonicalize_record,
    load_prompt_templates,
    read_jsonl,
    select_rows,
    token_stats,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _new_manifest(source_dir: Path, tokenizer_ref: str, seed: int) -> dict[str, Any]:
    return {
        "schema_version": "longbench-canonical-v1",
        "seed": seed,
        "tokenizer": tokenizer_ref,
        "prompt_config": str(PROMPT_CONFIG.relative_to(ROOT)),
        "prompt_config_sha256": sha256_file(PROMPT_CONFIG),
        "source_dir": str(source_dir),
        "source_counts": {},
        "selected_counts": {},
        "datasets": {},
        "selected_ids": {},
        "file_sha256": {},
    }


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_one_dataset(
    dataset: str,
    source_dir: Path,
    output_dir: Path,
    tokenizer: Callable[..., dict[str, Any]],
    target_count: int,
    seed: int,
    *,
    tokenizer_ref: str = "test-tokenizer",
) -> list[dict[str, Any]]:
    source_path = Path(source_dir) / f"{dataset}.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing local LongBench source: {source_path}")
    source_rows = read_jsonl(source_path)
    canonical_rows = [
        canonicalize_record(dataset, index, row, tokenizer)
        for index, row in enumerate(source_rows)
    ]
    selected = select_rows(canonical_rows, dataset, target_count, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}.jsonl"
    write_jsonl(output_path, selected)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _new_manifest(Path(source_dir), tokenizer_ref, seed)
    manifest["source_counts"][dataset] = len(source_rows)
    manifest["selected_counts"][dataset] = len(selected)
    manifest["datasets"][dataset] = token_stats(selected)
    manifest["selected_ids"][dataset] = [row["id"] for row in selected]
    manifest["file_sha256"][dataset] = sha256_file(output_path)
    manifest["total_selected"] = sum(manifest["selected_counts"].values())
    _save_manifest(manifest_path, manifest)
    return selected


def resolve_tokenizer(tokenizer_ref: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_ref, local_files_only=True, use_fast=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/longbench_200"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-dataset", type=int, default=200)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_dataset <= 0:
        raise SystemExit("--samples-per-dataset must be positive")
    if args.samples_per_dataset != 200 and not args.allow_partial:
        raise SystemExit("non-200 builds require --allow-partial")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}; use --force or a new directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_prompt_templates()
    tokenizer = resolve_tokenizer(args.tokenizer)
    manifest_path = args.output_dir / "manifest.json"
    manifest = _new_manifest(args.source_dir, args.tokenizer, args.seed)
    _save_manifest(manifest_path, manifest)

    for index, dataset in enumerate(DATASETS, start=1):
        source_path = args.source_dir / f"{dataset}.jsonl"
        source_count = len(read_jsonl(source_path))
        if args.samples_per_dataset == 200 and source_count != EXPECTED_SOURCE_COUNTS[dataset]:
            raise SystemExit(
                f"{dataset}: expected {EXPECTED_SOURCE_COUNTS[dataset]} source rows, got {source_count}"
            )
        print(f"[dataset {index}/{len(DATASETS)}] {dataset}: source={source_count}", flush=True)
        selected = build_one_dataset(
            dataset,
            args.source_dir,
            args.output_dir,
            tokenizer,
            args.samples_per_dataset,
            args.seed,
            tokenizer_ref=args.tokenizer,
        )
        print(
            f"[dataset {index}/{len(DATASETS)}] {dataset}: selected={len(selected)} "
            f"tokens={token_stats(selected)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
