"""Gộp ShareGPT/ArXiv, split deterministic và ghi manifest provenance."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from _common import read_jsonl, stable_id, write_json, write_jsonl
from progress import ProgressReporter, install_exception_hook


DEFAULT_SHAREGPT_COUNT = 50000
DEFAULT_ARXIV_COUNT = 50000


def _split_counts(total: int) -> Dict[str, int]:
    train = int(total * 0.90)
    val = int(total * 0.05)
    return {"train": train, "val": val, "test": total - train - val}


def _assign(rows: List[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    counts = _split_counts(len(rows))
    rng = random.Random(seed)
    # Length-stratified traversal for ArXiv; for ShareGPT this is simply a
    # deterministic shuffle because source_length is usually absent.
    strata: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        length = int(metadata.get("source_token_length", metadata.get("source_length", 0)))
        bucket = 0 if length <= 2048 else 1 if length <= 4096 else 2 if length <= 6144 else 3
        strata[bucket].append(row)
    for bucket in strata.values():
        rng.shuffle(bucket)
    result = {"train": [], "val": [], "test": []}
    remaining = dict(counts)
    bucket_items = sorted(strata.items())
    for bucket_index, (_key, bucket_rows) in enumerate(bucket_items):
        if bucket_index == len(bucket_items) - 1:
            quota = remaining
        else:
            remaining_rows = sum(remaining.values())
            quota = {
                split: min(
                    remaining[split],
                    int(len(bucket_rows) * remaining[split] / max(1, remaining_rows)),
                )
                for split in remaining
            }
            # Largest-remainder allocation keeps every length stratum
            # represented while preserving exact global 90/5/5 counts.
            slots = len(bucket_rows) - sum(quota.values())
            fractional = sorted(
                remaining,
                key=lambda split: (
                    len(bucket_rows) * remaining[split] / max(1, remaining_rows)
                    - quota[split],
                    remaining[split],
                ),
                reverse=True,
            )
            for split in fractional:
                if slots <= 0:
                    break
                if quota[split] < remaining[split]:
                    quota[split] += 1
                    slots -= 1
        cursor = 0
        for split in ("train", "val", "test"):
            take = int(quota[split])
            result[split].extend(bucket_rows[cursor : cursor + take])
            cursor += take
            remaining[split] -= take
    for split, split_rows in result.items():
        for row in split_rows:
            row["metadata"] = {**(row.get("metadata") or {}), "split": split}
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build the MR-DFlash pilot manifest")
    parser.add_argument("--sharegpt", required=True)
    parser.add_argument("--arxiv", required=True)
    parser.add_argument(
        "--source-sharegpt-input",
        default=None,
        help="raw source path để ghi provenance; mặc định dùng --sharegpt",
    )
    parser.add_argument(
        "--source-arxiv-input",
        default=None,
        help="raw source path để ghi provenance; mặc định dùng --arxiv",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sharegpt-count", type=int, default=DEFAULT_SHAREGPT_COUNT)
    parser.add_argument("--arxiv-count", type=int, default=DEFAULT_ARXIV_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-short", action="store_true")
    args = parser.parse_args(argv)

    reporter = ProgressReporter(None)
    previous_hook = install_exception_hook(reporter)
    reporter.update(
        "starting",
        input_sharegpt=str(args.sharegpt),
        input_arxiv=str(args.arxiv),
        completed_samples=reporter.completed_samples(),
    )

    sharegpt = list(read_jsonl(args.sharegpt))
    arxiv = list(read_jsonl(args.arxiv))
    if not args.allow_short and (len(sharegpt) < args.sharegpt_count or len(arxiv) < args.arxiv_count):
        raise ValueError(
            f"không đủ nguồn: ShareGPT {len(sharegpt)}/{args.sharegpt_count}, "
            f"ArXiv {len(arxiv)}/{args.arxiv_count}; dùng --allow-short cho smoke"
        )
    rng = random.Random(args.seed)
    rng.shuffle(sharegpt)
    rng.shuffle(arxiv)
    sharegpt = sharegpt[: args.sharegpt_count]
    arxiv = arxiv[: args.arxiv_count]
    total = len(sharegpt) + len(arxiv)
    if "MR_DFLASH_PROGRESS_TOTAL_SAMPLES" not in os.environ:
        reporter.set_context(total_samples=total)
    reporter.update("writing", completed_samples=reporter.completed_samples(), total_rows=total)
    splits = {"train": [], "val": [], "test": []}
    for source_rows, source_seed in ((sharegpt, args.seed + 11), (arxiv, args.seed + 29)):
        assigned = _assign(source_rows, source_seed)
        for key in splits:
            splits[key].extend(assigned[key])
    for split_rows in splits.values():
        rng.shuffle(split_rows)

    root = Path(args.output_root)
    normalized = root / "normalized"
    write_jsonl(normalized / "pilot_prompts.jsonl", [*sharegpt, *arxiv])
    for split, rows in splits.items():
        write_jsonl(normalized / f"{split}_prompts.jsonl", rows)
    counts = {split: len(rows) for split, rows in splits.items()}
    source_counts = {
        split: dict(Counter(str(row.get("source", "unknown")) for row in rows))
        for split, rows in splits.items()
    }
    observed_sharegpt = sum(1 for row in (*sharegpt, *arxiv) if row.get("source") == "sharegpt")
    observed_arxiv = sum(1 for row in (*sharegpt, *arxiv) if row.get("source") == "arxiv")
    observed_total = observed_sharegpt + observed_arxiv
    split_manifest = {
        "schema_version": "mr_dflash_split_v1",
        "seed": args.seed,
        "total": sum(counts.values()),
        "train": counts["train"],
        "val": counts["val"],
        "test": counts["test"],
        "source_ratio": {
            "sharegpt": observed_sharegpt / max(1, observed_total),
            "arxiv": observed_arxiv / max(1, observed_total),
        },
        "source_counts": source_counts,
        "stratification": {
            "arxiv": [2048, 4096, 6144, 8192],
            "metric": "source_token_length (fallback source_length_chars)",
        },
        "ids": {split: [str(row["id"]) for row in rows] for split, rows in splits.items()},
    }
    split_manifest["id_sha256"] = {
        split: hashlib.sha256(
            "\n".join(str(row["id"]) for row in splits[split]).encode("utf-8")
        ).hexdigest()
        for split in splits
    }
    write_json(root / "manifests" / "split_manifest.json", split_manifest)
    write_json(
        root / "manifests" / "source_manifest.json",
        {
            "schema_version": "mr_dflash_source_v1",
            "seed": args.seed,
            "sharegpt_count": len(sharegpt),
            "arxiv_count": len(arxiv),
            "total": len(sharegpt) + len(arxiv),
            "sharegpt_input": str(args.source_sharegpt_input or args.sharegpt),
            "arxiv_input": str(args.source_arxiv_input or args.arxiv),
        },
    )
    reporter.update("writing", completed_samples=total, total_rows=total)
    print(f"[build_pilot_dataset] total={sum(counts.values())} splits={counts}")
    reporter.update("done", completed_samples=total, total_rows=total)
    sys.excepthook = previous_hook


if __name__ == "__main__":
    main()
