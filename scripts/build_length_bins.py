#!/usr/bin/env python3
"""Build 5 length-partitioned benchmark datasets from canonical LongBench data.

Partitions:
  - bin1 (00k-02k): [0, 2000) tokens
  - bin2 (02k-04k): [2000, 4000) tokens
  - bin3 (04k-08k): [4000, 8000) tokens
  - bin4 (08k-12k): [8000, 12000) tokens
  - bin5 (12k-16k): [12000, 16000) tokens

Each bin contains 50 deterministically selected samples with balanced task representation.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

BINS = [
    ("bin1_00k_02k", 0, 2000),
    ("bin2_02k_04k", 2000, 4000),
    ("bin3_04k_08k", 4000, 8000),
    ("bin4_08k_12k", 8000, 12000),
    ("bin5_12k_16k", 12000, 16000),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_length_bins(
    source_dir: Path,
    output_dir: Path,
    samples_per_bin: int = 50,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    source_files = sorted(source_dir.glob("*.jsonl"))
    if not source_files:
        raise FileNotFoundError(f"No .jsonl files found in {source_dir}")

    all_records: list[dict[str, Any]] = []
    source_stats: dict[str, int] = {}
    for sf in source_files:
        dname = sf.stem
        count = 0
        with sf.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    all_records.append(rec)
                    count += 1
        source_stats[dname] = count

    manifest: dict[str, Any] = {
        "version": "1.0",
        "description": "5 length-partitioned benchmark datasets derived from LongBench",
        "source_dir": str(source_dir.relative_to(ROOT) if source_dir.is_relative_to(ROOT) else source_dir),
        "source_counts": source_stats,
        "seed": seed,
        "samples_per_bin": samples_per_bin,
        "bins": {},
    }

    for bin_idx, (bin_name, lo, hi) in enumerate(BINS):
        candidates = [
            r for r in all_records
            if lo <= int(r.get("input_tokens", 0)) < hi
        ]

        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in candidates:
            by_dataset[str(r.get("dataset", "unknown"))].append(r)

        d_names = sorted(by_dataset.keys())
        for d in d_names:
            # Deterministic shuffle with seeded RNG
            rng.shuffle(by_dataset[d])

        selected: list[dict[str, Any]] = []
        while len(selected) < samples_per_bin and any(by_dataset.values()):
            for d in d_names:
                if by_dataset[d] and len(selected) < samples_per_bin:
                    selected.append(by_dataset[d].pop(0))

        if len(selected) < samples_per_bin:
            print(f"Warning: {bin_name} only got {len(selected)}/{samples_per_bin} samples!")

        # Sort selected samples by input_tokens ascending for predictable iteration
        selected.sort(key=lambda x: (int(x.get("input_tokens", 0)), str(x.get("id", ""))))

        # Update bin tag in each record
        for item in selected:
            item["length_bin"] = bin_idx
            item["length_bin_name"] = bin_name

        out_path = output_dir / f"{bin_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in selected:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        tokens = [int(r["input_tokens"]) for r in selected]
        dataset_dist = defaultdict(int)
        task_dist = defaultdict(int)
        for r in selected:
            dataset_dist[r.get("dataset", "unknown")] += 1
            task_dist[r.get("task_type", "unknown")] += 1

        manifest["bins"][bin_name] = {
            "file": str(out_path.name),
            "range": [lo, hi],
            "count": len(selected),
            "min_tokens": min(tokens) if tokens else 0,
            "max_tokens": max(tokens) if tokens else 0,
            "mean_tokens": round(sum(tokens) / len(tokens), 2) if tokens else 0,
            "median_tokens": sorted(tokens)[len(tokens) // 2] if tokens else 0,
            "dataset_distribution": dict(sorted(dataset_dist.items())),
            "task_distribution": dict(sorted(task_dist.items())),
            "sha256": sha256_file(out_path),
            "sample_ids": [r["id"] for r in selected],
        }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated 5 length-partitioned datasets in: {output_dir}")
    print(f"Manifest written to: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 5 length-partitioned datasets")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data" / "longbench_200",
        help="Source directory containing canonical LongBench JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "length_bins",
        help="Target output directory",
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=50,
        help="Number of samples per bin (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection (default: 42)",
    )
    args = parser.parse_args()

    build_length_bins(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        samples_per_bin=args.samples_per_bin,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
