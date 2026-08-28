#!/usr/bin/env python3
"""Print human-review statistics and spot checks for a canonical dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.benchmark_data import DATASETS, read_jsonl, token_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--spot-checks", type=int, default=2)
    args = parser.parse_args()
    total = 0
    for dataset in DATASETS:
        rows = read_jsonl(args.data_dir / f"{dataset}.jsonl")
        total += len(rows)
        bins = Counter(row["length_bin"] for row in rows)
        print(f"{dataset}: count={len(rows)} stats={token_stats(rows)} bins={dict(sorted(bins.items(), key=lambda item: str(item[0])))}")
        for row in rows[: args.spot_checks]:
            print(
                f"  spot id={row['id']} type={row['task_type']} "
                f"context_chars={len(row['context'])} input_chars={len(row['input'])} "
                f"reference_chars={len(row['reference_output'])}"
            )
    print(f"total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
