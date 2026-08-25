#!/usr/bin/env python3
"""
Rebuild the complete summarization data layout in an isolated output root.

Datasets (test split only)
--------------------------
1. GovReport
2. Multi-News
3. CNN/DailyMail 3.0.0
4. XSum

The script intentionally downloads only the test data needed by the current
semantic-selection benchmark. It does NOT call legacy Hugging Face dataset
loading scripts and does NOT download CNN/DailyMail/XSum train splits.

Resulting layout
----------------
data/rebuild/
├── raw/
│   ├── cnn_dailymail/
│   │   └── 3.0.0/test-00000-of-00001.parquet
│   ├── govreport/
│   │   └── plain_text/gov_report-test.parquet
│   ├── multinews/
│   │   └── data/
│   │       ├── test.src.cleaned
│   │       └── test.tgt
│   └── xsum/
│       └── data/test-00000-of-00001.parquet
├── normalized/
│   ├── cnn_dailymail_test.jsonl
│   ├── govreport_test.jsonl
│   ├── multinews_test.jsonl
│   └── xsum_test.jsonl
├── debug/
│   ├── debug_real.jsonl
│   └── smoke_real.jsonl
└── build_manifest.json

Normalized schema
-----------------
{
    "id": "...",
    "dataset": "...",
    "source_split": "test",
    "document": "...",
    "reference": "..."
}

Debug-set policy
----------------
- Uses the Qwen/Qwen3-4B tokenizer by default.
- Never truncates a source document.
- Keeps only naturally occurring documents within the requested token range.
- Sorts candidates by target-token length and chooses evenly spaced examples.
- Default: 4 examples per dataset, producing:
      smoke_real.jsonl : 4 examples (one per dataset)
      debug_real.jsonl : 16 examples (four per dataset)

Dependencies
------------
uv add huggingface-hub pyarrow transformers

Fresh-server command
--------------------
python data/prepare_data.py \
    --root data/rebuild \
    --tokenizer Qwen/Qwen3-4B \
    --samples-per-dataset 4 \
    --min-debug-tokens 256 \
    --max-debug-tokens 4096
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


# Pinned revisions make the build reproducible across servers.
GOVREPORT_REVISION = "5884c3a5586093e6d571b0658b43ba02e2758929"
MULTINEWS_REVISION = "713d9814275decacf84b5b7dbe7a4cd3d21fc8d9"
CNN_DM_REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
XSUM_REVISION = "30802a38d3f89b2fa8f19276008459e8c2b8b8e6"

# Source test-row counts at the pinned revisions. Some source files contain
# blank/invalid test rows; those are intentionally excluded from normalized
# output by the script below.
EXPECTED_SOURCE_COUNTS = {
    "govreport": 973,
    "multinews": 5622,
    "cnn_dailymail": 11490,
    "xsum": 11334,
}

# Expected valid rows after removing empty source/reference records present in
# the pinned raw files. Keeping this separate from source counts makes the
# clean-build behavior explicit and reproducible.
EXPECTED_NORMALIZED_COUNTS = {
    "govreport": 973,
    "multinews": 5621,
    "cnn_dailymail": 11490,
    "xsum": 11333,
}

DEFAULT_ROOT = Path("data") / "build"

DATASET_ORDER = (
    "govreport",
    "multinews",
    "cnn_dailymail",
    "xsum",
)


@dataclass(frozen=True)
class HubFile:
    repo_id: str
    revision: str
    filename: str
    local_subdir: str


SOURCE_FILES: Dict[str, Tuple[HubFile, ...]] = {
    "govreport": (
        HubFile(
            "launch/gov_report",
            GOVREPORT_REVISION,
            "plain_text/gov_report-test.parquet",
            "govreport",
        ),
    ),
    "multinews": (
        HubFile(
            "alexfabbri/multi_news",
            MULTINEWS_REVISION,
            "data/test.src.cleaned",
            "multinews",
        ),
        HubFile(
            "alexfabbri/multi_news",
            MULTINEWS_REVISION,
            "data/test.tgt",
            "multinews",
        ),
    ),
    "cnn_dailymail": (
        HubFile(
            "abisee/cnn_dailymail",
            CNN_DM_REVISION,
            "3.0.0/test-00000-of-00001.parquet",
            "cnn_dailymail",
        ),
    ),
    "xsum": (
        HubFile(
            "EdinburghNLP/xsum",
            XSUM_REVISION,
            "data/test-00000-of-00001.parquet",
            "xsum",
        ),
    ),
}


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs(root: Path) -> None:
    for path in (
        root,
        root / "raw",
        root / "normalized",
        root / "debug",
        root / "raw" / "govreport",
        root / "raw" / "multinews",
        root / "raw" / "cnn_dailymail",
        root / "raw" / "xsum",
    ):
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}"
                ) from exc
            if not isinstance(item, dict):
                raise TypeError(
                    f"{path}:{line_number} is not a JSON object"
                )
            yield item


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def check_nonempty(
    dataset: str,
    document: str,
    reference: str,
    row_id: str,
) -> None:
    if not document:
        raise ValueError(f"{dataset}: empty document for {row_id}")
    if not reference:
        raise ValueError(f"{dataset}: empty reference for {row_id}")


def keep_valid_row(
    dataset: str,
    document: str,
    reference: str,
    row_id: str,
) -> bool:
    """Return whether a raw record can be used by infer.py.

    The public test files contain a small number of empty records. They are
    not useful for summarization or ROUGE, so skip them with an explicit log
    instead of aborting an otherwise complete four-dataset build.
    """
    if document and reference:
        return True

    missing = []
    if not document:
        missing.append("document")
    if not reference:
        missing.append("reference")
    log(
        f"[normalize] {dataset}: skipping {row_id}; "
        f"empty {', '.join(missing)}"
    )
    return False


def expected_count_check(dataset: str, actual: int) -> None:
    expected = EXPECTED_NORMALIZED_COUNTS[dataset]
    if actual == expected:
        log(f"[validate] {dataset}: row count OK ({actual})")
    else:
        source_expected = EXPECTED_SOURCE_COUNTS[dataset]
        log(
            f"[warning] {dataset}: expected {expected} valid rows, got "
            f"{actual} (pinned source has {source_expected} rows). "
            "Continuing, but inspect the source."
        )


def raw_path(root: Path, source: HubFile) -> Path:
    return root / "raw" / source.local_subdir / source.filename


def download_one(
    root: Path,
    source: HubFile,
    *,
    offline: bool,
    force_download: bool,
) -> Path:
    expected_path = raw_path(root, source)

    if offline:
        if not expected_path.exists():
            raise FileNotFoundError(
                f"Offline mode: missing raw file {expected_path}"
            )
        log(f"[download] offline: using {expected_path}")
        return expected_path

    log(
        f"[download] {source.repo_id}@{source.revision[:8]} "
        f":: {source.filename}"
    )

    downloaded = hf_hub_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        revision=source.revision,
        filename=source.filename,
        local_dir=root / "raw" / source.local_subdir,
        force_download=force_download,
    )

    path = Path(downloaded)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def download_all(
    root: Path,
    *,
    offline: bool,
    force_download: bool,
) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    for dataset in DATASET_ORDER:
        result[dataset] = [
            download_one(
                root,
                source,
                offline=offline,
                force_download=force_download,
            )
            for source in SOURCE_FILES[dataset]
        ]
    return result


def parquet_rows(
    path: Path,
    columns: Sequence[str],
    *,
    batch_size: int = 1024,
) -> Iterator[Dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    missing = set(columns) - available

    if missing:
        raise KeyError(
            f"{path} missing columns {sorted(missing)}; "
            f"available={sorted(available)}"
        )

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=list(columns),
    ):
        yield from batch.to_pylist()


def normalize_govreport(root: Path, source_path: Path) -> Path:
    out = root / "normalized" / "govreport_test.jsonl"

    def rows() -> Iterator[Dict[str, Any]]:
        for item in parquet_rows(
            source_path,
            ("id", "document", "summary"),
        ):
            source_id = clean_text(item["id"])
            document = clean_text(item["document"])
            reference = clean_text(item["summary"])
            row_id = f"govreport_{source_id}"
            if not keep_valid_row(
                "govreport", document, reference, row_id
            ):
                continue
            yield {
                "id": row_id,
                "dataset": "govreport",
                "source_split": "test",
                "document": document,
                "reference": reference,
            }

    count = write_jsonl(out, rows())
    expected_count_check("govreport", count)
    log(f"[normalize] GovReport -> {out}")
    return out


def normalize_multinews(
    root: Path,
    src_path: Path,
    tgt_path: Path,
) -> Path:
    out = root / "normalized" / "multinews_test.jsonl"

    with src_path.open("r", encoding="utf-8") as src_handle:
        sources = src_handle.readlines()
    with tgt_path.open("r", encoding="utf-8") as tgt_handle:
        targets = tgt_handle.readlines()

    if len(sources) != len(targets):
        raise RuntimeError(
            f"Multi-News source/target mismatch: "
            f"{len(sources)} vs {len(targets)}"
        )

    def rows() -> Iterator[Dict[str, Any]]:
        for i, (document, reference) in enumerate(zip(sources, targets)):
            # Reproduce official loader behavior.
            document = document.strip().replace("NEWLINE_CHAR", "\n")
            reference = reference.strip()
            row_id = f"multinews_{i}"
            if not keep_valid_row(
                "multinews", document, reference, row_id
            ):
                continue
            yield {
                "id": row_id,
                "dataset": "multinews",
                "source_split": "test",
                "document": document,
                "reference": reference,
            }

    count = write_jsonl(out, rows())
    expected_count_check("multinews", count)
    log(f"[normalize] Multi-News -> {out}")
    return out


def normalize_cnn_dailymail(root: Path, source_path: Path) -> Path:
    out = root / "normalized" / "cnn_dailymail_test.jsonl"

    def rows() -> Iterator[Dict[str, Any]]:
        for item in parquet_rows(
            source_path,
            ("article", "highlights", "id"),
        ):
            source_id = clean_text(item["id"])
            document = clean_text(item["article"])
            reference = clean_text(item["highlights"])
            row_id = f"cnn_dailymail_{source_id}"
            if not keep_valid_row(
                "cnn_dailymail", document, reference, row_id
            ):
                continue
            yield {
                "id": row_id,
                "dataset": "cnn_dailymail",
                "source_split": "test",
                "document": document,
                "reference": reference,
            }

    count = write_jsonl(out, rows())
    expected_count_check("cnn_dailymail", count)
    log(f"[normalize] CNN/DailyMail -> {out}")
    return out


def normalize_xsum(root: Path, source_path: Path) -> Path:
    out = root / "normalized" / "xsum_test.jsonl"

    def rows() -> Iterator[Dict[str, Any]]:
        for item in parquet_rows(
            source_path,
            ("document", "summary", "id"),
        ):
            source_id = clean_text(item["id"])
            document = clean_text(item["document"])
            reference = clean_text(item["summary"])
            row_id = f"xsum_{source_id}"
            if not keep_valid_row(
                "xsum", document, reference, row_id
            ):
                continue
            yield {
                "id": row_id,
                "dataset": "xsum",
                "source_split": "test",
                "document": document,
                "reference": reference,
            }

    count = write_jsonl(out, rows())
    expected_count_check("xsum", count)
    log(f"[normalize] XSum -> {out}")
    return out


def normalize_all(
    root: Path,
    raw_files: Mapping[str, Sequence[Path]],
) -> Dict[str, Path]:
    return {
        "govreport": normalize_govreport(
            root, raw_files["govreport"][0]
        ),
        "multinews": normalize_multinews(
            root,
            raw_files["multinews"][0],
            raw_files["multinews"][1],
        ),
        "cnn_dailymail": normalize_cnn_dailymail(
            root, raw_files["cnn_dailymail"][0]
        ),
        "xsum": normalize_xsum(
            root, raw_files["xsum"][0]
        ),
    }


def validate_normalized_files(
    normalized_files: Mapping[str, Path],
) -> Dict[str, int]:
    required = {
        "id",
        "dataset",
        "source_split",
        "document",
        "reference",
    }
    counts: Dict[str, int] = {}

    for dataset in DATASET_ORDER:
        seen_ids = set()
        count = 0

        for row in iter_jsonl(normalized_files[dataset]):
            missing = required - set(row)
            if missing:
                raise KeyError(
                    f"{normalized_files[dataset]} missing {sorted(missing)}"
                )
            if row["dataset"] != dataset:
                raise ValueError(
                    f"Wrong dataset label in {normalized_files[dataset]}"
                )
            if row["source_split"] != "test":
                raise ValueError("Only test rows are expected")
            if not row["document"] or not row["reference"]:
                raise ValueError(f"Empty text for {row['id']}")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate id {row['id']}")
            seen_ids.add(row["id"])
            count += 1

        counts[dataset] = count
        expected_count_check(dataset, count)

    return counts


def evenly_spaced_indices(
    population_size: int,
    sample_size: int,
) -> List[int]:
    if population_size < sample_size:
        raise ValueError(
            f"Cannot sample {sample_size} from {population_size}"
        )
    if sample_size == 1:
        return [population_size // 2]
    return [
        round(i * (population_size - 1) / (sample_size - 1))
        for i in range(sample_size)
    ]


def attach_token_lengths(
    rows: Sequence[Dict[str, Any]],
    tokenizer,
    *,
    batch_size: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        documents = [row["document"] for row in batch_rows]

        encoded = tokenizer(
            documents,
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )
        lengths = encoded["length"]

        for row, length in zip(batch_rows, lengths):
            copied = dict(row)
            copied["qwen_source_tokens"] = int(length)
            output.append(copied)

    return output


def choose_debug_examples(
    rows: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    n: int,
    min_tokens: int,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if min_tokens
        <= int(row["qwen_source_tokens"])
        <= max_tokens
    ]

    candidates.sort(
        key=lambda row: (
            int(row["qwen_source_tokens"]),
            str(row["id"]),
        )
    )

    if len(candidates) < n:
        shortest = sorted(
            int(row["qwen_source_tokens"]) for row in rows
        )[:5]
        raise RuntimeError(
            f"{dataset}: only {len(candidates)} examples are in "
            f"[{min_tokens}, {max_tokens}] tokens but {n} are required. "
            f"Shortest lengths={shortest}. Increase --max-debug-tokens "
            "or reduce --samples-per-dataset. No source is truncated."
        )

    return [
        candidates[i]
        for i in evenly_spaced_indices(len(candidates), n)
    ]


def build_debug_sets(
    root: Path,
    normalized_files: Mapping[str, Path],
    *,
    tokenizer_name: str,
    local_tokenizer_only: bool,
    samples_per_dataset: int,
    min_debug_tokens: int,
    max_debug_tokens: int,
    tokenizer_batch_size: int,
) -> Dict[str, Any]:
    log(f"[tokenizer] loading {tokenizer_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=local_tokenizer_only,
        trust_remote_code=True,
        use_fast=True,
    )

    debug_rows: List[Dict[str, Any]] = []
    smoke_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}

    for dataset in DATASET_ORDER:
        log(f"[debug] tokenizing {dataset} sources...")

        rows = read_jsonl(normalized_files[dataset])
        tokenized = attach_token_lengths(
            rows,
            tokenizer,
            batch_size=tokenizer_batch_size,
        )

        selected = choose_debug_examples(
            tokenized,
            dataset=dataset,
            n=samples_per_dataset,
            min_tokens=min_debug_tokens,
            max_tokens=max_debug_tokens,
        )

        selected_lengths = [
            int(row["qwen_source_tokens"]) for row in selected
        ]
        all_lengths = sorted(
            int(row["qwen_source_tokens"]) for row in tokenized
        )

        debug_rows.extend(selected)
        smoke_rows.append(selected[len(selected) // 2])

        stats[dataset] = {
            "test_rows": len(rows),
            "min_source_tokens": all_lengths[0],
            "median_source_tokens": all_lengths[len(all_lengths) // 2],
            "max_source_tokens": all_lengths[-1],
            "selected_source_tokens": selected_lengths,
        }

        log(
            f"[debug] {dataset:14s}: "
            f"selected lengths={selected_lengths}"
        )

    debug_path = root / "debug" / "debug_real.jsonl"
    smoke_path = root / "debug" / "smoke_real.jsonl"

    debug_count = write_jsonl(debug_path, debug_rows)
    smoke_count = write_jsonl(smoke_path, smoke_rows)

    expected_debug = samples_per_dataset * len(DATASET_ORDER)
    if debug_count != expected_debug:
        raise RuntimeError(
            f"Expected {expected_debug} debug rows, got {debug_count}"
        )
    if smoke_count != len(DATASET_ORDER):
        raise RuntimeError(
            f"Expected {len(DATASET_ORDER)} smoke rows, got {smoke_count}"
        )

    log(f"[debug] wrote {debug_count} rows -> {debug_path}")
    log(f"[smoke] wrote {smoke_count} rows -> {smoke_path}")

    return {
        "tokenizer": tokenizer_name,
        "min_debug_tokens": min_debug_tokens,
        "max_debug_tokens": max_debug_tokens,
        "samples_per_dataset": samples_per_dataset,
        "debug_count": debug_count,
        "smoke_count": smoke_count,
        "dataset_token_stats": stats,
    }


def build_manifest(
    root: Path,
    raw_files: Mapping[str, Sequence[Path]],
    normalized_files: Mapping[str, Path],
    *,
    normalized_counts: Mapping[str, int],
    debug_info: Mapping[str, Any],
    elapsed_seconds: float,
) -> Path:
    sources: Dict[str, Any] = {}

    for dataset in DATASET_ORDER:
        entries = []
        for source, path in zip(
            SOURCE_FILES[dataset],
            raw_files[dataset],
        ):
            entries.append(
                {
                    "repo_id": source.repo_id,
                    "revision": source.revision,
                    "filename": source.filename,
                    "local_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "expected_source_rows": EXPECTED_SOURCE_COUNTS[dataset],
                    "expected_normalized_rows": (
                        EXPECTED_NORMALIZED_COUNTS[dataset]
                    ),
                }
            )
        sources[dataset] = entries

    normalized = {}
    for dataset, path in normalized_files.items():
        normalized[dataset] = {
            "path": str(path),
            "rows": normalized_counts[dataset],
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    debug_path = root / "debug" / "debug_real.jsonl"
    smoke_path = root / "debug" / "smoke_real.jsonl"

    manifest = {
        "schema_version": 1,
        "created_unix_time": time.time(),
        "elapsed_seconds": elapsed_seconds,
        "sources": sources,
        "normalized": normalized,
        "debug": {
            **dict(debug_info),
            "debug_path": str(debug_path),
            "debug_sha256": sha256_file(debug_path),
            "smoke_path": str(smoke_path),
            "smoke_sha256": sha256_file(smoke_path),
        },
    }

    path = root / "build_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"[manifest] {path}")
    return path


def print_tree(root: Path) -> None:
    log("\n[done] data files:")
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        size_mb = path.stat().st_size / (1024 ** 2)
        log(f"  {relative} ({size_mb:.2f} MB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download, normalize, validate, and build deterministic "
            "Qwen-tokenized debug subsets for four summarization datasets."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Dedicated output root. The default is data/rebuild so the "
            "existing data/raw, data/normalized and data/debug are preserved."
        ),
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples-per-dataset", type=int, default=4)
    parser.add_argument("--min-debug-tokens", type=int, default=256)
    parser.add_argument("--max-debug-tokens", type=int, default=4096)
    parser.add_argument("--tokenizer-batch-size", type=int, default=8)

    parser.add_argument(
        "--local-tokenizer-only",
        action="store_true",
        help="Require the tokenizer to already be cached locally.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip network access; require all raw files under <root>/raw.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of pinned raw source files.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset must be > 0")
    if args.min_debug_tokens <= 0:
        raise ValueError("--min-debug-tokens must be > 0")
    if args.max_debug_tokens < args.min_debug_tokens:
        raise ValueError(
            "--max-debug-tokens must be >= --min-debug-tokens"
        )
    if args.tokenizer_batch_size <= 0:
        raise ValueError("--tokenizer-batch-size must be > 0")

    if args.root.resolve() == Path("data").resolve():
        raise ValueError(
            "Refusing to use the legacy data/ root directly. Choose a "
            "dedicated root such as data/rebuild."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    started = time.perf_counter()
    root = args.root.resolve()

    log("=" * 72)
    log("Rebuilding summarization benchmark data")
    log(f"root      : {root}")
    log(f"tokenizer : {args.tokenizer}")
    log(
        f"debug     : {args.samples_per_dataset}/dataset, "
        f"{args.min_debug_tokens}-{args.max_debug_tokens} source tokens"
    )
    log("=" * 72)

    ensure_dirs(root)

    # 1. Download only the required test files.
    raw_files = download_all(
        root,
        offline=args.offline,
        force_download=args.force_download,
    )

    # 2. Normalize all four datasets to the same JSONL schema.
    normalized_files = normalize_all(root, raw_files)

    # 3. Validate schema, row counts, IDs, and non-empty source/reference.
    normalized_counts = validate_normalized_files(normalized_files)

    # 4. Build deterministic smoke/debug sets using exact Qwen token lengths.
    debug_info = build_debug_sets(
        root,
        normalized_files,
        tokenizer_name=args.tokenizer,
        local_tokenizer_only=args.local_tokenizer_only,
        samples_per_dataset=args.samples_per_dataset,
        min_debug_tokens=args.min_debug_tokens,
        max_debug_tokens=args.max_debug_tokens,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )

    # 5. Hash inputs/outputs so another machine can verify the same build.
    elapsed = time.perf_counter() - started
    build_manifest(
        root,
        raw_files,
        normalized_files,
        normalized_counts=normalized_counts,
        debug_info=debug_info,
        elapsed_seconds=elapsed,
    )

    print_tree(root)
    log(f"\n[done] completed in {elapsed:.1f}s")
    log(
        "[next] smoke input: "
        f"{root / 'debug' / 'smoke_real.jsonl'}"
    )


if __name__ == "__main__":
    main()
