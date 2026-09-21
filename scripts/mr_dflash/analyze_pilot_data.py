"""Phân tích artifact MR-DFlash và tạo manifest điều phối theo độ dài.

Script không sửa input. Khi ``--limit`` không được truyền, toàn bộ input được
quét để có thống kê đủ cho batch controller; có thể dùng limit cho smoke.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from typing import Any, Dict, List

from _common import read_jsonl, write_json, write_jsonl
from progress import ProgressReporter, install_exception_hook


def _last_message(row: Dict[str, Any], role: str) -> str:
    for message in reversed(row.get("conversations") or []):
        if isinstance(message, dict) and str(message.get("role", "")).lower() == role:
            return str(message.get("content", ""))
    return ""


def _conversation_length_hint(row: Dict[str, Any]) -> int:
    """Estimate full prompt length when token metadata is unavailable."""
    messages = row.get("conversations") or []
    return len(
        "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
    )


def _ends_with_user(row: Dict[str, Any]) -> bool:
    messages = row.get("conversations") or []
    if not messages or not isinstance(messages[-1], dict):
        return False
    return str(messages[-1].get("role", "")).strip().lower() == "user"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Analyze MR-DFlash data and build length manifest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--length-manifest",
        default=None,
        help="JSONL manifest sorted tăng dần theo token length; mặc định không ghi",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit phải >= 1")

    input_total = sum(1 for _ in read_jsonl(args.input))
    total = input_total if args.limit is None else min(input_total, int(args.limit))
    reporter = ProgressReporter(None)
    reporter.set_context(total_samples=total)
    previous_hook = install_exception_hook(reporter)
    reporter.update(
        "starting",
        input=str(args.input),
        output=str(args.output),
        completed_samples=reporter.completed_samples(),
    )

    rows: List[Dict[str, Any]] = []
    seen = set()
    duplicate_ids: List[str] = []
    source_counts: Counter[str] = Counter()
    lengths: List[int] = []
    reference_rows = 0
    assistant_rows = 0
    prompt_only_rows = 0
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm có trong requirements server
        tqdm = lambda iterator, **_kwargs: iterator

    rows_to_process = read_jsonl(args.input)
    if args.limit is not None:
        from itertools import islice

        rows_to_process = islice(rows_to_process, int(args.limit))
    length_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(
        tqdm(rows_to_process, total=total, desc="Analyze MR-DFlash", unit="sample")
    ):
        sample_id = str(row.get("id", ""))
        if sample_id in seen:
            duplicate_ids.append(sample_id)
        seen.add(sample_id)
        rows.append(row)
        source_counts[str(row.get("source", "unknown"))] += 1
        metadata = row.get("metadata") or {}
        length = metadata.get("source_token_length", metadata.get("source_length"))
        if length is None:
            # This fallback is intentionally only a deterministic ordering
            # hint. Exact token lengths are supplied by prepare/tokenize when
            # available; the scheduler must never treat character count as a
            # VRAM measurement. Include assistant history because it is part
            # of the prompt context for multi-turn ShareGPT rows.
            length = _conversation_length_hint(row)
        length = max(0, int(length))
        lengths.append(length)
        length_rows.append(
            {
                "sample_id": sample_id,
                "index": int(index),
                "length": length,
                "source": str(row.get("source", "unknown")),
            }
        )
        conversations = row.get("conversations") or []
        has_assistant = any(
            isinstance(message, dict)
            and str(message.get("role", "")).lower() == "assistant"
            for message in conversations
        )
        assistant_rows += int(has_assistant)
        prompt_only_rows += int(not has_assistant)
        reference_rows += int(bool(str(metadata.get("reference_summary", "")).strip()))
        reporter.update(
            "processing",
            completed_samples=len(rows),
            sample_id=sample_id,
            row_index=int(index),
        )

    quantiles: Dict[str, int | None] = {}
    if lengths:
        ordered_lengths = sorted(lengths)
        for name, fraction in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
            position = min(len(ordered_lengths) - 1, int(round((len(ordered_lengths) - 1) * fraction)))
            quantiles[name] = int(ordered_lengths[position])
    else:
        quantiles = {name: None for name in ("p50", "p90", "p95", "p99")}

    report: Dict[str, Any] = {
        "schema_version": "mr_dflash_analysis_v1",
        "input": str(args.input),
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "duplicate_ids": duplicate_ids,
        "assistant_rows": assistant_rows,
        "prompt_only_rows": prompt_only_rows,
        "assistant_history_rows": assistant_rows,
        "prompt_ready_rows": sum(1 for row in rows if _ends_with_user(row)),
        "reference_rows": reference_rows,
        "length_metric": "source_token_length when available, otherwise source_length",
        "length_stats": {
            "count": len(lengths),
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": round(statistics.fmean(lengths), 2) if lengths else None,
            "quantiles": quantiles,
        },
        "sample_preview": [
            {
                "id": str(row.get("id", "")),
                "source": str(row.get("source", "unknown")),
                "user_chars": len(_last_message(row, "user")),
                "assistant_chars": len(_last_message(row, "assistant")),
                "has_reference_summary": bool(
                    str((row.get("metadata") or {}).get("reference_summary", "")).strip()
                ),
            }
            for row in rows[:5]
        ],
    }
    write_json(args.output, report)
    if args.length_manifest:
        write_jsonl(
            args.length_manifest,
            sorted(length_rows, key=lambda value: (int(value["length"]), int(value["index"]))),
        )
    print(
        f"[analyze_pilot_data] rows={len(rows)} sources={dict(source_counts)} "
        f"output={args.output}"
    )
    reporter.update(
        "done",
        completed_samples=len(rows),
        rows=len(rows),
    )
    sys.excepthook = previous_hook


if __name__ == "__main__":
    main()
