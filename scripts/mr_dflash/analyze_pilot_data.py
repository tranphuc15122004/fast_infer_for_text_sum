"""Phân tích một mẫu nhỏ của artifact MR-DFlash trước khi scale.

Script không sửa input. Mặc định chỉ đọc tối đa 1.000 dòng đầu để người dùng
spot-check schema, source ratio, độ dài và reference ArXiv trước khi cho phép
chạy toàn bộ dữ liệu server.
"""

from __future__ import annotations

import argparse
from itertools import islice
import statistics
import sys
from collections import Counter
from typing import Any, Dict, List

from _common import read_jsonl, write_json
from progress import ProgressReporter, install_exception_hook


def _last_message(row: Dict[str, Any], role: str) -> str:
    for message in reversed(row.get("conversations") or []):
        if isinstance(message, dict) and str(message.get("role", "")).lower() == role:
            return str(message.get("content", ""))
    return ""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Analyze a small MR-DFlash data sample")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.limit < 1:
        raise ValueError("--limit phải >= 1")

    input_total = sum(1 for _ in read_jsonl(args.input))
    total = min(input_total, int(args.limit))
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

    rows_to_process = islice(read_jsonl(args.input), int(args.limit))
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
        if length is not None:
            lengths.append(int(length))
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

    report: Dict[str, Any] = {
        "schema_version": "mr_dflash_analysis_v1",
        "input": str(args.input),
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "duplicate_ids": duplicate_ids,
        "assistant_rows": assistant_rows,
        "prompt_only_rows": prompt_only_rows,
        "reference_rows": reference_rows,
        "length_metric": "source_token_length when available, otherwise source_length",
        "length_stats": {
            "count": len(lengths),
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": round(statistics.fmean(lengths), 2) if lengths else None,
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
