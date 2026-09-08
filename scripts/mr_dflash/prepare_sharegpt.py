"""Chuẩn hóa ShareGPT thành prompt-only canonical JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from _common import (
    append_jsonl_durable,
    canonical_prompt,
    content_of,
    read_records,
    stable_id,
)


def _messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = row.get("conversations", row.get("messages", []))
    if not isinstance(raw, list):
        return []
    result: List[Dict[str, str]] = []
    # Prompt-only semantics: keep system messages and the final user turn;
    # original assistant answers are deliberately never used as supervision.
    last_user = None
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", item.get("from", ""))).strip().lower()
        if role in {"human", "user"}:
            last_user = content_of(item)
        elif role == "system":
            text = content_of(item).strip()
            if text:
                result.append({"role": "system", "content": text})
    # Một số record ShareGPT chỉ có system/metadata hoặc role không nằm trong
    # schema. Đây không phải prompt hợp lệ cho target regeneration; bỏ qua
    # record thay vì để canonical_prompt dừng toàn bộ phase prepare.
    if not last_user or not last_user.strip():
        return []
    result.append({"role": "user", "content": last_user})
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Normalize ShareGPT prompt data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append vào output hiện có và bỏ qua id đã chuẩn hóa",
    )
    args = parser.parse_args(argv)

    existing = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        raw = output_path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        valid_lines = []
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                valid_lines.append(line)
                continue
            try:
                existing.add(str(json.loads(line).get("id", "")))
            except json.JSONDecodeError as exc:
                if line_number != len(lines) or raw.endswith(("\n", "\r")):
                    raise ValueError(f"JSONL hỏng tại {output_path}:{line_number}") from exc
                temporary = output_path.with_name(f".{output_path.name}.repair.tmp")
                repaired = "\n".join(valid_lines)
                if repaired:
                    repaired += "\n"
                temporary.write_text(repaired, encoding="utf-8")
                os.replace(temporary, output_path)
                break
            valid_lines.append(line)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def rows():
        emitted = len(existing)
        for index, row in enumerate(read_records(args.input)):
            if args.limit is not None and emitted >= args.limit:
                break
            messages = _messages(row)
            if not messages:
                continue
            raw_id = row.get("id") or row.get("conversation_id") or stable_id("sharegpt", index)
            sample_id = str(raw_id) if str(raw_id).startswith("sharegpt_") else f"sharegpt_{raw_id}"
            if sample_id in existing:
                continue
            yield canonical_prompt(
                sample_id,
                "sharegpt",
                messages,
                source_index=index,
                original_turn_count=len(row.get("conversations", row.get("messages", [])) or []),
            )
            emitted += 1

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm có trong requirements server
        tqdm = lambda iterator, **_kwargs: iterator
    if not args.resume:
        output_path.unlink(missing_ok=True)
    count = 0
    pending = []
    for row in tqdm(rows(), desc="Normalize ShareGPT", unit="row"):
        pending.append(row)
        count += 1
        if len(pending) >= 256:
            append_jsonl_durable(output_path, pending)
            pending.clear()
    if pending:
        append_jsonl_durable(output_path, pending)
    if not output_path.exists():
        output_path.touch()
    print(f"[prepare_sharegpt] wrote={count} output={args.output}")


if __name__ == "__main__":
    main()
