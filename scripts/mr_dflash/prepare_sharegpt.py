"""Chuẩn hóa ShareGPT thành prompt-only canonical JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from _common import (
    append_jsonl_durable,
    canonical_prompt,
    content_of,
    normalize_role,
    read_records,
    stable_id,
)
from progress import ProgressReporter, install_exception_hook


def _last_user_index(raw: Any) -> int | None:
    if not isinstance(raw, list):
        return None
    last_user_index = None
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        if normalize_role(item.get("role", item.get("from", ""))) == "user" and content_of(item).strip():
            last_user_index = index
    return last_user_index


def _messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = row.get("conversations", row.get("messages", []))
    last_user_index = _last_user_index(raw)
    if last_user_index is None:
        return []
    # Prompt-only means không đưa target assistant cuối vào input, không có
    # nghĩa là loại bỏ assistant history. Giữ toàn bộ context trước user cuối
    # để model thấy đúng cuộc hội thoại ShareGPT; cắt mọi message sau user cuối
    # vì đó là response đích cũ (nếu raw record có response ở cuối).
    result: List[Dict[str, str]] = []
    allowed_roles = {"system", "user", "assistant", "tool"}
    for item in raw[: last_user_index + 1]:
        if not isinstance(item, dict):
            continue
        role = normalize_role(item.get("role", item.get("from", "")))
        text = content_of(item).strip()
        if role in allowed_roles and text:
            result.append({"role": role, "content": text})
    if not result or result[-1]["role"] != "user":
        return []
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

    reporter = ProgressReporter(None)
    if "MR_DFLASH_PROGRESS_TOTAL_SAMPLES" not in os.environ:
        source_total = (
            int(args.limit)
            if args.limit is not None
            else sum(1 for _ in read_records(args.input))
        )
        reporter.set_context(total_samples=source_total)
    previous_hook = install_exception_hook(reporter)

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
    progress_base = reporter.completed_samples() + len(existing)
    reporter.update(
        "starting",
        input=str(args.input),
        output=str(args.output),
        completed_samples=progress_base,
    )

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
            raw_messages = row.get("conversations", row.get("messages", []))
            target_source_turn_index = _last_user_index(raw_messages)
            if target_source_turn_index is None:
                continue
            source_length = sum(
                len(content_of(item).strip())
                for item in raw_messages[: target_source_turn_index + 1]
                if isinstance(item, dict)
            )
            yield canonical_prompt(
                sample_id,
                "sharegpt",
                messages,
                source_index=index,
                original_turn_count=len(raw_messages) if isinstance(raw_messages, list) else 0,
                retained_turn_count=len(messages),
                target_source_turn_index=target_source_turn_index,
                source_length=source_length,
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
    for row in tqdm(rows(), total=args.limit, desc="Normalize ShareGPT", unit="sample"):
        pending.append(row)
        count += 1
        reporter.update(
            "processing",
            completed_samples=progress_base + count,
            sample_id=str(row.get("id", "")),
        )
        if len(pending) >= 256:
            append_jsonl_durable(output_path, pending)
            pending.clear()
    if pending:
        append_jsonl_durable(output_path, pending)
    if not output_path.exists():
        output_path.touch()
    print(f"[prepare_sharegpt] wrote={count} output={args.output}")
    reporter.update(
        "done",
        completed_samples=progress_base + count,
        written=count,
    )
    sys.excepthook = previous_hook


if __name__ == "__main__":
    main()
