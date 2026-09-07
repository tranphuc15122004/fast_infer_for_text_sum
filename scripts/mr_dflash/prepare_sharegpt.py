"""Chuẩn hóa ShareGPT thành prompt-only canonical JSONL."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from _common import canonical_prompt, content_of, read_jsonl, stable_id, write_jsonl


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
    if last_user and last_user.strip():
        result.append({"role": "user", "content": last_user})
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Normalize ShareGPT prompt data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    def rows():
        emitted = 0
        for index, row in enumerate(read_jsonl(args.input)):
            if args.limit is not None and emitted >= args.limit:
                break
            messages = _messages(row)
            if not messages:
                continue
            raw_id = row.get("id") or row.get("conversation_id") or stable_id("sharegpt", index)
            sample_id = str(raw_id) if str(raw_id).startswith("sharegpt_") else f"sharegpt_{raw_id}"
            yield canonical_prompt(
                sample_id,
                "sharegpt",
                messages,
                source_index=index,
                original_turn_count=len(row.get("conversations", row.get("messages", [])) or []),
            )
            emitted += 1

    count = write_jsonl(args.output, rows())
    print(f"[prepare_sharegpt] wrote={count} output={args.output}")


if __name__ == "__main__":
    main()
