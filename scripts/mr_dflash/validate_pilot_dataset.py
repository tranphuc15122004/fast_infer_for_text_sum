"""Validate canonical regenerated pilot data và split invariants."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from _common import read_jsonl


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Validate MR-DFlash pilot JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--expected-target-model", default=None)
    parser.add_argument("--require-generated", action="store_true")
    args = parser.parse_args(argv)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    seen = set()
    counts = Counter()
    strata = Counter()
    valid = 0
    for index, row in enumerate(read_jsonl(args.input), 1):
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen:
            raise ValueError(f"row {index}: id rỗng/trùng {sample_id!r}")
        seen.add(sample_id)
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"row {index}: conversations không hợp lệ")
        roles = [str(m.get("role", "")).lower() for m in conversations if isinstance(m, dict)]
        if roles[-1:] != ["assistant"]:
            raise ValueError(f"row {index}: assistant cuối bị thiếu")
        if any(role == "assistant" for role in roles[:-1]):
            raise ValueError(f"row {index}: có assistant trajectory cũ trước response cuối")
        assistant = str(conversations[-1].get("content", "")).strip()
        if not assistant:
            raise ValueError(f"row {index}: assistant rỗng")
        metadata = row.get("metadata") or {}
        if args.require_generated and not metadata.get("generation_model"):
            raise ValueError(f"row {index}: thiếu metadata.generation_model")
        if args.expected_target_model and metadata.get("generation_model") != args.expected_target_model:
            raise ValueError(f"row {index}: generation model mismatch")
        source = str(row.get("source", "unknown"))
        counts[source] += 1
        source_length = int(metadata.get("source_length", 0))
        strata[min(source_length // 2048, 3)] += 1
        if tokenizer is not None:
            from MR_DFlash.data import render_conversation
            ids, mask = render_conversation(conversations, tokenizer, 10**9, supervision_mode="last_assistant")
            if len(ids) > args.max_length or sum(mask) < 2:
                raise ValueError(f"row {index}: tokenized length/mask không hợp lệ (len={len(ids)})")
        valid += 1
    print(f"[validate_pilot_dataset] valid={valid} sources={dict(counts)} strata={dict(strata)}")


if __name__ == "__main__":
    main()
