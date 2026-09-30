"""Validate canonical regenerated pilot data và split invariants."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from _common import read_jsonl, write_json
from progress import ProgressReporter, install_exception_hook


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Validate MR-DFlash pilot JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="chỉ đọc tokenizer từ snapshot local (mặc định bật trên server)",
    )
    parser.add_argument("--expected-target-model", default=None)
    parser.add_argument("--require-generated", action="store_true")
    parser.add_argument(
        "--invalid-sample-policy",
        choices=("error", "skip"),
        default="error",
        help="skip target không đủ điều kiện DFlash và ghi ID/lý do vào report",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="ghi báo cáo JSON để pipeline/CI đọc lại sau khi validate",
    )
    args = parser.parse_args(argv)
    total = sum(1 for _ in read_jsonl(args.input))
    reporter = ProgressReporter(None)
    reporter.set_context(total_samples=total)
    previous_hook = install_exception_hook(reporter)
    reporter.update(
        "starting",
        input=str(args.input),
        report=str(args.report) if args.report else None,
        completed_samples=reporter.completed_samples(),
    )
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            local_files_only=bool(args.local_files_only),
        )
    from MR_DFlash.data import has_consecutive_supervised_tokens, render_conversation

    seen = set()
    counts = Counter()
    strata = Counter()
    valid = 0
    processed = 0
    skipped_samples: List[Dict[str, Any]] = []
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
        assistant = str(conversations[-1].get("content", "")).strip()
        if not assistant:
            raise ValueError(f"row {index}: assistant rỗng")
        metadata = row.get("metadata") or {}
        if args.require_generated and not metadata.get("generation_model"):
            raise ValueError(f"row {index}: thiếu metadata.generation_model")
        if args.expected_target_model and metadata.get("generation_model") != args.expected_target_model:
            raise ValueError(f"row {index}: generation model mismatch")
        source = str(row.get("source", "unknown"))
        source_length = int(metadata.get("source_length", 0))
        invalid_reason = None
        tokenized_length = None
        supervised_tokens = None
        if tokenizer is not None:
            try:
                ids, mask = render_conversation(
                    conversations,
                    tokenizer,
                    10**9,
                    supervision_mode="last_assistant",
                )
            except (TypeError, ValueError, IndexError, KeyError) as exc:
                invalid_reason = f"render_error:{type(exc).__name__}:{exc}"
            else:
                tokenized_length = len(ids)
                supervised_tokens = int(sum(mask))
                if tokenized_length > args.max_length:
                    invalid_reason = "tokenized_length_exceeds_max_length"
                elif tokenized_length < 3:
                    invalid_reason = "tokenized_sequence_too_short"
                elif not has_consecutive_supervised_tokens(mask):
                    invalid_reason = "no_consecutive_supervised_target_tokens"

        if invalid_reason:
            if args.invalid_sample_policy == "error":
                raise ValueError(
                    f"row {index} id={sample_id!r}: {invalid_reason} "
                    f"(len={tokenized_length}, supervised={supervised_tokens})"
                )
            skipped_samples.append(
                {
                    "id": sample_id,
                    "row_index": int(index),
                    "kind": "invalid_training_sample",
                    "reason": invalid_reason,
                    "tokenized_length": tokenized_length,
                    "supervised_tokens": supervised_tokens,
                }
            )
            processed += 1
            reporter.update(
                "sample_skipped",
                completed_samples=processed,
                sample_id=sample_id,
                skip_reason=invalid_reason,
                valid_samples=valid,
            )
            continue

        counts[source] += 1
        strata[min(source_length // 2048, 3)] += 1
        valid += 1
        processed += 1
        reporter.update(
            "processing",
            completed_samples=processed,
            sample_id=sample_id,
            row_index=int(index - 1),
        )
    report = {
        "schema_version": "mr_dflash_validation_v1",
        "input": str(args.input),
        "max_length": int(args.max_length),
        "expected_target_model": args.expected_target_model,
        "require_generated": bool(args.require_generated),
        "valid": int(valid),
        "skipped": int(len(skipped_samples)),
        "skipped_by_reason": dict(
            sorted(Counter(str(item["reason"]) for item in skipped_samples).items())
        ),
        "skipped_samples": skipped_samples,
        "invalid_sample_policy": args.invalid_sample_policy,
        "unique_ids": int(len(seen)),
        "source_counts": dict(sorted(counts.items())),
        "length_strata": {str(key): int(value) for key, value in sorted(strata.items())},
    }
    if args.report:
        write_json(args.report, report)
    print(
        f"[validate_pilot_dataset] valid={valid} skipped={len(skipped_samples)} "
        f"skip_reasons={report['skipped_by_reason']} sources={dict(counts)} "
        f"strata={dict(strata)}"
    )
    reporter.update(
        "done",
        completed_samples=total,
        valid=valid,
        skipped=len(skipped_samples),
    )
    sys.excepthook = previous_hook


if __name__ == "__main__":
    main()
