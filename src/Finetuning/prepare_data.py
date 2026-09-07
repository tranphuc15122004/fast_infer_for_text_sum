"""Prepare local document/summary JSONL for offline feature capture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import SummaryRecord, load_summary_jsonl, render_summary_example


def prepare_summary_examples(
    path: str | Path,
    tokenizer: Any,
    *,
    max_length: int,
    chat_template: str = "qwen3",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    records = load_summary_jsonl(path, max_samples=max_samples)
    examples: list[dict[str, Any]] = []
    rejected: list[str] = []
    for record in records:
        try:
            rendered = render_summary_example(
                record,
                tokenizer,
                max_length=max_length,
                chat_template=chat_template,
            )
        except ValueError as exc:
            rejected.append(f"{record.id}: {exc}")
            continue
        examples.append({**rendered, "id": record.id})
    if not examples:
        suffix = f"; rejected={rejected[:3]}" if rejected else ""
        raise ValueError(f"no trainable summary examples after rendering{suffix}")
    return examples


__all__ = ["prepare_summary_examples"]
