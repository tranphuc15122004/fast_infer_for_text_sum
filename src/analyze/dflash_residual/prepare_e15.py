"""Prepare a small, reproducible task-adaptation set for E15.

The adaptation is deliberately a diagnostic: it keeps the original DFlash
architecture and objective and changes only the teacher-forced data
distribution.  Source documents are truncated from the right to match the
1K-context regime used by E11/E14; reference summaries are kept as assistant
targets so the MR-DFlash capture code can build the original loss mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: str, limit: int | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result.append(json.loads(line))
        if limit is not None and len(result) >= limit:
            break
    return result


def _truncate(tokenizer: Any, text: str, max_tokens: int) -> str:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    ids = list(encoded["input_ids"][:max_tokens])
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_dataset(
    inputs: list[str],
    output: str,
    *,
    tokenizer_path: str,
    max_samples_per_input: int | None,
    max_source_tokens: int = 760,
    max_summary_tokens: int = 180,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for input_path in inputs:
        source_rows = _rows(input_path, max_samples_per_input)
        source_counts[input_path] = len(source_rows)
        for row in source_rows:
            document = _truncate(tokenizer, str(row.get("document", "")), max_source_tokens)
            reference = _truncate(tokenizer, str(row.get("reference", "")), max_summary_tokens)
            if not document.strip() or not reference.strip():
                continue
            records.append({
                "id": str(row.get("id", len(records))),
                "dataset": str(row.get("dataset", "summarization")),
                "conversations": [
                    {"role": "user", "content": document},
                    {"role": "assistant", "content": reference},
                ],
            })
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "experiment": "E15",
        "purpose": "minimal_summarization_adaptation",
        "inputs": inputs,
        "source_counts": source_counts,
        "records": len(records),
        "tokenizer": tokenizer_path,
        "max_source_tokens": max_source_tokens,
        "max_summary_tokens": max_summary_tokens,
        "truncation": "right",
        "status": "ok",
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-samples-per-input", type=int, default=None)
    parser.add_argument("--max-source-tokens", type=int, default=760)
    parser.add_argument("--max-summary-tokens", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(build_dataset(
        args.input,
        args.output,
        tokenizer_path=args.tokenizer,
        max_samples_per_input=args.max_samples_per_input,
        max_source_tokens=args.max_source_tokens,
        max_summary_tokens=args.max_summary_tokens,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
