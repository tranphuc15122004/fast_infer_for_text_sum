"""Chuẩn hóa ArXiv/document records thành prompt-only canonical JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from _common import canonical_prompt, join_text, read_jsonl, stable_id


def _document(row: Dict[str, Any]) -> str:
    for key in ("text", "article", "document", "content", "body"):
        value = row.get(key)
        if value:
            return join_text(value)
    return ""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Normalize ArXiv prompt data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append vào output hiện có và bỏ qua id đã chuẩn hóa",
    )
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer local để ghi source_token_length")
    parser.add_argument(
        "--prompt-prefix",
        default="Summarize the following scientific document:\n\n",
    )
    args = parser.parse_args(argv)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    output_path = Path(args.output)
    existing = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing.add(str(json.loads(line).get("id", "")))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def rows():
        emitted = len(existing)
        for index, row in enumerate(read_jsonl(args.input)):
            if args.limit is not None and emitted >= args.limit:
                break
            document = _document(row).strip()
            if not document:
                continue
            raw_id = row.get("id") or row.get("paper_id") or row.get("arxiv_id") or stable_id("arxiv", index)
            sample_id = str(raw_id) if str(raw_id).startswith("arxiv_") else f"arxiv_{raw_id}"
            if sample_id in existing:
                continue
            prompt = args.prompt_prefix + document
            metadata = {
                "source_index": index,
                "source_length": len(document),
                "source_length_chars": len(document),
                "source_length_unit": "chars",
            }
            reference_summary = join_text(row.get("summary"))
            if reference_summary:
                metadata["reference_summary"] = reference_summary
            if "label" in row:
                metadata["label"] = row["label"]
            if tokenizer is not None:
                metadata["source_token_length"] = len(
                    tokenizer(document, add_special_tokens=False)["input_ids"]
                )
                # source_length luôn là chars để giữ schema ổn định; field
                # source_token_length mới là metric dùng cho length bucket.
            yield canonical_prompt(
                sample_id,
                "arxiv",
                [{"role": "user", "content": prompt}],
                **metadata,
            )
            emitted += 1

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm có trong requirements server
        tqdm = lambda iterator, **_kwargs: iterator
    mode = "a" if args.resume and output_path.exists() else "w"
    count = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for row in tqdm(rows(), desc="Normalize ArXiv", unit="row"):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    print(f"[prepare_arxiv] wrote={count} output={args.output}")


if __name__ == "__main__":
    main()
