"""Chuẩn hóa ArXiv/document records thành prompt-only canonical JSONL."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from _common import canonical_prompt, content_of, read_jsonl, stable_id, write_jsonl


def _document(row: Dict[str, Any]) -> str:
    for key in ("text", "article", "document", "content", "body"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Normalize ArXiv prompt data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
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

    def rows():
        emitted = 0
        for index, row in enumerate(read_jsonl(args.input)):
            if args.limit is not None and emitted >= args.limit:
                break
            document = _document(row).strip()
            if not document:
                continue
            raw_id = row.get("id") or row.get("paper_id") or row.get("arxiv_id") or stable_id("arxiv", index)
            sample_id = str(raw_id) if str(raw_id).startswith("arxiv_") else f"arxiv_{raw_id}"
            prompt = args.prompt_prefix + document
            metadata = {
                "source_index": index,
                "source_length": len(document),
            }
            if tokenizer is not None:
                metadata["source_token_length"] = len(
                    tokenizer(document, add_special_tokens=False)["input_ids"]
                )
            yield canonical_prompt(
                sample_id,
                "arxiv",
                [{"role": "user", "content": prompt}],
                **metadata,
            )
            emitted += 1

    count = write_jsonl(args.output, rows())
    print(f"[prepare_arxiv] wrote={count} output={args.output}")


if __name__ == "__main__":
    main()
