#!/usr/bin/env python3
"""Convert existing stratified MR-DFlash prompt splits for Finetuning Phase 1.

The input split is preserved. ArXiv references are carried through from
metadata.reference_summary; ShareGPT has no human summary reference, so its
summary field is intentionally empty. No generated response is treated as gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text).strip())
            elif item:
                parts.append(str(item).strip())
        return "\n".join(part for part in parts if part)
    return ""


def _messages(row: dict[str, Any], *, path: Path, line_number: int) -> list[dict[str, str]]:
    raw_messages = row.get("conversations")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"{path}:{line_number}: missing prompt conversations")
    aliases = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant"}
    messages: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: conversation message must be an object")
        role = aliases.get(str(item.get("role", "")).strip().lower(), str(item.get("role", "")).strip().lower())
        text = _content(item.get("content", item.get("value", "")))
        if role not in {"system", "user", "assistant", "tool"} or not text:
            raise ValueError(f"{path}:{line_number}: invalid role or empty conversation content")
        messages.append({"role": role, "content": text})
    if messages[-1]["role"] == "assistant":
        # Also accept regenerated rows, although the recommended source is the
        # stratified prompt-only split so Phase 1 regenerates it itself.
        messages = messages[:-1]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"{path}:{line_number}: prompt conversations must end with user")
    return messages


def _phase1_record(
    row: dict[str, Any], *, split: str, path: Path, line_number: int
) -> dict[str, Any]:
    sample_id = str(row.get("id", "")).strip()
    source = str(row.get("source", row.get("dataset", ""))).strip().lower()
    if not sample_id:
        raise ValueError(f"{path}:{line_number}: missing id")
    if source not in {"sharegpt", "arxiv"}:
        raise ValueError(f"{path}:{line_number}: expected source sharegpt/arxiv, got {source!r}")
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}:{line_number}: metadata must be an object")
    recorded_split = str(metadata.get("split", "")).strip().lower()
    allowed_splits = {"train"} if split == "train" else {"val", "validation"}
    if recorded_split and recorded_split not in allowed_splits:
        raise ValueError(
            f"{path}:{line_number}: row split {recorded_split!r} does not match {split} input"
        )
    messages = _messages(row, path=path, line_number=line_number)
    prompt_messages = messages

    if source == "arxiv":
        # ArXiv normalization stores the paper plus its original task prefix in
        # one user message; retain it verbatim so the prompt contract can be
        # reproduced with data.prompt_template: "{document}".
        document = prompt_messages[-1]["content"]
    else:
        # Preserve all ShareGPT history as visible role-marked context.  The
        # final assistant marker makes the continuation boundary explicit.
        document = "\n\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in prompt_messages
        ) + "\n\nASSISTANT:"

    reference = metadata.get("reference_summary", "") if source == "arxiv" else ""
    if not isinstance(reference, str):
        raise ValueError(f"{path}:{line_number}: reference_summary must be a string")
    reference = reference.strip()
    identity = (
        row.get("source_document_id")
        or row.get("document_id")
        or metadata.get("original_id")
        or sample_id
    )
    output_metadata = {
        "source": source,
        "source_split": split,
        "source_document_id": f"{source}:{identity}",
        "prompt_token_length": metadata.get("prompt_token_length"),
        "prompt_length_bin": metadata.get("prompt_length_bin"),
        "reference_kind": "arxiv_source_reference" if reference else "none",
        "source_stratified_file": path.name,
        "source_stratified_line": line_number,
    }
    return {
        "id": f"{source}:{sample_id}",
        "document": document,
        "summary": reference,
        "metadata": output_metadata,
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield line_number, row


def prepare_inputs(
    stratified_dir: str | Path,
    output_dir: str | Path,
    *,
    max_records_per_split: int | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Write fresh train/eval JSONL and a provenance report without re-splitting."""
    source_root = Path(stratified_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if max_records_per_split is not None and max_records_per_split < 1:
        raise ValueError("max_records_per_split must be positive")
    inputs = {
        "train": source_root / "train_prompts.jsonl",
        "validation": source_root / "val_prompts.jsonl",
    }
    for split, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing stratified {split} input: {path}")
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".phase1-inputs-", dir=destination.parent))
    seen_ids: dict[str, str] = {}
    seen_content: dict[str, str] = {}
    seen_source_docs: dict[str, str] = {}
    report: dict[str, Any] = {
        "schema": "finetuning-phase1-inputs-from-mr-stratified-v1",
        "source_stratified_dir": str(source_root),
        "max_records_per_split": max_records_per_split,
        "summary_policy": {
            "arxiv": "metadata.reference_summary when present; retained as reference only",
            "sharegpt": "empty; no human gold summary is available in the stratified corpus",
        },
        "prompt_policy": "preserve original ArXiv user prompt; flatten ShareGPT history with role labels; use prompt_template={document}",
        "inputs": {},
        "splits": {},
    }

    try:
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover - server profile includes tqdm
            tqdm = lambda iterable, **_kwargs: iterable  # type: ignore[assignment]

        for split, source_path in inputs.items():
            counts: Counter[str] = Counter()
            length_bins: Counter[str] = Counter()
            references: Counter[str] = Counter()
            output_name = "train.jsonl" if split == "train" else "eval.jsonl"
            output_path = stage / output_name
            with output_path.open("x", encoding="utf-8") as output:
                rows = _iter_jsonl(source_path)
                if show_progress:
                    rows = tqdm(rows, desc=f"Convert {split}", unit="row")
                for line_number, row in rows:
                    if row.get("record_type") == "summary":
                        counts["skipped_summary_metadata_rows"] += 1
                        continue
                    if max_records_per_split is not None and counts["records"] >= max_records_per_split:
                        break
                    record = _phase1_record(row, split=split, path=source_path, line_number=line_number)
                    sample_id = record["id"]
                    old_split = seen_ids.get(sample_id)
                    if old_split is not None:
                        raise ValueError(
                            f"ID overlap between train and validation: {sample_id} ({old_split}, {split})"
                        )
                    doc_hash = hashlib.sha256(record["document"].encode("utf-8")).hexdigest()
                    old_content_split = seen_content.get(doc_hash)
                    if old_content_split is not None:
                        raise ValueError(
                            f"prompt content overlap between train and validation ({old_content_split}, {split}): {sample_id}"
                        )
                    source_doc = str(record["metadata"]["source_document_id"])
                    old_doc_split = seen_source_docs.get(source_doc)
                    if old_doc_split is not None:
                        raise ValueError(
                            f"source-document overlap between train and validation ({old_doc_split}, {split}): {source_doc}"
                        )
                    seen_ids[sample_id] = split
                    seen_content[doc_hash] = split
                    seen_source_docs[source_doc] = split
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["records"] += 1
                    counts[record["metadata"]["source"]] += 1
                    length_bin = record["metadata"].get("prompt_length_bin")
                    if length_bin is not None:
                        length_bins[str(length_bin)] += 1
                    references[record["metadata"]["reference_kind"]] += 1

            if counts["records"] == 0:
                raise ValueError(f"no records converted from {source_path}")
            report["inputs"][split] = {
                "path": str(source_path),
                "sha256": _sha256(source_path),
            }
            report["splits"][split] = {
                "output": output_name,
                "records": counts["records"],
                "by_source": {
                    source: counts[source] for source in ("arxiv", "sharegpt") if counts[source]
                },
                "length_bin_counts": dict(sorted(length_bins.items())),
                "reference_counts": dict(sorted(references.items())),
                "skipped_summary_metadata_rows": counts["skipped_summary_metadata_rows"],
            }

        report_path = stage / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
        return report
    except Exception:
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Finetuning Phase 1 train/eval inputs from existing stratified MR-DFlash prompt splits"
    )
    parser.add_argument("--stratified-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-records-per-split",
        type=int,
        default=None,
        help="small pilot limit; omitted means convert all train and validation rows",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    report = prepare_inputs(
        args.stratified_dir,
        args.output_dir,
        max_records_per_split=args.max_records_per_split,
        show_progress=args.progress,
    )
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
