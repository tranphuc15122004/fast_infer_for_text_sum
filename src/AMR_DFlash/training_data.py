"""Prepare bounded, source-balanced AMR pilot manifests from regenerated JSONL."""

from __future__ import annotations

import heapq
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .artifacts import canonical_hash, sha256_file, write_jsonl
from .checkpoint import TOKENIZER_NAMES
from .prompts import PROMPT_POLICY, render_prompt, source_content_hash, validate_messages

SPLITS = ("train", "validation", "holdout")


def training_signal_summary(labels: list[dict[str, Any]], preferences: list[dict[str, Any]]) -> dict[str, Any]:
    """Count usable supervision by split; this is not GPU/runtime validation."""
    by_split = {}
    for split in SPLITS:
        split_labels = [row for row in labels if row.get("split") == split]
        pairs = [row for row in preferences if row.get("split") == split and not row.get("test_fixture")]
        teachers = [row for row in split_labels if row.get("status") == "success"
                    and not row.get("censored") and row.get("teacher_logits_ref")]
        by_split[split] = {
            "candidate_labels": len(split_labels),
            "states_labeled": len({row["state_id"] for row in split_labels}),
            "censored_labels": sum(bool(row.get("censored")) for row in split_labels),
            "preference_pairs": len(pairs),
            "states_with_preferences": len({row["state_id"] for row in pairs}),
            "uncensored_teacher_rows": len(teachers),
        }
    selector_signal = by_split["train"]["preference_pairs"] > 0
    compressor_signal = by_split["train"]["uncensored_teacher_rows"] > 0
    return {"by_split": by_split, "train_selector_signal_present": selector_signal,
            "train_compressor_signal_present": compressor_signal,
            "train_signal_present": selector_signal and compressor_signal}


def convert_regenerated_record(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    """Remove only the final generated answer; keep the preceding conversation."""
    original_id = row.get("id")
    try:
        if original_id is None or not str(original_id).strip():
            raise ValueError("missing id")
        if split not in SPLITS:
            raise ValueError(f"invalid split: {split}")
        source = str(row.get("source") or row.get("dataset") or "").lower()
        if source not in {"sharegpt", "arxiv"}:
            raise ValueError(f"expected source sharegpt/arxiv, found {source!r}")
        conversations = validate_messages(row.get("conversations"))
        if len(conversations) < 2 or conversations[-1]["role"] != "assistant":
            raise ValueError("regenerated conversations must end with an assistant answer")
        messages = conversations[:-1]
        if messages[-1]["role"] != "user":
            raise ValueError("prompt before regenerated answer must end with user")
        metadata = row.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        document_id = (row.get("source_document_id") or row.get("document_id")
                       or metadata.get("original_id") or original_id)
        converted = {
            "id": f"{source}:{original_id}",
            "original_id": str(original_id),
            "source_document_id": f"{source}:{document_id}",
            "dataset": source,
            "split": split,
            "messages": messages,
            # Compatibility with common.data_loader; AMR renders messages instead.
            "prompt": messages[-1]["content"],
            "data_schema": "amr-regenerated-prompts-v1",
        }
        reference = metadata.get("reference_summary")
        if source == "arxiv" and isinstance(reference, str) and reference.strip():
            converted["reference"] = reference
        return converted
    except ValueError as exc:
        raise ValueError(f"sample {original_id}: {exc}") from exc


def _write_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _tokenizer_identity(path: Path) -> dict[str, str]:
    files = {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and (file.name in TOKENIZER_NAMES or file.name == "vocab.txt"
                               or file.suffix in {".model", ".tiktoken", ".jinja"})
    }
    if not files:
        raise ValueError(f"no local tokenizer files at {path}")
    return files


def _balanced_rows(pools: dict[str, list], limit: int) -> list[dict[str, Any]]:
    ordered = {source: [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
               for source, heap in sorted(pools.items())}
    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for rows in ordered.values():
            if offset < len(rows) and len(selected) < limit:
                selected.append(rows[offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def prepare_manifest(
    *, inputs: dict[str, list[str | Path]], output_dir: str | Path,
    tokenizer_path: str | Path, limits: dict[str, int] | None = None,
    min_input_tokens: int = 4097, max_input_tokens: int = 16384,
    seed: int = 17, resume: bool = False, progress: bool = True,
) -> dict[str, Any]:
    """Scan all input identities, then emit at most the requested pilot counts.

    No weights or old hidden cache are loaded. Token lengths are checkpointed in
    SQLite; interrupted runs restart the scan but reuse completed tokenization.
    Memory holds only IDs/hashes and at most limit rows per split/source.
    """
    from transformers import AutoTokenizer
    from tqdm import tqdm

    limits = {**{"train": 200, "validation": 50, "holdout": 0}, **(limits or {})}
    if set(inputs) - set(SPLITS) or set(limits) - set(SPLITS):
        raise ValueError("inputs/limits must use train, validation, holdout splits")
    if limits["train"] < 1 or limits["validation"] < 1 or limits["holdout"] < 0:
        raise ValueError("train and validation limits must be positive; holdout must be nonnegative")
    if min_input_tokens < 1 or max_input_tokens < min_input_tokens:
        raise ValueError("require 1 <= min_input_tokens <= max_input_tokens")
    paths = [(split, Path(path).expanduser().resolve())
             for split in SPLITS for path in inputs.get(split, [])]
    for split in SPLITS:
        if limits[split] > 0 and not inputs.get(split):
            raise ValueError(f"missing {split} input")
    tokenizer_path = Path(tokenizer_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if any(path.is_relative_to(output_dir) for _, path in paths) or tokenizer_path.is_relative_to(output_dir):
        raise ValueError("output directory must be separate from input files and tokenizer")
    contract = {
        "schema": "amr-data-preparation-v1",
        "inputs": [{"split": split, "path": str(path), "sha256": sha256_file(path)}
                   for split, path in paths],
        "tokenizer_files": _tokenizer_identity(tokenizer_path),
        "prompt_policy": PROMPT_POLICY,
        "truncate_policy": "common_truncate_input_ids_v1",
        "limits": limits,
        "min_input_tokens": min_input_tokens,
        "max_input_tokens": max_input_tokens,
        "seed": seed,
        "sampling": "seeded_sha256_per_source_round_robin",
    }
    manifest_path, report_path = output_dir / "manifest.jsonl", output_dir / "report.json"
    contract_path = output_dir / "preparation_contract.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output_dir}; choose a new path or --resume")
        if not contract_path.is_file() or json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("data preparation resume contract differs; choose a new output directory")
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("contract") != contract or not manifest_path.is_file() or report.get("manifest_sha256") != sha256_file(manifest_path):
                raise ValueError("completed data manifest/report hash or contract was modified")
            return report
    else:
        output_dir.mkdir(parents=True)
        _write_json(contract_path, contract)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    ids: set[str] = set()
    sources: dict[str, str] = {}
    contents: dict[str, str] = {}
    eligible_sources: set[str] = set()
    scanned = defaultdict(Counter)
    eligible = defaultdict(Counter)
    skipped = Counter()
    pools = {split: defaultdict(list) for split in SPLITS}
    cache = sqlite3.connect(output_dir / "token_lengths.sqlite3")
    cache.execute("CREATE TABLE IF NOT EXISTS lengths (prompt_hash TEXT PRIMARY KEY, tokens INTEGER NOT NULL)")
    bar = tqdm(total=sum(path.stat().st_size for _, path in paths), unit="B", unit_scale=True,
               desc="AMR prompt audit", disable=not progress)
    try:
        for split, path in paths:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    bar.update(len(line.encode("utf-8")))
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                        if not isinstance(raw, dict):
                            raise ValueError("JSONL record must be an object")
                        if raw.get("record_type") == "summary":
                            continue
                        row = convert_regenerated_record(raw, split=split)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
                    sample_id, source_id, source = row["id"], row["source_document_id"], row["dataset"]
                    if sample_id in ids:
                        raise ValueError(f"duplicate document id: {sample_id}")
                    ids.add(sample_id)
                    scanned[split][source] += 1
                    content_hash = source_content_hash(row, row["prompt"])
                    for seen, identity in ((sources, source_id), (contents, content_hash)):
                        if identity in seen and seen[identity] != split:
                            raise ValueError(f"cross-split source/content overlap for document {sample_id}")
                    duplicate = content_hash in contents or source_id in eligible_sources
                    sources[source_id], contents[content_hash] = split, split
                    if duplicate:
                        skipped["duplicate_content_or_document"] += 1
                        continue
                    prompt_key = canonical_hash(row["messages"])
                    cached = cache.execute("SELECT tokens FROM lengths WHERE prompt_hash=?", (prompt_key,)).fetchone()
                    if cached is None:
                        prompt = render_prompt(tokenizer, {"id": sample_id, "prompt": row["prompt"], "raw": row})
                        token_count = len(tokenizer(prompt).input_ids)
                        cache.execute("INSERT INTO lengths VALUES (?, ?)", (prompt_key, token_count))
                    else:
                        token_count = int(cached[0])
                    if len(ids) % 100 == 0:
                        cache.commit()
                        bar.set_postfix(scanned=len(ids), eligible=sum(sum(counts.values()) for counts in eligible.values()))
                    if min(token_count, max_input_tokens) < min_input_tokens:
                        skipped["too_short"] += 1
                        continue
                    eligible_sources.add(source_id)
                    eligible[split][source] += 1
                    row.update(input_tokens_before_truncation=token_count,
                               input_tokens_after_truncation=min(token_count, max_input_tokens),
                               preparation_max_input_tokens=max_input_tokens,
                               provenance={"input_sha256": next(item["sha256"] for item in contract["inputs"]
                                                                if item["path"] == str(path) and item["split"] == split),
                                           "input_line": line_number})
                    limit = limits[split]
                    if limit:
                        rank = int(canonical_hash([seed, sample_id]), 16)
                        item = (-rank, sample_id, row)
                        heap = pools[split][source]
                        if len(heap) < limit:
                            heapq.heappush(heap, item)
                        elif item > heap[0]:
                            heapq.heapreplace(heap, item)
        rows = [row for split in SPLITS for row in _balanced_rows(pools[split], limits[split])]
        selected = defaultdict(Counter)
        for row in rows:
            selected[row["split"]][row["dataset"]] += 1
        for split in SPLITS:
            if limits[split] > 0 and not selected[split]:
                raise ValueError(f"no eligible {split} prompts after length/dedup filters")
        # Verify source files did not change during the scan.
        for item in contract["inputs"]:
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError("source JSONL changed during preparation; use a stable input copy")
        write_jsonl(manifest_path, rows)
        lengths = [row["input_tokens_after_truncation"] for row in rows]
        report = {
            "contract": contract,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "scanned_by_split_source": {key: dict(value) for key, value in scanned.items()},
            "eligible_by_split_source": {key: dict(value) for key, value in eligible.items()},
            "selected_by_split_source": {key: dict(value) for key, value in selected.items() if value},
            "skipped": dict(skipped),
            "selected": len(rows),
            "with_reference": sum("reference" in row for row in rows),
            "input_tokens_after_truncation": {"min": min(lengths), "max": max(lengths),
                                               "mean": sum(lengths) / len(lengths)},
            "training_labels_ready": False,
            "next_step": "capture -> candidates -> label",
        }
        _write_json(report_path, report)
        return report
    finally:
        cache.commit()
        cache.close()
        bar.close()
