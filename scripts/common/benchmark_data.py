"""Shared schema and deterministic utilities for the LongBench benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROMPT_CONFIG = Path(__file__).resolve().with_name("longbench_prompts.json")
DATASETS = ("gov_report", "qmsum", "multi_news", "lcc", "repobench-p")
CODE_DATASETS = frozenset(("lcc", "repobench-p"))
EXPECTED_SOURCE_COUNTS = {
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "lcc": 500,
    "repobench-p": 500,
}
REQUIRED_FIELDS = frozenset(
    (
        "id",
        "dataset",
        "source_split",
        "source_index",
        "task_type",
        "context",
        "input",
        "answers",
        "reference_output",
        "input_tokens",
        "length_bin",
    )
)


def load_prompt_templates(path: Path = PROMPT_CONFIG) -> dict[str, str]:
    templates = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(templates, dict):
        raise ValueError(f"Prompt config must be an object: {path}")
    missing = set(DATASETS) - set(templates)
    if missing:
        raise ValueError(f"Prompt config missing datasets: {sorted(missing)}")
    return {name: str(templates[name]) for name in DATASETS}


def stable_id(dataset: str, source_index: int, context: str, query: str) -> str:
    raw = f"{dataset}\0{source_index}\0{context}\0{query}".encode("utf-8")
    return f"{dataset}_{hashlib.sha1(raw).hexdigest()[:16]}"


def render_prompt(
    record: Mapping[str, Any],
    templates: Mapping[str, str] | None = None,
) -> str:
    templates = templates or load_prompt_templates()
    dataset = str(record["dataset"])
    try:
        template = templates[dataset]
    except KeyError as exc:
        raise ValueError(f"No prompt template for dataset {dataset!r}") from exc
    return template.format(
        context=str(record.get("context") or ""),
        input=str(record.get("input") or ""),
    )


def _token_count(tokenizer: Callable[..., Mapping[str, Any]], text: str) -> int:
    result = tokenizer(text, add_special_tokens=False)
    token_ids = result["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def canonicalize_record(
    dataset: str,
    source_index: int,
    source: Mapping[str, Any],
    tokenizer: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported LongBench dataset: {dataset}")
    context = "" if source.get("context") is None else str(source["context"])
    query = "" if source.get("input") is None else str(source["input"])
    answers = source.get("answers")
    if (
        not isinstance(answers, list)
        or not answers
        or any(not isinstance(answer, str) or not answer for answer in answers)
    ):
        raise ValueError(
            f"{dataset}[{source_index}]: answers must be a non-empty list of strings"
        )

    record: dict[str, Any] = {
        "id": stable_id(dataset, source_index, context, query),
        "dataset": dataset,
        "source_split": "test",
        "source_index": source_index,
        "task_type": "code_completion" if dataset in CODE_DATASETS else "summarization",
        "context": context,
        "input": query,
        "answers": list(answers),
        "reference_output": answers[0],
        "input_tokens": 0,
        "length_bin": None,
    }
    record["input_tokens"] = _token_count(tokenizer, render_prompt(record))

    metadata = {
        key: source[key]
        for key in ("language", "all_classes")
        if source.get(key) is not None
    }
    if metadata:
        record["metadata"] = metadata
    problems = validate_record(record)
    if problems:
        raise ValueError(f"{dataset}[{source_index}]: {'; '.join(problems)}")
    return record


def _balanced_bins(rows: Sequence[dict[str, Any]], n_bins: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["input_tokens"]),
            int(row.get("source_index", 0)),
            str(row["id"]),
        ),
    )
    base, remainder = divmod(len(ordered), n_bins)
    bins: list[list[dict[str, Any]]] = []
    start = 0
    for index in range(n_bins):
        size = base + (1 if index < remainder else 0)
        bins.append(ordered[start : start + size])
        start += size
    return bins


def stratified_sample(
    rows: Sequence[dict[str, Any]],
    n: int,
    n_bins: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if n <= 0 or n > len(rows):
        raise ValueError(f"Cannot select {n} rows from {len(rows)} source rows")
    if n_bins <= 0 or n % n_bins:
        raise ValueError(f"Selection count {n} must be divisible by {n_bins} bins")
    bins = _balanced_bins(rows, n_bins)
    per_bin = n // n_bins
    rng = random.Random(f"longbench:{seed}")
    selected: list[dict[str, Any]] = []
    for bin_index, candidates in enumerate(bins):
        if len(candidates) < per_bin:
            raise ValueError(
                f"Length bin {bin_index} has {len(candidates)} rows; "
                f"need {per_bin}"
            )
        for row in rng.sample(candidates, per_bin):
            item = dict(row)
            item["length_bin"] = bin_index
            selected.append(item)
    return sorted(
        selected,
        key=lambda row: (int(row.get("source_index", 0)), str(row["id"])),
    )


def select_rows(
    rows: Sequence[dict[str, Any]],
    dataset: str,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n == len(rows) and dataset not in CODE_DATASETS:
        return [dict(row, length_bin=None) for row in rows]
    return stratified_sample(rows, n=n, n_bins=5, seed=seed)


def validate_record(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        problems.append(f"missing fields: {sorted(missing)}")
        return problems
    if not isinstance(record["id"], str) or not record["id"]:
        problems.append("id must be a non-empty string")
    if record["dataset"] not in DATASETS:
        problems.append(f"unknown dataset: {record['dataset']!r}")
    expected_type = "code_completion" if record["dataset"] in CODE_DATASETS else "summarization"
    if record["task_type"] != expected_type:
        problems.append(f"task_type must be {expected_type!r}")
    if record["source_split"] != "test":
        problems.append("source_split must be 'test'")
    if not isinstance(record["source_index"], int) or isinstance(record["source_index"], bool):
        problems.append("source_index must be an integer")
    if not isinstance(record["context"], str) or not record["context"].strip():
        problems.append("context must be non-empty text")
    if not isinstance(record["input"], str):
        problems.append("input must be text")
    answers = record["answers"]
    if not isinstance(answers, list) or not answers or any(
        not isinstance(answer, str) or not answer for answer in answers
    ):
        problems.append("answers must be a non-empty list of non-empty strings")
    elif record["reference_output"] != answers[0]:
        problems.append("reference_output must equal answers[0]")
    if not isinstance(record["input_tokens"], int) or isinstance(record["input_tokens"], bool):
        problems.append("input_tokens must be an integer")
    elif record["input_tokens"] < 0:
        problems.append("input_tokens must be non-negative")
    length_bin = record["length_bin"]
    if length_bin is not None and (
        not isinstance(length_bin, int)
        or isinstance(length_bin, bool)
        or not 0 <= length_bin < 5
    ):
        problems.append("length_bin must be null or an integer in [0, 4]")
    return problems


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def token_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    values = [int(row["input_tokens"]) for row in rows]
    if not values:
        return {"num_samples": 0, "min": 0, "p25": 0, "median": 0, "mean": 0.0, "p75": 0, "p90": 0, "max": 0}
    return {
        "num_samples": len(values),
        "min": min(values),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.50),
        "mean": round(sum(values) / len(values), 2),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }


def metric_family(task_type: str) -> str:
    if task_type == "summarization":
        return "rouge"
    if task_type == "code_completion":
        return "code_completion"
    raise ValueError(f"Unknown task type: {task_type!r}")


def validate_output_dir(output_dir: Path, expected_count: int = 200) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hashes = manifest.get("file_sha256")
    if not isinstance(manifest_hashes, dict):
        raise ValueError("manifest missing file_sha256")
    all_ids: set[str] = set()
    summary: dict[str, Any] = {"datasets": {}, "total": 0}
    for dataset in DATASETS:
        path = output_dir / f"{dataset}.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing dataset file: {path}")
        expected_hash = manifest_hashes.get(dataset)
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(f"manifest missing checksum for {dataset}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"checksum mismatch for {dataset}: {digest} != {expected_hash}")
        rows = read_jsonl(path)
        if len(rows) != expected_count:
            raise ValueError(f"{dataset}: expected {expected_count} rows, got {len(rows)}")
        for row in rows:
            problems = validate_record(row)
            if problems:
                raise ValueError(f"{dataset}: {'; '.join(problems)}")
            if row["dataset"] != dataset:
                raise ValueError(f"{dataset}: record has dataset {row['dataset']!r}")
            if row["id"] in all_ids:
                raise ValueError(f"duplicate id: {row['id']}")
            all_ids.add(row["id"])
        if dataset in CODE_DATASETS:
            bins = {index: 0 for index in range(5)}
            for row in rows:
                bins[row["length_bin"]] += 1
            if set(bins.values()) != {expected_count // 5}:
                raise ValueError(f"{dataset}: invalid length-bin distribution: {bins}")
        summary["datasets"][dataset] = token_stats(rows)
        summary["total"] += len(rows)
    manifest_counts = manifest.get("selected_counts", {})
    for dataset in DATASETS:
        if manifest_counts.get(dataset) != expected_count:
            raise ValueError(
                f"manifest selected_counts[{dataset!r}] does not equal {expected_count}"
            )
    return summary
