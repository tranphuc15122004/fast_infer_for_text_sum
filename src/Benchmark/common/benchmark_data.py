"""Shared schema and deterministic utilities for the LongBench benchmark."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from Benchmark.common.paths import ROOT


PROMPT_CONFIG = Path(__file__).resolve().with_name("longbench_prompts.json")
LONGBENCH_DATASETS = ("gov_report", "qmsum", "multi_news", "lcc", "repobench-p")
VIETNAMESE_DATASETS = ("vietnews", "wikilingua", "vims", "vlsp")
DATASETS = LONGBENCH_DATASETS + VIETNAMESE_DATASETS
CODE_DATASETS = frozenset(("lcc", "repobench-p"))
EXPECTED_SOURCE_COUNTS = {
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "lcc": 500,
    "repobench-p": 500,
    "vietnews": 100,
    "wikilingua": 100,
    "vims": 100,
    "vlsp": 100,
}
VIETNAMESE_PROMPT_TEMPLATE = (
    "Hãy tóm tắt văn bản sau bằng tiếng Việt. Chỉ trả lời bằng bản tóm tắt:\n\n"
    "{document}"
)
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
    if not path.is_file():
        path = Path(__file__).resolve().with_name("longbench_prompts.json")
    templates = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(templates, dict):
        raise ValueError(f"Prompt config must be an object: {path}")
    missing = set(LONGBENCH_DATASETS) - set(templates)
    if missing:
        raise ValueError(f"Prompt config missing datasets: {sorted(missing)}")
    res = {name: str(templates[name]) for name in LONGBENCH_DATASETS}
    for vn_name in VIETNAMESE_DATASETS:
        res[vn_name] = VIETNAMESE_PROMPT_TEMPLATE
    return res


def render_prompt(
    record: Mapping[str, Any],
    templates: Mapping[str, str] | None = None,
) -> str:
    dataset = str(record.get("dataset", ""))
    if dataset in VIETNAMESE_DATASETS:
        document = str(record.get("document") or record.get("context") or "")
        return VIETNAMESE_PROMPT_TEMPLATE.format(document=document)
    templates = templates or load_prompt_templates()
    if dataset in templates:
        return templates[dataset].format(
            context=str(record.get("context") or ""),
            input=str(record.get("input") or ""),
        )
    for key in ("prompt", "question", "instruction", "document", "text"):
        if record.get(key):
            return str(record[key])
    return str(record.get("context") or "")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
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


def _length(row: Mapping[str, Any]) -> int:
    for key in ("input_tokens", "document_words"):
        value = row.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return len(str(row.get("context") or row.get("document") or "").split())


def _balanced_bins(rows: Sequence[dict[str, Any]], n_bins: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (_length(row), str(row.get("id", ""))))
    base, remainder = divmod(len(ordered), n_bins)
    bins: list[list[dict[str, Any]]] = []
    start = 0
    for index in range(n_bins):
        size = base + (1 if index < remainder else 0)
        bins.append(ordered[start : start + size])
        start += size
    return bins


def stratified_sample(
    rows: Sequence[dict[str, Any]], n: int, n_bins: int = 5, seed: int = 42
) -> list[dict[str, Any]]:
    if n <= 0 or n > len(rows):
        raise ValueError(f"Cannot select {n} rows from {len(rows)} source rows")
    if n_bins <= 0 or n % n_bins:
        raise ValueError(f"Selection count {n} must be divisible by {n_bins} bins")
    rng = random.Random(f"longbench:{seed}")
    selected: list[dict[str, Any]] = []
    per_bin = n // n_bins
    for bin_index, candidates in enumerate(_balanced_bins(rows, n_bins)):
        if len(candidates) < per_bin:
            raise ValueError(f"Length bin {bin_index} has too few rows")
        for row in rng.sample(candidates, per_bin):
            selected.append(dict(row, length_bin=bin_index))
    return sorted(selected, key=lambda row: str(row.get("id", "")))


def select_rows(rows: Sequence[dict[str, Any]], dataset: str, n: int, seed: int) -> list[dict[str, Any]]:
    if n == len(rows):
        return [dict(row) for row in rows]
    if n == 1:
        return [dict(rows[0])]
    return stratified_sample(rows, n=n, n_bins=5, seed=seed)


def validate_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
