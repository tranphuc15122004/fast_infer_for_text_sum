"""Workload contract shared by the Qwen3-4B paired benchmark.

This module intentionally contains no model imports.  It is used by the
launcher, unit tests, and inference adapters to keep the exact input/output
contract identical across all baselines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from common.input_utils import truncate_input_ids
from common.paired_reference import config_hash


QWEN3_PAIRED_DATASETS = (
    "gov_report",
    "qmsum",
    "multi_news",
    "lcc",
    "repobench-p",
)
QWEN3_PAIRED_BASELINES = (
    "vanilla_hf",
    "vanilla_fa",
    "dflash",
    "domino",
    "eagle3",
)
DEFAULT_INPUT_CAP = 14_000
DEFAULT_SPEED_OUTPUT_TOKENS = 1_024
DEFAULT_SMOKE_OUTPUT_TOKENS = 32
DEFAULT_FULL_SAMPLES = 100
DEFAULT_SMOKE_SAMPLES = 2


@dataclass(frozen=True)
class PreparedInput:
    input_ids: torch.Tensor
    original_input_tokens: int
    input_tokens: int
    input_truncated: bool


def profile_defaults(mode: str) -> dict[str, int | str]:
    normalized = str(mode).lower()
    if normalized not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    smoke = normalized == "smoke"
    return {
        "mode": normalized,
        "max_samples": DEFAULT_SMOKE_SAMPLES if smoke else DEFAULT_FULL_SAMPLES,
        "max_input_tokens": DEFAULT_INPUT_CAP,
        "max_new_tokens": (
            DEFAULT_SMOKE_OUTPUT_TOKENS if smoke else DEFAULT_SPEED_OUTPUT_TOKENS
        ),
    }


def prepare_input_ids(input_ids: torch.Tensor, max_input_tokens: int) -> PreparedInput:
    """Apply the common deterministic head+tail input cap and report metadata."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("paired benchmark requires input_ids with shape [1, seq]")
    original = int(input_ids.shape[1])
    prepared = truncate_input_ids(input_ids, int(max_input_tokens))
    length = int(prepared.shape[1])
    return PreparedInput(
        input_ids=prepared,
        original_input_tokens=original,
        input_tokens=length,
        input_truncated=length != original,
    )


def build_run_config(
    *,
    dataset: str,
    target_model: str,
    input_cap: int,
    output_tokens: int,
    seed: int,
    dtype: str,
    attention_backend: str,
    batch_size: int = 1,
    temperature: float = 0.0,
    tokenizer_mode: str = "longbench_rendered_prompt",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return workload fields and a stable hash for strict reference pairing."""

    config: dict[str, Any] = {
        "benchmark": "qwen3_4b_paired",
        "dataset": str(dataset),
        "target_model": str(target_model),
        "input_cap": int(input_cap),
        "output_tokens": int(output_tokens),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "temperature": float(temperature),
        "dtype": str(dtype),
        "attention_backend": str(attention_backend),
        "tokenizer_mode": str(tokenizer_mode),
    }
    # Run IDs, host names, and method names are deliberately not part of the
    # pairing contract. They describe provenance, not workload identity.
    if extra:
        for key, value in extra.items():
            if key not in {"run_id", "method", "hostname", "output"}:
                config[key] = value
    config["run_config_hash"] = config_hash(config)
    return config


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else None,
    }


def input_distribution(data_dir: Path) -> dict[str, Any]:
    """Read precomputed ``input_tokens`` metadata from a LongBench directory."""

    data_dir = Path(data_dir)
    datasets: dict[str, dict[str, int | float | None]] = {}
    all_values: list[int] = []
    for path in sorted(data_dir.glob("*.jsonl")):
        values: list[int] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get("input_tokens")
                if isinstance(value, (int, float)) and int(value) > 0:
                    values.append(int(value))
        if values:
            datasets[path.stem] = _summarize(values)
            all_values.extend(values)
    return {"data_dir": str(data_dir), "datasets": datasets, "global": _summarize(all_values)}


__all__ = [
    "DEFAULT_INPUT_CAP",
    "DEFAULT_SPEED_OUTPUT_TOKENS",
    "QWEN3_PAIRED_BASELINES",
    "QWEN3_PAIRED_DATASETS",
    "PreparedInput",
    "build_run_config",
    "input_distribution",
    "prepare_input_ids",
    "profile_defaults",
]
