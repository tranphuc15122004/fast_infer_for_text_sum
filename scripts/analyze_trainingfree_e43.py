#!/usr/bin/env python3
"""Aggregate E43 GQA-aware temporal support reuse traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Sequence[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numbers) if numbers else 0.0,
        "p95": _percentile(numbers, 95.0),
        "p99": _percentile(numbers, 99.0),
    }


def _rows(
    records: Iterable[Mapping[str, Any]],
    *,
    group: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        for step in record.get("steps", []):
            temporal = step.get("temporal", step)
            if not isinstance(temporal, Mapping):
                continue
            values = temporal.get(group, {})
            if not isinstance(values, Mapping):
                continue
            for config, metrics in values.items():
                if not isinstance(metrics, Mapping):
                    raise ValueError("E43 metric rows must be objects")
                row = {
                    "dataset": str(record.get("dataset", "unknown")),
                    "sample_id": str(record.get("sample_id", "unknown")),
                    "step": int(step.get("step", 0)),
                    "config": str(config),
                    "group": group,
                }
                for key, value in metrics.items():
                    if isinstance(value, bool):
                        row[key] = value
                    elif isinstance(value, (int, float)):
                        numeric = float(value)
                        if not math.isfinite(numeric):
                            raise ValueError("E43 metrics must be finite")
                        row[key] = numeric
                rows.append(row)
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"observation_count": 0}
    missed = [float(row.get("mean_missed_mass", 0.0)) for row in rows]
    p99_missed = [
        float(row.get("p99_missed_mass", value))
        for row, value in zip(rows, missed)
    ]
    cost = [
        float(row.get("source_cost_ratio", row.get("gqa_expansion_fraction", 0.0)))
        for row in rows
    ]
    working_cost = [
        float(row.get("working_source_cost_ratio", row.get("gqa_expansion_fraction", 0.0)))
        for row in rows
    ]
    refresh = [float(bool(row.get("refresh", False))) for row in rows]
    return {
        "observation_count": len(rows),
        "document_count": len({str(row["sample_id"]) for row in rows}),
        "source_cost_ratio": _stats(cost),
        "working_source_cost_ratio": _stats(working_cost),
        "mean_missed_mass": statistics.fmean(missed),
        "p99_missed_mass": max(p99_missed) if p99_missed else 0.0,
        "p99_step_mean_missed_mass": _percentile(missed, 99.0),
        "refresh_rate": statistics.fmean(refresh),
        "gqa_expansion_fraction": statistics.fmean(
            float(row.get("gqa_expansion_fraction", row.get("working_source_cost_ratio", 0.0)))
            for row in rows
        ),
    }


def _aggregate_by_config(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    configs = sorted({str(row["config"]) for row in rows})
    return {
        config: _aggregate([row for row in rows if str(row["config"]) == config])
        for config in configs
    }


def summarize_e43(
    records: Sequence[Mapping[str, Any]],
    *,
    min_documents_per_dataset: int = 10,
) -> dict[str, Any]:
    recurrent_rows = _rows(records, group="recurrent")
    if not recurrent_rows:
        return {
            "schema_version": "recap.e43.metrics.v1",
            "status": "INCONCLUSIVE",
            "reason": "no_temporal_rows",
            "record_count": 0,
            "datasets": {},
            "recurrent_candidates": {},
            "best_candidate": None,
        }
    datasets = defaultdict(set)
    for row in recurrent_rows:
        datasets[str(row["dataset"])].add(str(row["sample_id"]))
    coverage = {dataset: len(sample_ids) for dataset, sample_ids in sorted(datasets.items())}
    candidates: dict[str, dict[str, Any]] = {}
    for config in sorted({str(row["config"]) for row in recurrent_rows}):
        config_rows = [row for row in recurrent_rows if str(row["config"]) == config]
        if len({str(row["dataset"]) for row in config_rows}) < 2:
            continue
        summary = _aggregate(config_rows)
        summary["dataset_metrics"] = {
            dataset: _aggregate(
                [row for row in config_rows if str(row["dataset"]) == dataset]
            )
            for dataset in sorted({str(row["dataset"]) for row in config_rows})
        }
        summary["gate"] = {
            "cost_lt_50pct": summary["source_cost_ratio"]["mean"] < 0.50,
            "mean_missed_mass_le_1pct": summary["mean_missed_mass"] <= 0.01,
            "p99_missed_mass_le_5pct": summary["p99_missed_mass"] <= 0.05,
        }
        candidates[config] = summary
    covered = len(coverage) >= 2 and min(coverage.values(), default=0) >= min_documents_per_dataset
    passing = [
        config
        for config, summary in candidates.items()
        if all(summary["gate"].values())
    ]
    ranked = sorted(
        candidates,
        key=lambda config: (
            not all(candidates[config]["gate"].values()),
            candidates[config]["source_cost_ratio"]["mean"],
            candidates[config]["mean_missed_mass"],
        ),
    )
    best = passing[0] if passing else None
    status = (
        "INCONCLUSIVE"
        if not covered
        else "GO_MASKED_GENERATION"
        if best is not None
        else "STOP_SOURCE_SELECTION"
    )
    return {
        "schema_version": "recap.e43.metrics.v1",
        "status": status,
        "record_count": len(
            {str(record.get("sample_id")) for record in records if record.get("status") == "ok"}
        ),
        "datasets": coverage,
        "recurrent_candidates": candidates,
        "best_candidate": best,
        "ranked_candidates": ranked,
        "gqa_oracle": _aggregate_by_config(_rows(records, group="gqa_oracle")),
        "lag_transfer": _aggregate_by_config(_rows(records, group="lag")),
        "adaptive_lag_transfer": _aggregate_by_config(
            _rows(records, group="adaptive_lag")
        ),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_outputs(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e43_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "e43_recurrent.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "config",
            "source_cost_mean",
            "working_cost_mean",
            "mean_missed_mass",
            "p99_missed_mass",
            "refresh_rate",
            "gate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for config, values in summary.get("recurrent_candidates", {}).items():
            writer.writerow(
                {
                    "config": config,
                    "source_cost_mean": values["source_cost_ratio"]["mean"],
                    "working_cost_mean": values["working_source_cost_ratio"]["mean"],
                    "mean_missed_mass": values["mean_missed_mass"],
                    "p99_missed_mass": values["p99_missed_mass"],
                    "refresh_rate": values["refresh_rate"],
                    "gate": json.dumps(values["gate"], sort_keys=True),
                }
            )
    lines = [
        "# E43 — Temporal Support Reuse",
        "",
        f"- Status: **{summary.get('status')}**",
        f"- Trace records: {summary.get('record_count', 0)}",
        f"- Datasets: {summary.get('datasets', {})}",
        f"- Best candidate: {summary.get('best_candidate')}",
        "",
        "## Recurrent candidates",
        "",
        "| Config | Source cost | Working cost | Mean missed | P99 missed | Refresh rate | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for config, values in summary.get("recurrent_candidates", {}).items():
        lines.append(
            f"| {config} | {values['source_cost_ratio']['mean']:.4f} | "
            f"{values['working_source_cost_ratio']['mean']:.4f} | "
            f"{values['mean_missed_mass']:.6f} | {values['p99_missed_mass']:.6f} | "
            f"{values['refresh_rate']:.4f} | {values['gate']} |"
        )
    (output_dir / "e43_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-documents-per-dataset", type=int, default=10)
    args = parser.parse_args()
    summary = summarize_e43(
        _load_jsonl(args.trace),
        min_documents_per_dataset=args.min_documents_per_dataset,
    )
    _write_outputs(summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] != "INCONCLUSIVE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
