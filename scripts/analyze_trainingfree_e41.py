#!/usr/bin/env python3
"""Aggregate E41 per-head source concentration traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LEVELS = (90, 95, 99)
GLOBAL_THRESHOLDS = (0.1, 0.25, 0.5)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "variance": 0.0, "p95": 0.0, "p99": 0.0}
    numbers = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numbers),
        "variance": statistics.pvariance(numbers) if len(numbers) > 1 else 0.0,
        "p95": _percentile(numbers, 95.0),
        "p99": _percentile(numbers, 99.0),
    }


def _observations(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        dataset = str(record.get("dataset", "unknown"))
        sample_id = str(record.get("sample_id", "unknown"))
        for step in record.get("steps", []):
            rows = step.get("head_concentration", [])
            if not isinstance(rows, list):
                raise ValueError("head_concentration must be a list")
            for row in rows:
                try:
                    observation = {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "step": int(step.get("step", 0)),
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "source_mass": float(row["source_mass"]),
                    }
                    for level in LEVELS:
                        observation[f"k{level}_fraction"] = float(
                            row[f"k{level}_fraction"]
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("invalid E41 head concentration row") from exc
                if any(
                    not math.isfinite(float(value))
                    for key, value in observation.items()
                    if key not in {"dataset", "sample_id"}
                ):
                    raise ValueError("E41 rows must contain finite values")
                observations.append(observation)
    return observations


def _dataset_summary(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    global_fraction = {
        f"{threshold:g}": sum(
            float(row["k95_fraction"]) > threshold for row in observations
        )
        / len(observations)
        for threshold in GLOBAL_THRESHOLDS
    }
    heatmap: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)
    for (layer, head), rows in sorted(grouped.items()):
        cell: dict[str, Any] = {"layer": layer, "head": head, "observations": len(rows)}
        for level in LEVELS:
            values = [float(row[f"k{level}_fraction"]) for row in rows]
            stats = _stats(values)
            cell[f"mean_k{level}_fraction"] = stats["mean"]
            cell[f"variance_k{level}_fraction"] = stats["variance"]
            cell[f"p95_k{level}_fraction"] = stats["p95"]
            cell[f"p99_k{level}_fraction"] = stats["p99"]
        source_masses = [float(row["source_mass"]) for row in rows]
        cell["mean_source_mass"] = statistics.fmean(source_masses)
        heatmap.append(cell)

    by_document: dict[str, list[float]] = defaultdict(list)
    for row in observations:
        by_document[str(row["sample_id"])].append(float(row["k95_fraction"]))
    document_means = [statistics.fmean(values) for values in by_document.values()]
    return {
        "observation_count": len(observations),
        "document_count": len(by_document),
        "global_head_fraction": global_fraction,
        "k90_fraction": _stats([float(row["k90_fraction"]) for row in observations]),
        "k95_fraction": _stats([float(row["k95_fraction"]) for row in observations]),
        "k99_fraction": _stats([float(row["k99_fraction"]) for row in observations]),
        "document_mean_k95_fraction": _stats(document_means),
        "heatmap": heatmap,
    }


def summarize_e41(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return E41 aggregate metrics grouped by dataset and layer/head."""

    observations = _observations(records)
    if not observations:
        return {
            "schema_version": "recap.e41.metrics.v1",
            "status": "inconclusive",
            "reason": "no_head_concentration_rows",
            "record_count": 0,
            "datasets": {},
        }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[str(observation["dataset"])].append(observation)
    return {
        "schema_version": "recap.e41.metrics.v1",
        "status": "ok",
        "record_count": len({str(record.get("sample_id")) for record in records if record.get("status") == "ok"}),
        "observation_count": len(observations),
        "datasets": {
            dataset: _dataset_summary(rows) for dataset, rows in sorted(grouped.items())
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_outputs(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e41_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "e41_heatmap.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "dataset", "layer", "head", "observations", "mean_k90_fraction",
            "mean_k95_fraction", "mean_k99_fraction", "variance_k95_fraction",
            "p95_k95_fraction", "p99_k95_fraction", "mean_source_mass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, values in summary.get("datasets", {}).items():
            for cell in values.get("heatmap", []):
                writer.writerow({"dataset": dataset, **{field: cell.get(field, "") for field in fields[1:]}})
    lines = [
        "# E41 — Head-wise source concentration",
        "",
        f"- Status: **{summary.get('status')}**",
        f"- Trace records: `{summary.get('record_count', 0)}`",
        f"- Head-step observations: `{summary.get('observation_count', 0)}`",
        "",
    ]
    for dataset, values in summary.get("datasets", {}).items():
        lines.extend([
            f"## Dataset: `{dataset}`",
            "",
            "| Metric | Mean | Variance | P95 | P99 |",
            "|---|---:|---:|---:|---:|",
        ])
        for name in ("k90_fraction", "k95_fraction", "k99_fraction", "document_mean_k95_fraction"):
            stats = values[name]
            lines.append(
                f"| {name} | {stats['mean']:.6f} | {stats['variance']:.6f} | "
                f"{stats['p95']:.6f} | {stats['p99']:.6f} |"
            )
        lines.extend([
            "",
            "GlobalHeadFraction (based on K95/source_tokens):",
            "",
            "| Threshold | Fraction |",
            "|---:|---:|",
        ])
        for threshold, fraction in values["global_head_fraction"].items():
            lines.append(f"| {threshold} | {fraction:.6f} |")
        lines.append("")
    (output_dir / "e41_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_e41(_load_jsonl(args.trace))
    _write_outputs(summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
