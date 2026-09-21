#!/usr/bin/env python3
"""Aggregate E42 oracle hybrid-head sweeps from hierarchy traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.analyze_trainingfree_e41 import _stats
except ModuleNotFoundError:  # direct execution: python scripts/analyze_trainingfree_e42.py
    from analyze_trainingfree_e41 import _stats


def _collect_oracle_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        for step in record.get("steps", []):
            for row in step.get("head_oracle_metrics", []):
                if not isinstance(row, Mapping):
                    raise ValueError("head_oracle_metrics rows must be objects")
                try:
                    item = {
                        "dataset": str(record.get("dataset", "unknown")),
                        "sample_id": str(record.get("sample_id", "unknown")),
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "k95_fraction": float(row["k95_fraction"]),
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("invalid E42 oracle row") from exc
                for key, value in row.items():
                    if str(key).startswith("routed_"):
                        try:
                            item[str(key)] = float(value)
                        except (TypeError, ValueError) as exc:
                            raise ValueError("invalid E42 routed metric") from exc
                if any(
                    not math.isfinite(float(value))
                    for key, value in item.items()
                    if key not in {"dataset", "sample_id"}
                ):
                    raise ValueError("E42 oracle rows must contain finite values")
                rows.append(item)
    return rows


def _sweep(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_fraction: float,
    routed_fraction: float,
) -> dict[str, Any]:
    grouped_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped_scores[(int(row["layer"]), int(row["head"]))].append(
            float(row["k95_fraction"])
        )
    identities = sorted(grouped_scores)
    selected_count = max(1, math.ceil(len(identities) * global_fraction))
    ranked = sorted(
        identities,
        key=lambda identity: (-sum(grouped_scores[identity]) / len(grouped_scores[identity]), identity),
    )
    selected = set(ranked[:selected_count])
    label = str(routed_fraction).rstrip("0").rstrip(".")
    missed: list[float] = []
    output_error: list[float] = []
    expansion: list[float] = []
    for row in rows:
        identity = (int(row["layer"]), int(row["head"]))
        if identity in selected:
            missed.append(0.0)
            output_error.append(0.0)
            expansion.append(1.0)
        else:
            missed.append(float(row[f"routed_missed_mass_{label}"]))
            output_error.append(float(row[f"routed_output_error_{label}"]))
            expansion.append(float(row[f"routed_fraction_{label}"]))
    missed_stats = _stats(missed)
    error_stats = _stats(output_error)
    expansion_value = sum(expansion) / len(expansion)
    return {
        "global_fraction_requested": global_fraction,
        "global_fraction_actual": selected_count / len(identities),
        "routed_fraction_requested": routed_fraction,
        "selected_global_heads": [[layer, head] for layer, head in sorted(selected)],
        "head_count": len(identities),
        "exact_expansion_fraction": expansion_value,
        "missed_attention_mass": missed_stats,
        "attention_output_error": error_stats,
        "gate": {
            "expansion_le_30pct": expansion_value <= 0.30,
            "mean_missed_mass_le_1pct": missed_stats["mean"] <= 0.01,
            "p99_missed_mass_le_5pct": missed_stats["p99"] <= 0.05,
        },
    }


def summarize_e42(
    records: Sequence[Mapping[str, Any]],
    *,
    global_fractions: Sequence[float] = (0.10, 0.20, 0.30, 0.40),
    routed_fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> dict[str, Any]:
    """Run the fixed-head oracle sweep without implementing a router."""

    rows = _collect_oracle_rows(records)
    if not rows:
        return {
            "schema_version": "recap.e42.metrics.v1",
            "status": "inconclusive",
            "reason": "no_head_oracle_rows",
            "record_count": 0,
            "datasets": {},
        }
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    datasets: dict[str, Any] = {}
    for dataset, dataset_rows in sorted(by_dataset.items()):
        sweeps: dict[str, Any] = {}
        for global_fraction in global_fractions:
            for routed_fraction in routed_fractions:
                key = f"global_{global_fraction:g}_routed_{routed_fraction:g}"
                sweeps[key] = _sweep(
                    dataset_rows,
                    global_fraction=float(global_fraction),
                    routed_fraction=float(routed_fraction),
                )
        datasets[dataset] = {
            "observation_count": len(dataset_rows),
            "document_count": len({str(row["sample_id"]) for row in dataset_rows}),
            "sweeps": sweeps,
        }
    return {
        "schema_version": "recap.e42.metrics.v1",
        "status": "ok",
        "record_count": len({str(record.get("sample_id")) for record in records if record.get("status") == "ok"}),
        "observation_count": len(rows),
        "global_fractions": [float(value) for value in global_fractions],
        "routed_fractions": [float(value) for value in routed_fractions],
        "datasets": datasets,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_outputs(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e42_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "dataset", "sweep", "global_fraction_actual", "routed_fraction_requested",
        "exact_expansion_fraction", "missed_mean", "missed_p99", "output_error_mean",
        "output_error_p99", "gate_pass",
    ]
    with (output_dir / "e42_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, dataset_values in summary.get("datasets", {}).items():
            for name, sweep in dataset_values.get("sweeps", {}).items():
                writer.writerow({
                    "dataset": dataset,
                    "sweep": name,
                    "global_fraction_actual": sweep["global_fraction_actual"],
                    "routed_fraction_requested": sweep["routed_fraction_requested"],
                    "exact_expansion_fraction": sweep["exact_expansion_fraction"],
                    "missed_mean": sweep["missed_attention_mass"]["mean"],
                    "missed_p99": sweep["missed_attention_mass"]["p99"],
                    "output_error_mean": sweep["attention_output_error"]["mean"],
                    "output_error_p99": sweep["attention_output_error"]["p99"],
                    "gate_pass": all(sweep["gate"].values()),
                })
    lines = [
        "# E42 — Oracle hybrid-head sweep",
        "",
        f"- Status: **{summary.get('status')}**",
        f"- Trace records: `{summary.get('record_count', 0)}`",
        "",
        "Oracle only: dense global heads and exact top-source routing; chưa phải physical executor.",
        "",
    ]
    for dataset, dataset_values in summary.get("datasets", {}).items():
        lines.extend([
            f"## Dataset: `{dataset}`",
            "",
            "| Sweep | Expansion | Missed mean | Missed P99 | Output error mean | Gate |",
            "|---|---:|---:|---:|---:|:---:|",
        ])
        for name, sweep in dataset_values["sweeps"].items():
            lines.append(
                f"| {name} | {sweep['exact_expansion_fraction']:.4f} | "
                f"{sweep['missed_attention_mass']['mean']:.6f} | "
                f"{sweep['missed_attention_mass']['p99']:.6f} | "
                f"{sweep['attention_output_error']['mean']:.6f} | "
                f"{'PASS' if all(sweep['gate'].values()) else 'FAIL'} |"
            )
        lines.append("")
    (output_dir / "e42_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_e42(_load_jsonl(args.trace))
    _write_outputs(summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
