"""Metrics and Phase 1 gate for RECAP-KV V3 hierarchy traces."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HierarchyEvaluationConfig:
    mass_budget: float = 0.01
    max_missed_attention_mass: float = 0.01
    max_exact_expansion: float = 0.30
    max_index_overhead: float = 0.10
    min_documents_per_dataset: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.mass_budget <= 1.0:
            raise ValueError("mass_budget must be in [0, 1]")
        if not 0.0 <= self.max_missed_attention_mass <= 1.0:
            raise ValueError("max_missed_attention_mass must be in [0, 1]")
        if not 0.0 <= self.max_exact_expansion <= 1.0:
            raise ValueError("max_exact_expansion must be in [0, 1]")
        if self.max_index_overhead < 0.0:
            raise ValueError("max_index_overhead must be non-negative")
        if self.min_documents_per_dataset <= 0:
            raise ValueError("min_documents_per_dataset must be positive")


def _mean(steps: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(step[key]) for step in steps]
    return statistics.fmean(values) if values else 0.0


def evaluate_hierarchy_trace(
    record: Mapping[str, Any], config: HierarchyEvaluationConfig
) -> dict[str, Any]:
    """Evaluate one trace; exact attention in the trace is audit-only."""

    if record.get("status") != "ok":
        return {
            "status": "error",
            "sample_id": record.get("sample_id", "unknown"),
            "dataset": record.get("dataset", "unknown"),
            "error": record.get("error", "trace status is not ok"),
        }
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        return {
            "status": "inconclusive",
            "sample_id": record.get("sample_id", "unknown"),
            "dataset": record.get("dataset", "unknown"),
            "reason": "hierarchy trace has no decode steps",
        }
    metrics = {
        "decode_steps": len(steps),
        "missed_attention_mass": _mean(steps, "missed_attention_mass"),
        "max_missed_attention_mass": max(float(step["max_missed_attention_mass"]) for step in steps),
        "exact_expansion_fraction": _mean(steps, "exact_expansion_fraction"),
        "index_overhead": _mean(steps, "index_overhead"),
        "routing_fraction": _mean(steps, "routing_fraction"),
        "upper_bound_violations": sum(int(step["upper_bound_violations"]) for step in steps),
        "active_qk_tokens": _mean(steps, "active_qk_tokens"),
        "full_qk_tokens": _mean(steps, "full_qk_tokens"),
        "routing_representatives": _mean(steps, "routing_representatives"),
        "upper_missed_mass_bound": _mean(steps, "upper_missed_mass_bound"),
        "routing_time_ms": _mean(steps, "routing_time_ms"),
    }
    gate = {
        "missed_mass": metrics["max_missed_attention_mass"] <= config.max_missed_attention_mass,
        "exact_expansion": metrics["exact_expansion_fraction"] <= config.max_exact_expansion,
        "index_overhead": metrics["index_overhead"] <= config.max_index_overhead,
        "upper_bound_sound": metrics["upper_bound_violations"] == 0,
    }
    return {
        "status": "ok" if all(gate.values()) else "gate_fail",
        "sample_id": str(record.get("sample_id", "unknown")),
        "dataset": str(record.get("dataset", "unknown")),
        "metrics": metrics,
        "gate": gate,
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    names = (
        "decode_steps",
        "missed_attention_mass",
        "max_missed_attention_mass",
        "exact_expansion_fraction",
        "index_overhead",
        "routing_fraction",
        "upper_bound_violations",
        "active_qk_tokens",
        "full_qk_tokens",
        "routing_representatives",
        "upper_missed_mass_bound",
        "routing_time_ms",
    )
    return {
        name: sum(float(row["metrics"][name]) for row in rows) / len(rows)
        for name in names
    }


def aggregate_hierarchy_results(
    rows: Sequence[Mapping[str, Any]], config: HierarchyEvaluationConfig
) -> dict[str, Any]:
    """Aggregate traces and decide whether physical follow-up is allowed."""

    ok_rows = [row for row in rows if row.get("status") in {"ok", "gate_fail"}]
    datasets: dict[str, int] = {}
    for row in ok_rows:
        dataset = str(row.get("dataset", "unknown"))
        datasets[dataset] = datasets.get(dataset, 0) + 1
    if len(datasets) < 2 or min(datasets.values(), default=0) < config.min_documents_per_dataset:
        return {
            "status": "INCONCLUSIVE",
            "reason": "minimum_dataset_or_document_gate_not_met",
            "datasets": datasets,
            "ok_documents": len(ok_rows),
        }
    metrics = _aggregate_rows(ok_rows)
    dataset_metrics = {
        dataset: _aggregate_rows(
            [row for row in ok_rows if str(row.get("dataset", "unknown")) == dataset]
        )
        for dataset in datasets
    }
    gate = {
        "missed_mass": metrics["max_missed_attention_mass"] <= config.max_missed_attention_mass,
        "exact_expansion": metrics["exact_expansion_fraction"] <= config.max_exact_expansion,
        "index_overhead": metrics["index_overhead"] <= config.max_index_overhead,
        "upper_bound_sound": all(
            values["upper_bound_violations"] == 0 for values in dataset_metrics.values()
        ),
    }
    return {
        "status": "GO_ROUTING_FOLLOWUP" if all(gate.values()) else "STOP_BEFORE_PHYSICAL",
        "datasets": datasets,
        "ok_documents": len(ok_rows),
        "metrics": metrics,
        "dataset_metrics": dataset_metrics,
        "gate": gate,
        "thresholds": {
            "mass_budget": config.mass_budget,
            "max_missed_attention_mass": config.max_missed_attention_mass,
            "max_exact_expansion": config.max_exact_expansion,
            "max_index_overhead": config.max_index_overhead,
        },
    }
