"""Evaluation and kill gate for RECAP-KV V2 lease traces."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class LeaseEvaluationConfig:
    delta_anchor: float = 0.05
    delta_cert: float = 0.10
    max_violation_rate: float = 1e-3
    violation_tolerance: float = 1e-4
    min_cold_fraction: float = 0.20
    min_lease_length: float = 4.0
    min_documents_per_dataset: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.delta_anchor < 1.0:
            raise ValueError("delta_anchor must be in [0, 1)")
        if not 0.0 < self.delta_cert < 1.0:
            raise ValueError("delta_cert must be in (0, 1)")
        if self.max_violation_rate < 0.0:
            raise ValueError("max_violation_rate must be non-negative")
        if self.violation_tolerance < 0.0:
            raise ValueError("violation_tolerance must be non-negative")
        if not 0.0 <= self.min_cold_fraction <= 1.0:
            raise ValueError("min_cold_fraction must be in [0, 1]")
        if self.min_lease_length <= 0.0:
            raise ValueError("min_lease_length must be positive")
        if self.min_documents_per_dataset <= 0:
            raise ValueError("min_documents_per_dataset must be positive")


def _lease_lengths(steps: Sequence[Mapping[str, Any]]) -> list[int]:
    lengths: list[int] = []
    current = 0
    for step in steps:
        valid = bool(step.get("valid", False))
        expired = bool(step.get("expired", False))
        if valid:
            current += 1
        if expired:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def evaluate_lease_trace(
    record: Mapping[str, Any], config: LeaseEvaluationConfig
) -> dict[str, Any]:
    """Summarize one V2 trace without using future attention as an input."""

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
            "reason": "lease trace has no decode steps",
        }

    valid_steps = [step for step in steps if bool(step.get("valid", False))]
    bounds = [_finite(step.get("bound"), "bound") for step in steps]
    actual = [_finite(step.get("actual_cold_mass"), "actual_cold_mass") for step in steps]
    if any(value < 0.0 or value > 1.0 for value in actual):
        raise ValueError("actual_cold_mass must be in [0, 1]")
    violations = [
        current > bound + config.violation_tolerance
        for current, bound in zip(actual, bounds)
    ]
    cold_values = [
        _finite(step.get("cold_fraction"), "cold_fraction")
        for step in valid_steps
    ]
    if any(value < 0.0 or value > 1.0 for value in cold_values):
        raise ValueError("cold_fraction must be in [0, 1]")
    lengths = _lease_lengths(steps)
    # ``expired`` itself implies a refresh at that step.  This fallback keeps
    # traces written by the first V2 collector version auditable even though
    # they did not set ``full_source_score`` on the expiry row.
    full_source_scores = sum(
        bool(step.get("full_source_score", False)) or bool(step.get("expired", False))
        for step in steps
    )
    refresh_rate = full_source_scores / len(steps)
    return {
        "status": "ok",
        "sample_id": str(record.get("sample_id", "unknown")),
        "dataset": str(record.get("dataset", "unknown")),
        "steps": len(steps),
        "metrics": {
            "decode_steps": len(steps),
            "valid_steps": len(valid_steps),
            "lease_count": len(lengths),
            "lease_lengths": lengths,
            "median_lease_length": statistics.median(lengths) if lengths else 0.0,
            "mean_lease_length": statistics.fmean(lengths) if lengths else 0.0,
            "certified_cold_fraction": statistics.fmean(cold_values) if cold_values else 0.0,
            "violation_rate": sum(violations) / len(violations),
            "mean_bound": statistics.fmean(bounds),
            "mean_actual_cold_mass": statistics.fmean(actual),
            "mean_bound_tightness": statistics.fmean(
                bound - current for bound, current in zip(bounds, actual)
            ),
            "refresh_rate": refresh_rate,
            "source_score_avoidance": 1.0 - refresh_rate,
        },
    }


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row["metrics"][key]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def aggregate_lease_results(
    rows: Sequence[Mapping[str, Any]], config: LeaseEvaluationConfig
) -> dict[str, Any]:
    """Aggregate per-sample lease metrics and apply the physical follow-up gate."""

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    datasets: dict[str, int] = {}
    for row in ok_rows:
        name = str(row.get("dataset", "unknown"))
        datasets[name] = datasets.get(name, 0) + 1
    if len(datasets) < 2 or min(datasets.values(), default=0) < config.min_documents_per_dataset:
        return {
            "status": "INCONCLUSIVE",
            "reason": "minimum_dataset_or_document_gate_not_met",
            "datasets": datasets,
            "ok_documents": len(ok_rows),
        }

    metric_names = (
        "decode_steps",
        "valid_steps",
        "lease_count",
        "median_lease_length",
        "mean_lease_length",
        "certified_cold_fraction",
        "violation_rate",
        "mean_bound",
        "mean_actual_cold_mass",
        "mean_bound_tightness",
        "refresh_rate",
        "source_score_avoidance",
    )
    metrics = {name: _mean_metric(ok_rows, name) for name in metric_names}
    dataset_metrics = {
        dataset: {
            name: _mean_metric(
                [row for row in ok_rows if str(row.get("dataset", "unknown")) == dataset],
                name,
            )
            for name in metric_names
        }
        for dataset in datasets
    }
    gate = {
        "certificate_valid": all(
            values["violation_rate"] <= config.max_violation_rate
            for values in dataset_metrics.values()
        ),
        "cold_fraction": metrics["certified_cold_fraction"] >= config.min_cold_fraction,
        "median_lease_length": metrics["median_lease_length"] >= config.min_lease_length,
        "refresh_saving": metrics["source_score_avoidance"] > 0.0,
    }
    return {
        "status": "GO_PHYSICAL_FOLLOWUP" if all(gate.values()) else "STOP_BEFORE_PHYSICAL",
        "datasets": datasets,
        "ok_documents": len(ok_rows),
        "metrics": metrics,
        "dataset_metrics": dataset_metrics,
        "gate": gate,
        "thresholds": {
            "max_violation_rate": config.max_violation_rate,
            "violation_tolerance": config.violation_tolerance,
            "min_cold_fraction": config.min_cold_fraction,
            "min_lease_length": config.min_lease_length,
        },
    }
