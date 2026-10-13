"""Paired source-group aggregation and stratified cluster bootstrap."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile from no bootstrap samples")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _aggregate_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_field: str,
    variant_field: str,
) -> dict[str, dict[str, tuple[float, float]]]:
    values: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        dataset = str(row.get("dataset", ""))
        group = str(row.get("source_group_id", ""))
        if not dataset or not group:
            raise ValueError("every paired row requires dataset and source_group_id")
        try:
            baseline = float(row[baseline_field])
            variant = float(row[variant_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("paired rows are missing numeric baseline/variant values") from exc
        if not math.isfinite(baseline) or not math.isfinite(variant):
            raise ValueError("paired metric values must be finite")
        values[(dataset, group)].append((baseline, variant))
    grouped: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for (dataset, group), pairs in values.items():
        grouped[dataset][group] = (
            sum(pair[0] for pair in pairs) / len(pairs),
            sum(pair[1] for pair in pairs) / len(pairs),
        )
    return dict(grouped)


def _statistic(
    sampled: Mapping[str, Sequence[tuple[float, float]]],
    metric: str,
) -> float:
    per_dataset: list[float] = []
    for dataset in sorted(sampled):
        pairs = sampled[dataset]
        if not pairs:
            continue
        if metric == "difference":
            per_dataset.append(sum(variant - baseline for baseline, variant in pairs) / len(pairs))
        elif metric == "speedup":
            if any(baseline <= 0 or variant <= 0 for baseline, variant in pairs):
                raise ValueError("speedup latency values must be positive")
            per_dataset.append(sum(math.log(baseline / variant) for baseline, variant in pairs) / len(pairs))
        else:
            raise ValueError("metric must be 'difference' or 'speedup'")
    if not per_dataset:
        raise ValueError("at least one dataset with source groups is required")
    macro = sum(per_dataset) / len(per_dataset)
    return macro if metric == "difference" else math.exp(macro)


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_field: str,
    variant_field: str,
    metric: str,
    resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap source groups within each dataset, keeping pairs intact.

    Repeated prefixes/repetitions for a group are averaged before resampling,
    preventing them from being treated as independent documents. For speedup,
    each source group's paired mean latency ratio is aggregated geometrically
    within dataset, then macro-geometrically across datasets.
    """
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    grouped = _aggregate_groups(rows, baseline_field=baseline_field, variant_field=variant_field)
    if not grouped:
        raise ValueError("no paired observations were supplied")
    source_counts = {dataset: len(groups) for dataset, groups in grouped.items()}
    if any(count < 1 for count in source_counts.values()):
        raise ValueError("every dataset stratum must contain at least one source group")
    original = {
        dataset: tuple(groups[group] for group in sorted(groups))
        for dataset, groups in grouped.items()
    }
    estimate = _statistic(original, metric)
    rng = random.Random(int(seed))
    bootstrap: list[float] = []
    for _ in range(resamples):
        sample: dict[str, list[tuple[float, float]]] = {}
        for dataset, pairs in original.items():
            sample[dataset] = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        bootstrap.append(_statistic(sample, metric))
    alpha = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "estimate": estimate,
        "ci_low": _quantile(bootstrap, alpha),
        "ci_high": _quantile(bootstrap, 1.0 - alpha),
        "confidence": confidence,
        "resamples": resamples,
        "seed": int(seed),
        "n_source_groups": sum(source_counts.values()),
        "source_groups_by_dataset": source_counts,
        "datasets": sorted(source_counts),
        "resampling_unit": "source_group_id",
        "stratified_by": "dataset",
    }
