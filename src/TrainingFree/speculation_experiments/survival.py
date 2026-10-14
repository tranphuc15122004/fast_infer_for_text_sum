"""Small, grouped and calibrated survival estimators for B2."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from TrainingFree.speculation_experiments.protocol import assign_group_folds


DEFAULT_VARIANTS: dict[str, tuple[str, ...]] = {
    "R": (),
    "R-H": ("parent_entropy",),
    "R-C": ("parent_confidence",),
    "R-A": ("history_ema", "history_missing"),
    "R-HA": ("parent_entropy", "history_ema", "history_missing"),
    "R-PreAll": ("parent_entropy", "parent_confidence", "history_ema", "history_missing"),
    "R-DH": ("draft_entropy_max",),
    "R-DC": ("draft_confidence_min",),
    "R-A-DH": ("history_ema", "history_missing", "draft_entropy_max"),
    "R-A-DC": ("history_ema", "history_missing", "draft_confidence_min"),
    "R-DAll": ("history_ema", "history_missing", "parent_entropy", "parent_confidence", "draft_entropy_max", "draft_confidence_min"),
    "R-DAll-noDH": ("history_ema", "history_missing", "parent_entropy", "parent_confidence", "draft_confidence_min"),
}

LAGGED_VARIANTS: dict[str, tuple[str, ...]] = {
    "R-A-LDH": ("history_ema", "history_missing", "lagged_draft_entropy_max"),
    "R-A-LDC": ("history_ema", "history_missing", "lagged_draft_confidence_min"),
    "R-A-LDHC": (
        "history_ema", "history_missing", "lagged_draft_entropy_max",
        "lagged_draft_confidence_min",
    ),
}
SURVIVAL_VARIANTS = {**DEFAULT_VARIANTS, **LAGGED_VARIANTS}


def _context_bin(value: int) -> int:
    return min(max(int(value), 0) // 4096, 3)


def _progress_bin(row: Mapping[str, Any]) -> int:
    processed = max(0, int(row.get("processed_output_tokens", 0)))
    return 0 if processed < 128 else (1 if processed < 512 else 2)


def _signal_value(row: Mapping[str, Any], feature: str, threshold: int) -> float:
    uncertainty = row.get("draft_uncertainty") or {}
    lagged = row.get("lagged_signals") or {}
    if feature.startswith("lagged_"):
        lagged_feature = feature.removeprefix("lagged_")
        try:
            value = float(lagged[lagged_feature])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"candidate_signals row is missing finite {feature}") from exc
        if not math.isfinite(value):
            raise ValueError(f"candidate_signals row is missing finite {feature}")
        return value
    if feature == "draft_entropy_max":
        values = uncertainty.get("normalized_entropy")
        if values is None or len(values) < threshold:
            raise ValueError(f"candidate_signals row requires {threshold} draft entropy values")
        return max(float(value) for value in values[:threshold])
    if feature == "draft_confidence_min":
        values = uncertainty.get("confidence")
        if values is None or len(values) < threshold:
            raise ValueError(f"candidate_signals row requires {threshold} draft confidence values")
        return min(float(value) for value in values[:threshold])
    if feature == "parent_entropy":
        signal = row.get("parent_signal") or {}
        value = signal.get("normalized_entropy")
    elif feature == "parent_confidence":
        signal = row.get("parent_signal") or {}
        value = signal.get("confidence")
    elif feature == "history_missing":
        return 1.0 if bool(row.get("history_missing", row.get("history_ema") is None)) else 0.0
    else:
        value = row.get(feature)
        if feature == "history_ema" and value is None:
            return -1.0
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"candidate_signals row is missing finite {feature}")
    return float(value)


def _quantile_edges(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return ()
    edges = []
    for fraction in (0.25, 0.50, 0.75):
        index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))
        edge = ordered[index]
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return tuple(edges)


def _quartile(value: float, edges: Sequence[float]) -> int:
    # Fit boundaries only from training folds. `bisect_right` places a value
    # equal to a boundary in the next interval, consistently for train/test.
    import bisect

    return bisect.bisect_right(edges, float(value))


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[start]]):
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index in range(start, end):
            ranks[order[index]] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = _average_ranks(left)
    y = _average_ranks(right)
    x_mean = mean(x)
    y_mean = mean(y)
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    denominator = math.sqrt(sum(value * value for value in x_centered) * sum(value * value for value in y_centered))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(x_centered, y_centered)) / denominator


def _descriptive_signal_correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feature_rows: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row in rows:
        uncertainty = row["draft_uncertainty"]
        entropy = [float(value) for value in uncertainty["normalized_entropy"]]
        confidence = [float(value) for value in uncertainty["confidence"]]
        parent = row.get("parent_signal") or {}
        values = {
            "draft_entropy_mean": mean(entropy),
            "draft_entropy_max": max(entropy),
            "draft_confidence_mean": mean(confidence),
            "draft_confidence_min": min(confidence),
        }
        if parent.get("normalized_entropy") is not None:
            values["parent_entropy"] = float(parent["normalized_entropy"])
        if parent.get("confidence") is not None:
            values["parent_confidence"] = float(parent["confidence"])
        if row.get("history_ema") is not None:
            values["history_ema"] = float(row["history_ema"])
        for name, value in values.items():
            feature_rows[name].append((row, value))

    output: dict[str, Any] = {}
    for name, pairs in feature_rows.items():
        pooled_x = [value for _, value in pairs]
        pooled_y = [float(row["accepted_candidates"]) for row, _ in pairs]
        by_dataset: dict[str, list[tuple[float, float]]] = defaultdict(list)
        by_document: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for row, value in pairs:
            y = float(row["accepted_candidates"])
            by_dataset[str(row["dataset"])].append((value, y))
            by_document[(str(row["dataset"]), str(row["source_group_id"]))].append((value, y))
        within_document = [
            correlation
            for values in by_document.values()
            if (correlation := _spearman([item[0] for item in values], [item[1] for item in values])) is not None
        ]
        output[name] = {
            "pooled": _spearman(pooled_x, pooled_y),
            "by_dataset": {
                dataset: _spearman([item[0] for item in values], [item[1] for item in values])
                for dataset, values in by_dataset.items()
            },
            "within_document_mean": mean(within_document) if within_document else None,
            "within_document_groups": len(within_document),
        }
    return output


def _record(row: Mapping[str, Any], threshold: int) -> dict[str, Any]:
    accepted = int(row.get("accepted_candidates", -1))
    if not 0 <= accepted <= 15:
        raise ValueError("candidate_signals accepted_candidates must be within [0, 15]")
    return {
        "label": int(accepted >= threshold),
        "regime": (_context_bin(int(row.get("prompt_tokens", row.get("context_tokens", 0)))), _progress_bin(row)),
    }


def _fit_threshold(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: int,
    features: Sequence[str],
    min_support_states: int,
    min_support_groups: int,
) -> dict[str, Any]:
    feature_edges = {
        feature: _quantile_edges([_signal_value(row, feature, threshold) for row in train_rows])
        for feature in features
    }
    counts: dict[tuple[Any, ...], list[Any]] = {}

    def add(key: tuple[Any, ...], label: int, group: str) -> None:
        cell = counts.setdefault(key, [0, 0, set()])
        cell[0] += int(label)
        cell[1] += 1
        cell[2].add(group)

    global_key = ("global",)
    for row in train_rows:
        datum = _record(row, threshold)
        group = str(row["source_group_id"])
        regime_key = ("regime", *datum["regime"])
        add(global_key, datum["label"], group)
        add(regime_key, datum["label"], group)
        if features:
            signal_bins = tuple(
                _quartile(_signal_value(row, feature, threshold), feature_edges[feature])
                for feature in features
            )
            add(("signal", *datum["regime"], *signal_bins), datum["label"], group)

    return {
        "threshold": threshold,
        "features": tuple(features),
        "feature_edges": feature_edges,
        "counts": counts,
        "min_support_states": min_support_states,
        "min_support_groups": min_support_groups,
    }


def _predict_threshold(model: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[float, str]:
    threshold = int(model["threshold"])
    datum = _record(row, threshold)
    counts = model["counts"]
    regime_key = ("regime", *datum["regime"])
    selected_key = global_key = ("global",)
    fallback = "global"
    features = model["features"]
    if features:
        signal_bins = tuple(
            _quartile(_signal_value(row, feature, threshold), model["feature_edges"][feature])
            for feature in features
        )
        candidate = ("signal", *datum["regime"], *signal_bins)
        support = counts.get(candidate)
        if support and support[1] >= model["min_support_states"] and len(support[2]) >= model["min_support_groups"]:
            selected_key = candidate
            fallback = "signal"
    if fallback == "global":
        support = counts.get(regime_key)
        if support and support[1] >= model["min_support_states"] and len(support[2]) >= model["min_support_groups"]:
            selected_key = regime_key
            fallback = "regime"
    successes, total, _ = counts[selected_key]
    return (successes + 1.0) / (total + 2.0), fallback


def fit_survival_curve(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    max_candidates: int = 15,
    min_support_states: int = 8,
    min_support_groups: int = 3,
) -> dict[str, Any]:
    """Fit training-only binned Beta(1,1)-smoothed survival curves."""

    if variant not in SURVIVAL_VARIANTS:
        raise ValueError(f"unsupported survival variant {variant!r}")
    if not train_rows or max_candidates < 1 or min_support_states < 1 or min_support_groups < 1:
        raise ValueError("survival fit requires rows and positive support settings")
    features = SURVIVAL_VARIANTS[variant]
    models = [
        _fit_threshold(
            train_rows,
            threshold=index,
            features=features,
            min_support_states=min_support_states,
            min_support_groups=min_support_groups,
        )
        for index in range(1, max_candidates + 1)
    ]
    return {
        "variant": variant,
        "training_groups": sorted({str(row["source_group_id"]) for row in train_rows}),
        "max_candidates": max_candidates,
        "models": models,
    }


def serialize_survival_curve(model: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a fitted curve into a stable JSON-safe deployment artifact."""

    variant = str(model.get("variant", ""))
    if variant not in SURVIVAL_VARIANTS:
        raise ValueError(f"unsupported survival variant {variant!r}")
    serialized_models = []
    for threshold_model in model.get("models", ()):
        counts = []
        for key, value in threshold_model["counts"].items():
            successes, total, groups = value
            counts.append({
                "key": list(key),
                "successes": int(successes),
                "total": int(total),
                "groups": sorted(str(group) for group in groups),
            })
        serialized_models.append({
            "threshold": int(threshold_model["threshold"]),
            "features": list(threshold_model["features"]),
            "feature_edges": {
                str(name): list(edges)
                for name, edges in threshold_model["feature_edges"].items()
            },
            "counts": counts,
            "min_support_states": int(threshold_model["min_support_states"]),
            "min_support_groups": int(threshold_model["min_support_groups"]),
        })
    if len(serialized_models) != int(model.get("max_candidates", 0)):
        raise ValueError("fitted survival curve has an inconsistent threshold count")
    return {
        "schema_version": "dflash-speculation-b.survival-model.v1",
        "variant": variant,
        "training_groups": [str(value) for value in model.get("training_groups", ())],
        "max_candidates": int(model["max_candidates"]),
        "models": serialized_models,
    }


def deserialize_survival_curve(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Load and validate a JSON-safe fitted survival curve."""

    if payload.get("schema_version") != "dflash-speculation-b.survival-model.v1":
        raise ValueError("unsupported survival-model schema")
    variant = str(payload.get("variant", ""))
    if variant not in SURVIVAL_VARIANTS:
        raise ValueError(f"unsupported survival variant {variant!r}")
    models = []
    for raw_model in payload.get("models", ()):
        counts: dict[tuple[Any, ...], list[Any]] = {}
        for item in raw_model.get("counts", ()):
            key = tuple(item["key"])
            if key in counts:
                raise ValueError("survival-model artifact has duplicate count keys")
            total = int(item["total"])
            successes = int(item["successes"])
            groups = {str(value) for value in item.get("groups", ())}
            if total < 0 or not 0 <= successes <= total or len(groups) > total:
                raise ValueError("survival-model artifact has invalid counts")
            counts[key] = [successes, total, groups]
        models.append({
            "threshold": int(raw_model["threshold"]),
            "features": tuple(str(value) for value in raw_model.get("features", ())),
            "feature_edges": {
                str(name): tuple(float(value) for value in edges)
                for name, edges in raw_model.get("feature_edges", {}).items()
            },
            "counts": counts,
            "min_support_states": int(raw_model["min_support_states"]),
            "min_support_groups": int(raw_model["min_support_groups"]),
        })
    max_candidates = int(payload.get("max_candidates", 0))
    if max_candidates < 1 or len(models) != max_candidates:
        raise ValueError("survival-model artifact has an invalid candidate count")
    if [row["threshold"] for row in models] != list(range(1, max_candidates + 1)):
        raise ValueError("survival-model thresholds must be the ordered range 1..max_candidates")
    return {
        "variant": variant,
        "training_groups": [str(value) for value in payload.get("training_groups", ())],
        "max_candidates": max_candidates,
        "models": models,
    }


def predict_survival_curve(
    model: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[list[float], list[str]]:
    probabilities: list[float] = []
    fallback: list[str] = []
    for threshold_model in model["models"]:
        probability, level = _predict_threshold(threshold_model, row)
        probabilities.append(probability)
        fallback.append(level)
    # Survival is nonincreasing in candidate position. Repair independently
    # predicted bins with a cumulative minimum, and expose the repair count.
    for index in range(1, len(probabilities)):
        probabilities[index] = min(probabilities[index], probabilities[index - 1])
    return probabilities, fallback


def _document_brier(predictions: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    by_document: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        probabilities = row["predictions"][variant]
        labels = row["labels"]
        score = mean((float(probability) - int(label)) ** 2 for probability, label in zip(probabilities, labels))
        by_document[(row["dataset"], row["source_group_id"])].append(score)
    for (dataset, group), values in by_document.items():
        by_dataset[dataset][group] = values
    dataset_means = [
        mean([mean(values) for values in groups.values()])
        for groups in by_dataset.values()
        if groups
    ]
    return {"integrated_brier": mean(dataset_means), "per_dataset": {
        dataset: mean([mean(values) for values in groups.values()])
        for dataset, groups in by_dataset.items()
    }}


def _bootstrap_brier_skill(
    predictions: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    baseline: str,
    replicates: int,
    seed: int,
) -> list[float]:
    documents: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        documents[str(row["dataset"])][str(row["source_group_id"])].append(row)
    document_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for dataset, groups in documents.items():
        for group, group_rows in groups.items():
            per_variant = {}
            for name in (variant, baseline):
                state_scores = []
                for row in group_rows:
                    probabilities = row["predictions"][name]
                    state_scores.append(mean(
                        (float(probability) - int(label)) ** 2
                        for probability, label in zip(probabilities, row["labels"])
                    ))
                per_variant[name] = mean(state_scores)
            document_scores[dataset][group] = per_variant
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        dataset_scores: dict[str, dict[str, list[float]]] = {
            dataset: {variant: [], baseline: []} for dataset in documents
        }
        for dataset, groups in documents.items():
            sampled = [rng.choice(list(groups)) for _ in groups]
            for group in sampled:
                for name in (variant, baseline):
                    dataset_scores[dataset][name].append(document_scores[dataset][group][name])
        # Match _document_brier: average source groups within each dataset,
        # then macro-average datasets so larger cohorts do not dominate.
        base_score = mean(
            mean(scores[baseline])
            for scores in dataset_scores.values()
            if scores[baseline]
        )
        signal_score = mean(
            mean(scores[variant])
            for scores in dataset_scores.values()
            if scores[variant]
        )
        values.append((base_score - signal_score) / base_score if base_score else 0.0)
    return values


def cross_validated_survival(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 4,
    seed: int = 42,
    variants: Sequence[str] = tuple(DEFAULT_VARIANTS),
    min_support_states: int = 8,
    min_support_groups: int = 3,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261013,
) -> dict[str, Any]:
    """Evaluate survival curves out of fold, holding whole source groups out."""

    signals = [row for row in rows if row.get("type") == "candidate_signals"]
    if not signals:
        raise ValueError("no candidate_signals rows were provided")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be nonnegative")
    variant_values = tuple(variants)
    if "R" not in variant_values or any(value not in SURVIVAL_VARIANTS for value in variant_values):
        raise ValueError("variants must include R and use supported model names")
    for row in signals:
        uncertainty = row.get("draft_uncertainty") or {}
        for name in ("normalized_entropy", "confidence"):
            if uncertainty.get(name) is None or len(uncertainty[name]) != 15:
                raise ValueError(f"candidate_signals rows require 15 candidates in {name}")
    state_ids = [str(row.get("state_id", "")) for row in signals]
    if any(not value for value in state_ids) or len(state_ids) != len(set(state_ids)):
        raise ValueError("candidate_signals rows require unique nonempty state_id values")

    group_folds = assign_group_folds(signals, n_splits=n_splits, seed=seed)
    prediction_rows: list[dict[str, Any]] = []
    fallback_counts: dict[str, dict[str, int]] = {variant: defaultdict(int) for variant in variant_values}
    for fold in sorted(set(group_folds.values())):
        train_rows = [row for row in signals if group_folds[str(row["source_group_id"])] != fold]
        test_rows = [row for row in signals if group_folds[str(row["source_group_id"])] == fold]
        train_groups = sorted({str(row["source_group_id"]) for row in train_rows})
        fitted = {
            variant: fit_survival_curve(
                train_rows,
                variant=variant,
                min_support_states=min_support_states,
                min_support_groups=min_support_groups,
            )
            for variant in variant_values
        }
        for row in test_rows:
            predictions: dict[str, list[float]] = {}
            for variant in variant_values:
                curve, levels = predict_survival_curve(fitted[variant], row)
                predictions[variant] = curve
                for level in levels:
                    fallback_counts[variant][level] += 1
            labels = [int(int(row["accepted_candidates"]) >= index) for index in range(1, 16)]
            prediction_rows.append({
                "schema_version": "dflash-speculation-b.oof-prediction.v1",
                "state_id": str(row["state_id"]),
                "dataset": str(row["dataset"]),
                "source_group_id": str(row["source_group_id"]),
                "fold": fold,
                "training_groups": train_groups,
                "labels": labels,
                "predictions": predictions,
            })

    brier = {variant: _document_brier(prediction_rows, variant) for variant in variant_values}
    candidate_risk = []
    for candidate_index in range(15):
        reached = [row for row in signals if int(row["accepted_candidates"]) >= candidate_index]
        rejected = [row for row in reached if int(row["accepted_candidates"]) == candidate_index]
        high_confidence_reached = [
            row for row in reached
            if float(row["draft_uncertainty"]["confidence"][candidate_index]) >= 0.9
        ]
        high_confidence_rejected = [
            row for row in high_confidence_reached
            if int(row["accepted_candidates"]) == candidate_index
        ]
        candidate_risk.append({
            "candidate_index": candidate_index + 1,
            "reached_states": len(reached),
            "rejections": len(rejected),
            "rejection_rate": len(rejected) / len(reached) if reached else None,
            "mean_normalized_entropy": (
                mean(float(row["draft_uncertainty"]["normalized_entropy"][candidate_index]) for row in reached)
                if reached else None
            ),
            "mean_confidence": (
                mean(float(row["draft_uncertainty"]["confidence"][candidate_index]) for row in reached)
                if reached else None
            ),
            "confidence_ge_0_9_reached": len(high_confidence_reached),
            "confidence_ge_0_9_rejections": len(high_confidence_rejected),
            "confidence_ge_0_9_rejection_rate": (
                len(high_confidence_rejected) / len(high_confidence_reached)
                if high_confidence_reached else None
            ),
        })
    skills = {}
    intervals = {}
    for variant in variant_values:
        if variant == "R":
            continue
        control = brier["R"]["integrated_brier"]
        signal = brier[variant]["integrated_brier"]
        skills[variant] = (control - signal) / control if control else None
        if bootstrap_replicates:
            draws = _bootstrap_brier_skill(
                prediction_rows,
                variant=variant,
                baseline="R",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            ordered = sorted(draws)
            low = ordered[max(0, int(0.025 * len(ordered)))]
            high = ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))]
            intervals[variant] = [low, high]
        else:
            intervals[variant] = None
    paired_controls = {
        variant: control
        for variant, control in (
            ("R-A-DH", "R-A"),
            ("R-A-DC", "R-A"),
            ("R-DH", "R-C"),
            ("R-DAll", "R-DAll-noDH"),
            ("R-A-LDH", "R-A"),
            ("R-A-LDC", "R-A"),
        )
        if variant in variant_values and control in variant_values
    }
    control_skills: dict[str, float | None] = {}
    control_intervals: dict[str, list[float] | None] = {}
    for variant, control in paired_controls.items():
        base_score = brier[control]["integrated_brier"]
        signal_score = brier[variant]["integrated_brier"]
        control_skills[f"{variant}_vs_{control}"] = (
            (base_score - signal_score) / base_score if base_score else None
        )
        if bootstrap_replicates:
            draws = _bootstrap_brier_skill(
                prediction_rows,
                variant=variant,
                baseline=control,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            ordered = sorted(draws)
            control_intervals[f"{variant}_vs_{control}"] = [
                ordered[max(0, int(0.025 * len(ordered)))],
                ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))],
            ]
        else:
            control_intervals[f"{variant}_vs_{control}"] = None
    return {
        "schema_version": "dflash-speculation-b.b2-analysis.v1",
        "status": "complete",
        "state_count": len(signals),
        "source_group_count": len(group_folds),
        "fold_assignments": group_folds,
        "variants": list(variant_values),
        "brier_by_variant": {variant: value["integrated_brier"] for variant, value in brier.items()},
        "per_dataset_brier": {variant: value["per_dataset"] for variant, value in brier.items()},
        "brier_skill_vs_R": skills,
        "brier_skill_ci95": intervals,
        "brier_skill_vs_control": control_skills,
        "brier_skill_ci95_vs_control": control_intervals,
        "candidate_risk_diagnostics": candidate_risk,
        "descriptive_spearman_vs_accepted": _descriptive_signal_correlations(signals),
        "fallback_counts": {variant: dict(counts) for variant, counts in fallback_counts.items()},
        "oof_predictions": prediction_rows,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "min_support_states": min_support_states,
        "min_support_groups": min_support_groups,
        "unit_of_analysis": "source_group_document_macro_stratified_by_dataset",
    }
