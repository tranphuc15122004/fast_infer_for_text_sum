"""Statistical summaries and report helpers for GroundSync traces."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from csv import DictWriter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core import (
    bootstrap_mean_ci,
    grounding_horizon,
    js_divergence,
    normalize_distribution,
    persistence_summary,
    policy_k,
)


def split_by_document(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split rows by sorted document id, never by individual token/step."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError("dev_fraction must be non-negative and below one")
    if train_fraction + dev_fraction >= 1.0:
        raise ValueError("train_fraction + dev_fraction must be below one")
    document_ids = sorted({str(row["document_id"]) for row in rows})
    if not document_ids:
        raise ValueError("rows must not be empty")
    train_count = max(1, int(len(document_ids) * train_fraction))
    remaining = len(document_ids) - train_count
    dev_count = int(len(document_ids) * dev_fraction)
    dev_count = min(max(dev_count, 0), max(remaining - 1, 0))
    train_ids = set(document_ids[:train_count])
    dev_ids = set(document_ids[train_count : train_count + dev_count])
    train = [row for row in rows if str(row["document_id"]) in train_ids]
    dev = [row for row in rows if str(row["document_id"]) in dev_ids]
    test = [
        row
        for row in rows
        if str(row["document_id"]) not in train_ids
        and str(row["document_id"]) not in dev_ids
    ]
    return train, dev, test


def _finite_scores(values: Iterable[float]) -> list[float]:
    scores = [float(value) for value in values]
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("prediction scores must be finite")
    return [min(max(value, 0.0), 1.0) for value in scores]


def binary_prediction_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
) -> dict[str, Any]:
    """Compute robust binary metrics, including explicit single-class status."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        return {
            "status": "empty",
            "count": 0,
            "auroc": None,
            "auprc": None,
            "log_loss": None,
            "brier": None,
        }
    y_true = [int(bool(value)) for value in labels]
    y_score = _finite_scores(scores)
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    varied = len(set(y_true)) == 2
    return {
        "status": "ok" if varied else "insufficient_class_variation",
        "count": len(y_true),
        "positive_rate": statistics.fmean(y_true),
        "auroc": float(roc_auc_score(y_true, y_score)) if varied else None,
        "auprc": float(average_precision_score(y_true, y_score)) if varied else None,
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def decide_status(
    *,
    value: float | None,
    threshold: float,
    direction: str,
    count: int,
    min_count: int,
) -> str:
    """Convert one metric and coverage into a conservative hypothesis status."""

    if value is None or not math.isfinite(float(value)):
        return "UNAVAILABLE"
    if count < min_count:
        return "INCONCLUSIVE"
    value = float(value)
    if direction == ">=":
        passed = value >= threshold
    elif direction == ">":
        passed = value > threshold
    elif direction == "<=":
        passed = value <= threshold
    elif direction == "<":
        passed = value < threshold
    elif direction == "==":
        passed = value == threshold
    else:
        raise ValueError(f"unsupported direction: {direction}")
    return "PASS" if passed else "FAIL"


def fit_controlled_predictor(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    label_key: str = "label",
) -> dict[str, Any]:
    """Fit a small document-split logistic predictor and report test metrics."""

    if not feature_names:
        raise ValueError("feature_names must not be empty")
    train, dev, test = split_by_document(rows)
    if not train or not test:
        return {
            "status": "insufficient_document_split",
            "feature_names": list(feature_names),
            "train_count": len(train),
            "dev_count": len(dev),
            "test_count": len(test),
        }
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def matrix(part: Sequence[Mapping[str, Any]]) -> tuple[Any, list[int]]:
        values: list[list[float]] = []
        labels: list[int] = []
        for row in part:
            try:
                vector = [float(row[name]) for name in feature_names]
                label = int(bool(row[label_key]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"row missing finite predictor fields: {exc}") from exc
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("predictor features must be finite")
            values.append(vector)
            labels.append(label)
        return np.asarray(values, dtype=float), labels

    x_train, y_train = matrix(train)
    x_test, y_test = matrix(test)
    if len(set(y_train)) < 2:
        return {
            "status": "insufficient_training_class_variation",
            "feature_names": list(feature_names),
            "train_count": len(train),
            "dev_count": len(dev),
            "test_count": len(test),
        }
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=300, random_state=0),
    )
    model.fit(x_train, y_train)
    dev_scores = (
        model.predict_proba(matrix(dev)[0])[:, 1].tolist()
        if dev else []
    )
    scores = model.predict_proba(x_test)[:, 1].tolist()
    classifier = model[-1]
    metrics = binary_prediction_metrics(y_test, scores)

    def prediction_details(
        part: Sequence[Mapping[str, Any]],
        values: Sequence[float],
        labels: Sequence[int],
    ) -> list[dict[str, Any]]:
        return [
            {
                "document_id": str(row.get("document_id")),
                "start_position": row.get("start_position"),
                "step_position": row.get("step_position"),
                "score": float(score),
                "label": int(label),
            }
            for row, score, label in zip(part, values, labels)
        ]

    return {
        "status": "ok",
        "feature_names": list(feature_names),
        "train_count": len(train),
        "dev_count": len(dev),
        "test_count": len(test),
        "train_documents": sorted({str(row["document_id"]) for row in train}),
        "dev_documents": sorted({str(row["document_id"]) for row in dev}),
        "test_documents": sorted({str(row["document_id"]) for row in test}),
        "metrics": metrics,
        "dev_predictions": prediction_details(dev, dev_scores, matrix(dev)[1]) if dev else [],
        "test_predictions": prediction_details(test, scores, y_test),
        "coefficients": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write an indented UTF-8 JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2), encoding="utf-8")


def _ok_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def _mean_or_none(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(values) if values else None


def _trace_variant(row: Mapping[str, Any], variant: str) -> list[list[float]]:
    trace: list[list[float]] = []
    for step in row.get("attention", []):
        value = step.get(variant)
        if value is None:
            continue
        trace.append([float(item) for item in value])
    return trace


def _resample_distribution(values: Sequence[float], width: int) -> list[float]:
    """Interpolate a distribution on a fixed relative-position grid."""

    if width <= 0:
        raise ValueError("width must be positive")
    source = normalize_distribution(values)
    if len(source) == width:
        return source
    result: list[float] = []
    scale = len(source) / width
    for index in range(width):
        coordinate = (index + 0.5) * scale - 0.5
        left = max(0, min(len(source) - 1, int(math.floor(coordinate))))
        right = max(0, min(len(source) - 1, left + 1))
        weight = coordinate - math.floor(coordinate)
        result.append(source[left] * (1.0 - weight) + source[right] * weight)
    return normalize_distribution(result)


def fit_relative_positional_prior(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str = "raw",
    bins: int = 32,
) -> list[float]:
    """Learn a document-balanced source-position prior from target traces."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    document_means: list[list[float]] = []
    for row in _ok_rows(rows):
        vectors = [
            _resample_distribution(step[variant], bins)
            for step in row.get("attention", [])
            if step.get(variant)
        ]
        if not vectors:
            continue
        document_means.append([
            statistics.fmean(vector[index] for vector in vectors)
            for index in range(bins)
        ])
    if not document_means:
        return []
    prior = [
        max(statistics.fmean(values[index] for values in document_means), 1e-8)
        for index in range(bins)
    ]
    return normalize_distribution(prior)


def calibrated_trace(
    trace: Sequence[Sequence[float]],
    positional_prior: Sequence[float],
) -> list[list[float]]:
    """Divide each source distribution by a relative-position prior."""

    if not trace:
        return []
    prior = normalize_distribution(positional_prior)
    result: list[list[float]] = []
    for vector in trace:
        normalized = normalize_distribution(vector)
        local_prior = _resample_distribution(prior, len(normalized))
        adjusted = [
            value / max(baseline, 1e-8)
            for value, baseline in zip(normalized, local_prior)
        ]
        result.append(normalize_distribution(adjusted))
    return result


def _attach_trace_variant(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    traces: Mapping[str, Sequence[Sequence[float]]],
) -> list[dict[str, Any]]:
    """Return target rows with one derived attention variant attached."""

    result: list[dict[str, Any]] = []
    for row in rows:
        document_id = str(row.get("document_id"))
        derived = traces.get(document_id)
        if derived is None:
            result.append(dict(row))
            continue
        updated = dict(row)
        attention: list[dict[str, Any]] = []
        for index, step in enumerate(row.get("attention", [])):
            updated_step = dict(step)
            if index < len(derived):
                updated_step[variant] = list(derived[index])
            attention.append(updated_step)
        updated["attention"] = attention
        result.append(updated)
    return result


def _shuffled_adjacent_similarity(trace: Sequence[Sequence[float]], seed: int) -> float | None:
    if len(trace) < 2:
        return None
    shuffled = [list(value) for value in trace]
    random.Random(seed).shuffle(shuffled)
    values = [1.0 - js_divergence(left, right) for left, right in zip(shuffled, shuffled[1:])]
    return statistics.fmean(values) if values else None


def summarize_target_traces(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
    variant: str = "nosink",
    min_documents: int = 5,
) -> dict[str, Any]:
    """Aggregate H1 persistence and a shuffled null at document level."""

    summaries: list[dict[str, Any]] = []
    for row in _ok_rows(rows):
        trace = _trace_variant(row, variant)
        if not trace:
            continue
        summary = persistence_summary(trace, threshold=threshold, lags=lags)
        digest = hashlib.sha256(str(row.get("document_id")).encode("utf-8")).digest()
        null = _shuffled_adjacent_similarity(trace, seed=int.from_bytes(digest[:4], "big"))
        summary["document_id"] = str(row.get("document_id"))
        summary["null_adjacent_similarity"] = null
        summaries.append(summary)
    adjacent = _mean_or_none(
        item["lag_similarity"]["1"]["mean"]
        for item in summaries
        if item["lag_similarity"].get("1", {}).get("count", 0)
    )
    null_adjacent = _mean_or_none(
        item["null_adjacent_similarity"]
        for item in summaries
        if item["null_adjacent_similarity"] is not None
    )
    excess = None if adjacent is None or null_adjacent is None else adjacent - null_adjacent
    excess_by_document = [
        float(item["lag_similarity"]["1"]["mean"] - item["null_adjacent_similarity"])
        for item in summaries
        if item["null_adjacent_similarity"] is not None
        and item["lag_similarity"].get("1", {}).get("count", 0)
    ]
    excess_ci = bootstrap_mean_ci(excess_by_document, seed=42) if excess_by_document else None
    lag_means: dict[str, float | None] = {}
    for lag in lags:
        key = str(lag)
        lag_means[key] = _mean_or_none(
            item["lag_similarity"][key]["mean"]
            for item in summaries
            if item["lag_similarity"].get(key, {}).get("count", 0)
        )
    return {
        "status": "ok" if summaries else "UNAVAILABLE",
        "variant": variant,
        "documents": len(summaries),
        "token_steps": sum(len(_trace_variant(row, variant)) for row in _ok_rows(rows)),
        "mean_segment_length": _mean_or_none(item["mean_segment_length"] for item in summaries),
        "median_segment_length": _mean_or_none(item["median_segment_length"] for item in summaries),
        "lag_similarity": lag_means,
        "adjacent_similarity": adjacent,
        "null_adjacent_similarity": null_adjacent,
        "persistence_excess": excess,
        "persistence_excess_ci": excess_ci,
        "decision": decide_status(
            value=excess_ci["low"] if excess_ci else excess,
            threshold=0.02,
            direction=">=",
            count=len(summaries),
            min_count=min_documents,
        ),
        "documents_detail": summaries,
    }


def _rejection(row: Mapping[str, Any]) -> int:
    return int(not bool(row.get("fully_accepted", False)))


def summarize_speculative_traces(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_group_count: int = 5,
) -> dict[str, Any]:
    """Summarize drift/rejection association for H2."""

    usable = [
        row for row in _ok_rows(rows)
        if row.get("drift_at_start") is not None
        and math.isfinite(float(row["drift_at_start"]))
    ]
    if not usable:
        return {"status": "UNAVAILABLE", "count": 0, "decision": "UNAVAILABLE"}
    drifts = [float(row["drift_at_start"]) for row in usable]
    pivot = statistics.median(drifts)
    low = [row for row in usable if float(row["drift_at_start"]) <= pivot]
    high = [row for row in usable if float(row["drift_at_start"]) > pivot]
    low_rate = _mean_or_none(_rejection(row) for row in low)
    high_rate = _mean_or_none(_rejection(row) for row in high)
    delta = None if low_rate is None or high_rate is None else high_rate - low_rate
    return {
        "status": "ok",
        "count": len(usable),
        "drift_median": pivot,
        "low_count": len(low),
        "high_count": len(high),
        "low_rejection_rate": low_rate,
        "high_rejection_rate": high_rate,
        "high_minus_low_rejection_rate": delta,
        "decision": (
            "UNAVAILABLE" if delta is None else
            "INCONCLUSIVE" if min(len(low), len(high)) < min_group_count else
            "PASS" if delta > 0.0 else "FAIL"
        ),
    }


def summarize_rejection_hazard(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
) -> dict[str, Any]:
    """Compute a discrete first-rejection hazard by relative draft position.

    The descriptive risk-set table is accompanied by a small logistic hazard
    model.  Its drift coefficient is estimated after controlling for relative
    draft position, and its confidence interval resamples whole documents so
    repeated positions from one document are not treated as independent
    observations.
    """

    if max_k <= 0:
        raise ValueError("max_k must be positive")
    usable = _ok_rows(rows)
    by_position: dict[str, dict[str, float | int | None]] = {}
    for relative_position in range(1, max_k + 1):
        at_risk = 0
        events = 0
        for row in usable:
            row_max_k = int(row.get("max_k", max_k))
            first_reject = row.get("first_reject_rel")
            if row_max_k < relative_position:
                continue
            if first_reject is not None and int(first_reject) < relative_position:
                continue
            at_risk += 1
            if first_reject is not None and int(first_reject) == relative_position:
                events += 1
        by_position[str(relative_position)] = {
            "at_risk": at_risk,
            "events": events,
            "hazard": events / at_risk if at_risk else None,
        }
    result = {
        "status": "ok" if usable else "UNAVAILABLE",
        "count": len(usable),
        "max_k": max_k,
        "by_relative_position": by_position,
    }
    result["hazard_model"] = _fit_rejection_hazard_model(usable, max_k=max_k)
    return result


def _expand_rejection_hazard_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
) -> list[dict[str, float | int | str]]:
    """Expand one first-rejection observation into its risk-set rows."""

    expanded: list[dict[str, float | int | str]] = []
    for row_index, row in enumerate(rows):
        drift = row.get("drift_at_start")
        if drift is None:
            continue
        drift = float(drift)
        if not math.isfinite(drift):
            continue
        row_max_k = min(max_k, int(row.get("max_k", max_k)))
        first_reject = row.get("first_reject_rel")
        if first_reject is not None:
            first_reject = int(first_reject)
            if first_reject < 1 or first_reject > row_max_k:
                first_reject = None
        document_id = str(row.get("document_id", f"row-{row_index}"))
        for relative_position in range(1, row_max_k + 1):
            if first_reject is not None and first_reject < relative_position:
                continue
            expanded.append({
                "document_id": document_id,
                "drift_at_start": drift,
                "relative_position": float(relative_position),
                "event": int(first_reject == relative_position),
            })
    return expanded


def _fit_hazard_coefficient(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
    standardization: tuple[Sequence[float], Sequence[float]] | None = None,
) -> float | None:
    """Fit the standardized drift coefficient of the discrete hazard model."""

    expanded = _expand_rejection_hazard_rows(rows, max_k=max_k)
    labels = [int(row["event"]) for row in expanded]
    if len(expanded) < 4 or len(set(labels)) < 2:
        return None
    import numpy as np

    features = np.asarray([
        [float(row["drift_at_start"]), float(row["relative_position"])]
        for row in expanded
    ], dtype=float)
    if standardization is None:
        means = features.mean(axis=0)
        scales = features.std(axis=0)
    else:
        means = np.asarray(standardization[0], dtype=float)
        scales = np.asarray(standardization[1], dtype=float)
    scales[scales == 0.0] = 1.0
    features = (features - means) / scales
    design = np.column_stack((np.ones(len(features)), features))
    targets = np.asarray(labels, dtype=float)
    weights = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag([0.0, 1.0, 1.0])
    for _ in range(60):
        logits = np.clip(design @ weights, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-8)
        hessian = design.T @ (curvature[:, None] * design) + penalty
        gradient = design.T @ (targets - probabilities) - penalty @ weights
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        weights += step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    coefficient = float(weights[1])
    return coefficient if math.isfinite(coefficient) else None


def _fit_rejection_hazard_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Fit H2's position-adjusted drift hazard with document bootstrap."""

    expanded = _expand_rejection_hazard_rows(rows, max_k=max_k)
    document_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        document_id = str(row.get("document_id", len(document_groups)))
        document_groups.setdefault(document_id, []).append(row)
    document_count = len(document_groups)
    standardization = None
    if expanded:
        import numpy as np

        scale_features = np.asarray([
            [float(row["drift_at_start"]), float(row["relative_position"])]
            for row in expanded
        ], dtype=float)
        standardization = (
            scale_features.mean(axis=0),
            scale_features.std(axis=0),
        )
    point = _fit_hazard_coefficient(
        rows, max_k=max_k, standardization=standardization
    )
    if point is None or document_count < 5:
        return {
            "status": "UNAVAILABLE",
            "reason": "need finite drift, both event classes, and at least five documents",
            "bootstrap_unit": "document",
            "document_count": document_count,
            "expanded_risk_set_count": len(expanded),
            "features": ["drift_at_start", "relative_position"],
            "drift_coefficient": point,
            "drift_coefficient_ci": None,
            "decision": "UNAVAILABLE",
        }

    rng = random.Random(42)
    groups = list(document_groups.values())
    coefficients: list[float] = []
    for _ in range(bootstrap_samples):
        resampled = [
            row
            for _ in groups
            for row in rng.choice(groups)
        ]
        try:
            coefficient = _fit_hazard_coefficient(
                resampled,
                max_k=max_k,
                standardization=standardization,
            )
        except (ValueError, RuntimeError):
            coefficient = None
        if coefficient is not None and math.isfinite(coefficient):
            coefficients.append(coefficient)
    coefficient_ci = (
        bootstrap_mean_ci(coefficients, seed=42, samples=2000)
        if coefficients else None
    )
    decision_value = (
        coefficient_ci.get("low") if coefficient_ci is not None else point
    )
    return {
        "status": "ok",
        "bootstrap_unit": "document",
        "document_count": document_count,
        "expanded_risk_set_count": len(expanded),
        "features": ["drift_at_start", "relative_position"],
        "standardization": "within_fit z-score; coefficient is per drift standard deviation",
        "drift_coefficient": point,
        "drift_odds_ratio_per_sd": math.exp(point),
        "drift_coefficient_ci": coefficient_ci,
        "bootstrap_valid_fits": len(coefficients),
        "bootstrap_requested": bootstrap_samples,
        "decision_gate": "lower 95% document-bootstrap CI > 0",
        "decision": decide_status(
            value=decision_value,
            threshold=0.0,
            direction=">",
            count=document_count,
            min_count=5,
        ),
    }


def summarize_position_relocation(
    rows: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[str] = ("raw_chunk_16", "nosink_8_chunk_16"),
) -> dict[str, Any]:
    """Summarize an E0 fixture with the same evidence span relocated in source.

    Each row must identify the evidence token interval in the original source
    and the source-side skip used by a sink-controlled variant.  The helper is
    deliberately descriptive: a position effect is evidence of a confounder,
    not a pass/fail claim about grounding.
    """

    usable = _ok_rows(rows)
    if not usable:
        return {"status": "UNAVAILABLE", "count": 0, "variants": {}}
    result_variants: dict[str, Any] = {}
    for variant in variants:
        case_values: dict[str, list[float]] = {}
        for row in usable:
            start = row.get("evidence_token_start")
            end = row.get("evidence_token_end")
            if start is None or end is None or int(end) <= int(start):
                continue
            skip = (
                int(row.get("evidence_skip_source_tokens", 0))
                if variant.startswith("nosink_") else 0
            )
            chunk_size = int(row.get("evidence_chunk_size", 16))
            relative_start = max(int(start) - skip, 0)
            relative_end = max(int(end) - skip, relative_start + 1)
            step_values: list[float] = []
            for step in row.get("attention", []):
                vector = step.get(variant)
                if not vector:
                    continue
                first_chunk = relative_start // chunk_size
                last_chunk = (relative_end - 1) // chunk_size
                first_chunk = max(0, min(first_chunk, len(vector) - 1))
                last_chunk = max(0, min(last_chunk, len(vector) - 1))
                step_values.append(sum(float(value) for value in vector[first_chunk : last_chunk + 1]))
            if step_values:
                case = str(row.get("relocation_case", row.get("document_id")))
                case_values.setdefault(case, []).append(statistics.fmean(step_values))
        case_summary = {
            case: {
                "count": len(values),
                "mean_evidence_mass": statistics.fmean(values),
                "median_evidence_mass": statistics.median(values),
            }
            for case, values in sorted(case_values.items())
        }
        means = [value["mean_evidence_mass"] for value in case_summary.values()]
        result_variants[variant] = {
            "status": "ok" if case_summary else "UNAVAILABLE",
            "cases": case_summary,
            "position_range": max(means) - min(means) if means else None,
            "position_ratio_max_over_min": (
                max(means) / min(means) if means and min(means) > 0.0 else None
            ),
        }
    return {
        "status": "ok" if any(item["status"] == "ok" for item in result_variants.values()) else "UNAVAILABLE",
        "count": len(usable),
        "variants": result_variants,
        "interpretation": "descriptive position sensitivity; not a grounding pass/fail gate",
    }


def build_predictor_rows(
    target_rows: Sequence[Mapping[str, Any]],
    speculative_rows: Sequence[Mapping[str, Any]],
    *,
    horizon_threshold: float,
    max_horizon: int = 16,
    attention_variant: str = "nosink",
) -> list[dict[str, Any]]:
    """Join online target features to controlled acceptance labels."""

    targets = {
        str(row.get("document_id")): row
        for row in _ok_rows(target_rows)
    }
    result: list[dict[str, Any]] = []
    acceptance_history: dict[str, list[float]] = {}
    for row in _ok_rows(speculative_rows):
        document_id = str(row.get("document_id"))
        target = targets.get(document_id)
        if target is None:
            continue
        start = int(row.get("start_position", 0))
        entropies = target.get("target_entropy", [])
        boundaries = target.get("sentence_boundary", [])
        copy_flags = target.get("copyability", [])
        attention = target.get("attention", [])
        if (
            not 0 <= start < len(attention)
            or start >= len(boundaries)
            or start >= len(copy_flags)
        ):
            continue
        current = attention[start].get(attention_variant)
        if not current:
            continue
        confidence = [float(value) for value in row.get("draft_confidence", [])]
        if not confidence:
            continue
        drift_missing = row.get("drift_at_start") is None
        drift_value = float(row.get("drift_at_start") or 0.0)
        if not math.isfinite(drift_value):
            continue
        lag_drift_value = None
        if start >= 2:
            previous_step = attention[start - 1].get(attention_variant)
            lag_step = attention[start - 2].get(attention_variant)
            if previous_step and lag_step:
                lag_drift_value = js_divergence(previous_step, lag_step)
        previous_acceptance = acceptance_history.get(document_id, [])
        horizon = grounding_horizon(
            _trace_variant(target, attention_variant),
            start=start,
            threshold=horizon_threshold,
            max_horizon=max_horizon,
        )
        output_tokens = max(int(target.get("output_tokens", len(attention))), 1)

        def concentration_at(
            position: int, variant: str = attention_variant
        ) -> float | None:
            if not 0 <= position < len(attention):
                return None
            values = attention[position].get(variant)
            if not values:
                return None
            return max(float(value) for value in values)

        accepted_len = int(row.get("accepted_len", 0))
        first_token_rejected = int(
            row.get("first_reject_rel") == 1 or accepted_len == 0
        )
        raw_concentration = concentration_at(start, "raw")
        result.append({
            "document_id": document_id,
            "start_position": start,
            "label": _rejection(row),
            "rejected": _rejection(row),
            "first_token_rejected": first_token_rejected,
            "target_entropy": float(entropies[start]) if start < len(entropies) else 0.0,
            "position_fraction": start / output_tokens,
            "draft_confidence_mean": statistics.fmean(confidence),
            "recent_acceptance": (
                statistics.fmean(previous_acceptance) if previous_acceptance else 0.0
            ),
            "sentence_boundary": float(boundaries[start]),
            "copyability": float(copy_flags[start]),
            "max_k": float(row.get("max_k", len(confidence))),
            "drift_at_start": drift_value,
            "drift_missing": float(drift_missing),
            "lag_drift": float(lag_drift_value or 0.0),
            "lag_drift_missing": float(lag_drift_value is None),
            "source_concentration": max(float(value) for value in current),
            "raw_source_concentration": (
                float(raw_concentration) if raw_concentration is not None else 0.0
            ),
            "shift10_source_concentration": concentration_at(start + 10),
            "shift20_source_concentration": concentration_at(start + 20),
            "shift50_source_concentration": concentration_at(start + 50),
            "horizon_normalized": (
                float(horizon if horizon is not None else max_horizon + 1) / max_horizon
            ),
            "grounding_horizon": horizon,
            "step_position": start,
        })
        acceptance_history.setdefault(document_id, []).append(
            float(row.get("accepted_len", 0)) / max(float(row.get("max_k", 1)), 1.0)
        )
    return result


def _finite_feature_rows(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if all(
            row.get(name) is not None
            and math.isfinite(float(row[name]))
            for name in feature_names
        )
    ]


def _shuffle_features_within_document(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Create a deterministic request-level shuffle negative control."""

    result = [dict(row) for row in rows]
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(result):
        groups.setdefault(str(row.get("document_id")), []).append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        for name in feature_names:
            values = [result[index][name] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values):
                result[index][name] = value
    return result


def summarize_h3_predictor(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_key: str = "label",
) -> dict[str, Any]:
    """Compare controls against grounding-aware features with document splits."""

    baseline_features = [
        "target_entropy",
        "position_fraction",
        "draft_confidence_mean",
        "recent_acceptance",
        "sentence_boundary",
        "copyability",
        "max_k",
    ]
    grounding_features = [
        "source_concentration",
        "drift_at_start",
        "drift_missing",
        "lag_drift",
        "lag_drift_missing",
        "horizon_normalized",
    ]
    full_features = baseline_features + grounding_features
    if not rows:
        return {"status": "UNAVAILABLE", "decision": "UNAVAILABLE", "count": 0}
    baseline = fit_controlled_predictor(
        rows, feature_names=baseline_features, label_key=label_key
    )
    full = fit_controlled_predictor(
        rows, feature_names=full_features, label_key=label_key
    )
    baseline_auc = baseline.get("metrics", {}).get("auroc")
    full_auc = full.get("metrics", {}).get("auroc")
    gain = None if baseline_auc is None or full_auc is None else full_auc - baseline_auc
    usable_count = min(
        int(baseline.get("test_count", 0)),
        int(full.get("test_count", 0)),
    )
    controls: dict[str, Any] = {
        "position_only": fit_controlled_predictor(
            rows, feature_names=["position_fraction"], label_key=label_key
        ),
    }
    for shift in (10, 20, 50):
        feature = f"shift{shift}_source_concentration"
        shifted = _finite_feature_rows(rows, [feature])
        controls[f"temporal_shift{shift}"] = (
            fit_controlled_predictor(
                shifted, feature_names=[feature], label_key=label_key
            )
            if shifted else {
                "status": "UNAVAILABLE", "feature_names": [feature], "count": 0
            }
        )
    shuffled_rows = _shuffle_features_within_document(
        rows, grounding_features, seed=17
    )
    controls["shuffle_grounding"] = fit_controlled_predictor(
        shuffled_rows, feature_names=full_features, label_key=label_key
    )
    return {
        "status": "ok" if baseline.get("status") == "ok" and full.get("status") == "ok" else "INCONCLUSIVE",
        "count": len(rows),
        "label_key": label_key,
        "baseline": baseline,
        "full": full,
        "controls": controls,
        "feature_names": {
            "baseline": baseline_features,
            "grounding_increment": grounding_features,
        },
        "auroc_gain": gain,
        "decision": decide_status(
            value=gain,
            threshold=0.02,
            direction=">=",
            count=usable_count,
            min_count=10,
        ),
    }


def build_horizon_rows(
    target_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    max_horizon: int = 16,
    speculative_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build document-split online labels for H5 horizon prediction."""

    result: list[dict[str, Any]] = []
    if speculative_rows is not None:
        targets = {
            str(row.get("document_id")): row for row in _ok_rows(target_rows)
        }
        acceptance_history: dict[str, list[float]] = {}
        for spec in _ok_rows(speculative_rows):
            document_id = str(spec.get("document_id"))
            target = targets.get(document_id)
            if target is None:
                continue
            trace = _trace_variant(target, "nosink")
            start = int(spec.get("start_position", 0))
            if not 0 <= start < len(trace):
                continue
            target_entropies = target.get("target_entropy", [])
            horizon = grounding_horizon(
                trace, start=start, threshold=threshold, max_horizon=max_horizon
            )
            confidence = [float(value) for value in spec.get("draft_confidence", [])]
            if not confidence:
                continue
            previous = acceptance_history.get(document_id, [])
            result.append({
                "status": "ok",
                "document_id": document_id,
                "start_position": start,
                "label": int(horizon is not None and horizon >= 2),
                "grounding_horizon": horizon,
                "target_entropy": (
                    float(target_entropies[start])
                    if start < len(target_entropies) else 0.0
                ),
                "position_fraction": start / max(len(trace) - 1, 1),
                "source_concentration": max(trace[start]),
                "drift_at_start": float(spec.get("drift_at_start") or 0.0),
                "drift_missing": float(spec.get("drift_at_start") is None),
                "lag_drift": (
                    js_divergence(trace[start - 1], trace[start - 2])
                    if start >= 2 else 0.0
                ),
                "lag_drift_missing": float(start < 2),
                "draft_confidence_mean": statistics.fmean(confidence),
                "recent_acceptance": statistics.fmean(previous) if previous else 0.0,
                "max_k": float(spec.get("max_k", len(confidence))),
                "accepted_len": int(spec.get("accepted_len", 0)),
                "draft_time_by_k_ms": spec.get("draft_time_by_k_ms"),
                "verification_time_by_k_ms": spec.get("verification_time_by_k_ms"),
            })
            acceptance_history.setdefault(document_id, []).append(
                int(spec.get("accepted_len", 0)) / max(float(spec.get("max_k", 1)), 1.0)
            )
        return result
    for row in _ok_rows(target_rows):
        trace = _trace_variant(row, "nosink")
        if len(trace) < 2:
            continue
        entropies = row.get("target_entropy", [])
        for start in range(len(trace) - 1):
            horizon = grounding_horizon(
                trace, start=start, threshold=threshold, max_horizon=max_horizon
            )
            result.append({
                "document_id": str(row.get("document_id")),
                "label": int(horizon is not None and horizon >= 2),
                "target_entropy": float(entropies[start]) if start < len(entropies) else 0.0,
                "position_fraction": start / max(len(trace) - 1, 1),
                "source_concentration": max(trace[start]),
            })
    return result


def _timing_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row.get("draft_time_by_k_ms") and row.get("verification_time_by_k_ms")
    ]


def summarize_h5_predictor(
    rows: Sequence[Mapping[str, Any]],
    *,
    timing_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    features = [
        "target_entropy",
        "position_fraction",
        "source_concentration",
        "drift_at_start",
        "drift_missing",
        "lag_drift",
        "lag_drift_missing",
        "draft_confidence_mean",
        "recent_acceptance",
    ]
    if not rows:
        return {"status": "UNAVAILABLE", "decision": "UNAVAILABLE", "count": 0}
    usable = _finite_feature_rows(rows, features)
    if not usable:
        return {
            "status": "UNAVAILABLE",
            "decision": "INCONCLUSIVE",
            "count": len(rows),
            "feature_names": features,
            "reason": "rows do not contain the complete online feature set",
        }
    result = fit_controlled_predictor(usable, feature_names=features)
    metric = result.get("metrics", {}).get("auroc")
    report: dict[str, Any] = {
        "status": result.get("status", "INCONCLUSIVE"),
        "count": len(usable),
        "predictor": result,
        "feature_names": features,
        "decision": decide_status(
            value=metric,
            threshold=0.5,
            direction=">=",
            count=int(result.get("test_count", 0)),
            min_count=10,
        ),
    }
    timing_source = [
        row for row in _timing_rows(timing_rows if timing_rows is not None else usable)
        if len(row.get("draft_time_by_k_ms") or []) >= 8
        and len(row.get("verification_time_by_k_ms") or []) >= 8
    ]
    report["timing_coverage"] = {
        "rows": len(timing_source),
        "status": "ok" if timing_source else "UNAVAILABLE",
        "source": "timing_subset" if timing_rows is not None else "horizon_rows",
    }
    if timing_source and result.get("status") == "ok":
        scores = {
            (
                item.get("document_id"),
                item.get("start_position"),
                item.get("step_position"),
            ): float(item["score"])
            for item in result.get("test_predictions", [])
        }
        test_rows = [
            row for row in timing_source
            if (
                str(row.get("document_id")),
                row.get("start_position"),
                row.get("step_position"),
            ) in scores
        ]
        threshold = 0.5
        predicted_rows = [
            {
                **row,
                "predicted_horizon": 2 if scores[(
                    str(row.get("document_id")),
                    row.get("start_position"),
                    row.get("step_position"),
                )] >= threshold else 1,
            }
            for row in test_rows
        ]
        fixed = replay_policy(
            predicted_rows, policy="fixed", max_k=8, fixed_k=4, require_timing=True
        )
        predicted = replay_policy(
            predicted_rows, policy="predicted", max_k=8, require_timing=True
        )
        oracle = replay_policy(
            predicted_rows, policy="oracle", max_k=8, require_timing=True
        )
        fixed_speed = fixed.get("tokens_per_ms")
        predicted_speed = predicted.get("tokens_per_ms")
        oracle_speed = oracle.get("tokens_per_ms")
        denominator = (
            None if fixed_speed in (None, 0) or oracle_speed is None
            else float(oracle_speed) - float(fixed_speed)
        )
        recovery = (
            None
            if (
                denominator in (None, 0)
                or predicted_speed is None
                or fixed_speed is None
                or oracle_speed is None
                or float(oracle_speed) <= float(fixed_speed)
            )
            else (float(predicted_speed) - float(fixed_speed)) / denominator
        )
        report["policy"] = {
            "probability_threshold": threshold,
            "test_count": len(test_rows),
            "fixed_k4": fixed,
            "predicted_horizon": predicted,
            "oracle_horizon": oracle,
            "oracle_gain_recovery": recovery,
            "decision_basis": "test document split; recovery is defined only when oracle is faster than fixed",
        }
        report["decision"] = decide_status(
            value=recovery,
            threshold=0.50,
            direction=">=",
            count=len(test_rows),
            min_count=10,
        )
    else:
        report["reason"] = "H5 needs document-split predictor rows with measured draft and verifier timing"
        report["decision"] = "INCONCLUSIVE"
    return report


def select_horizon_threshold(
    target_rows: Sequence[Mapping[str, Any]],
    speculative_rows: Sequence[Mapping[str, Any]],
    *,
    requested: float,
    max_horizon: int,
) -> dict[str, Any]:
    """Select a horizon threshold on train/dev documents only.

    The selection objective is to obtain both horizon classes while staying
    closest to a balanced calibration label rate.  The test documents are not
    inspected during this selection.
    """

    usable_targets = _ok_rows(target_rows)
    if not usable_targets or not _ok_rows(speculative_rows):
        return {
            "requested": requested,
            "selected": requested,
            "status": "UNAVAILABLE",
            "calibration_documents": 0,
        }
    train, dev, _ = split_by_document(usable_targets)
    calibration_targets = train + dev
    calibration_ids = {str(row.get("document_id")) for row in calibration_targets}
    calibration_specs = [
        row for row in _ok_rows(speculative_rows)
        if str(row.get("document_id")) in calibration_ids
    ]
    candidates: list[float] = []
    for value in (requested, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
        value = float(value)
        if value >= 0.0 and math.isfinite(value) and value not in candidates:
            candidates.append(value)
    scored: list[tuple[float, float, int, float]] = []
    for candidate in candidates:
        labels = [
            row["label"]
            for row in build_horizon_rows(
                calibration_targets,
                threshold=candidate,
                max_horizon=max_horizon,
                speculative_rows=calibration_specs,
            )
        ]
        if not labels or len(set(labels)) < 2:
            continue
        positive_rate = statistics.fmean(labels)
        scored.append((abs(positive_rate - 0.5), -len(labels), candidate, positive_rate))
    if not scored:
        return {
            "requested": requested,
            "selected": requested,
            "status": "insufficient_class_variation",
            "calibration_documents": len(calibration_ids),
        }
    _, _, selected, positive_rate = min(scored)
    return {
        "requested": requested,
        "selected": selected,
        "status": "ok",
        "calibration_documents": len(calibration_ids),
        "calibration_positive_rate": positive_rate,
        "candidates_with_two_classes": [item[2] for item in scored],
    }


def replay_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    max_k: int,
    fixed_k: int = 1,
    require_timing: bool = False,
) -> dict[str, Any]:
    """Replay acceptance policies from exact proposal/canonical prefixes.

    With ``require_timing=True`` this function refuses to turn analytical
    acceptance into a speed claim unless each row has measured verification and
    draft timing.  The acceptance-only result remains useful for H4 planning.
    """

    if max_k <= 0 or fixed_k <= 0:
        raise ValueError("max_k and fixed_k must be positive")
    usable = _ok_rows(rows)
    if not usable:
        return {"status": "UNAVAILABLE", "policy": policy, "count": 0}
    if require_timing and any(
        not row.get("draft_time_by_k_ms")
        or not row.get("verification_time_by_k_ms")
        for row in usable
    ):
        return {
            "status": "UNAVAILABLE",
            "policy": policy,
            "count": len(usable),
            "reason": "measured draft_time_ms and verification_time_ms are required",
        }
    entropy_values = [
        float(row["target_entropy_at_start"])
        for row in usable
        if row.get("target_entropy_at_start") is not None
        and math.isfinite(float(row["target_entropy_at_start"]))
    ]
    history_values = [
        float(row["recent_acceptance"])
        for row in usable
        if row.get("recent_acceptance") is not None
        and math.isfinite(float(row["recent_acceptance"]))
    ]
    if policy in {"entropy", "adaptive_entropy"} and not entropy_values:
        return {
            "status": "UNAVAILABLE", "policy": policy, "count": len(usable),
            "reason": "target_entropy_at_start is required for adaptive entropy policy",
        }
    if policy in {"history", "adaptive_history"} and not history_values:
        return {
            "status": "UNAVAILABLE", "policy": policy, "count": len(usable),
            "reason": "recent_acceptance is required for adaptive history policy",
        }
    entropy_threshold = statistics.median(entropy_values) if entropy_values else None
    history_threshold = statistics.median(history_values) if history_values else None
    committed: list[float] = []
    draft_tokens: list[float] = []
    costs: list[float] = []
    for row in usable:
        if policy == "fixed":
            k = policy_k(fixed_k, max_k=max_k)
        elif policy == "oracle":
            k = policy_k(row.get("grounding_horizon"), max_k=max_k)
        elif policy == "predicted":
            k = policy_k(row.get("predicted_horizon"), max_k=max_k)
        elif policy in {"entropy", "adaptive_entropy"}:
            k = max_k if float(row["target_entropy_at_start"]) >= entropy_threshold else max(1, max_k // 2)
        elif policy in {"history", "adaptive_history"}:
            k = max_k if float(row["recent_acceptance"]) >= history_threshold else max(1, max_k // 2)
        else:
            raise ValueError(f"unsupported policy: {policy}")
        accepted = min(int(row.get("accepted_len", 0)), k)
        committed.append(float(min(k, accepted + 1)))
        draft_tokens.append(float(accepted))
        if require_timing:
            draft_times = row.get("draft_time_by_k_ms", [])
            verification_times = row.get("verification_time_by_k_ms", [])
            if k > len(draft_times) or k > len(verification_times):
                return {
                    "status": "UNAVAILABLE",
                    "policy": policy,
                    "count": len(usable),
                    "reason": f"timing arrays do not cover selected k={k}",
                }
            costs.append(float(draft_times[k - 1]) + float(verification_times[k - 1]))
    result: dict[str, Any] = {
        "status": "ok",
        "policy": policy,
        "count": len(usable),
        "mean_committed_tokens": statistics.fmean(committed),
        "mean_accepted_draft_tokens": statistics.fmean(draft_tokens),
    }
    if require_timing:
        result["mean_cost_ms"] = statistics.fmean(costs)
        result["tokens_per_ms"] = sum(committed) / sum(costs) if sum(costs) else None
        result["timing_basis"] = "measured_draft_plus_verification"
    else:
        result["timing_basis"] = "acceptance_only_no_speed_claim"
    return result


def replay_policy_sweep(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
    fixed_ks: Sequence[int] = (2, 4, 8),
    require_timing: bool = False,
) -> dict[str, Any]:
    """Replay fixed-k policies and a hindsight true-cost oracle."""

    policies: dict[str, dict[str, Any]] = {}
    for fixed_k in fixed_ks:
        if fixed_k <= 0:
            raise ValueError("fixed_ks must contain positive values")
        policies[f"fixed_k{fixed_k}"] = replay_policy(
            rows,
            policy="fixed",
            max_k=max_k,
            fixed_k=fixed_k,
            require_timing=require_timing,
        )
    policies["adaptive_entropy"] = replay_policy(
        rows, policy="adaptive_entropy", max_k=max_k, require_timing=require_timing
    )
    policies["adaptive_history"] = replay_policy(
        rows, policy="adaptive_history", max_k=max_k, require_timing=require_timing
    )
    usable = _ok_rows(rows)
    if not usable:
        policies["true_cost_oracle"] = {
            "status": "UNAVAILABLE", "policy": "true_cost_oracle", "count": 0,
        }
        return {"status": "UNAVAILABLE", "policies": policies}
    if not require_timing:
        policies["true_cost_oracle"] = {
            "status": "UNAVAILABLE",
            "policy": "true_cost_oracle",
            "count": len(usable),
            "reason": "true-cost oracle requires measured timing",
        }
        return {"status": "ok", "policies": policies}

    committed: list[float] = []
    draft_tokens: list[float] = []
    costs: list[float] = []
    for row in usable:
        draft_times = row.get("draft_time_by_k_ms") or []
        verification_times = row.get("verification_time_by_k_ms") or []
        available = min(max_k, len(draft_times), len(verification_times))
        if available <= 0:
            policies["true_cost_oracle"] = {
                "status": "UNAVAILABLE",
                "policy": "true_cost_oracle",
                "count": len(usable),
                "reason": "timing arrays do not cover any k",
            }
            return {"status": "ok", "policies": policies}
        candidates: list[tuple[float, int, float, float, float]] = []
        for candidate_k in range(1, available + 1):
            accepted = min(int(row.get("accepted_len", 0)), candidate_k)
            committed_tokens = float(min(candidate_k, accepted + 1))
            cost = float(draft_times[candidate_k - 1]) + float(
                verification_times[candidate_k - 1]
            )
            if cost <= 0.0 or not math.isfinite(cost):
                continue
            candidates.append((committed_tokens / cost, candidate_k, committed_tokens, float(accepted), cost))
        if not candidates:
            return {
                "status": "ok",
                "policies": {
                    **policies,
                    "true_cost_oracle": {
                        "status": "UNAVAILABLE",
                        "policy": "true_cost_oracle",
                        "count": len(usable),
                        "reason": "no finite positive timing cost",
                    },
                },
            }
        _, _, committed_tokens, accepted_tokens, cost = max(
            candidates, key=lambda item: (item[0], -item[1])
        )
        committed.append(committed_tokens)
        draft_tokens.append(accepted_tokens)
        costs.append(cost)
    policies["true_cost_oracle"] = {
        "status": "ok",
        "policy": "true_cost_oracle",
        "count": len(usable),
        "mean_committed_tokens": statistics.fmean(committed),
        "mean_accepted_draft_tokens": statistics.fmean(draft_tokens),
        "mean_cost_ms": statistics.fmean(costs),
        "tokens_per_ms": sum(committed) / sum(costs) if sum(costs) else None,
        "timing_basis": "measured_draft_plus_verification",
    }
    return {"status": "ok", "policies": policies}


def build_hypothesis_report(
    target_rows: Sequence[Mapping[str, Any]],
    speculative_rows: Sequence[Mapping[str, Any]],
    *,
    timing_speculative_rows: Sequence[Mapping[str, Any]] | None = None,
    threshold: float,
    horizon_threshold: float,
    max_horizon: int = 16,
    max_k: int = 8,
    min_documents: int = 5,
) -> dict[str, Any]:
    """Create the machine-readable H1–H5 report from raw trace rows."""

    h1 = summarize_target_traces(
        target_rows,
        threshold=threshold,
        min_documents=min_documents,
    )
    h1_raw_control = summarize_target_traces(
        target_rows,
        threshold=threshold,
        variant="raw",
        min_documents=min_documents,
    )
    train_target, _, _ = split_by_document(_ok_rows(target_rows)) if _ok_rows(target_rows) else ([], [], [])
    prior = fit_relative_positional_prior(train_target, variant="raw", bins=32)
    calibrated_rows = _attach_trace_variant(
        target_rows,
        variant="calibrated",
        traces={
            str(row.get("document_id")): calibrated_trace(
                _trace_variant(row, "raw"), prior
            )
            for row in train_target + _ok_rows(target_rows)
            if prior and _trace_variant(row, "raw")
        },
    )
    h1_calibrated_control = summarize_target_traces(
        calibrated_rows,
        threshold=threshold,
        variant="calibrated",
        min_documents=min_documents,
    )
    h1["controls"] = {
        "raw": h1_raw_control,
        "calibrated": h1_calibrated_control,
    }
    h1["primary_nosink_decision"] = h1.get("decision")
    h1["robust_decision_basis"] = (
        "PASS only when the calibrated estimator also passes the same gate"
    )
    if h1_calibrated_control.get("decision") != "PASS":
        h1["decision"] = (
            "UNAVAILABLE"
            if h1_calibrated_control.get("decision") == "UNAVAILABLE"
            else "FAIL"
        )
    h1["calibration"] = {
        "prior_bins": len(prior),
        "fit_documents": len({str(row.get("document_id")) for row in train_target}),
        "status": "ok" if prior else "UNAVAILABLE",
    }
    h1["sensitivity"] = {
        variant: summarize_target_traces(
            target_rows,
            threshold=threshold,
            variant=variant,
            min_documents=min_documents,
        )
        for variant in (
            "raw_chunk_64",
            "raw_chunk_128",
            "raw_chunk_256",
            "nosink_4_chunk_128",
            "nosink_8_chunk_128",
            "nosink_16_chunk_128",
        )
    }
    h2 = summarize_speculative_traces(speculative_rows)
    h2["coarse_decision"] = h2.get("decision")
    h2["hazard"] = summarize_rejection_hazard(speculative_rows, max_k=max_k)
    hazard_model = h2["hazard"].get("hazard_model", {})
    h2["decision_basis"] = (
        "position-adjusted drift coefficient; PASS requires the lower 95% "
        "document-bootstrap CI to be strictly positive"
    )
    if hazard_model.get("status") == "ok":
        h2["decision"] = hazard_model.get("decision", "UNAVAILABLE")
    predictor_rows = build_predictor_rows(
        target_rows,
        speculative_rows,
        horizon_threshold=horizon_threshold,
        max_horizon=max_horizon,
    )
    h3 = summarize_h3_predictor(
        predictor_rows, label_key="first_token_rejected"
    )
    h3["alternative_fully_rejected"] = summarize_h3_predictor(
        predictor_rows, label_key="label"
    )
    calibrated_predictor_rows = build_predictor_rows(
        calibrated_rows,
        speculative_rows,
        horizon_threshold=horizon_threshold,
        max_horizon=max_horizon,
        attention_variant="calibrated",
    )
    h3_nosink_summary = dict(h3)
    h3["estimator_variants"] = {
        "nosink": h3_nosink_summary,
        "calibrated": summarize_h3_predictor(
            calibrated_predictor_rows, label_key="first_token_rejected"
        ),
    }
    horizon_selection = select_horizon_threshold(
        target_rows,
        speculative_rows,
        requested=horizon_threshold,
        max_horizon=max_horizon,
    )
    h5_horizon_threshold = float(horizon_selection["selected"])
    horizon_rows = build_horizon_rows(
        target_rows,
        threshold=h5_horizon_threshold,
        max_horizon=max_horizon,
        speculative_rows=speculative_rows if _ok_rows(speculative_rows) else None,
    )
    timing_horizon_rows = (
        build_horizon_rows(
            target_rows,
            threshold=h5_horizon_threshold,
            max_horizon=max_horizon,
            speculative_rows=timing_speculative_rows,
        )
        if timing_speculative_rows is not None
        else None
    )
    h5 = summarize_h5_predictor(
        horizon_rows,
        timing_rows=timing_horizon_rows,
    )
    h5["threshold_selection"] = horizon_selection
    if not _ok_rows(speculative_rows):
        if horizon_rows:
            h5["status"] = "INCONCLUSIVE"
            h5["decision"] = "INCONCLUSIVE"
            h5["reason"] = "H2/E2 controlled acceptance data is required by the protocol"
        else:
            h5["reason"] = "no usable target trace"

    timing_rows = (
        timing_speculative_rows
        if timing_speculative_rows is not None
        else speculative_rows
    )
    timing_rows = [
        row for row in _ok_rows(timing_rows)
        if len(row.get("draft_time_by_k_ms") or []) >= max_k
        and len(row.get("verification_time_by_k_ms") or []) >= max_k
    ]
    fixed = replay_policy(
        timing_rows, policy="fixed", max_k=max_k, fixed_k=max_k
    )
    oracle = replay_policy(
        [
            {**row, "grounding_horizon": row.get("grounding_horizon")}
            for row in timing_rows
        ],
        policy="oracle",
        max_k=max_k,
    )
    fixed_mean = fixed.get("mean_committed_tokens")
    oracle_mean = oracle.get("mean_committed_tokens")
    acceptance_gain = (
        None
        if fixed_mean in (None, 0) or oracle_mean is None
        else float(oracle_mean) / float(fixed_mean) - 1.0
    )
    timed_fixed = replay_policy(
        timing_rows, policy="fixed", max_k=max_k, fixed_k=max_k, require_timing=True
    )
    timed_oracle = replay_policy(
        timing_rows, policy="oracle", max_k=max_k, require_timing=True
    )
    fixed_speed = timed_fixed.get("tokens_per_ms")
    oracle_speed = timed_oracle.get("tokens_per_ms")
    speed_gain = (
        None
        if fixed_speed in (None, 0) or oracle_speed is None
        else float(oracle_speed) / float(fixed_speed) - 1.0
    )
    h4_timed_ok = timed_fixed.get("status") == "ok" and timed_oracle.get("status") == "ok"
    h4 = {
        "status": "ok" if h4_timed_ok else "UNAVAILABLE",
        "fixed": fixed,
        "oracle": oracle,
        "acceptance_only_gain": acceptance_gain,
        "timed_fixed": timed_fixed,
        "timed_oracle": timed_oracle,
        "speed_gain": speed_gain,
        "timing_basis": (
            "measured_cached_target_check"
            if h4_timed_ok
            else "acceptance_only_no_speed_claim"
        ),
        "decision": decide_status(
            value=speed_gain,
            threshold=0.08,
            direction=">=",
            count=min(int(timed_fixed.get("count", 0)), int(timed_oracle.get("count", 0))),
            min_count=10,
        ),
        "reason": None if h4_timed_ok else "true draft plus target-verification timing is not present",
        "timing_rows_source": (
            "speculative_timing_traces.jsonl"
            if timing_speculative_rows is not None
            else "speculative_traces.jsonl"
        ),
    }
    h4["policy_sweep"] = replay_policy_sweep(
        timing_rows,
        max_k=max_k,
        fixed_ks=(2, 4, 8),
        require_timing=True,
    )
    return {
        "schema_version": "groundsync.report.v1",
        "target_rows": len(target_rows),
        "speculative_rows": len(speculative_rows),
        "predictor_rows": len(predictor_rows),
        "horizon_rows": len(horizon_rows),
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5},
    }


def _write_simple_plot(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    """Write a tiny dependency-light PNG for audit-friendly reports."""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    width, height = 900, 500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 20), title, fill="black")
    if not values:
        draw.text((30, 60), "No finite data", fill="black")
        image.save(path)
        return
    margin_left, margin_bottom = 70, 70
    plot_width = width - margin_left - 30
    plot_height = height - 110
    lo = min(0.0, min(values))
    hi = max(1.0, max(values))
    scale = plot_height / (hi - lo or 1.0)
    zero_y = 60 + int((hi - 0.0) * scale)
    draw.line((margin_left, 60, margin_left, height - margin_bottom), fill="black")
    draw.line((margin_left, zero_y, width - 30, zero_y), fill="black")
    step = plot_width / max(len(values), 1)
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + int((index + 0.5) * step)
        y = 60 + int((hi - value) * scale)
        draw.line((x, zero_y, x, y), fill=(35, 95, 160), width=5)
        draw.text((x - 20, height - 55), str(label), fill="black")
    image.save(path)


def write_report_artifacts(output_dir: Path, report: Mapping[str, Any]) -> None:
    """Persist JSON, flat CSV, PNG summaries, and a Vietnamese Markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", report)
    rows: list[dict[str, Any]] = []
    for name, value in report.get("hypotheses", {}).items():
        if name == "H1":
            primary_metric = value.get("persistence_excess")
            count = value.get("documents")
        elif name == "H2":
            primary_metric = value.get("high_minus_low_rejection_rate")
            count = value.get("count")
        elif name == "H3":
            primary_metric = value.get("auroc_gain")
            count = value.get("count")
        elif name == "H4":
            primary_metric = value.get("speed_gain")
            count = value.get("fixed", {}).get("count")
        else:
            primary_metric = value.get("predictor", {}).get("metrics", {}).get("auroc")
            count = value.get("count")
        rows.append({
            "hypothesis": name,
            "decision": value.get("decision"),
            "status": value.get("status"),
            "count": count,
            "primary_metric": primary_metric,
        })
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=["hypothesis", "decision", "status", "count", "primary_metric"])
        writer.writeheader()
        writer.writerows(rows)

    h1 = report.get("hypotheses", {}).get("H1", {})
    lag = h1.get("lag_similarity", {})
    _write_simple_plot(
        output_dir / "persistence.png",
        "H1 source-state lag similarity",
        [key for key, value in lag.items() if value is not None],
        [float(value) for value in lag.values() if value is not None],
    )
    h2 = report.get("hypotheses", {}).get("H2", {})
    rejection_labels = ["low", "high"]
    rejection_values = [h2.get("low_rejection_rate"), h2.get("high_rejection_rate")]
    _write_simple_plot(
        output_dir / "drift_rejection.png",
        "H2 rejection rate by drift group",
        [label for label, value in zip(rejection_labels, rejection_values) if value is not None],
        [float(value) for value in rejection_values if value is not None],
    )
    h4 = report.get("hypotheses", {}).get("H4", {})
    policy_labels = ["fixed", "oracle"]
    policy_values = [
        h4.get("fixed", {}).get("mean_committed_tokens"),
        h4.get("oracle", {}).get("mean_committed_tokens"),
    ]
    _write_simple_plot(
        output_dir / "policy_utility.png",
        "H4 committed tokens (acceptance-only or measured)",
        [label for label, value in zip(policy_labels, policy_values) if value is not None],
        [float(value) for value in policy_values if value is not None],
    )
    h5 = report.get("hypotheses", {}).get("H5", {})
    horizon_rate = h5.get("predictor", {}).get("metrics", {}).get("positive_rate")
    _write_simple_plot(
        output_dir / "horizon_labels.png",
        "H5 fraction with grounding horizon >= 2",
        ["long", "short"] if horizon_rate is not None else [],
        [float(horizon_rate), 1.0 - float(horizon_rate)] if horizon_rate is not None else [],
    )
    lines = [
        "# GroundSync — báo cáo kiểm chứng hypothesis",
        "",
        "Báo cáo này tách acceptance/attention analytics khỏi timing E2E. "
        "`PASS` chỉ có nghĩa là metric và coverage vượt tiêu chí đã định; "
        "`UNAVAILABLE` nghĩa là chưa có dữ liệu cần thiết.",
        "",
        "## Tóm tắt H1–H5",
        "",
        "| Hypothesis | Quyết định | Số mẫu/đơn vị | Metric chính |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['hypothesis']} | {row['decision']} | {row['count']} | {row['primary_metric']} |"
        )
    lines.extend([
        "",
        "## Diễn giải và giới hạn",
        "",
        "- H1 dùng similarity `1 - JS` giữa các vector source-attention đã gom chunk; null là thứ tự token bị shuffle theo document.",
        "- H2 dùng canonical greedy continuation của target để tính first rejection của draft; quyết định chính dùng drift coefficient trong hazard điều chỉnh relative position với document bootstrap; đây là controlled acceptance, không phải throughput E2E.",
        "- H3/H5 chia train/test theo document để tránh coi các token trong cùng tài liệu là các quan sát độc lập.",
        "- H4 chỉ được kết luận về tốc độ khi trace có cả `draft_time_ms` và `verification_time_ms`; acceptance-only không phải speedup.",
        "- Attention là tín hiệu quan sát được, không được xem là ground-truth attribution; E0 position relocation phải được đọc như confounder diagnostic.",
        "",
        "Artifacts: `metrics.json`, `metrics.csv`, `persistence.png`, `drift_rejection.png`, `policy_utility.png`, `horizon_labels.png`.",
        "",
    ])
    (output_dir / "hypothesis_report.md").write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "build_predictor_rows",
    "calibrated_trace",
    "build_horizon_rows",
    "build_hypothesis_report",
    "binary_prediction_metrics",
    "decide_status",
    "fit_controlled_predictor",
    "fit_relative_positional_prior",
    "replay_policy",
    "replay_policy_sweep",
    "split_by_document",
    "summarize_h3_predictor",
    "summarize_h5_predictor",
    "summarize_position_relocation",
    "summarize_rejection_hazard",
    "summarize_speculative_traces",
    "summarize_target_traces",
    "write_report_artifacts",
    "write_json",
]
