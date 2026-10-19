"""Small, grouped and calibrated survival estimators for B2."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

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

HAZARD_VARIANTS: dict[str, tuple[str, ...]] = {
    "R-Hazard-Pos": ("position",),
    "R-Hazard-Present": ("position", "draft_present_log_margin"),
    "R-Hazard-Prefix": ("position", "draft_present_log_margin", "draft_prefix_tv"),
    "R-Hazard-Present-Future": (
        "position",
        "draft_present_log_margin",
        "draft_suffix_margin",
    ),
    "R-Hazard-Suffix-Margin": (
        "position",
        "draft_present_log_margin",
        "draft_prefix_tv",
        "draft_suffix_margin",
    ),
    "R-Hazard-Suffix-Entropy": (
        "position",
        "draft_present_log_margin",
        "draft_prefix_tv",
        "draft_suffix_entropy",
    ),
}
SURVIVAL_VARIANTS = {**DEFAULT_VARIANTS, **LAGGED_VARIANTS, **HAZARD_VARIANTS}


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


def _extract_hazard_signals(
    row: Mapping[str, Any], max_candidates: int = 15
) -> tuple[list[float], list[float], list[float]]:
    """Extract margins (m_j), confidences (c_j), and normalized entropies (H_j)."""
    uncertainty = row.get("draft_uncertainty") or {}
    confidences = [float(v) for v in (uncertainty.get("confidence") or [])]
    entropies = [float(v) for v in (uncertainty.get("normalized_entropy") or [])]
    prob_margins = [float(v) for v in (uncertainty.get("probability_margin") or [])]
    logit_margins = [float(v) for v in (uncertainty.get("logit_margin") or [])]

    c = [confidences[i] if i < len(confidences) else 0.5 for i in range(max_candidates)]
    h = [entropies[i] if i < len(entropies) else 0.5 for i in range(max_candidates)]

    m = []
    for i in range(max_candidates):
        if i < len(prob_margins) and prob_margins[i] > 1e-6:
            m.append(prob_margins[i])
        elif i < len(logit_margins) and logit_margins[i] > 0 and c[i] > 0:
            delta = max(logit_margins[i], 0.0)
            m.append(c[i] * (1.0 - math.exp(-delta)))
        else:
            m.append(max(2.0 * c[i] - 1.0, 0.0))
    return m, c, h


def _hazard_features_for_position(
    position: int,
    m: Sequence[float],
    c: Sequence[float],
    h: Sequence[float],
    variant: str,
    max_candidates: int = 15,
) -> list[float]:
    """Compute feature values for position j in 1..max_candidates for a given hazard variant."""
    j = int(position)
    features = HAZARD_VARIANTS[variant]
    values: list[float] = []
    for feature in features:
        if feature == "position":
            values.append(float(j))
        elif feature == "draft_present_log_margin":
            margin = m[j - 1] if j - 1 < len(m) else 0.0
            values.append(math.log(max(margin, 1e-6)))
        elif feature == "draft_prefix_tv":
            val = sum(1.0 - c[i] for i in range(j - 1)) if j > 1 else 0.0
            values.append(float(val))
        elif feature == "draft_suffix_margin":
            val = (
                sum(1.0 - m[r] for r in range(j, max_candidates)) / float(max_candidates)
                if j < max_candidates
                else 0.0
            )
            values.append(float(val))
        elif feature == "draft_suffix_entropy":
            val = (
                sum(h[r] for r in range(j, max_candidates)) / float(max_candidates)
                if j < max_candidates
                else 0.0
            )
            values.append(float(val))
        else:
            raise ValueError(f"unknown hazard feature {feature!r}")
    return values


def optimize_hazard_for_e2e_speedup(
    beta_init: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    variant: str,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    cost_by_prefix: Mapping[int | str, float] | None = None,
    allowed_prefixes: Sequence[int] = (7, 11, 15),
    max_candidates: int = 15,
    overhead_ms: float = 0.0,
    reg_to_init: float = 0.01,
) -> tuple[np.ndarray, dict[str, float]]:
    """Directly fine-tunes hazard parameters beta to maximize empirical token throughput (E2E speedup)
    across consecutive warmup draft rounds."""
    if not train_rows:
        return beta_init, {"initial_cost_per_commit": 0.0, "optimized_cost_per_commit": 0.0}

    costs: dict[int, float] = {}
    if cost_by_prefix:
        for k, v in cost_by_prefix.items():
            costs[int(k)] = float(v)
    else:
        # Default B200 FA4 flat costs
        costs = {3: 17.5, 7: 18.2, 11: 19.0, 15: 20.0}

    valid_prefixes = tuple(sorted(p for p in allowed_prefixes if p in costs and p <= max_candidates))
    if not valid_prefixes:
        valid_prefixes = (min(max_candidates, max(costs.keys())),)

    # Pre-extract features for all rounds to make evaluation vector-fast
    rounds_data = []
    for row in train_rows:
        accepted = int(row.get("accepted_candidates", 0))
        m, c, h = _extract_hazard_signals(row, max_candidates=max_candidates)
        features_list = [
            _hazard_features_for_position(j, m, c, h, variant, max_candidates=max_candidates)
            for j in range(1, max_candidates + 1)
        ]
        X_pred = np.asarray(features_list, dtype=np.float64)
        standardized = (X_pred - center) / scale
        Phi = np.column_stack([np.ones(len(X_pred), dtype=np.float64), standardized])
        rounds_data.append((Phi, accepted))

    def evaluate_cost_per_commit(beta: np.ndarray) -> float:
        total_commits = 0.0
        total_cost = 0.0
        for Phi, accepted in rounds_data:
            logits = Phi @ beta
            log_p = -np.logaddexp(0.0, -logits)
            log_S = np.cumsum(log_p)
            S = np.exp(log_S)
            S = np.clip(S, 0.0, 1.0 - 1e-12)

            best_k = valid_prefixes[-1]
            min_expected_cost = float("inf")
            for k in valid_prefixes:
                expected_commits = 1.0 + float(np.sum(S[:k]))
                cost_k = costs[k] + overhead_ms
                score = cost_k / max(expected_commits, 1e-6)
                if score < min_expected_cost:
                    min_expected_cost = score
                    best_k = k

            actual_commits = 1.0 + min(best_k, accepted)
            actual_cost = costs[best_k] + overhead_ms
            total_commits += actual_commits
            total_cost += actual_cost

        return total_cost / max(total_commits, 1e-6)

    initial_cpc = evaluate_cost_per_commit(beta_init)
    best_beta = beta_init.copy()
    best_cpc = initial_cpc

    # Multi-pass coordinate scan to find optimal basins across plateaus
    for _ in range(2):
        for dim in range(len(best_beta)):
            cur = best_beta[dim]
            deltas = np.linspace(-2.0, 2.0, 41) if dim == 0 else np.linspace(-1.5, 1.5, 31)
            for d in deltas:
                cand = best_beta.copy()
                cand[dim] = cur + d
                cpc = evaluate_cost_per_commit(cand)
                if cpc < best_cpc - 1e-6:
                    best_cpc = cpc
                    best_beta = cand

    def objective(beta: np.ndarray) -> float:
        cpc = evaluate_cost_per_commit(beta)
        penalty = 0.5 * reg_to_init * float(np.sum((beta - best_beta) ** 2))
        return cpc + penalty

    res = minimize(
        objective,
        best_beta,
        method="Nelder-Mead",
        options={"maxiter": 150, "xatol": 1e-4, "fatol": 1e-4},
    )

    if res.success and evaluate_cost_per_commit(res.x) <= best_cpc:
        opt_beta = res.x
        opt_cpc = evaluate_cost_per_commit(res.x)
    else:
        opt_beta = best_beta
        opt_cpc = best_cpc

    metrics = {
        "initial_cost_per_commit": float(initial_cpc),
        "optimized_cost_per_commit": float(opt_cpc),
        "speedup_gain_percent": float(max(0.0, (initial_cpc - opt_cpc) / max(initial_cpc, 1e-6) * 100.0)),
    }
    return opt_beta, metrics


def fit_hazard_model(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    max_candidates: int = 15,
    ridge: float = 1.0,
    optimize_speedup: bool = False,
    cost_by_prefix: Mapping[int | str, float] | None = None,
    allowed_prefixes: Sequence[int] = (7, 11, 15),
    overhead_ms: float = 0.0,
) -> dict[str, Any]:
    """Fit regularized discrete hazard model on observed risk set transitions with optional direct speedup tuning."""
    if variant not in HAZARD_VARIANTS:
        raise ValueError(f"unsupported hazard variant {variant!r}")
    if not train_rows:
        raise ValueError("hazard fit requires train_rows")

    x_list: list[list[float]] = []
    y_list: list[float] = []

    for row in train_rows:
        accepted = int(row.get("accepted_candidates", -1))
        if not 0 <= accepted <= max_candidates:
            raise ValueError(f"candidate_signals accepted_candidates must be within [0, {max_candidates}]")
        m, c, h = _extract_hazard_signals(row, max_candidates=max_candidates)
        risk_end = min(accepted + 1, max_candidates)
        for j in range(1, risk_end + 1):
            features = _hazard_features_for_position(
                j, m, c, h, variant, max_candidates=max_candidates
            )
            label = 1.0 if j <= accepted else 0.0
            x_list.append(features)
            y_list.append(label)

    if not x_list:
        raise ValueError("no risk set transitions available to fit hazard model")

    X = np.asarray(x_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)

    center = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale < 1e-8] = 1.0

    standardized_X = (X - center) / scale
    Phi = np.column_stack([np.ones(len(X), dtype=np.float64), standardized_X])
    reg = float(ridge)

    def loss(beta: np.ndarray) -> float:
        z = Phi @ beta
        nll = float(np.logaddexp(0.0, z).sum() - y @ z)
        penalty = 0.5 * reg * float(beta @ beta)
        return nll + penalty

    def grad(beta: np.ndarray) -> np.ndarray:
        z = Phi @ beta
        p = expit(z)
        return Phi.T @ (p - y) + reg * beta

    initial_beta = np.zeros(Phi.shape[1], dtype=np.float64)
    fit = minimize(
        loss,
        initial_beta,
        jac=grad,
        method="L-BFGS-B",
        options={"maxiter": 500, "gtol": 1e-7},
    )
    if not fit.success:
        raise RuntimeError(f"Hazard logistic fit failed for {variant}: {fit.message}")

    opt_beta = fit.x
    speedup_metrics: dict[str, float] = {}
    if optimize_speedup and len(train_rows) > 0:
        opt_beta, speedup_metrics = optimize_hazard_for_e2e_speedup(
            fit.x,
            train_rows,
            variant,
            center,
            scale,
            cost_by_prefix=cost_by_prefix,
            allowed_prefixes=allowed_prefixes,
            max_candidates=max_candidates,
            overhead_ms=overhead_ms,
        )

    return {
        "model_type": "hazard",
        "variant": variant,
        "training_groups": sorted({str(row.get("source_group_id", "default")) for row in train_rows}),
        "max_candidates": max_candidates,
        "features": list(HAZARD_VARIANTS[variant]),
        "beta": [float(v) for v in opt_beta],
        "center": [float(v) for v in center],
        "scale": [float(v) for v in scale],
        "ridge": reg,
        "train_risk_states": len(y),
        "speedup_metrics": speedup_metrics,
    }


def predict_hazard_curve(
    model: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[list[float], list[str]]:
    """Predict survival curve S(k) = prod_{j=1..k} p_j^{cond} using fitted hazard model."""
    variant = str(model["variant"])
    max_candidates = int(model.get("max_candidates", 15))
    beta = np.asarray(model["beta"], dtype=np.float64)
    center = np.asarray(model["center"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)

    m, c, h = _extract_hazard_signals(row, max_candidates=max_candidates)
    features_list = [
        _hazard_features_for_position(j, m, c, h, variant, max_candidates=max_candidates)
        for j in range(1, max_candidates + 1)
    ]
    X_pred = np.asarray(features_list, dtype=np.float64)
    standardized = (X_pred - center) / scale
    Phi = np.column_stack([np.ones(len(X_pred), dtype=np.float64), standardized])

    logits = Phi @ beta
    log_p_cond = -np.logaddexp(0.0, -logits)
    log_S = np.cumsum(log_p_cond)
    S = np.exp(log_S)
    S = np.clip(S, 0.0, 1.0 - 1e-12)

    return [float(v) for v in S], ["hazard"] * max_candidates


def fit_survival_curve(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    max_candidates: int = 15,
    min_support_states: int = 8,
    min_support_groups: int = 3,
) -> dict[str, Any]:
    """Fit training-only binned or hazard survival curves."""

    if variant not in SURVIVAL_VARIANTS:
        raise ValueError(f"unsupported survival variant {variant!r}")
    if variant in HAZARD_VARIANTS:
        return fit_hazard_model(
            train_rows,
            variant=variant,
            max_candidates=max_candidates,
        )
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
    if model.get("model_type") == "hazard" or variant in HAZARD_VARIANTS:
        return {
            "schema_version": "dflash-speculation-b.survival-model.v1",
            "model_type": "hazard",
            "variant": variant,
            "training_groups": [str(value) for value in model.get("training_groups", ())],
            "max_candidates": int(model["max_candidates"]),
            "features": list(model["features"]),
            "beta": [float(value) for value in model["beta"]],
            "center": [float(value) for value in model["center"]],
            "scale": [float(value) for value in model["scale"]],
            "ridge": float(model.get("ridge", 1.0)),
            "train_risk_states": int(model.get("train_risk_states", 0)),
        }
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
    if payload.get("model_type") == "hazard" or "beta" in payload:
        max_candidates = int(payload.get("max_candidates", 0))
        if max_candidates < 1:
            raise ValueError("hazard model artifact has invalid max_candidates")
        beta = [float(v) for v in payload.get("beta", ())]
        center = [float(v) for v in payload.get("center", ())]
        scale = [float(v) for v in payload.get("scale", ())]
        if len(beta) != len(center) + 1 or len(center) != len(scale):
            raise ValueError("hazard model artifact has mismatched parameter dimensions")
        return {
            "model_type": "hazard",
            "variant": variant,
            "training_groups": [str(value) for value in payload.get("training_groups", ())],
            "max_candidates": max_candidates,
            "features": tuple(str(value) for value in payload.get("features", ())),
            "beta": beta,
            "center": center,
            "scale": scale,
            "ridge": float(payload.get("ridge", 1.0)),
            "train_risk_states": int(payload.get("train_risk_states", 0)),
        }
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
    if model.get("model_type") == "hazard" or str(model.get("variant", "")) in HAZARD_VARIANTS:
        return predict_hazard_curve(model, row)
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
            ("R-Hazard-Present", "R-Hazard-Pos"),
            ("R-Hazard-Prefix", "R-Hazard-Present"),
            ("R-Hazard-Suffix-Margin", "R-Hazard-Prefix"),
            ("R-Hazard-Suffix-Entropy", "R-Hazard-Prefix"),
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
