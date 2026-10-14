"""Causal adjacent-round feature construction for the optional B4 study."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from TrainingFree.speculation_experiments.protocol import assign_group_folds


def pair_adjacent_round_signals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attach round t-1 signals to t only when output progress is causal."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str, int]] = set()
    for row in rows:
        if row.get("type", "candidate_signals") != "candidate_signals":
            continue
        dataset = str(row.get("dataset", "unknown"))
        sample = str(row.get("sample_id", ""))
        group = str(row.get("source_group_id", ""))
        state_id = str(row.get("state_id", ""))
        round_index = int(row.get("round_index", -1))
        if not sample or not group or not state_id or round_index < 0:
            raise ValueError("candidate signals require sample/group/state identity and a nonnegative round")
        identity = (dataset, sample, group, round_index)
        if identity in seen:
            raise ValueError(f"duplicate candidate signal for request round: {identity}")
        seen.add(identity)
        groups[(dataset, sample, group)].append(row)

    paired: list[dict[str, Any]] = []
    for identity, request_rows in sorted(groups.items()):
        ordered = sorted(request_rows, key=lambda row: int(row["round_index"]))
        for previous, current in zip(ordered, ordered[1:]):
            previous_round = int(previous["round_index"])
            current_round = int(current["round_index"])
            if current_round != previous_round + 1:
                continue
            uncertainty = previous.get("draft_uncertainty") or {}
            entropy = [float(value) for value in uncertainty.get("normalized_entropy", ())]
            confidence = [float(value) for value in uncertainty.get("confidence", ())]
            if not entropy or len(entropy) != len(confidence):
                continue
            previous_processed = int(previous.get("processed_output_tokens", -1))
            current_processed = int(current.get("processed_output_tokens", -1))
            previous_accepted = int(previous.get("accepted_candidates", -1))
            if previous_processed < 0 or current_processed < 0:
                raise ValueError("candidate signals require processed_output_tokens")
            if not 0 <= previous_accepted <= len(entropy):
                raise ValueError(f"invalid accepted_candidates at {previous.get('state_id')}")
            # A verifier round commits the accepted draft prefix plus one target
            # token (the pending anchor). Reject rows that merely have adjacent
            # round labels but do not represent this same output trajectory.
            if current_processed != previous_processed + 1 + previous_accepted:
                continue
            if any(not math.isfinite(value) for value in (*entropy, *confidence)):
                raise ValueError(f"non-finite candidate signal at {previous.get('state_id')}")
            current_row = dict(current)
            current_row["lagged_signals"] = {
                "source_state_id": str(previous["state_id"]),
                "source_round_index": previous_round,
                "draft_entropy_mean": mean(entropy),
                "draft_entropy_max": max(entropy),
                "draft_confidence_min": min(confidence),
                "draft_confidence_mean": mean(confidence),
                "accepted_fraction": float(previous_accepted) / len(entropy),
                "all_candidates_accepted": previous_accepted == len(entropy),
                "candidate_count": len(entropy),
                "history_ema": previous.get("history_ema"),
            }
            current_row["lagged_source_state_id"] = str(previous["state_id"])
            current_row["lagged_source_round_index"] = previous_round
            current_row["lagged_draft_entropy_mean"] = mean(entropy)
            current_row["lagged_draft_entropy_max"] = max(entropy)
            current_row["lagged_draft_confidence_min"] = min(confidence)
            current_row["lagged_draft_confidence_mean"] = mean(confidence)
            current_row["lagged_accepted_fraction"] = float(previous_accepted) / len(entropy)
            current_row["lagged_all_candidates_accepted"] = bool(
                previous_accepted == len(entropy)
            )
            paired.append(current_row)
    return paired


_COST_VARIANTS = {
    "R": ("context", "progress"),
    "R-A": ("context", "progress", "history"),
    "R-A-LDH": ("context", "progress", "history", "lagged_draft_entropy_max"),
    "R-A-LDC": ("context", "progress", "history", "lagged_draft_confidence_min"),
    "R-A-LDHC": (
        "context", "progress", "history",
        "lagged_draft_entropy_max", "lagged_draft_confidence_min",
    ),
}


def _base_features(row: Mapping[str, Any]) -> tuple[int, int, int]:
    context = int(row.get("context_tokens", row.get("prompt_tokens", 0)))
    processed = max(0, int(row.get("processed_output_tokens", 0)))
    history = row.get("history_ema")
    history_bucket = 0 if history is None else (1 if float(history) < 0.5 else 2)
    return min(max(context, 0) // 4096, 3), (0 if processed < 128 else 1 if processed < 512 else 2), history_bucket


def _feature_value(row: Mapping[str, Any], name: str) -> float:
    lagged = row.get("lagged_signals") or {}
    raw = row.get(name, lagged.get(name.removeprefix("lagged_")))
    if raw is None or not math.isfinite(float(raw)):
        raise ValueError(f"B4 row is missing finite {name}")
    return float(raw)


def _quartile_edges(values: Sequence[float]) -> list[float]:
    ordered = sorted(float(value) for value in values)
    edges = []
    for fraction in (0.25, 0.5, 0.75):
        edge = ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return edges


def _feature_key(
    row: Mapping[str, Any],
    variant: str,
    edges: Mapping[str, Sequence[float]],
) -> tuple[int, ...]:
    names = _COST_VARIANTS[variant]
    context, progress, history = _base_features(row)
    key = [context, progress]
    if "history" in names:
        key.append(history)
    for name in names:
        if name.startswith("lagged_"):
            import bisect

            key.append(bisect.bisect_right(edges.get(name, ()), _feature_value(row, name)))
    return tuple(key)


def fit_draft_cost_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    gammas: Sequence[int] = (3, 7, 11, 15),
    min_support_states: int = 8,
    min_support_groups: int = 3,
) -> dict[str, Any]:
    """Fit a small train-only binned cost-per-commit policy for B4."""
    if variant not in _COST_VARIANTS:
        raise ValueError(f"unsupported B4 draft-cost variant {variant!r}")
    if not rows or min_support_states < 1 or min_support_groups < 1:
        raise ValueError("B4 fit requires rows and positive support thresholds")
    gamma_values = tuple(int(value) for value in gammas)
    if not gamma_values or tuple(sorted(set(gamma_values))) != gamma_values:
        raise ValueError("B4 gamma actions must be unique and sorted")
    lagged_names = [name for name in _COST_VARIANTS[variant] if name.startswith("lagged_")]
    edges = {
        name: _quartile_edges([_feature_value(row, name) for row in rows])
        for name in lagged_names
    }
    full_cells: dict[tuple[int, ...], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    regime_cells: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    full_groups: dict[tuple[int, ...], set[str]] = defaultdict(set)
    regime_groups: dict[tuple[int, int], set[str]] = defaultdict(set)
    global_costs: dict[int, list[float]] = defaultdict(list)
    train_groups = set()
    for row in rows:
        group = str(row.get("source_group_id", ""))
        if not group:
            raise ValueError("B4 cost rows require source_group_id")
        train_groups.add(group)
        full_key = _feature_key(row, variant, edges)
        regime_key = _base_features(row)[:2]
        costs = row.get("selection_costs_ms_per_commit_by_gamma") or {}
        for gamma in gamma_values:
            value = float(costs.get(str(gamma), costs.get(gamma, math.nan)))
            if not math.isfinite(value) or value <= 0:
                continue
            full_cells[full_key][gamma].append(value)
            regime_cells[regime_key][gamma].append(value)
            global_costs[gamma].append(value)
            full_groups[full_key].add(group)
            regime_groups[regime_key].add(group)
    def serialize_cells(cells: Mapping[Any, Mapping[int, Sequence[float]]], groups: Mapping[Any, set[str]]) -> list[dict[str, Any]]:
        output = []
        for key, costs_by_gamma in sorted(cells.items(), key=lambda item: str(item[0])):
            output.append({
                "key": list(key),
                "cost_by_gamma": {
                    str(gamma): mean(values)
                    for gamma, values in sorted(costs_by_gamma.items())
                    if values
                },
                "state_count": max((len(values) for values in costs_by_gamma.values()), default=0),
                "group_count": len(groups.get(key, set())),
            })
        return output
    return {
        "schema_version": "dflash-speculation-b.b4-cost-model.v1",
        "variant": variant,
        "gammas": list(gamma_values),
        "training_groups": sorted(train_groups),
        "min_support_states": int(min_support_states),
        "min_support_groups": int(min_support_groups),
        "supported_previous_gamma": 15 if lagged_names else None,
        "feature_edges": edges,
        "full_cells": serialize_cells(full_cells, full_groups),
        "regime_cells": serialize_cells(regime_cells, regime_groups),
        "global_cost_by_gamma": {
            str(gamma): mean(values) for gamma, values in sorted(global_costs.items()) if values
        },
    }


def predict_draft_costs(
    model: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[dict[int, float], str]:
    """Predict supported gamma costs with a transparent regime/global backoff."""
    if model.get("schema_version") != "dflash-speculation-b.b4-cost-model.v1":
        raise ValueError("unsupported B4 cost-model schema")
    variant = str(model.get("variant", ""))
    edges = model.get("feature_edges") or {}
    full_key = _feature_key(row, variant, edges)
    regime_key = list(_base_features(row)[:2])
    def find_cell(cells: Sequence[Mapping[str, Any]], key: Sequence[int], minimum_states: int, minimum_groups: int) -> Mapping[str, Any] | None:
        for cell in cells:
            if list(cell.get("key", ())) == list(key):
                if int(cell.get("state_count", 0)) >= minimum_states and int(cell.get("group_count", 0)) >= minimum_groups:
                    return cell
        return None
    full = find_cell(
        model.get("full_cells", ()), full_key,
        int(model["min_support_states"]), int(model["min_support_groups"]),
    )
    regime = find_cell(model.get("regime_cells", ()), regime_key, 1, 1)
    if full is not None:
        selected = full.get("cost_by_gamma", {})
        level = "signal"
    elif regime is not None:
        selected = regime.get("cost_by_gamma", {})
        level = "regime"
    else:
        selected = model.get("global_cost_by_gamma", {})
        level = "global"
    costs = {int(key): float(value) for key, value in selected.items() if math.isfinite(float(value)) and float(value) > 0}
    return costs, level


def cross_validated_draft_cost_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[str] = tuple(_COST_VARIANTS),
    n_splits: int = 4,
    seed: int = 42,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261013,
) -> dict[str, Any]:
    """Evaluate gamma selection on held-out source groups and timing repeats."""
    if not rows:
        raise ValueError("B4 evaluation requires rows")
    assignments = assign_group_folds(rows, n_splits=n_splits, seed=seed)
    variant_values = tuple(variants)
    if "R" not in variant_values or any(value not in _COST_VARIANTS for value in variant_values):
        raise ValueError("B4 variants must include R and be supported draft-cost models")
    predictions: list[dict[str, Any]] = []
    fallback_counts: dict[str, dict[str, int]] = {variant: defaultdict(int) for variant in variant_values}
    for fold in sorted(set(assignments.values())):
        train = [row for row in rows if assignments[str(row["source_group_id"])] != fold]
        test = [row for row in rows if assignments[str(row["source_group_id"])] == fold]
        models = {variant: fit_draft_cost_policy(train, variant=variant) for variant in variant_values}
        for row in test:
            predicted = {}
            selected_gamma = {}
            for variant, model in models.items():
                costs, level = predict_draft_costs(model, row)
                fallback_counts[variant][level] += 1
                if not costs:
                    continue
                predicted[variant] = {str(gamma): cost for gamma, cost in costs.items()}
                selected_gamma[variant] = min(costs, key=lambda gamma: (costs[gamma], gamma))
            actual = {int(key): float(value) for key, value in (row.get("evaluation_costs_ms_per_commit_by_gamma") or {}).items()}
            if not actual or any(gamma not in actual for gamma in selected_gamma.values()):
                continue
            oracle_gamma = min(actual, key=lambda gamma: (actual[gamma], gamma))
            predictions.append({
                "dataset": str(row["dataset"]),
                "sample_id": str(row["sample_id"]),
                "source_group_id": str(row["source_group_id"]),
                "state_id": str(row["state_id"]),
                "fold": fold,
                "training_groups": sorted({str(value["source_group_id"]) for value in train}),
                "actual_costs_ms_per_commit_by_gamma": {str(key): value for key, value in actual.items()},
                "predicted_costs_ms_per_commit_by_variant": predicted,
                "selected_gamma_by_variant": {key: int(value) for key, value in selected_gamma.items()},
                "oracle_gamma_eval": oracle_gamma,
                "cost_by_variant": {
                    "oracle": actual[oracle_gamma],
                    **{
                        variant: actual[gamma]
                        for variant, gamma in selected_gamma.items()
                    },
                },
            })
    def macro_cost(values: Sequence[Mapping[str, Any]], variant: str) -> float | None:
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in values:
            grouped[str(row["dataset"])][str(row["source_group_id"])].append(float(row["cost_by_variant"][variant]))
        dataset_means = [mean(mean(costs) for costs in groups.values()) for groups in grouped.values() if groups]
        return mean(dataset_means) if dataset_means else None
    metrics = {variant: macro_cost(predictions, variant) for variant in variant_values}
    metrics["eval_oracle"] = macro_cost(predictions, "oracle")
    regret_by_variant = {}
    gamma_distribution: dict[str, dict[str, int]] = {}
    for variant in variant_values:
        grouped_regret: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        counts: dict[str, int] = defaultdict(int)
        for row in predictions:
            oracle_cost = float(row["cost_by_variant"]["oracle"])
            grouped_regret[str(row["dataset"])][str(row["source_group_id"])].append(
                float(row["cost_by_variant"][variant]) / oracle_cost - 1.0
            )
            gamma = int(row["selected_gamma_by_variant"][variant])
            counts[str(gamma)] += 1
        dataset_regrets = [
            mean(mean(values) for values in documents.values())
            for documents in grouped_regret.values() if documents
        ]
        regret_by_variant[variant] = mean(dataset_regrets) if dataset_regrets else None
        gamma_distribution[variant] = dict(counts)
    controls = {}
    for variant in variant_values:
        if variant in {"R", "R-A"}:
            continue
        base_variant = "R-A" if variant.startswith("R-A-") else "R"
        base = metrics.get(base_variant)
        candidate = metrics.get(variant)
        controls[f"{variant}_vs_{base_variant}"] = (base - candidate) / base if base and candidate is not None else None
    intervals: dict[str, list[float] | None] = {}
    for contrast in controls:
        variant, base_variant = contrast.split("_vs_", 1)
        per_document: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for row in predictions:
            if variant in row["cost_by_variant"] and base_variant in row["cost_by_variant"]:
                per_document[str(row["dataset"])][str(row["source_group_id"])].append((
                    float(row["cost_by_variant"][base_variant]),
                    float(row["cost_by_variant"][variant]),
                ))
        draws = []
        rng = random.Random(bootstrap_seed)
        if bootstrap_replicates and per_document and all(len(groups) >= 2 for groups in per_document.values()):
            for _ in range(bootstrap_replicates):
                base_means = []
                signal_means = []
                for groups in per_document.values():
                    sampled_groups = [rng.choice(list(groups)) for _ in groups]
                    sampled = [item for group in sampled_groups for item in groups[group]]
                    base_means.append(mean(pair[0] for pair in sampled))
                    signal_means.append(mean(pair[1] for pair in sampled))
                base_mean = mean(base_means)
                signal_mean = mean(signal_means)
                draws.append((base_mean - signal_mean) / base_mean if base_mean else 0.0)
        if draws:
            draws.sort()
            intervals[contrast] = [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]
        else:
            intervals[contrast] = None
    return {
        "schema_version": "dflash-speculation-b.b4-analysis.v1",
        "status": "complete" if predictions else "inconclusive",
        "state_count": len(predictions),
        "source_group_count": len({str(row["source_group_id"]) for row in predictions}),
        "fold_assignments": assignments,
        "variants": list(variant_values),
        "macro_cost_ms_per_commit": metrics,
        "mean_regret_vs_eval_oracle": regret_by_variant,
        "selected_gamma_counts": gamma_distribution,
        "cost_skill_vs_control": controls,
        "cost_skill_ci95_vs_control": intervals,
        "fallback_counts": {variant: dict(counts) for variant, counts in fallback_counts.items()},
        "oof_predictions": predictions,
        "unit_of_analysis": "source_group_document_macro_stratified_by_dataset",
        "note": "B4 fixed-reference predictive screen; E2E deployment still requires dev lock and exact confirm.",
    }


class LaggedDraftLengthController:
    """Select a feasible gamma from a frozen B4 lagged cost model."""

    def __init__(self, model: Mapping[str, Any], *, fallback_gamma: int = 15) -> None:
        if model.get("schema_version") != "dflash-speculation-b.b4-cost-model.v1":
            raise ValueError("unsupported B4 cost model")
        self.model = model
        self.fallback_gamma = int(fallback_gamma)
        self.requires_uncertainty = any(
            name.startswith("lagged_") for name in _COST_VARIANTS[str(model.get("variant", ""))]
        )

    def __call__(
        self,
        *,
        state: Any,
        previous_draft_uncertainty: Mapping[str, Sequence[float]] | None,
        previous_draft_gamma: int | None,
        feasible_gammas: Sequence[int],
        fallback_gamma: int,
    ) -> dict[str, Any]:
        uncertainty = previous_draft_uncertainty or {}
        entropy = [float(value) for value in uncertainty.get("normalized_entropy", ())]
        confidence = [float(value) for value in uncertainty.get("confidence", ())]
        needs_lagged = any(name.startswith("lagged_") for name in _COST_VARIANTS[str(self.model["variant"])])
        if needs_lagged and int(previous_draft_gamma or 0) != int(self.model.get("supported_previous_gamma", 15)):
            feasible = tuple(int(value) for value in feasible_gammas)
            chosen = int(fallback_gamma if fallback_gamma in feasible else max(feasible))
            return {
                "gamma": chosen,
                "reason": "fallback_unsupported_previous_gamma",
                "support_level": "fallback",
                "predicted_cost_per_commit": None,
                "fallback_gamma": int(fallback_gamma),
                "previous_draft_gamma": previous_draft_gamma,
            }
        if needs_lagged and (not entropy or not confidence or len(entropy) != len(confidence)):
            feasible = tuple(int(value) for value in feasible_gammas)
            chosen = int(fallback_gamma if fallback_gamma in feasible else max(feasible))
            return {
                "gamma": chosen,
                "reason": "fallback_missing_lagged_signal",
                "support_level": "fallback",
                "predicted_cost_per_commit": None,
                "fallback_gamma": int(fallback_gamma),
                "lagged_source_round": None,
            }
        row = {
            "context_tokens": int(state.logical_length),
            "processed_output_tokens": int(state.processed_output_tokens),
            "history_ema": state.history_acceptance,
            "lagged_signals": {},
        }
        if entropy and confidence and len(entropy) == len(confidence):
            row["lagged_signals"] = {
                "draft_entropy_max": max(entropy),
                "draft_confidence_min": min(confidence),
            }
        costs, level = predict_draft_costs(self.model, row)
        feasible = tuple(int(value) for value in feasible_gammas)
        supported = {gamma: cost for gamma, cost in costs.items() if gamma in feasible}
        if not supported:
            chosen = int(fallback_gamma if fallback_gamma in feasible else max(feasible))
            reason = "fallback_no_supported_action"
        else:
            chosen = min(supported, key=lambda gamma: (supported[gamma], gamma))
            reason = "minimum_predicted_cost_per_commit"
        return {
            "gamma": chosen,
            "reason": reason,
            "support_level": level,
            "predicted_cost_per_commit": supported.get(chosen),
            "fallback_gamma": int(fallback_gamma),
            "lagged_source_round": (int(state.round_index) - 1) if entropy else None,
        }
