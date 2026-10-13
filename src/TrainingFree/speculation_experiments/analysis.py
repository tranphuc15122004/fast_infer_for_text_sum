"""Document-clustered analysis for same-prefix B1 draft-shape sweeps."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _macro_mean(
    values_by_state: Mapping[str, float],
    state_info: Mapping[str, tuple[str, str]],
    group_draws: Mapping[str, Sequence[str]] | None = None,
) -> float:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for dataset, groups in (group_draws or {}).items():
        for group in groups:
            state_values = [
                values_by_state[state_id]
                for state_id, info in state_info.items()
                if info == (dataset, group) and state_id in values_by_state
            ]
            if state_values:
                by_dataset[dataset].append(sum(state_values) / len(state_values))
    if not by_dataset:
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for state_id, value in values_by_state.items():
            dataset, group = state_info[state_id]
            grouped[dataset][group].append(value)
        for dataset, group_values in grouped.items():
            by_dataset[dataset] = [sum(values) / len(values) for values in group_values.values()]
    dataset_means = [sum(values) / len(values) for values in by_dataset.values() if values]
    if not dataset_means:
        raise ValueError("cannot compute a document-macro mean without valid states")
    return sum(dataset_means) / len(dataset_means)


def _weighted_group_mean(
    values_by_state: Mapping[str, float],
    state_info: Mapping[str, tuple[str, str]],
    group_draws: Mapping[str, Sequence[str]],
) -> float:
    means: list[float] = []
    for dataset, groups in group_draws.items():
        draw_values = []
        for group in groups:
            state_values = [
                values_by_state[state_id]
                for state_id, info in state_info.items()
                if info == (dataset, group) and state_id in values_by_state
            ]
            if state_values:
                draw_values.append(sum(state_values) / len(state_values))
        if draw_values:
            means.append(sum(draw_values) / len(draw_values))
    if not means:
        raise ValueError("bootstrap sample has no states")
    return sum(means) / len(means)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile from no values")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze_draft_shape(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 20261013,
    near_optimal_ratio: float = 1.02,
) -> dict[str, Any]:
    """Estimate B1 oracle/fixed gaps with timing selection/evaluation split.

    The oracle and best fixed gamma are selected only from rows marked
    ``selection``. Their cost is then evaluated on the independent repetition
    split. Bootstrap resampling is stratified by dataset and source document;
    timing repetitions and states are never treated as independent samples.
    """

    if bootstrap_replicates < 0 or near_optimal_ratio < 1.0:
        raise ValueError("bootstrap_replicates must be nonnegative and near_optimal_ratio >= 1")
    actions = [row for row in rows if row.get("type") == "draft_shape_action"]
    if not actions:
        raise ValueError("no draft_shape_action rows were provided")

    by_cell: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    state_info: dict[str, tuple[str, str]] = {}
    repetitions_by_state: dict[str, set[int]] = defaultdict(set)
    gammas: set[int] = set()
    for row in actions:
        state_id = str(row.get("state_id", ""))
        group = str(row.get("source_group_id", ""))
        dataset = str(row.get("dataset", ""))
        if not state_id or not group or not dataset:
            raise ValueError("B1 action rows require state_id, dataset, and source_group_id")
        gamma = int(row["gamma"])
        repetition = int(row["repetition"])
        key = (state_id, gamma, repetition)
        if key in by_cell:
            raise ValueError(f"duplicate paired action cell {key}")
        state_pair = (dataset, group)
        if state_id in state_info and state_info[state_id] != state_pair:
            raise ValueError(f"state_id {state_id} maps to multiple source groups")
        state_info[state_id] = state_pair
        by_cell[key] = row
        repetitions_by_state[state_id].add(repetition)
        gammas.add(gamma)
    gamma_values = tuple(sorted(gammas))
    states = sorted(state_info)
    datasets = sorted({dataset for dataset, _ in state_info.values()})
    if len(gamma_values) < 2:
        raise ValueError("B1 requires at least two gamma actions")

    state_reps: dict[str, set[int]] = {}
    for state in states:
        state_reps[state] = repetitions_by_state[state]
        for gamma in gamma_values:
            present = {rep for rep in state_reps[state] if (state, gamma, rep) in by_cell}
            if present != state_reps[state]:
                raise ValueError(f"unpaired repetitions for state={state}, gamma={gamma}")
        for rep in state_reps[state]:
            splits = {str(by_cell[(state, gamma, rep)].get("repetition_split", "")) for gamma in gamma_values}
            if len(splits) != 1 or not splits.issubset({"selection", "evaluation"}):
                raise ValueError(f"paired repetition has inconsistent/unknown split for state={state}, rep={rep}")

    # The same repetition IDs must cover the whole paired state/action matrix.
    rep_ids = set.intersection(*(state_reps[state] for state in states))
    if not rep_ids:
        raise ValueError("no common paired repetitions across states")
    if any(state_reps[state] != rep_ids for state in states):
        raise ValueError("paired B1 repetitions differ across states")
    if rep_ids != set(range(min(rep_ids), max(rep_ids) + 1)):
        raise ValueError("B1 repetition IDs must be contiguous")
    selection_reps = {
        rep for rep in rep_ids
        if all(str(by_cell[(state, gamma_values[0], rep)].get("repetition_split")) == "selection" for state in states)
    }
    evaluation_reps = {
        rep for rep in rep_ids
        if all(str(by_cell[(state, gamma_values[0], rep)].get("repetition_split")) == "evaluation" for state in states)
    }
    if not selection_reps or not evaluation_reps:
        raise ValueError("paired actions need at least one selection and one evaluation repetition")

    selection_j: dict[str, dict[int, float]] = defaultdict(dict)
    evaluation_j: dict[str, dict[int, float]] = defaultdict(dict)
    evaluation_t: dict[str, dict[int, float]] = defaultdict(dict)
    selection_state_gamma_j: dict[tuple[str, int], float] = {}
    evaluation_state_gamma_j: dict[tuple[str, int], float] = {}
    evaluation_state_gamma_t: dict[tuple[str, int], float] = {}
    mean_accepted: dict[int, list[float]] = defaultdict(list)
    for state in states:
        for gamma in gamma_values:
            sel_rows = [by_cell[(state, gamma, rep)] for rep in selection_reps]
            eval_rows = [by_cell[(state, gamma, rep)] for rep in evaluation_reps]
            selection_state_gamma_j[(state, gamma)] = sum(
                float(row["round_ms"]) / (1.0 + int(row["accepted_candidates"])) for row in sel_rows
            ) / len(sel_rows)
            evaluation_state_gamma_j[(state, gamma)] = sum(
                float(row["round_ms"]) / (1.0 + int(row["accepted_candidates"])) for row in eval_rows
            ) / len(eval_rows)
            evaluation_state_gamma_t[(state, gamma)] = sum(float(row["round_ms"]) for row in eval_rows) / len(eval_rows)
            mean_accepted[gamma].append(sum(int(row["accepted_candidates"]) for row in eval_rows) / len(eval_rows))

    oracle_gamma: dict[str, int] = {}
    for state in states:
        oracle_gamma[state] = min(
            gamma_values,
            key=lambda gamma: (selection_state_gamma_j[(state, gamma)], gamma),
        )

    group_datasets: dict[str, list[str]] = defaultdict(list)
    for dataset, group in sorted(set(state_info.values())):
        group_datasets[dataset].append(group)

    def select_fixed(draws: Mapping[str, Sequence[str]] | None = None) -> int:
        costs = {}
        for gamma in gamma_values:
            state_values = {
                state: selection_state_gamma_j[(state, gamma)] for state in states
            }
            costs[gamma] = (
                _weighted_group_mean(state_values, state_info, draws)
                if draws is not None else _macro_mean(state_values, state_info)
            )
        return min(gamma_values, key=lambda gamma: (costs[gamma], gamma))

    best_fixed = select_fixed()
    oracle_eval_j = {
        state: evaluation_state_gamma_j[(state, oracle_gamma[state])] for state in states
    }
    fixed_eval_j = {
        state: evaluation_state_gamma_j[(state, best_fixed)] for state in states
    }
    oracle_eval_t = {
        state: evaluation_state_gamma_t[(state, oracle_gamma[state])] for state in states
    }
    fixed_eval_t = {
        state: evaluation_state_gamma_t[(state, best_fixed)] for state in states
    }

    def gaps(draws: Mapping[str, Sequence[str]] | None = None) -> tuple[float, float]:
        macro = _weighted_group_mean if draws is not None else _macro_mean
        oracle_j = macro(oracle_eval_j, state_info, draws) if draws is not None else macro(oracle_eval_j, state_info)
        fixed_j = macro(fixed_eval_j, state_info, draws) if draws is not None else macro(fixed_eval_j, state_info)
        oracle_t = macro(oracle_eval_t, state_info, draws) if draws is not None else macro(oracle_eval_t, state_info)
        fixed_t = macro(fixed_eval_t, state_info, draws) if draws is not None else macro(fixed_eval_t, state_info)
        return 1.0 - oracle_j / fixed_j, 1.0 - oracle_t / fixed_t

    gap_j, gap_t = gaps()
    bootstrap_j: list[float] = []
    bootstrap_t: list[float] = []
    rng = random.Random(seed)
    if bootstrap_replicates:
        for _ in range(bootstrap_replicates):
            draws = {
                dataset: [rng.choice(groups) for _ in groups]
                for dataset, groups in group_datasets.items()
            }
            boot_fixed = select_fixed(draws)
            boot_fixed_eval_j = {state: evaluation_state_gamma_j[(state, boot_fixed)] for state in states}
            boot_fixed_eval_t = {state: evaluation_state_gamma_t[(state, boot_fixed)] for state in states}
            oracle_j = _weighted_group_mean(oracle_eval_j, state_info, draws)
            fixed_j = _weighted_group_mean(boot_fixed_eval_j, state_info, draws)
            oracle_t = _weighted_group_mean(oracle_eval_t, state_info, draws)
            fixed_t = _weighted_group_mean(boot_fixed_eval_t, state_info, draws)
            bootstrap_j.append(1.0 - oracle_j / fixed_j)
            bootstrap_t.append(1.0 - oracle_t / fixed_t)

    histogram = Counter(str(gamma) for gamma in oracle_gamma.values())
    near_optimal_coverage = {}
    for gamma in gamma_values:
        near_optimal_coverage[str(gamma)] = sum(
            min(selection_state_gamma_j[(state, value)] for value in gamma_values)
            * near_optimal_ratio >= selection_state_gamma_j[(state, gamma)]
            for state in states
        ) / len(states)

    marginal = {}
    for left, right in zip(gamma_values, gamma_values[1:]):
        pairs = [(state, left, right) for state in states]
        delta_l = [
            sum(int(by_cell[(state, right, rep)]["accepted_candidates"]) for rep in evaluation_reps) / len(evaluation_reps)
            - sum(int(by_cell[(state, left, rep)]["accepted_candidates"]) for rep in evaluation_reps) / len(evaluation_reps)
            for state, _, _ in pairs
        ]
        delta_t = [
            evaluation_state_gamma_t[(state, right)] - evaluation_state_gamma_t[(state, left)]
            for state, _, _ in pairs
        ]
        marginal[f"{left}_to_{right}"] = {
            "mean_delta_accepted": sum(delta_l) / len(delta_l),
            "mean_delta_round_ms": sum(delta_t) / len(delta_t),
            "fraction_longer_accepts_no_more": sum(value <= 0 for value in delta_l) / len(delta_l),
            "proposal_prefix_equal_fraction": sum(
                tuple(by_cell[(state, right, rep)]["proposed_token_ids"][:left])
                == tuple(by_cell[(state, left, rep)]["proposed_token_ids"][:left])
                for state in states for rep in evaluation_reps
            ) / (len(states) * len(evaluation_reps)),
        }

    j_ci = [_percentile(bootstrap_j, 0.025), _percentile(bootstrap_j, 0.975)] if bootstrap_j else None
    t_ci = [_percentile(bootstrap_t, 0.025), _percentile(bootstrap_t, 0.975)] if bootstrap_t else None
    return {
        "schema_version": "dflash-speculation-b.b1-analysis.v1",
        "status": "complete",
        "state_count": len(states),
        "source_group_count": len(set(state_info.values())),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "gamma_values": list(gamma_values),
        "selection_repetitions": sorted(selection_reps),
        "evaluation_repetitions": sorted(evaluation_reps),
        "best_fixed_gamma": best_fixed,
        "oracle_gamma_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "near_optimal_coverage": near_optimal_coverage,
        "mean_accepted_by_gamma": {
            str(gamma): sum(values) / len(values) for gamma, values in sorted(mean_accepted.items())
        },
        "gamma15_tail": ({
            "fraction_accepts_at_most_3": sum(
                int(by_cell[(state, 15, rep)]["accepted_candidates"]) <= 3
                for state in states for rep in evaluation_reps
            ) / (len(states) * len(evaluation_reps)),
            "fraction_accepts_at_least_11": sum(
                int(by_cell[(state, 15, rep)]["accepted_candidates"]) >= 11
                for state in states for rep in evaluation_reps
            ) / (len(states) * len(evaluation_reps)),
        } if 15 in gamma_values else None),
        "marginal": marginal,
        "evaluation": {
            "oracle_gap_j": gap_j,
            "oracle_gap_j_ci95": j_ci,
            "oracle_gap_t": gap_t,
            "oracle_gap_t_ci95": t_ci,
        },
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "unit_of_analysis": "source_group_document_macro_stratified_by_dataset",
    }
