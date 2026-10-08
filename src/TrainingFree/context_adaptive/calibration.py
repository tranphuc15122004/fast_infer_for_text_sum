"""Action-replay calibration and controller-overhead profiling."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .config import AdaptiveConfig
from .controller import AdaptiveController
from .statistics import AdaptiveStatistics, fit_calibration
from .types import Action, RoundState


CONTROLLER_VARIANTS = (
    "fixed",
    "dflash_full_fixed",
    "best_fixed_pair",
    "a_only",
    "b_only_history",
    "b_only_entropy",
    "independent_ab",
    "joint",
    "joint_no_entropy",
    "joint_no_source_relevance",
)


def fit_action_priors(
    rows: Iterable[Mapping[str, Any]],
    *,
    signature: Mapping[str, Any],
    config: AdaptiveConfig,
) -> dict[str, Any]:
    """Fit prefix survival and action costs from frozen-prefix replays."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("calibration action replay produced no observations")
    calibration = fit_calibration(
        materialized,
        signature=signature,
        latency_ema_alpha=config.latency_ema_alpha,
    )
    calibration["collection_scope"] = "immutable_prefix_action_replay"
    calibration["selector_id"] = config.selector
    calibration["timing_mode"] = config.timing_mode
    calibration["min_state_support"] = config.min_state_support
    calibration["min_cost_repetitions"] = 3
    calibration["action_replay_rows"] = len(materialized)
    calibration["calibration_source_groups"] = sorted({
        str(row["source_group_id"]) for row in materialized if row.get("source_group_id") is not None
    })
    calibration["calibration_sample_count"] = len({
        str(row.get("sample_id")) for row in materialized if row.get("sample_id") is not None
    })
    return calibration


def _state_from_row(row: Mapping[str, Any]) -> RoundState:
    return RoundState(
        round_index=int(row.get("round_index", 0)),
        logical_length=int(row.get("logical_length", 1)),
        processed_output_tokens=max(0, int(row.get("logical_length", 1)) - int(row.get("prompt_length", 0))),
        remaining_output_tokens=int(row.get("remaining_output_tokens", 1)),
        parent_entropy=(float(row["parent_entropy"]) if row.get("parent_entropy") is not None else None),
        source_concentration=(float(row["source_concentration"]) if row.get("source_concentration") is not None else None),
        history_acceptance=(float(row["history_acceptance"]) if row.get("history_acceptance") is not None else None),
        ranking_age=int(row.get("ranking_age", 10**9)),
        refresh_required=bool(row.get("refresh_required", False)),
        source_origin_round=row.get("source_origin_round"),
    )


def _feasible_actions(config: AdaptiveConfig, state: RoundState) -> tuple[Action, ...]:
    unique: dict[str, Action] = {}
    for budget in config.budgets:
        canonical: int | str = "full" if budget == "full" or int(budget) >= state.logical_length else int(budget)
        if state.refresh_required:
            canonical = "full"
        for gamma in config.gammas:
            if gamma <= max(0, state.remaining_output_tokens - 1):
                action = Action(canonical, gamma, config.length_mode)
                unique[action.key()] = action
    return tuple(unique.values())


def _state_identity(row: Mapping[str, Any]) -> str:
    values = {
        key: row.get(key)
        for key in (
            "round_index",
            "logical_length",
            "remaining_output_tokens",
            "parent_entropy",
            "source_concentration",
            "history_acceptance",
            "refresh_required",
            "ranking_age",
            "source_origin_round",
        )
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def profile_controller_costs(
    calibration: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    config: AdaptiveConfig,
    *,
    fixed_budget: int | str = "full",
    fixed_gamma: int = 15,
    gamma_reference: int = 15,
    repeats: int = 20,
) -> dict[str, Any]:
    """Measure each policy's CPU-side choose overhead on calibration states."""
    if repeats < 1:
        raise ValueError("controller profiling repeats must be positive")
    placeholders = {
        variant: {"choose_ms": 0.0, "repetitions": 0, "profiling_scope": "causal calibration states"}
        for variant in CONTROLLER_VARIANTS
    }
    calibration["controller_cost_priors"] = placeholders
    statistics = AdaptiveStatistics(
        calibration,
        prior_strength=config.prior_strength,
        decay=config.statistics_decay,
        online=False,
        min_state_support=1,
        min_cost_repetitions=1,
    )
    unique_states: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unique_states.setdefault(_state_identity(row), row)
    costs: dict[str, Any] = {}
    for variant in CONTROLLER_VARIANTS:
        samples: list[float] = []
        controller = AdaptiveController(
            config,
            statistics,
            variant=variant,
            fixed_budget=fixed_budget,
            fixed_gamma=fixed_gamma,
            gamma_reference=gamma_reference,
        )
        for row in unique_states.values():
            state = _state_from_row(row)
            actions = _feasible_actions(config, state)
            if not actions:
                continue
            started = time.perf_counter()
            for _ in range(repeats):
                controller.choose(state, actions)
            samples.append((time.perf_counter() - started) * 1000.0 / repeats)
        if not samples:
            continue
        samples = [value for value in samples if math.isfinite(value) and value >= 0]
        costs[variant] = {
            "choose_ms": sum(samples) / len(samples),
            "p95_choose_ms": sorted(samples)[max(0, math.ceil(0.95 * len(samples)) - 1)],
            "repetitions": len(samples) * repeats,
            "state_count": len(samples),
            "profiling_scope": "CPU controller.choose; action tables frozen",
        }
    calibration["controller_cost_priors"] = costs
    return calibration


def calibration_signature(
    *,
    target_signature: str,
    draft_signature: str,
    tokenizer_signature: str,
    runtime_signature: str,
    config: AdaptiveConfig,
    sampling_temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "target_signature": target_signature,
        "draft_signature": draft_signature,
        "tokenizer_signature": tokenizer_signature,
        "runtime_signature": runtime_signature,
        "selector_id": config.selector,
        "target_layer": config.target_layer,
        "refresh_period": config.refresh_period,
        "signal_update_period": config.signal_update_period,
        "source_chunk_size": config.source_chunk_size,
        "source_anchors": config.source_anchors,
        "recent_output": config.recent_output,
        "budgets": list(config.budgets),
        "gammas": list(config.gammas),
        "length_mode": config.length_mode,
        "entropy_signal_temperature": float(config.entropy_signal_temperature),
        "temperature": float(sampling_temperature),
        "max_new_tokens": int(max_new_tokens),
    }


def assert_signature_matches(calibration: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    actual = calibration.get("signature")
    if not isinstance(actual, Mapping):
        raise ValueError("calibration artifact has no signature")
    differences = [key for key, value in expected.items() if actual.get(key) != value]
    if differences:
        raise ValueError("calibration signature mismatch: " + ", ".join(sorted(differences)))
