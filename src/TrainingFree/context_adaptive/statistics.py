"""Prefix-survival and profiling tables; no model training or weights involved."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .types import Action, RoundState


def _entropy_bin(value: float | None, cutpoints: list[float] | tuple[float, ...]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "unknown"
    return str(sum(float(value) > float(cutpoint) for cutpoint in cutpoints))


def state_bucket(
    state: RoundState,
    entropy_cutpoints: list[float] | tuple[float, ...],
) -> dict[str, str]:
    concentration = state.source_concentration
    history = state.history_acceptance
    return {
        "entropy": _entropy_bin(state.parent_entropy, entropy_cutpoints),
        "concentration": (
            "unknown" if concentration is None else ("low" if concentration < 0.5 else "high")
        ),
        "history": "unknown" if history is None else ("low" if history < 0.5 else "high"),
        "context": str(math.ceil(max(state.logical_length, 1) / 2048)),
        "refresh": "1" if state.refresh_required else "0",
    }


def bucket_key(bucket: Mapping[str, str], *, include: tuple[str, ...] | None = None) -> str:
    keys = include or ("entropy", "concentration", "history", "context", "refresh")
    return "|".join(f"{key}={bucket.get(key, 'unknown')}" for key in keys)


def action_key(action: Action) -> str:
    return action.key()


@dataclass
class OnlinePrefixCount:
    count: float
    successes: list[float]


class AdaptiveStatistics:
    def __init__(
        self,
        calibration: Mapping[str, Any],
        *,
        prior_strength: float = 16.0,
        decay: float = 0.95,
        online: bool = True,
        min_state_support: int = 8,
        min_cost_repetitions: int = 3,
    ) -> None:
        if calibration.get("schema_version") != "cadflash.calibration.v1":
            raise ValueError("unsupported calibration schema")
        self.calibration = calibration
        self.prior_strength = float(prior_strength)
        self.decay = float(decay)
        self.online = bool(online)
        self.min_state_support = int(min_state_support)
        self.min_cost_repetitions = int(min_cost_repetitions)
        self.cutpoints = tuple(float(v) for v in calibration.get("entropy_cutpoints", ()))
        if len(self.cutpoints) != 3 or any(not math.isfinite(v) for v in self.cutpoints):
            raise ValueError("calibration must contain three finite entropy cutpoints")
        if tuple(sorted(self.cutpoints)) != self.cutpoints:
            raise ValueError("entropy cutpoints must be nondecreasing")
        self._online: dict[tuple[str, str], OnlinePrefixCount] = {}
        self._online_cost: dict[tuple[str, str], tuple[float, int]] = {}

    def bucket(self, state: RoundState) -> dict[str, str]:
        return state_bucket(state, self.cutpoints)

    def _prefix_row(self, action: Action, bucket: Mapping[str, str]) -> Mapping[str, Any] | None:
        rows = self.calibration.get("prefix_priors", {})
        akey = action_key(action)
        candidate_buckets = (
            ("entropy", "concentration", "history", "context", "refresh"),
            ("entropy", "history", "context", "refresh"),
            ("concentration", "history", "context", "refresh"),
            ("history", "context", "refresh"),
            ("entropy", "context", "refresh"),
            ("context", "refresh"),
        )
        for dimensions in candidate_buckets:
            key = bucket_key(bucket, include=dimensions)
            row = rows.get(f"{akey}::{key}")
            online = self._online.get((akey, bucket_key(bucket)))
            support = int(row.get("unique_state_count", 0)) if row else 0
            support += int(online.count) if online and self.online else 0
            if row and row.get("survival") and support >= self.min_state_support:
                return row
        row = rows.get(f"{akey}::global")
        online = self._online.get((akey, bucket_key(bucket)))
        support = int(row.get("unique_state_count", 0)) if row else 0
        support += int(online.count) if online and self.online else 0
        return row if row and support >= self.min_state_support else None

    def expected_commits(self, action: Action, state: RoundState) -> float | None:
        bucket = self.bucket(state)
        row = self._prefix_row(action, bucket)
        if row is None:
            return None
        survival = [float(v) for v in row.get("survival", ())]
        if len(survival) != action.gamma:
            return None
        if any(not 0.0 <= value <= 1.0 for value in survival):
            return None
        key = (action_key(action), bucket_key(bucket))
        online = self._online.get(key)
        if online and self.online:
            prior = survival
            total = online.count + self.prior_strength
            survival = [
                (success + self.prior_strength * prior[index]) / total
                for index, success in enumerate(online.successes)
            ]
        for index in range(1, len(survival)):
            survival[index] = min(survival[index], survival[index - 1])
        return 1.0 + sum(survival)

    def cost_ms(self, action: Action, state: RoundState, controller_id: str) -> float | None:
        bucket = self.bucket(state)
        table = self.calibration.get("cost_priors", {})
        akey = action_key(action)
        keys = (
            f"{akey}::context={bucket['context']}::refresh={bucket['refresh']}",
            f"{akey}::context={bucket['context']}::refresh=*",
            f"{akey}::context=*::refresh={bucket['refresh']}",
            f"{akey}::global",
        )
        row = next(
            (
                table[key]
                for key in keys
                if key in table
                and int(table[key].get("repetitions", 0)) >= self.min_cost_repetitions
            ),
            None,
        )
        if row is None:
            return None
        controller_rows = self.calibration.get("controller_cost_priors", {})
        controller = controller_rows.get(controller_id)
        if row.get("controller_id") == controller_id and row.get("total_round_ms") is not None:
            cost = float(row["total_round_ms"])
        else:
            if "action_cost_ms" in row:
                cost = float(row["action_cost_ms"])
            else:
                components = row.get("components_ms", {})
                cost = sum(float(value) for value in components.values())
            if controller is None:
                return None
            cost += float(controller.get("choose_ms", 0.0))
        if not math.isfinite(cost) or cost <= 0:
            return None
        update_key = (akey, bucket_key(bucket))
        online = self._online_cost.get(update_key)
        if online and self.online:
            cost = online[0] / online[1] + (float(controller.get("choose_ms", 0.0)) if controller else 0.0)
        return cost

    def observe(self, action: Action, state: RoundState, accepted_prefix: int, completed_cost_ms: float | None = None) -> None:
        if accepted_prefix < 0 or accepted_prefix > action.gamma:
            raise ValueError("accepted_prefix must be within the executed draft shape")
        bucket = self.bucket(state)
        key = (action_key(action), bucket_key(bucket))
        if self.online:
            observation = self._online.get(key)
            if observation is None:
                observation = OnlinePrefixCount(0.0, [0.0] * action.gamma)
                self._online[key] = observation
            if len(observation.successes) != action.gamma:
                raise ValueError("an action's online prefix shape changed")
            observation.count = self.decay * observation.count + 1.0
            observation.successes = [
                self.decay * success + float(accepted_prefix >= index + 1)
                for index, success in enumerate(observation.successes)
            ]
            if completed_cost_ms is not None and math.isfinite(completed_cost_ms) and completed_cost_ms > 0:
                cost, count = self._online_cost.get(key, (0.0, 0))
                alpha = float(self.calibration.get("latency_ema_alpha", 0.1))
                updated = completed_cost_ms if count == 0 else (1 - alpha) * cost + alpha * completed_cost_ms
                self._online_cost[key] = (updated, count + 1)


def fit_calibration(
    round_rows: Iterable[Mapping[str, Any]],
    *,
    signature: Mapping[str, Any],
    controller_cost_priors: Mapping[str, Any] | None = None,
    latency_ema_alpha: float = 0.1,
) -> dict[str, Any]:
    rows = [dict(row) for row in round_rows if row.get("status") == "ok"]
    entropy_by_state: dict[str, float] = {}
    for index, row in enumerate(rows):
        value = row.get("parent_entropy")
        if value is None or not math.isfinite(float(value)):
            continue
        identity = "|".join(str(row.get(key, "")) for key in (
            "source_group_id", "prompt_hash", "calibration_checkpoint", "logical_length", "round_index"
        )) or str(index)
        entropy_by_state.setdefault(identity, float(value))
    entropies = sorted(entropy_by_state.values())
    def quantile(q: float) -> float:
        if not entropies:
            return 0.0
        index = min(len(entropies) - 1, max(0, math.ceil(q * len(entropies)) - 1))
        return entropies[index]

    cutpoints = [quantile(0.25), quantile(0.5), quantile(0.75)]

    def enrich(row: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(row)
        entropy = row.get("parent_entropy")
        concentration = row.get("source_concentration")
        history = row.get("history_acceptance")
        enriched["state_entropy_bin"] = _entropy_bin(
            float(entropy) if entropy is not None else None, cutpoints
        )
        enriched["state_concentration_bin"] = (
            "unknown" if concentration is None else ("low" if float(concentration) < 0.5 else "high")
        )
        enriched["state_history_bin"] = (
            "unknown" if history is None else ("low" if float(history) < 0.5 else "high")
        )
        enriched["state_context_bucket"] = str(
            row.get("state_context_bucket") or math.ceil(max(int(row.get("logical_length", 1)), 1) / 2048)
        )
        enriched["state_refresh"] = "1" if bool(row.get("refresh_required", False)) else "0"
        return enriched

    rows = [enrich(row) for row in rows]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cost_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    action_state: dict[str, tuple[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("gamma_executed", 0) < 1:
            continue
        action = Action(row["requested_budget"], int(row["gamma_executed"]), str(row.get("length_mode", "draft_shape")))
        akey = action_key(action)
        bucket = {
            "entropy": str(row.get("state_entropy_bin", "unknown")),
            "concentration": str(row.get("state_concentration_bin", "unknown")),
            "history": str(row.get("state_history_bin", "unknown")),
            "context": str(row.get("state_context_bucket", "1")),
            "refresh": str(row.get("state_refresh", "0")),
        }
        state_hash = str(row.get("state_hash") or hashlib.sha256(json.dumps(bucket, sort_keys=True).encode()).hexdigest())
        unique_key = f"{akey}::{bucket_key(bucket)}"
        grouped[unique_key].append({**row, "state_hash": state_hash})
        enriched = {**row, "state_hash": state_hash}
        grouped[f"{akey}::global"].append(enriched)
        # Explicit marginal tables support ablations that remove a state
        # signal, instead of using the signal's bin under an unknown label.
        for dimensions in (
            ("entropy", "history", "context", "refresh"),
            ("concentration", "history", "context", "refresh"),
            ("history", "context", "refresh"),
            ("entropy", "context", "refresh"),
            ("context", "refresh"),
        ):
            grouped[f"{akey}::{bucket_key(bucket, include=dimensions)}"].append(enriched)
        action_state[unique_key] = (akey, bucket)
        context_key = f"{akey}::context={bucket['context']}::refresh={bucket['refresh']}"
        cost_rows[context_key].append(row)
        cost_rows[f"{akey}::global"].append(row)

    prefix_priors: dict[str, Any] = {}
    for key, observations in grouped.items():
        unique: dict[str, dict[str, Any]] = {}
        for row in observations:
            unique.setdefault(str(row["state_hash"]), row)
        max_gamma = max(int(row["gamma_executed"]) for row in unique.values())
        successes = [0.0] * max_gamma
        n = 0
        for row in unique.values():
            accepted = int(row["accepted_candidates"])
            gamma = int(row["gamma_executed"])
            n += 1
            for index in range(min(max_gamma, gamma)):
                successes[index] += float(accepted >= index + 1)
        survival = [success / n for success in successes] if n else []
        for index in range(1, len(survival)):
            survival[index] = min(survival[index], survival[index - 1])
        prefix_priors[key] = {"survival": survival, "unique_state_count": n, "backoff_origin": key}

    costs: dict[str, Any] = {}
    for key, observations in cost_rows.items():
        values = [
            float(row.get("action_cost_ms", row.get("round_total_ms")))
            for row in observations
            if row.get("action_cost_ms", row.get("round_total_ms")) is not None
            and math.isfinite(float(row.get("action_cost_ms", row.get("round_total_ms"))))
            and float(row.get("action_cost_ms", row.get("round_total_ms"))) > 0
        ]
        if values:
            mean_cost = sum(values) / len(values)
            variance = sum((value - mean_cost) ** 2 for value in values) / len(values)
            costs[key] = {
                "action_cost_ms": mean_cost,
                "std_ms": math.sqrt(variance),
                "repetitions": len(values),
                "components_ms": {"action_total": mean_cost},
                "timing_scope": str(observations[0].get("timing_mode", "diagnostic")),
            }
    return {
        "schema_version": "cadflash.calibration.v1",
        "signature": dict(signature),
        "entropy_cutpoints": cutpoints,
        "prefix_priors": prefix_priors,
        "cost_priors": costs,
        "controller_cost_priors": dict(controller_cost_priors or {}),
        "supported_actions": sorted({key.split("::", 1)[0] for key in costs}),
        "observations_hash": hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest(),
        "collection_scope": "production-compatible action replay",
        "latency_ema_alpha": latency_ema_alpha,
    }
