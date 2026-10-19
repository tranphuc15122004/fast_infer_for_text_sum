"""Safe action selection for adaptive speculative verification."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def resolve_verification_prefix(decision: Any, *, maximum_prefix: int) -> tuple[int, dict[str, Any]]:
    """Validate a policy response before it reaches the target verifier."""

    if isinstance(decision, Mapping):
        details = dict(decision)
        raw_prefix = details.get("prefix")
    else:
        details = {}
        raw_prefix = decision
    try:
        prefix = int(raw_prefix)
    except (TypeError, ValueError) as exc:
        raise ValueError("verification policy must select an integer prefix") from exc
    if prefix < 1 or prefix > int(maximum_prefix):
        raise ValueError("verification policy selected a prefix outside the drafted block")
    return prefix, details


def choose_verification_prefix(
    *,
    survival_probabilities: Sequence[float],
    cost_by_prefix: Mapping[int | str, float],
    allowed_prefixes: Sequence[int] = (3, 7, 11, 15),
    overhead_ms: float = 0.0,
    fallback_prefix: int = 15,
    opportunity_threshold_eta: float = 0.0,
) -> dict[str, Any]:
    """Choose k by predicted full-round milliseconds per committed token.

    ``survival_probabilities[i - 1]`` estimates P(L >= i) for the current
    gamma-15 proposal. Prefix costs must include the draft, verification,
    cache/bank update, and any measured extraction/controller overhead.
    Unsupported actions are excluded; no extrapolation is performed.
    """

    prefixes = tuple(int(value) for value in allowed_prefixes)
    if not prefixes or tuple(sorted(set(prefixes))) != prefixes or prefixes[0] < 1:
        raise ValueError("allowed_prefixes must be positive, unique, and sorted")
    if fallback_prefix not in prefixes:
        raise ValueError("fallback_prefix must belong to allowed_prefixes")
    if not math.isfinite(float(overhead_ms)) or overhead_ms < 0:
        raise ValueError("overhead_ms must be finite and nonnegative")
    if not math.isfinite(float(opportunity_threshold_eta)) or opportunity_threshold_eta < 0:
        raise ValueError("opportunity_threshold_eta must be finite and nonnegative")

    probabilities = [float(value) for value in survival_probabilities]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("survival probabilities must be finite values in [0, 1]")
    if any(right > left + 1e-12 for left, right in zip(probabilities, probabilities[1:])):
        raise ValueError("survival probabilities must be nonincreasing")

    costs: dict[int, float] = {}
    for key, raw_cost in cost_by_prefix.items():
        prefix = int(key)
        cost = float(raw_cost)
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("prefix costs must be finite positive milliseconds")
        costs[prefix] = cost

    # Cost-opportunity gate: if max potential savings delta <= eta, fallback to K.
    if opportunity_threshold_eta > 0.0 and fallback_prefix in costs:
        supported_costs = [
            costs[p] for p in prefixes if p in costs and p <= len(probabilities)
        ]
        if supported_costs:
            c_min = min(supported_costs)
            c_fallback = costs[fallback_prefix]
            delta = 1.0 - (c_min / c_fallback)
            if delta <= float(opportunity_threshold_eta):
                return {
                    "prefix": int(fallback_prefix),
                    "reason": "cost_opportunity_gate_below_threshold",
                    "predicted_cost_per_commit": None,
                    "expected_commits": None,
                    "unsupported_prefixes": [p for p in prefixes if p > len(probabilities) or p not in costs],
                    "overhead_ms": float(overhead_ms),
                    "opportunity_delta": float(delta),
                    "opportunity_threshold_eta": float(opportunity_threshold_eta),
                }

    scored: list[tuple[float, int, float]] = []
    unsupported: list[int] = []
    for prefix in prefixes:
        if prefix > len(probabilities) or prefix not in costs:
            unsupported.append(prefix)
            continue
        expected_commits = 1.0 + sum(probabilities[:prefix])
        adjusted_cost = costs[prefix] + float(overhead_ms)
        scored.append((adjusted_cost / expected_commits, prefix, expected_commits))

    if not scored:
        return {
            "prefix": int(fallback_prefix),
            "reason": "fallback_no_supported_action",
            "predicted_cost_per_commit": None,
            "expected_commits": None,
            "unsupported_prefixes": unsupported,
            "overhead_ms": float(overhead_ms),
            "opportunity_threshold_eta": float(opportunity_threshold_eta),
        }

    cost_per_commit, prefix, expected_commits = min(scored, key=lambda item: (item[0], item[1]))

    # Hysteresis threshold: only choose shorter k < K if J(k) < (1 - eta) * J(K).
    reason = "minimum_predicted_cost_per_commit"
    if opportunity_threshold_eta > 0.0 and prefix != fallback_prefix:
        fallback_scored = next((item for item in scored if item[1] == fallback_prefix), None)
        if fallback_scored is not None:
            j_fallback = fallback_scored[0]
            if cost_per_commit >= (1.0 - float(opportunity_threshold_eta)) * j_fallback:
                prefix = fallback_prefix
                cost_per_commit = j_fallback
                expected_commits = fallback_scored[2]
                reason = "hysteresis_insufficient_gain"

    return {
        "prefix": prefix,
        "reason": reason,
        "predicted_cost_per_commit": cost_per_commit,
        "expected_commits": expected_commits,
        "unsupported_prefixes": unsupported,
        "overhead_ms": float(overhead_ms),
        "opportunity_threshold_eta": float(opportunity_threshold_eta),
        "candidate_scores": {
            str(k): {
                "cost_per_commit_ms": score,
                "expected_commits": commits,
            }
            for score, k, commits in scored
        },
    }


def context_cost_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_prefixes: Sequence[int] = (3, 7, 11, 15),
    minimum_repetitions: int = 3,
) -> dict[str, dict[str, float | int]]:
    """Estimate deployable per-context costs from selection repetitions only."""

    if minimum_repetitions < 1:
        raise ValueError("minimum_repetitions must be positive")
    selected = [
        row for row in rows
        if row.get("repetition_split") == "selection"
        and row.get("status", "ok") == "ok"
        and row.get("acceptance_identity_valid", True) is True
    ]
    profiles: dict[str, dict[str, Any]] = {}
    # Dataset is included because target cost can vary materially by workload;
    # caller may merge only after confirming runtime-equivalent profiles.
    dataset_buckets: dict[tuple[str, int], dict[int, list[float]]] = {}
    for row in selected:
        prefix = int(row.get("k", 0))
        if prefix not in allowed_prefixes or row.get("total_round_ms") is None:
            continue
        context = int(row.get("context_tokens", row.get("prompt_tokens", 0)))
        bucket = min(max(context, 0) // 4096, 3)
        cost = float(row["total_round_ms"])
        if not math.isfinite(cost) or cost <= 0:
            continue
        dataset_buckets.setdefault((str(row.get("dataset", "unknown")), bucket), {}).setdefault(prefix, []).append(cost)
    for (dataset, bucket), actions in dataset_buckets.items():
        measured = {
            str(prefix): sum(values) / len(values)
            for prefix, values in sorted(actions.items())
            if len(values) >= minimum_repetitions
        }
        counts = {
            str(prefix): len(values)
            for prefix, values in sorted(actions.items())
            if len(values) >= minimum_repetitions
        }
        if measured:
            profiles[f"{dataset}::context={bucket}"] = {
                "cost_by_prefix": measured,
                "repetitions": counts,
            }
    return profiles


class VerificationPrefixController:
    """B3 policy driven by a frozen B2 survival model and measured costs."""

    def __init__(
        self,
        *,
        model: Mapping[str, Any],
        cost_profiles: Mapping[str, Mapping[str, Any]],
        dataset: str,
        allowed_prefixes: Sequence[int] = (3, 7, 11, 15),
        overhead_ms: float = 0.0,
        fallback_prefix: int = 15,
        opportunity_threshold_eta: float = 0.0,
    ) -> None:
        self.model = model
        self.cost_profiles = cost_profiles
        self.dataset = str(dataset)
        self.allowed_prefixes = tuple(int(value) for value in allowed_prefixes)
        self.overhead_ms = float(overhead_ms)
        self.fallback_prefix = int(fallback_prefix)
        if not math.isfinite(float(opportunity_threshold_eta)) or opportunity_threshold_eta < 0:
            raise ValueError("opportunity_threshold_eta must be finite and nonnegative")
        self.opportunity_threshold_eta = float(opportunity_threshold_eta)
        from TrainingFree.speculation_experiments.survival import SURVIVAL_VARIANTS

        variant_features = SURVIVAL_VARIANTS.get(str(model.get("variant", "")), ())
        self.requires_uncertainty = any(
            feature.startswith("draft_")
            for feature in variant_features
        )
        self.requires_entropy = any(
            feature.startswith("draft_") and "entropy" in feature
            for feature in variant_features
        )
        self.observed_policy_ms: list[float] = []

    def set_policy_overhead(self, overhead_ms: float) -> None:
        """Set the measured callback cost from dev-only warmup rollouts."""
        value = float(overhead_ms)
        if not math.isfinite(value) or value < 0:
            raise ValueError("policy overhead must be finite and nonnegative")
        self.overhead_ms = value

    def should_bypass_signals(self, logical_length: int) -> bool:
        """Check if cost profile for current context has savings delta <= eta."""
        if self.opportunity_threshold_eta <= 0.0:
            return False
        context_bucket = min(max(int(logical_length), 0) // 4096, 3)
        profile = self.cost_profiles.get(f"{self.dataset}::context={context_bucket}", {})
        measured_costs = profile.get("cost_by_prefix", {})
        if not measured_costs:
            return True
        k_key = self.fallback_prefix if self.fallback_prefix in measured_costs else str(self.fallback_prefix)
        if k_key not in measured_costs:
            return True
        k_cost = float(measured_costs[k_key])
        if k_cost <= 0:
            return True
        supported = [
            float(v) for k, v in measured_costs.items()
            if int(k) in self.allowed_prefixes and float(v) > 0
        ]
        if not supported:
            return True
        c_min = min(supported)
        delta = 1.0 - (c_min / k_cost)
        return delta <= self.opportunity_threshold_eta

    def __call__(
        self,
        *,
        state: Any,
        draft_uncertainty: Mapping[str, Sequence[float]],
        draft_gamma: int,
        available_prefixes: Sequence[int],
    ) -> dict[str, Any]:
        from TrainingFree.speculation_experiments.survival import predict_survival_curve

        logical_length = int(state.logical_length)
        context_bucket = min(max(logical_length, 0) // 4096, 3)
        available = tuple(
            value for value in self.allowed_prefixes
            if value in set(int(item) for item in available_prefixes) and value <= int(draft_gamma)
        )
        if not available:
            available = (min(int(draft_gamma), self.fallback_prefix),)
        clamped_fallback = self.fallback_prefix if self.fallback_prefix in available else available[-1]
        if self.should_bypass_signals(logical_length):
            return {
                "prefix": clamped_fallback,
                "reason": "cost_opportunity_gate_below_threshold",
                "predicted_cost_per_commit": None,
                "expected_commits": None,
                "unsupported_prefixes": [],
                "overhead_ms": float(self.overhead_ms),
                "opportunity_threshold_eta": float(self.opportunity_threshold_eta),
                "survival_fallback_levels": [],
                "survival_model_variant": self.model.get("variant"),
                "cost_context_key": f"{self.dataset}::context={context_bucket}",
            }

        processed = int(state.processed_output_tokens)
        row = {
            "accepted_candidates": 0,  # Decision occurs before verifier outcomes are known.
            "prompt_tokens": max(0, logical_length - processed),
            "context_tokens": logical_length,
            "processed_output_tokens": processed,
            "parent_entropy": state.parent_entropy,
            "parent_signal": {
                "normalized_entropy": state.parent_entropy,
                "confidence": getattr(state, "parent_confidence", None),
            },
            "history_ema": state.history_acceptance,
            "history_missing": state.history_acceptance is None,
            "draft_uncertainty": {
                "normalized_entropy": list(draft_uncertainty.get("normalized_entropy") or ()),
                "confidence": list(draft_uncertainty.get("confidence") or ()),
                "probability_margin": list(draft_uncertainty.get("probability_margin") or ()),
                "logit_margin": list(draft_uncertainty.get("logit_margin") or ()),
            },
        }
        survival, fallback = predict_survival_curve(self.model, row)
        profile = self.cost_profiles.get(f"{self.dataset}::context={context_bucket}", {})
        measured_costs = profile.get("cost_by_prefix", {})
        available = tuple(
            value for value in self.allowed_prefixes
            if value in set(int(item) for item in available_prefixes) and value <= int(draft_gamma)
        )
        if not available:
            available = (min(int(draft_gamma), self.fallback_prefix),)
        extraction_ms = float(draft_uncertainty.get("signal_extraction_ms", 0.0))
        if not math.isfinite(extraction_ms) or extraction_ms < 0:
            raise ValueError("signal extraction overhead must be finite and nonnegative")
        decision = choose_verification_prefix(
            survival_probabilities=survival,
            cost_by_prefix=measured_costs,
            allowed_prefixes=available,
            overhead_ms=self.overhead_ms + extraction_ms,
            fallback_prefix=(self.fallback_prefix if self.fallback_prefix in available else available[-1]),
            opportunity_threshold_eta=self.opportunity_threshold_eta,
        )
        return {
            **decision,
            "survival_fallback_levels": fallback,
            "survival_model_variant": self.model.get("variant"),
            "cost_context_key": f"{self.dataset}::context={context_bucket}",
        }


class FixedVerificationPrefixController:
    """Causal fixed-k control that adapts only at an infeasible tail shape."""

    requires_uncertainty = False

    def __init__(self, prefix: int) -> None:
        if isinstance(prefix, bool) or int(prefix) not in (3, 7, 11, 15):
            raise ValueError("fixed verification prefix must be one of 3, 7, 11, 15")
        self.prefix = int(prefix)

    def __call__(
        self,
        *,
        state: Any,
        draft_uncertainty: Mapping[str, Sequence[float]],
        draft_gamma: int,
        available_prefixes: Sequence[int],
    ) -> dict[str, Any]:
        supported = tuple(int(value) for value in available_prefixes)
        choices = [value for value in supported if value <= self.prefix]
        selected = max(choices) if choices else max(supported)
        return {
            "prefix": selected,
            "reason": "fixed_prefix" if selected == self.prefix else "fixed_prefix_tail_fallback",
            "requested_prefix": self.prefix,
            "available_prefixes": list(supported),
        }
