"""Pre-draft admission and post-draft cost/survival controllers."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math
from typing import Iterable, Mapping

import torch

from .config import BudgetProfile, DEFAULT_BUDGET_PROFILES


def context_bin(context_length: int) -> str:
    """Map request length to the finite scheduler bins used by the design."""
    value = int(context_length)
    return "long" if value >= 1024 else "medium" if value >= 256 else "short"


@dataclass
class RuntimeFeedback:
    """Request-local EMA state used by the v1.1 runtime feedback loop."""

    alpha: float = 0.2
    rounds: int = 0
    ar_rounds: int = 0
    acceptance_ema: float = 0.0
    accepted_length_ema: float = 0.0
    draft_latency_ema_ms: float = 0.0
    selector_latency_ema_ms: float = 0.0
    survival_latency_ema_ms: float = 0.0
    verify_latency_ema_ms: float = 0.0
    scheduler_latency_ema_ms: float = 0.0
    ar_latency_ema_ms: float = 0.0
    last_accepted_length: int = 0
    last_proposed_length: int = 0

    def __post_init__(self) -> None:
        self.alpha = float(self.alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("runtime feedback alpha must satisfy 0 < alpha <= 1")

    def _ema(self, previous: float, value: float, first: bool) -> float:
        return float(value) if first else (
            (1.0 - self.alpha) * float(previous) + self.alpha * float(value)
        )

    def update(
        self, accepted_length: int, proposed_length: int,
        timings_ms: Mapping[str, float] | None = None,
    ) -> None:
        accepted = int(accepted_length)
        proposed = int(proposed_length)
        if proposed <= 0:
            raise ValueError("runtime feedback proposed length must be positive")
        if accepted < 0 or accepted > proposed:
            raise ValueError("runtime feedback accepted length must be within proposal length")
        first = self.rounds == 0
        self.rounds += 1
        self.acceptance_ema = self._ema(
            self.acceptance_ema, accepted / proposed, first,
        )
        self.accepted_length_ema = self._ema(
            self.accepted_length_ema, accepted, first,
        )
        self.last_accepted_length = accepted
        self.last_proposed_length = proposed
        values = timings_ms or {}
        for key, attribute in (
            ("draft", "draft_latency_ema_ms"),
            ("selector", "selector_latency_ema_ms"),
            ("survival", "survival_latency_ema_ms"),
            ("verify", "verify_latency_ema_ms"),
            ("scheduler", "scheduler_latency_ema_ms"),
        ):
            if key not in values:
                continue
            value = max(0.0, float(values[key]))
            setattr(self, attribute, self._ema(getattr(self, attribute), value, first))

    def update_ar_latency(self, latency_ms: float) -> None:
        first = self.ar_rounds == 0
        self.ar_rounds += 1
        self.ar_latency_ema_ms = self._ema(
            self.ar_latency_ema_ms, max(0.0, float(latency_ms)), first,
        )

    def adjusted_gain(self, base_gain: float) -> float:
        """Conservatively down-weight the prior after observed poor acceptance."""
        base = max(0.0, float(base_gain))
        if self.rounds == 0:
            return base
        return base * max(0.0, min(1.0, float(self.acceptance_ema)))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "alpha": self.alpha,
            "rounds": self.rounds,
            "ar_rounds": self.ar_rounds,
            "acceptance_ema": self.acceptance_ema,
            "accepted_length_ema": self.accepted_length_ema,
            "draft_latency_ema_ms": self.draft_latency_ema_ms,
            "selector_latency_ema_ms": self.selector_latency_ema_ms,
            "survival_latency_ema_ms": self.survival_latency_ema_ms,
            "verify_latency_ema_ms": self.verify_latency_ema_ms,
            "scheduler_latency_ema_ms": self.scheduler_latency_ema_ms,
            "ar_latency_ema_ms": self.ar_latency_ema_ms,
            "last_accepted_length": self.last_accepted_length,
            "last_proposed_length": self.last_proposed_length,
        }


def fit_empirical_gate_table(traces: Iterable[Mapping]) -> dict[str, float]:
    """Fit the Stage-4 empirical pre-gate table from paired serving traces.

    Each trace supplies either ``realized_gain`` or both
    ``throughput_tok_s`` and ``ar_throughput_tok_s``. The output is directly
    consumable as ``SyncSpecConfig.gate_table``.
    """
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        context_length = trace.get("context_length", trace.get("input_tokens"))
        batch_size = trace.get("batch_size", 1)
        if context_length is None:
            continue
        try:
            batch = int(batch_size)
            if batch <= 0:
                continue
            if trace.get("realized_gain") is not None:
                gain = float(trace["realized_gain"])
            else:
                spec_rate = float(trace["throughput_tok_s"])
                ar_rate = float(trace["ar_throughput_tok_s"])
                if spec_rate < 0 or ar_rate <= 0:
                    continue
                gain = spec_rate / ar_rate - 1.0
            if not torch.isfinite(torch.tensor(gain)):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        key = f"{context_bin(int(context_length))}:batch{batch}"
        profile_kd = trace.get("kd")
        if profile_kd is None and isinstance(trace.get("budget"), Mapping):
            profile_kd = trace["budget"].get("kd")
        if profile_kd is None and isinstance(trace.get("budgets"), list):
            budget_kds: set[int] = set()
            for item in trace["budgets"]:
                if not isinstance(item, Mapping) or item.get("kd") is None:
                    continue
                try:
                    budget_kds.add(int(item["kd"]))
                except (TypeError, ValueError):
                    continue
            if len(budget_kds) == 1:
                profile_kd = budget_kds.pop()
        if profile_kd is not None:
            try:
                profile_kd = int(profile_kd)
            except (TypeError, ValueError):
                profile_kd = None
            if profile_kd is not None and profile_kd > 0:
                key = f"{key}:kd{profile_kd}"
        grouped[key].append(gain)
    if not grouped:
        raise ValueError("no valid paired serving traces for gate calibration")
    return {key: sum(values) / len(values) for key, values in sorted(grouped.items())}


@dataclass(frozen=True)
class GateChoice:
    profile: BudgetProfile
    reason: str

    @property
    def kd(self) -> int:
        return self.profile.kd


class PreDraftGate:
    def __init__(self, epsilon: float = 0.03, profiles=DEFAULT_BUDGET_PROFILES,
                 default_gain: float = 0.15, gain_table: Mapping[str, float] | None = None):
        self.epsilon = float(epsilon)
        self.profiles = tuple(profiles)
        self.default_gain = float(default_gain)
        self.gain_table = dict(gain_table or {})

    def _profile_key(self, context_length: int, batch_size: int, kd: int) -> str:
        return f"{context_bin(context_length)}:batch{int(batch_size)}:kd{int(kd)}"

    def _base_key(self, context_length: int, batch_size: int) -> str:
        return f"{context_bin(context_length)}:batch{int(batch_size)}"

    def _profile_gain(
        self,
        context_length: int,
        batch_size: int,
        kd: int,
        predicted_gain: float | Mapping[int | str, float] | None,
    ) -> float:
        if isinstance(predicted_gain, Mapping):
            value = predicted_gain.get(kd, predicted_gain.get(str(kd)))
            if value is not None:
                return float(value)
        elif predicted_gain is not None:
            return float(predicted_gain)
        specific = self.gain_table.get(self._profile_key(context_length, batch_size, kd))
        if specific is not None:
            return float(specific)
        return float(self.gain_table.get(self._base_key(context_length, batch_size), self.default_gain))

    def _estimated_gain(
        self,
        context_length: int,
        batch_size: int,
        predicted_gain: float | Mapping[int | str, float] | None,
        kd: int | None = None,
    ) -> float:
        if kd is not None:
            return self._profile_gain(context_length, batch_size, kd, predicted_gain)
        if isinstance(predicted_gain, Mapping):
            values = [float(value) for value in predicted_gain.values()]
            return sum(values) / len(values) if values else 0.0
        if predicted_gain is not None:
            return float(predicted_gain)
        return float(self.gain_table.get(self._base_key(context_length, batch_size), self.default_gain))

    def estimated_gain(
        self, context_length: int, batch_size: int,
        predicted_gain: float | Mapping[int | str, float] | None = None,
        kd: int | None = None,
    ) -> float:
        """Expose the calibrated prior for request-local feedback adjustment."""
        return self._estimated_gain(context_length, batch_size, predicted_gain, kd=kd)

    def choose(
        self,
        context_length: int,
        batch_size: int,
        predicted_gain: float | Mapping[int | str, float] | None = None,
        allowed_kds: Iterable[int] | None = None,
    ) -> GateChoice:
        ar = next(p for p in self.profiles if p.kd == 0)
        allowed = None if allowed_kds is None else {int(value) for value in allowed_kds}
        spec_profiles = [p for p in self.profiles if p.kd > 0]
        if not spec_profiles:
            return GateChoice(ar, "no_spec_profile")
        available_specific = {
            p.kd for p in spec_profiles
            if self._profile_key(context_length, batch_size, p.kd) in self.gain_table
        }
        if available_specific:
            # A partially calibrated table must not silently promote an
            # unmeasured K_d using the global default prior.
            spec_profiles = [p for p in spec_profiles if p.kd in available_specific]
        if allowed is not None:
            spec_profiles = [p for p in spec_profiles if p.kd in allowed]
        if not spec_profiles:
            return GateChoice(ar, "no_usable_spec_profile")
        best: tuple[float, BudgetProfile] | None = None
        for kd in sorted({p.kd for p in spec_profiles}):
            candidates = [p for p in spec_profiles if p.kd == kd]
            profile = max(candidates, key=lambda p: p.kv)
            gain = self._estimated_gain(context_length, batch_size, predicted_gain, kd=kd)
            if gain <= self.epsilon:
                continue
            if best is None or (gain, profile.kd, profile.kv) > (
                best[0], best[1].kd, best[1].kv,
            ):
                best = (gain, profile)
        if best is None:
            return GateChoice(ar, "below_safety_margin")
        return GateChoice(best[1], "predicted_utility_gain")


@dataclass(frozen=True)
class ControllerChoice:
    kd: int
    kv: int
    utility: float


class PostDraftController:
    def choose(
        self,
        kd: int,
        survival: torch.Tensor,
        costs: Mapping[int, float],
        profiles=DEFAULT_BUDGET_PROFILES,
        max_kv: int | None = None,
        ar_cost: float | None = None,
        ar_margin: float = 0.03,
    ) -> ControllerChoice:
        if float(ar_margin) < 0.0:
            raise ValueError("ar_margin must be non-negative")
        candidates = [
            p for p in profiles
            if p.kd == kd and p.kv > 0 and p.kv <= survival.numel()
            and (max_kv is None or p.kv <= int(max_kv))
        ]
        if not candidates:
            return ControllerChoice(kd=kd, kv=0, utility=1.0)
        best: tuple[float, BudgetProfile] | None = None
        for profile in candidates:
            cost = float(costs.get(profile.kv, float("inf")))
            if cost <= 0 or not torch.isfinite(torch.tensor(cost)):
                continue
            expected = 1.0 + float(survival[: profile.kv].sum().item())
            utility = expected / cost
            key = (utility, -profile.kv)
            if best is None or key > (best[0], -best[1].kv):
                best = (utility, profile)
        if best is None:
            return ControllerChoice(kd=kd, kv=0, utility=1.0)
        if ar_cost is not None:
            measured_ar_cost = float(ar_cost)
            if measured_ar_cost > 0.0 and math.isfinite(measured_ar_cost):
                ar_utility = 1.0 / measured_ar_cost
                if best[0] <= (1.0 + float(ar_margin)) * ar_utility:
                    return ControllerChoice(kd=kd, kv=0, utility=ar_utility)
        return ControllerChoice(kd=kd, kv=best[1].kv, utility=best[0])
