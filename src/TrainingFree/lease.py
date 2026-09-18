"""Framework-independent Source-State Lease certificate math.

The V2 certificate operates on log partition contributions collected at an
anchor query.  It never predicts future attention and does not mutate a model
cache; the model adapter decides when to refresh a lease.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    if maximum == float("-inf"):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _validate_indices(indices: Sequence[int], width: int, name: str) -> tuple[int, ...]:
    result = tuple(int(index) for index in indices)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate indices")
    if any(index < 0 or index >= width for index in result):
        raise ValueError(f"{name} contains an out-of-range index")
    return result


def select_hot_cold(
    anchor_mass: Sequence[float], *, delta_anchor: float
) -> tuple[list[int], list[int]]:
    """Select the smallest stable hot set retaining ``1 - delta_anchor`` mass."""

    values = [_finite(value, "anchor_mass") for value in anchor_mass]
    if not values or any(value < 0.0 for value in values) or sum(values) <= 0.0:
        raise ValueError("anchor_mass must be finite, non-negative and non-zero")
    delta = _finite(delta_anchor, "delta_anchor")
    if not 0.0 <= delta < 1.0:
        raise ValueError("delta_anchor must be in [0, 1)")

    total = sum(values)
    target = total * (1.0 - delta)
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    hot: list[int] = []
    retained = 0.0
    for index in order:
        hot.append(index)
        retained += values[index]
        if retained + 1e-12 >= target:
            break
    hot_set = set(hot)
    cold = [index for index in range(len(values)) if index not in hot_set]
    return hot, cold


def actual_mass(logits: Sequence[float], indices: Sequence[int]) -> float:
    """Return softmax mass assigned to ``indices`` using stable log-sum-exp."""

    values = [_finite(value, "logits") for value in logits]
    if not values:
        raise ValueError("logits must not be empty")
    selected = _validate_indices(indices, len(values), "indices")
    if not selected:
        return 0.0
    denominator = _logsumexp(values)
    numerator = _logsumexp([values[index] for index in selected])
    result = math.exp(numerator - denominator)
    return min(1.0, max(0.0, result))


def certificate_from_anchor(
    log_z0: Sequence[float],
    kappa: Sequence[float],
    hot: Sequence[int],
    cold: Sequence[int],
    live_log_z: float | None,
    drift: float,
) -> float:
    """Upper-bound cold source attention mass after query displacement.

    ``log_z0[i]`` is ``log(sum(exp(z_j(q_anchor))))`` for source block ``i``.
    ``live_log_z`` is the exact log contribution of non-source/live keys; it
    may be ``None`` when the adapter intentionally uses the conservative zero
    lower-bound for live keys.
    """

    partitions = [_finite(value, "log_z0") for value in log_z0]
    kappas = [_finite(value, "kappa") for value in kappa]
    if not partitions or len(partitions) != len(kappas):
        raise ValueError("log_z0 and kappa must be non-empty and aligned")
    if any(value < 0.0 for value in kappas):
        raise ValueError("kappa must be non-negative")
    distance = _finite(drift, "drift")
    if distance < 0.0:
        raise ValueError("drift must be non-negative")
    hot_indices = _validate_indices(hot, len(partitions), "hot")
    cold_indices = _validate_indices(cold, len(partitions), "cold")
    if set(hot_indices) & set(cold_indices):
        raise ValueError("hot and cold must be disjoint")

    cold_upper = _logsumexp([
        partitions[index] + distance * kappas[index]
        for index in cold_indices
    ])
    hot_lower = _logsumexp([
        partitions[index] - distance * kappas[index]
        for index in hot_indices
    ])
    if live_log_z is None or float(live_log_z) == float("-inf"):
        live = float("-inf")
    else:
        live = _finite(live_log_z, "live_log_z")
    if cold_upper == float("-inf"):
        return 0.0
    denominator = _logsumexp([cold_upper, hot_lower, live])
    result = math.exp(cold_upper - denominator)
    return min(1.0, max(0.0, result))


@dataclass
class LeaseState:
    """Track one event-driven lease without changing the underlying cache."""

    anchor_query: Sequence[float]
    log_z0: Sequence[float]
    kappa: Sequence[float]
    hot: Sequence[int]
    cold: Sequence[int]
    delta_cert: float
    anchor_step: int = 0
    observed_steps: int = 0

    def __post_init__(self) -> None:
        self.anchor_query = tuple(_finite(value, "anchor_query") for value in self.anchor_query)
        if not self.anchor_query:
            raise ValueError("anchor_query must not be empty")
        self.log_z0 = tuple(_finite(value, "log_z0") for value in self.log_z0)
        self.kappa = tuple(_finite(value, "kappa") for value in self.kappa)
        if len(self.log_z0) != len(self.kappa) or not self.log_z0:
            raise ValueError("log_z0 and kappa must be non-empty and aligned")
        self.hot = _validate_indices(self.hot, len(self.log_z0), "hot")
        self.cold = _validate_indices(self.cold, len(self.log_z0), "cold")
        if set(self.hot) & set(self.cold):
            raise ValueError("hot and cold must be disjoint")
        self.delta_cert = _finite(self.delta_cert, "delta_cert")
        if not 0.0 < self.delta_cert < 1.0:
            raise ValueError("delta_cert must be in (0, 1)")
        self.anchor_step = int(self.anchor_step)
        if self.anchor_step < 0:
            raise ValueError("anchor_step must be non-negative")

    def step(
        self,
        current_query: Sequence[float],
        live_log_z: float | None,
        *,
        actual_cold_mass: float,
        step_index: int,
    ) -> dict[str, float | int | bool]:
        """Evaluate one query against this lease and report expiry."""

        query = tuple(_finite(value, "current_query") for value in current_query)
        if len(query) != len(self.anchor_query):
            raise ValueError("current_query dimension must match anchor_query")
        actual = _finite(actual_cold_mass, "actual_cold_mass")
        if not 0.0 <= actual <= 1.0:
            raise ValueError("actual_cold_mass must be in [0, 1]")
        step = int(step_index)
        if step < self.anchor_step:
            raise ValueError("step_index must not precede anchor_step")
        drift = math.sqrt(sum(
            (current - anchor) ** 2
            for current, anchor in zip(query, self.anchor_query)
        ))
        bound = certificate_from_anchor(
            self.log_z0,
            self.kappa,
            self.hot,
            self.cold,
            live_log_z,
            drift,
        )
        valid = bound <= self.delta_cert
        self.observed_steps += 1
        return {
            "step": step,
            "anchor_step": self.anchor_step,
            "drift": drift,
            "bound": bound,
            "actual_cold_mass": actual,
            "slack": bound - actual,
            "valid": valid,
            "expired": not valid,
        }
