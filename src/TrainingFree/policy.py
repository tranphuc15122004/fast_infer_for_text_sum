"""Framework-independent RECAP-KV residual-evidence policy.

The functions in this module operate on ordinary Python sequences so that the
policy can be tested without loading a model.  Model adapters convert tensors
to this small interface at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


NumberVector = Sequence[float]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _vectors(values: Sequence[NumberVector], name: str) -> list[list[float]]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    result = [[_finite(item, name) for item in vector] for vector in values]
    width = len(result[0])
    if width <= 0 or any(len(vector) != width for vector in result):
        raise ValueError(f"{name} must contain equal-width non-empty vectors")
    return result


def _dot(left: NumberVector, right: NumberVector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal dimensions")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _cosine(left: NumberVector, right: NumberVector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal dimensions")
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("logits must not be empty")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("softmax denominator is invalid")
    return [value / denominator for value in exponentials]


def compute_current_relevance(
    query_vectors: Sequence[NumberVector],
    source_prototypes: Sequence[Sequence[NumberVector]],
    *,
    temperature: float,
) -> list[float]:
    """Compute ``pi`` from the best query/prototype cosine per source block."""

    queries = _vectors(query_vectors, "query_vectors")
    if not source_prototypes:
        raise ValueError("source_prototypes must not be empty")
    temperature = _finite(temperature, "temperature")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    dimensions = len(queries[0])
    logits: list[float] = []
    for block_index, prototypes in enumerate(source_prototypes):
        block = _vectors(prototypes, f"source_prototypes[{block_index}]")
        if len(block[0]) != dimensions:
            raise ValueError("query and prototype dimensions must match")
        best = max(_cosine(query, prototype) for query in queries for prototype in block)
        logits.append(best / temperature)
    return _softmax(logits)


def compute_consumption(
    relevance: Sequence[float],
    segment_vector: NumberVector,
    source_vectors: Sequence[NumberVector],
) -> list[float]:
    """Compute output-conditioned evidence consumption for each source block."""

    segment = [_finite(value, "segment_vector") for value in segment_vector]
    if not segment:
        raise ValueError("segment_vector must not be empty")
    sources = _vectors(source_vectors, "source_vectors")
    if len(relevance) != len(sources):
        raise ValueError("relevance and source_vectors must have equal length")
    if any(_finite(value, "relevance") < 0.0 for value in relevance):
        raise ValueError("relevance must be non-negative")
    if len(sources[0]) != len(segment):
        raise ValueError("segment and source vector dimensions must match")
    return [
        float(relevance[index]) * max(0.0, _cosine(segment, source))
        for index, source in enumerate(sources)
    ]


def update_residual(
    residual: Sequence[float],
    consumption: Sequence[float],
    *,
    eta: float,
    floor: float,
) -> list[float]:
    """Apply monotone exponential demotion while retaining a recovery floor."""

    if len(residual) != len(consumption) or not residual:
        raise ValueError("residual and consumption must be non-empty and aligned")
    eta = _finite(eta, "eta")
    floor = _finite(floor, "floor")
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    if not 0.0 < floor <= 1.0:
        raise ValueError("floor must be in (0, 1]")
    result: list[float] = []
    for old, used in zip(residual, consumption):
        old_value = _finite(old, "residual")
        used_value = _finite(used, "consumption")
        if not 0.0 <= old_value <= 1.0 or used_value < 0.0:
            raise ValueError("residual must be in [0, 1] and consumption non-negative")
        result.append(max(floor, old_value * math.exp(-eta * used_value)))
    return result


def residual_utility(
    relevance: Sequence[float], residual: Sequence[float], *, beta: float
) -> list[float]:
    """Combine current demand and remaining evidence utility."""

    if len(relevance) != len(residual) or not relevance:
        raise ValueError("relevance and residual must be non-empty and aligned")
    beta = _finite(beta, "beta")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    result: list[float] = []
    for demand, remaining in zip(relevance, residual):
        demand_value = _finite(demand, "relevance")
        remaining_value = _finite(remaining, "residual")
        if demand_value < 0.0 or not 0.0 <= remaining_value <= 1.0:
            raise ValueError("relevance must be non-negative and residual in [0, 1]")
        result.append(demand_value * (1.0 - beta * (1.0 - remaining_value)))
    return result


def adaptive_budget(
    probabilities: Sequence[float], k_min: int, k_max: int
) -> int:
    """Map demand entropy to an integer budget bounded by available blocks."""

    values = [_finite(value, "probabilities") for value in probabilities]
    if not values or any(value < 0.0 for value in values) or sum(values) <= 0.0:
        raise ValueError("probabilities must be finite, non-negative and non-zero")
    if k_min <= 0 or k_max < k_min:
        raise ValueError("k_min and k_max must be positive and ordered")
    count = len(values)
    lower = min(int(k_min), count)
    upper = min(int(k_max), count)
    total = sum(values)
    normalized = [value / total for value in values]
    entropy = -sum(value * math.log(value) for value in normalized if value > 0.0)
    max_entropy = math.log(count) if count > 1 else 1.0
    fraction = min(1.0, max(0.0, entropy / max_entropy))
    return max(lower, min(upper, int(round(lower + (upper - lower) * fraction))))


def select_topk(scores: Sequence[float], k: int) -> list[int]:
    """Return stable descending-score indices, breaking ties by source order."""

    values = [_finite(value, "scores") for value in scores]
    if not values or k <= 0:
        raise ValueError("scores must be non-empty and k positive")
    return [
        index
        for index, _ in sorted(
            enumerate(values), key=lambda item: (-item[1], item[0])
        )[: min(int(k), len(values))]
    ]


@dataclass(frozen=True)
class RecapConfig:
    """Registered V0 hyperparameters."""

    temperature: float = 0.1
    beta: float = 0.7
    eta: float = 2.0
    residual_floor: float = 0.1
    k_min: int = 1
    k_max: int = 8

    def __post_init__(self) -> None:
        compute_current_relevance([[1.0]], [[[1.0]]], temperature=self.temperature)
        residual_utility([1.0], [1.0], beta=self.beta)
        update_residual([1.0], [0.0], eta=self.eta, floor=self.residual_floor)
        if self.k_min <= 0 or self.k_max < self.k_min:
            raise ValueError("k_min and k_max must be positive and ordered")


class RecapState:
    """Online residual state with a pure-serializable ``step`` method."""

    def __init__(self, config: RecapConfig, residual: Sequence[float]) -> None:
        if not residual:
            raise ValueError("residual must not be empty")
        self.config = config
        self.residual = update_residual(
            residual,
            [0.0] * len(residual),
            eta=config.eta,
            floor=config.residual_floor,
        )

    def step(
        self,
        query_vectors: Sequence[NumberVector],
        segment_vector: NumberVector,
        source_vectors: Sequence[NumberVector],
        source_prototypes: Sequence[Sequence[NumberVector]],
    ) -> dict[str, object]:
        relevance = compute_current_relevance(
            query_vectors,
            source_prototypes,
            temperature=self.config.temperature,
        )
        return self.step_with_relevance(relevance, segment_vector, source_vectors)

    def step_with_relevance(
        self,
        relevance: Sequence[float],
        segment_vector: NumberVector,
        source_vectors: Sequence[NumberVector],
    ) -> dict[str, object]:
        """Advance state from an observed relevance distribution.

        This is used by the V0 ablation that isolates residual dynamics from
        the semantic hidden-state indexer.
        """

        if len(relevance) != len(self.residual):
            raise ValueError("relevance width must match residual state")
        utility = residual_utility(relevance, self.residual, beta=self.config.beta)
        budget = adaptive_budget(relevance, self.config.k_min, self.config.k_max)
        active = select_topk(utility, budget)
        consumption = compute_consumption(relevance, segment_vector, source_vectors)
        previous = list(self.residual)
        self.residual = update_residual(
            self.residual,
            consumption,
            eta=self.config.eta,
            floor=self.config.residual_floor,
        )
        return {
            "relevance": relevance,
            "utility": utility,
            "budget": budget,
            "active": active,
            "consumption": consumption,
            "residual_before": previous,
            "residual_after": list(self.residual),
        }
