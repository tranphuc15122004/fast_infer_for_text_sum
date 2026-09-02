"""Dependency-light metrics used by the GroundSync experiments.

The functions in this module operate on ordinary Python sequences and mappings.
They deliberately do not know about a model, GPU, or a particular trace file so
that the hypothesis tests can be exercised with deterministic CPU fixtures.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def finite_metric(value: float) -> float:
    """Return ``value`` as a float, rejecting NaN and infinities."""

    result = float(value)
    if not math.isfinite(result):
        raise ValueError("metric must be finite")
    return result


def stable_sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid."""

    value = float(value)
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _validated_nonnegative(values: Sequence[float], *, name: str) -> list[float]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    if any(value < 0.0 for value in result):
        raise ValueError(f"{name} must contain non-negative values")
    if sum(result) <= 0.0:
        raise ValueError(f"{name} must have a positive sum")
    return result


def normalize_distribution(values: Sequence[float]) -> list[float]:
    """Normalize a non-negative finite sequence to a probability distribution."""

    result = _validated_nonnegative(values, name="distribution")
    total = sum(result)
    return [value / total for value in result]


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Compute the base-2 Jensen–Shannon divergence in ``[0, 1]``."""

    if len(p) != len(q):
        raise ValueError("distributions must have the same length")
    p_norm = normalize_distribution(p)
    q_norm = normalize_distribution(q)
    midpoint = [(left + right) / 2.0 for left, right in zip(p_norm, q_norm)]

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        total = 0.0
        for left_value, right_value in zip(left, right):
            if left_value > 0.0:
                total += left_value * math.log2(left_value / right_value)
        return total

    result = 0.5 * (kl(p_norm, midpoint) + kl(q_norm, midpoint))
    return min(max(result, 0.0), 1.0)


def aggregate_source_mass(
    mass: Sequence[float],
    *,
    chunk_size: int,
    skip_tokens: int = 0,
    positional_prior: Sequence[float] | None = None,
) -> list[float]:
    """Aggregate source-token mass into normalized chunks.

    ``skip_tokens`` models a source-side sink control.  A positional prior is
    applied before aggregation and must describe the retained token positions.
    Zero adjusted mass is represented by a uniform distribution so downstream
    divergence metrics remain defined while the caller can still inspect the
    original trace metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if skip_tokens < 0 or skip_tokens >= len(mass):
        raise ValueError("skip_tokens must leave at least one source token")
    retained = _validated_nonnegative(mass[skip_tokens:], name="source mass")
    if positional_prior is not None:
        prior = [float(value) for value in positional_prior]
        if len(prior) != len(retained):
            raise ValueError("positional_prior must match retained mass length")
        if any(not math.isfinite(value) or value <= 0.0 for value in prior):
            raise ValueError("positional_prior must contain positive finite values")
        retained = [value / baseline for value, baseline in zip(retained, prior)]

    chunks = [
        sum(retained[start : start + chunk_size])
        for start in range(0, len(retained), chunk_size)
    ]
    if sum(chunks) <= 0.0:
        return [1.0 / len(chunks)] * len(chunks)
    total = sum(chunks)
    return [value / total for value in chunks]


def _validate_trace(trace: Sequence[Sequence[float]]) -> None:
    if not trace:
        raise ValueError("trace must not be empty")
    width = len(trace[0])
    if width == 0:
        raise ValueError("trace distributions must not be empty")
    for distribution in trace:
        if len(distribution) != width:
            raise ValueError("all trace distributions must have the same length")
        normalize_distribution(distribution)


def lag_similarity(
    trace: Sequence[Sequence[float]],
    lags: Iterable[int],
) -> dict[str, dict[str, float | int]]:
    """Summarize mean ``1 - JS`` similarity at each non-negative lag."""

    _validate_trace(trace)
    result: dict[str, dict[str, float | int]] = {}
    for lag in lags:
        lag = int(lag)
        if lag <= 0:
            raise ValueError("lags must be positive")
        values = [
            1.0 - js_divergence(trace[index], trace[index + lag])
            for index in range(len(trace) - lag)
        ]
        if not values:
            result[str(lag)] = {"count": 0, "mean": 0.0, "stdev": 0.0}
            continue
        result[str(lag)] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return result


def segment_lengths(
    trace: Sequence[Sequence[float]],
    *,
    threshold: float,
) -> list[int]:
    """Return segment lengths split where adjacent JS drift exceeds threshold."""

    _validate_trace(trace)
    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("threshold must be a finite non-negative value")
    lengths: list[int] = []
    current = 1
    for previous, current_distribution in zip(trace, trace[1:]):
        if js_divergence(previous, current_distribution) > threshold:
            lengths.append(current)
            current = 1
        else:
            current += 1
    lengths.append(current)
    return lengths


def grounding_horizon(
    trace: Sequence[Sequence[float]],
    *,
    start: int,
    threshold: float,
    max_horizon: int | None = None,
) -> int | None:
    """Return the first future offset whose drift from ``start`` exceeds threshold."""

    _validate_trace(trace)
    if not 0 <= start < len(trace):
        raise IndexError("start is outside trace")
    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("threshold must be a finite non-negative value")
    end = len(trace) - start - 1
    if max_horizon is not None:
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive")
        end = min(end, max_horizon)
    for offset in range(1, end + 1):
        if js_divergence(trace[start], trace[start + offset]) > threshold:
            return offset
    return None


def persistence_summary(
    trace: Sequence[Sequence[float]],
    *,
    threshold: float,
    lags: Iterable[int] = (1, 2, 4, 8, 16, 32),
) -> dict[str, Any]:
    """Return the document-level summary used for the H1 report."""

    lengths = segment_lengths(trace, threshold=threshold)
    lag_values = lag_similarity(trace, lags)
    return {
        "segment_count": len(lengths),
        "segment_lengths": lengths,
        "mean_segment_length": statistics.fmean(lengths),
        "median_segment_length": statistics.median(lengths),
        "lag_similarity": lag_values,
    }


def accepted_prefix_length(
    proposed: Sequence[Any],
    canonical: Sequence[Any],
) -> int:
    """Return the greedy longest matching prefix length."""

    accepted = 0
    for draft_token, target_token in zip(proposed, canonical):
        if draft_token != target_token:
            break
        accepted += 1
    return accepted


def document_level_mean(
    rows: Iterable[Mapping[str, Any]],
    value_key: str,
    *,
    document_key: str = "document_id",
) -> float:
    """Average per-document means, preventing long traces dominating the result."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if document_key not in row or value_key not in row:
            raise ValueError(f"row must contain {document_key!r} and {value_key!r}")
        grouped[str(row[document_key])].append(finite_metric(float(row[value_key])))
    if not grouped:
        raise ValueError("rows must not be empty")
    return statistics.fmean(statistics.fmean(values) for values in grouped.values())


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 42,
    samples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Return a deterministic percentile bootstrap CI for a mean.

    The caller is responsible for passing one observation per independent
    cluster (GroundSync reports use one observation per document), rather than
    treating every generated token as an independent sample.
    """

    if not values:
        raise ValueError("values must not be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    clean = [finite_metric(value) for value in values]
    rng = random.Random(seed)
    means = [
        statistics.fmean(clean[rng.randrange(len(clean))] for _ in clean)
        for _ in range(samples)
    ]
    means.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(means) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return means[lower]
        weight = position - lower
        return means[lower] * (1.0 - weight) + means[upper] * weight

    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": statistics.fmean(clean),
        "low": percentile(alpha),
        "high": percentile(1.0 - alpha),
        "count": len(clean),
        "bootstrap_samples": samples,
        "confidence": confidence,
    }


def policy_k(
    horizon: int | None,
    *,
    max_k: int,
    fallback: int = 1,
) -> int:
    """Clip an oracle/predicted horizon to a valid speculative block length."""

    if max_k <= 0:
        raise ValueError("max_k must be positive")
    candidate = fallback if horizon is None else horizon
    if candidate <= 0:
        candidate = 1
    return min(int(candidate), max_k)


__all__ = [
    "accepted_prefix_length",
    "aggregate_source_mass",
    "bootstrap_mean_ci",
    "document_level_mean",
    "finite_metric",
    "grounding_horizon",
    "js_divergence",
    "lag_similarity",
    "normalize_distribution",
    "persistence_summary",
    "policy_k",
    "segment_lengths",
    "stable_sigmoid",
]
