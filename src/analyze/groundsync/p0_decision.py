"""Decision experiments for corrected GroundSync/BurstSpec hypotheses.

This module is deliberately separate from the legacy H1--H5 report.  It makes
the three decision questions explicit:

* H2 uses the drift *crossed inside* a speculative block, not drift at block
  start;
* a missing transition through ``Kmax`` means horizon ``Kmax``;
* the opportunity for BurstSpec is measured with ``k=0`` (AR admission) and
  ``k=16`` in addition to the old fixed blocks.

All functions operate on JSON-safe trace mappings.  Model loading remains in
``trace_target.py``/``trace_speculative.py``; this file is safe to exercise on
CPU with deterministic fixtures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core import js_divergence, normalize_distribution


def _ok_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("values must not be empty")
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _trace(row: Mapping[str, Any], variant: str = "nosink") -> list[list[float]]:
    result: list[list[float]] = []
    for step in row.get("attention", []):
        if not isinstance(step, Mapping):
            continue
        values = step.get(variant)
        if values is not None:
            result.append(normalize_distribution([float(value) for value in values]))
    return result


def corrected_grounding_horizon(
    trace: Sequence[Sequence[float]],
    *,
    start: int,
    threshold: float,
    max_horizon: int,
) -> int:
    """Return first within-block transition, or ``max_horizon`` if absent.

    The caller must provide ``max_horizon`` future transitions.  A short trace
    is rejected rather than silently treating missing future evidence as a
    long horizon.
    """

    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive")
    if not 0 <= int(start) < len(trace):
        raise IndexError("start is outside trace")
    if threshold < 0.0 or not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite and non-negative")
    if start + max_horizon >= len(trace):
        raise ValueError("trace does not cover max_horizon future transitions")
    for relative_position in range(1, max_horizon + 1):
        drift = js_divergence(
            trace[start + relative_position - 1],
            trace[start + relative_position],
        )
        if drift > threshold:
            return relative_position
    return max_horizon


def _first_reject(row: Mapping[str, Any]) -> int | None:
    value = row.get("first_reject_rel")
    if value is not None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None
    max_k = int(row.get("max_k", 0) or 0)
    accepted = int(row.get("accepted_len", 0) or 0)
    return accepted + 1 if max_k > 0 and accepted < max_k else None


def transition_hazard_rows(
    target_rows: Sequence[Mapping[str, Any]],
    speculative_rows: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
    variant: str = "nosink",
) -> list[dict[str, Any]]:
    """Expand proposals into a discrete hazard risk set.

    For proposal start ``t`` and relative position ``j``, the main predictor is
    ``JS(g[t+j-1], g[t+j])``.  Rows after the first rejection are excluded from
    the risk set.  Entropy is taken at ``t+j`` and draft confidence at ``j``.
    """

    if max_k <= 0:
        raise ValueError("max_k must be positive")
    targets = {
        str(row.get("document_id")): row
        for row in _ok_rows(target_rows)
    }
    result: list[dict[str, Any]] = []
    for proposal_index, spec in enumerate(_ok_rows(speculative_rows)):
        document_id = str(spec.get("document_id"))
        target = targets.get(document_id)
        if target is None:
            continue
        trace = _trace(target, variant)
        try:
            start = int(spec.get("start_position", 0))
            row_max_k = min(max_k, int(spec.get("max_k", max_k)))
        except (TypeError, ValueError):
            continue
        if start < 0 or row_max_k <= 0:
            continue
        first_reject = _first_reject(spec)
        if first_reject is not None and first_reject > row_max_k:
            first_reject = None
        confidence = spec.get("draft_confidence") or []
        entropies = target.get("target_entropy") or []
        available = min(row_max_k, len(confidence), len(trace) - start - 1)
        if available <= 0 or start >= len(trace):
            continue
        for relative_position in range(1, available + 1):
            if first_reject is not None and relative_position > first_reject:
                break
            entropy = _finite(
                entropies[start + relative_position]
                if start + relative_position < len(entropies) else None
            )
            draft_confidence = _finite(confidence[relative_position - 1])
            if entropy is None or draft_confidence is None:
                continue
            left = trace[start + relative_position - 1]
            right = trace[start + relative_position]
            result.append({
                "document_id": document_id,
                "proposal_index": proposal_index,
                "start_position": start,
                "relative_position": relative_position,
                "output_position": start + relative_position,
                "d_transition": js_divergence(left, right),
                "target_entropy": entropy,
                "draft_confidence": draft_confidence,
                "event": int(first_reject == relative_position),
            })
    return result


_HAZARD_FEATURES = (
    "d_transition",
    "target_entropy",
    "draft_confidence",
    "relative_position",
    "output_position",
)


def _hazard_feature_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[Any, Any] | None:
    usable: list[Mapping[str, Any]] = []
    for row in rows:
        if all(_finite(row.get(name)) is not None for name in _HAZARD_FEATURES):
            if row.get("event") in (0, 1, False, True):
                usable.append(row)
    if not usable or len({int(bool(row["event"])) for row in usable}) < 2:
        return None
    import numpy as np

    matrix = np.asarray(
        [[float(row[name]) for name in _HAZARD_FEATURES] for row in usable],
        dtype=float,
    )
    labels = np.asarray([int(bool(row["event"])) for row in usable], dtype=float)
    return matrix, labels


def _fit_transition_coefficient(
    rows: Sequence[Mapping[str, Any]],
    *,
    standardization: tuple[Any, Any] | None = None,
) -> float | None:
    """Fit ridge-logistic hazard and return standardized drift coefficient."""

    prepared = _hazard_feature_matrix(rows)
    if prepared is None:
        return None
    import numpy as np

    matrix, labels = prepared
    if standardization is None:
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
    else:
        means, scales = standardization
    scales = np.asarray(scales, dtype=float).copy()
    scales[scales == 0.0] = 1.0
    features = (matrix - np.asarray(means, dtype=float)) / scales
    design = np.column_stack((np.ones(len(features)), features))
    weights = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag([0.0] + [1.0] * len(_HAZARD_FEATURES))
    for _ in range(80):
        logits = np.clip(design @ weights, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-8)
        hessian = design.T @ (curvature[:, None] * design) + penalty
        gradient = design.T @ (labels - probabilities) - penalty @ weights
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        weights += step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    value = float(weights[1])
    return value if math.isfinite(value) else None


def fit_transition_hazard(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit corrected H2 and bootstrap whole documents, not token rows."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    prepared = _hazard_feature_matrix(rows)
    if prepared is None:
        return {
            "status": "UNAVAILABLE",
            "decision": "UNAVAILABLE",
            "risk_set_rows": 0,
            "features": list(_HAZARD_FEATURES),
        }
    import numpy as np

    matrix, labels = prepared
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    usable = [
        row for row in rows
        if all(_finite(row.get(name)) is not None for name in _HAZARD_FEATURES)
        and row.get("event") in (0, 1, False, True)
    ]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        groups[str(row.get("document_id"))].append(row)
    document_count = len(groups)
    event_count = int(labels.sum())
    point = _fit_transition_coefficient(usable, standardization=(means, scales))
    base = {
        "features": list(_HAZARD_FEATURES),
        "standardization": "z-score over full risk set; drift coefficient per SD",
        "risk_set_rows": len(usable),
        "event_rows": event_count,
        "document_count": document_count,
        "bootstrap_unit": "document",
        "drift_coefficient": point,
    }
    if point is None or document_count < 5:
        return {
            **base,
            "status": "UNAVAILABLE",
            "decision": "UNAVAILABLE",
            "reason": "need both event classes and at least five documents",
            "drift_coefficient_ci": None,
        }
    rng = random.Random(seed)
    document_groups = list(groups.values())
    coefficients: list[float] = []
    for _ in range(bootstrap_samples):
        resampled = [
            row
            for _ in document_groups
            for row in rng.choice(document_groups)
        ]
        coefficient = _fit_transition_coefficient(
            resampled, standardization=(means, scales)
        )
        if coefficient is not None and math.isfinite(coefficient):
            coefficients.append(coefficient)
    ci = None
    if coefficients:
        ci = {
            "low": _percentile(coefficients, 0.025),
            "high": _percentile(coefficients, 0.975),
            "bootstrap_valid": len(coefficients),
            "bootstrap_requested": bootstrap_samples,
            "confidence": 0.95,
        }
    decision_value = ci["low"] if ci else point
    return {
        **base,
        "status": "ok" if ci else "UNAVAILABLE",
        "drift_coefficient_ci": ci,
        "drift_odds_ratio_per_sd": math.exp(point),
        "decision_gate": "lower 95% document-bootstrap CI > 0",
        "decision": "PASS" if ci and decision_value > 0.0 else "FAIL" if ci else "UNAVAILABLE",
    }


def summarize_within_block_burstiness(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
) -> dict[str, Any]:
    """Report ``h_j=P(R=j | R>=j)`` and the entry/continuation asymmetry."""

    usable = _ok_rows(rows)
    by_position: dict[str, dict[str, Any]] = {}
    for relative_position in range(1, max_k + 1):
        at_risk = 0
        events = 0
        for row in usable:
            proposal = row.get("proposal_token_ids")
            confidence = row.get("draft_confidence")
            observed_k = int(row.get("max_k", max_k) or 0)
            if isinstance(proposal, Sequence):
                observed_k = min(observed_k, len(proposal))
            if isinstance(confidence, Sequence):
                observed_k = min(observed_k, len(confidence))
            if observed_k < relative_position:
                continue
            first_reject = _first_reject(row)
            if first_reject is not None and first_reject < relative_position:
                continue
            at_risk += 1
            if first_reject == relative_position:
                events += 1
        by_position[str(relative_position)] = {
            "at_risk": at_risk,
            "events": events,
            "hazard": events / at_risk if at_risk else None,
        }
    h1 = by_position.get("1", {}).get("hazard")
    later = [
        value["hazard"] for key, value in by_position.items()
        if key != "1" and value["hazard"] is not None
    ]
    later_mean = statistics.fmean(later) if later else None
    ratio = h1 / later_mean if h1 is not None and later_mean and later_mean > 0 else None
    return {
        "status": "ok" if usable else "UNAVAILABLE",
        "count": len(usable),
        "by_relative_position": by_position,
        "h1": h1,
        "later_hazard_mean": later_mean,
        "h1_to_later_hazard_ratio": ratio,
        "decision": (
            "UNAVAILABLE" if ratio is None else
            "PASS" if ratio > 1.0 else "FAIL"
        ),
    }


def across_round_persistence(
    rows: Sequence[Mapping[str, Any]],
    *,
    deltas: Sequence[int] = (1, 2, 4, 8),
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure persistence of ``S_t=1[accepted_len>0]`` by document.

    The excess is bootstrapped by document so a tiny positive pooled excess is
    not promoted to a persistence finding without uncertainty evidence.
    """

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _ok_rows(rows):
        grouped[str(row.get("document_id"))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row.get("start_position", 0)))
    all_states = [int(int(row.get("accepted_len", 0) or 0) > 0) for row in _ok_rows(rows)]
    marginal = statistics.fmean(all_states) if all_states else None
    output: dict[str, Any] = {}
    for delta in deltas:
        if delta <= 0:
            raise ValueError("deltas must be positive")
        document_pairs = [
            [
                (int(int(values[index].get("accepted_len", 0) or 0) > 0),
                 int(int(values[index + delta].get("accepted_len", 0) or 0) > 0))
                for index in range(max(0, len(values) - delta))
            ]
            for values in grouped.values()
        ]
        document_pairs = [pairs for pairs in document_pairs if pairs]
        pairs = [pair for pairs_for_document in document_pairs for pair in pairs_for_document]
        conditioned = [right for left, right in pairs if left == 1]
        conditional_probability = statistics.fmean(conditioned) if conditioned else None
        target_probability = statistics.fmean(right for _, right in pairs) if pairs else None
        excess = (
            conditional_probability - target_probability
            if conditional_probability is not None and target_probability is not None
            else None
        )
        excess_ci = None
        if len(document_pairs) >= 5:
            rng = random.Random(seed + int(delta))
            bootstrapped: list[float] = []
            for _ in range(bootstrap_samples):
                resampled = [
                    pair
                    for _ in document_pairs
                    for pair in rng.choice(document_pairs)
                ]
                resampled_conditioned = [right for left, right in resampled if left == 1]
                if not resampled_conditioned or not resampled:
                    continue
                bootstrapped.append(
                    statistics.fmean(resampled_conditioned)
                    - statistics.fmean(right for _, right in resampled)
                )
            if bootstrapped:
                excess_ci = {
                    "low": _percentile(bootstrapped, 0.025),
                    "high": _percentile(bootstrapped, 0.975),
                    "bootstrap_valid": len(bootstrapped),
                    "bootstrap_requested": bootstrap_samples,
                    "bootstrap_unit": "document",
                    "confidence": 0.95,
                }
        output[str(delta)] = {
            "pair_count": len(pairs),
            "conditioned_pair_count": len(conditioned),
            "conditional_probability": conditional_probability,
            "marginal_probability": target_probability if target_probability is not None else marginal,
            "excess": excess,
            "excess_ci": excess_ci,
            "persistence_gate": "lower 95% document-bootstrap CI > 0",
            "lift": (
                conditional_probability / target_probability
                if conditional_probability is not None and target_probability not in (None, 0)
                else None
            ),
        }
    return {
        "status": "ok" if grouped else "UNAVAILABLE",
        "documents": len(grouped),
        "marginal_probability": marginal,
        "by_delta": output,
    }


def _timing_cost(row: Mapping[str, Any], selected_k: int) -> float | None:
    if selected_k == 0:
        value = _finite(row.get("autoregressive_time_ms"))
        return value if value is not None and value > 0.0 else None
    draft = row.get("draft_time_by_k_ms") or []
    verify = row.get("verification_time_by_k_ms") or []
    if len(draft) < selected_k or len(verify) < selected_k:
        return None
    cost = _finite(draft[selected_k - 1])
    verify_cost = _finite(verify[selected_k - 1])
    if cost is None or verify_cost is None or cost <= 0.0 or verify_cost <= 0.0:
        return None
    return cost + verify_cost


def _committed(row: Mapping[str, Any], selected_k: int) -> float:
    if selected_k == 0:
        return 1.0
    accepted = max(0, int(row.get("accepted_len", 0) or 0))
    return float(min(selected_k, accepted + 1))


def _aggregate_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    selected_k: Sequence[int],
    require_timing: bool,
) -> dict[str, Any]:
    committed: list[float] = []
    costs: list[float] = []
    selected: list[int] = []
    for row, k in zip(rows, selected_k):
        committed.append(_committed(row, k))
        selected.append(int(k))
        if require_timing:
            cost = _timing_cost(row, int(k))
            if cost is None:
                continue
            costs.append(cost)
    result: dict[str, Any] = {
        "status": "ok" if committed and (not require_timing or len(costs) == len(committed)) else "UNAVAILABLE",
        "policy": name,
        "count": len(committed),
        "mean_committed_tokens": statistics.fmean(committed) if committed else None,
        "selected_k_counts": dict(sorted(Counter(selected).items())),
        "timing_basis": "measured_draft_plus_verification_and_ar" if require_timing else "acceptance_only_no_speed_claim",
    }
    if require_timing and len(costs) == len(committed) and sum(costs) > 0.0:
        result.update({
            "mean_cost_ms": statistics.fmean(costs),
            "tokens_per_ms": sum(committed) / sum(costs),
        })
    elif require_timing:
        result["reason"] = "one or more selected policy costs are unavailable"
    return result


def replay_oracle_ladder(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_ks: Sequence[int] = (0, 2, 4, 8, 16),
    require_timing: bool = False,
) -> dict[str, Any]:
    """Replay fixed blocks, admission-free AR, and true-cost hindsight oracle."""

    candidates = tuple(sorted(set(int(value) for value in candidate_ks)))
    if not candidates or candidates[0] < 0:
        raise ValueError("candidate_ks must contain non-negative values")
    usable = _ok_rows(rows)
    policies: dict[str, dict[str, Any]] = {}
    valid_rows_by_k: dict[int, list[Mapping[str, Any]]] = {}
    for k in candidates:
        valid = [
            row for row in usable
            if (not require_timing or _timing_cost(row, k) is not None)
        ]
        valid_rows_by_k[k] = valid
        policies[f"fixed_k{k}"] = _aggregate_policy(
            valid,
            name=f"fixed_k{k}",
            selected_k=[k] * len(valid),
            require_timing=require_timing,
        )
    if not usable:
        return {"status": "UNAVAILABLE", "candidate_ks": candidates, "policies": policies}

    common_timing_count = 0
    if require_timing:
        # All ladder comparisons and the hindsight oracle must use the same
        # rows.  Otherwise k=16, which is most likely to be missing on long
        # contexts, would be compared on an easier/different subset.
        common_rows = [
            row for row in usable
            if all(_timing_cost(row, k) is not None for k in candidates)
        ]
        for k in candidates:
            policies[f"fixed_k{k}"] = _aggregate_policy(
                common_rows,
                name=f"fixed_k{k}",
                selected_k=[k] * len(common_rows),
                require_timing=True,
            )
        oracle_rows: list[Mapping[str, Any]] = []
        oracle_ks: list[int] = []
        for row in common_rows:
            scored: list[tuple[float, int]] = []
            for k in candidates:
                cost = _timing_cost(row, k)
                if cost is not None and cost > 0.0:
                    scored.append((_committed(row, k) / cost, k))
            if scored:
                _, selected_k = max(scored, key=lambda item: (item[0], -item[1]))
                oracle_rows.append(row)
                oracle_ks.append(selected_k)
        policies["true_cost_oracle"] = _aggregate_policy(
            oracle_rows,
            name="true_cost_oracle",
            selected_k=oracle_ks,
            require_timing=True,
        )
        common_timing_count = len(common_rows)
    else:
        policies["true_cost_oracle"] = {
            "status": "UNAVAILABLE",
            "policy": "true_cost_oracle",
            "count": len(usable),
            "reason": "true-cost oracle requires timing",
        }
    fixed_positive = [
        value for key, value in policies.items()
        if key.startswith("fixed_k") and key != "fixed_k0" and value.get("tokens_per_ms") is not None
    ]
    best_fixed = max(fixed_positive, key=lambda item: float(item["tokens_per_ms"])) if fixed_positive else None
    return {
        "status": "ok",
        "candidate_ks": candidates,
        "count": len(usable),
        "policies": policies,
        "best_fixed_positive": best_fixed,
        "common_timing_count": common_timing_count,
    }


def first_token_admission_oracle(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_k: int,
    require_timing: bool = False,
) -> dict[str, Any]:
    """Use only ``accepted_len > 0`` to choose AR versus ``SPEC(candidate_k)``."""

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    usable = [
        row for row in _ok_rows(rows)
        if not require_timing or (
            _timing_cost(row, 0) is not None and _timing_cost(row, candidate_k) is not None
        )
    ]
    policy_rows: list[dict[str, Any]] = []
    selected: list[int] = []
    for row in usable:
        admission_bit = int(int(row.get("accepted_len", 0) or 0) > 0)
        selected_k = candidate_k if admission_bit else 0
        selected.append(selected_k)
        policy_rows.append({
            "document_id": str(row.get("document_id")),
            "start_position": int(row.get("start_position", 0)),
            "admission_bit": admission_bit,
            "selected_k": selected_k,
            "committed_tokens": _committed(row, selected_k),
        })
    result = _aggregate_policy(
        usable,
        name=f"first_token_admission_k{candidate_k}",
        selected_k=selected,
        require_timing=require_timing,
    )
    result.update({
        "candidate_k": candidate_k,
        "admitted_speculation": sum(item["admission_bit"] for item in policy_rows),
        "admission_rate": statistics.fmean(item["admission_bit"] for item in policy_rows) if policy_rows else None,
        "policy_rows": policy_rows,
    })
    return result


def _split_by_document(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    document_ids = sorted({str(row.get("document_id")) for row in rows})
    train_count = max(1, int(len(document_ids) * train_fraction)) if document_ids else 0
    dev_count = int(len(document_ids) * dev_fraction)
    train_ids = set(document_ids[:train_count])
    dev_ids = set(document_ids[train_count : train_count + dev_count])
    train = [row for row in rows if str(row.get("document_id")) in train_ids]
    dev = [row for row in rows if str(row.get("document_id")) in dev_ids]
    test = [
        row for row in rows
        if str(row.get("document_id")) not in train_ids
        and str(row.get("document_id")) not in dev_ids
    ]
    return train, dev, test


def _with_corrected_horizon(
    target_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    max_k: int,
    variant: str = "nosink",
) -> list[dict[str, Any]]:
    targets = {str(row.get("document_id")): row for row in _ok_rows(target_rows)}
    result: list[dict[str, Any]] = []
    for row in _ok_rows(rows):
        target = targets.get(str(row.get("document_id")))
        if target is None:
            continue
        trace = _trace(target, variant)
        start = int(row.get("start_position", 0))
        if start < 0 or start + max_k >= len(trace):
            continue
        updated = dict(row)
        updated["corrected_grounding_horizon"] = corrected_grounding_horizon(
            trace, start=start, threshold=threshold, max_horizon=max_k
        )
        result.append(updated)
    return result


def _replay_horizon_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
    horizon_key: str = "corrected_grounding_horizon",
    require_timing: bool = True,
) -> dict[str, Any]:
    selected_rows: list[Mapping[str, Any]] = []
    selected_ks: list[int] = []
    for row in _ok_rows(rows):
        horizon = row.get(horizon_key)
        if horizon is None:
            continue
        k = min(max(1, int(horizon)), max_k)
        if not require_timing or _timing_cost(row, k) is not None:
            selected_rows.append(row)
            selected_ks.append(k)
    return _aggregate_policy(
        selected_rows,
        name="corrected_grounding_oracle",
        selected_k=selected_ks,
        require_timing=require_timing,
    )


def tune_horizon_threshold(
    target_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    *,
    requested: float,
    max_k: int,
    candidates: Sequence[float] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> dict[str, Any]:
    """Select threshold using train+dev utility, leaving test untouched."""

    usable = _with_corrected_horizon(
        target_rows, timing_rows, threshold=float(requested), max_k=max_k
    )
    if not usable:
        return {"status": "UNAVAILABLE", "requested": requested, "selected": requested}
    train, dev, test = _split_by_document(usable)
    calibration = train + dev
    scored: list[dict[str, Any]] = []
    for candidate in tuple(dict.fromkeys([float(requested), *candidates])):
        if candidate < 0.0 or not math.isfinite(candidate):
            continue
        candidate_rows = _with_corrected_horizon(
            target_rows, calibration, threshold=candidate, max_k=max_k
        )
        policy = _replay_horizon_policy(candidate_rows, max_k=max_k, require_timing=True)
        scored.append({
            "threshold": candidate,
            "calibration_rows": len(candidate_rows),
            "calibration_utility_tokens_per_ms": policy.get("tokens_per_ms"),
            "calibration_mean_cost_ms": policy.get("mean_cost_ms"),
        })
    valid = [item for item in scored if item["calibration_utility_tokens_per_ms"] is not None]
    selected_item = max(valid, key=lambda item: float(item["calibration_utility_tokens_per_ms"])) if valid else None
    return {
        "status": "ok" if selected_item else "UNAVAILABLE",
        "requested": requested,
        "selected": selected_item["threshold"] if selected_item else requested,
        "calibration_documents": len({str(row.get("document_id")) for row in calibration}),
        "test_documents": len({str(row.get("document_id")) for row in test}),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "candidates": scored,
    }


def _relative_speed_gain(policy: Mapping[str, Any], baseline: Mapping[str, Any]) -> float | None:
    value = _finite(policy.get("tokens_per_ms"))
    base = _finite(baseline.get("tokens_per_ms"))
    if value is None or base in (None, 0.0):
        return None
    return value / base - 1.0


def _best_timed_policy(policies: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the fastest policy among results with a measured utility."""

    usable = [
        dict(policy) for policy in policies
        if _finite(policy.get("tokens_per_ms")) is not None
    ]
    return max(usable, key=lambda item: float(item["tokens_per_ms"])) if usable else {}


def _annotate_ladder_levels(
    common_rows: Sequence[Mapping[str, Any]],
    *,
    max_k: int,
) -> dict[str, Any]:
    """Compute O1/O2/O3 on one common timing population."""

    levels = {
        "O1_positive_only": (2, 4, 8),
        "O2_with_admission": (0, 2, 4, 8),
        "O3_with_long_burst": (0, 2, 4, 8, 16),
    }
    result: dict[str, Any] = {}
    for name, candidates in levels.items():
        if any(candidate > max_k for candidate in candidates):
            continue
        ladder = replay_oracle_ladder(
            common_rows, candidate_ks=candidates, require_timing=True
        )
        policies = ladder.get("policies", {})
        fixed = [
            value for key, value in policies.items()
            if key.startswith("fixed_k") and key != "fixed_k0"
            and value.get("tokens_per_ms") is not None
        ]
        best_fixed = max(fixed, key=lambda item: float(item["tokens_per_ms"])) if fixed else {}
        oracle = policies.get("true_cost_oracle", {})
        gain = _relative_speed_gain(oracle, best_fixed) if best_fixed else None
        result[name] = {
            "candidate_ks": list(candidates),
            "ladder": ladder,
            "best_fixed_positive": best_fixed,
            "true_cost_oracle": oracle,
            "headroom_vs_best_fixed": gain,
            "decision": (
                "PASS" if gain is not None and gain >= 0.08
                else "FAIL" if gain is not None
                else "UNAVAILABLE"
            ),
        }
    return result


def summarize_cross_regime(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize whether a P0 decision replicates across datasets."""

    def status(path: str) -> str:
        values = [
            str(result.get(path, {}).get("decision", "UNAVAILABLE"))
            for result in results.values()
        ]
        if not values:
            return "UNAVAILABLE"
        if all(value == "PASS" for value in values):
            return "PASS"
        if all(value == "FAIL" for value in values):
            return "FAIL"
        return "MIXED"

    levels = [
        result.get("P0-3_oracle_ladder", {}).get("levels", {}).get("O3_with_long_burst", {})
        for result in results.values()
    ]
    o3_values = [item.get("headroom_vs_best_fixed") for item in levels]
    o3_status = (
        "PASS" if o3_values and all(value is not None and value >= 0.08 for value in o3_values)
        else "FAIL" if o3_values and all(value is not None and value < 0.08 for value in o3_values)
        else "MIXED" if o3_values else "UNAVAILABLE"
    )
    summary = {
        "P0-1_corrected_H2": status("P0-1_corrected_H2"),
        "P0-2_corrected_H4_grounding_oracle": status("P0-2_corrected_H4_grounding_oracle"),
        "P0-3_oracle_ladder_O3": o3_status,
        "P0-4_first_token_admission": status("P0-4_first_token_admission"),
        "P0-5_burstiness": status("P0-5_burstiness"),
    }
    if summary["P0-2_corrected_H4_grounding_oracle"] == "FAIL":
        summary["overall_decision"] = (
            "NO_GO_GROUNDSYNC_GENERAL; conditional BurstSpec follow-up only where admission passes"
        )
    elif summary["P0-4_first_token_admission"] == "PASS":
        summary["overall_decision"] = "PIVOT_BURSTSPEC"
    else:
        summary["overall_decision"] = "NO_GO_ADAPTIVE_SPECULATION"
    return summary


def analyze_p0_dataset(
    target_rows: Sequence[Mapping[str, Any]],
    speculative_rows: Sequence[Mapping[str, Any]],
    *,
    timing_rows: Sequence[Mapping[str, Any]] | None = None,
    multistart_rows: Sequence[Mapping[str, Any]] | None = None,
    max_k: int = 16,
    candidate_ks: Sequence[int] = (0, 2, 4, 8, 16),
    requested_horizon_threshold: float = 0.2,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Run P0-1..P0-5 for one dataset and preserve all coverage details."""

    targets = _ok_rows(target_rows)
    specs = _ok_rows(speculative_rows)
    timings = _ok_rows(timing_rows if timing_rows is not None else speculative_rows)
    transition_rows = transition_hazard_rows(targets, specs, max_k=max_k)
    hazard = fit_transition_hazard(transition_rows, bootstrap_samples=bootstrap_samples)

    threshold_selection = tune_horizon_threshold(
        targets,
        timings,
        requested=requested_horizon_threshold,
        max_k=max_k,
    )
    selected_threshold = float(threshold_selection.get("selected", requested_horizon_threshold))
    corrected_timing = _with_corrected_horizon(
        targets, timings, threshold=selected_threshold, max_k=max_k
    )
    corrected_all = _with_corrected_horizon(
        targets, specs, threshold=selected_threshold, max_k=max_k
    )
    common_timing_rows = [
        row for row in timings
        if all(_timing_cost(row, int(k)) is not None for k in candidate_ks)
    ]
    ladder = replay_oracle_ladder(
        common_timing_rows,
        candidate_ks=candidate_ks,
        require_timing=True,
    )
    ladder["levels"] = _annotate_ladder_levels(common_timing_rows, max_k=max_k)
    positive_fixed = [
        value for key, value in ladder.get("policies", {}).items()
        if key.startswith("fixed_k") and key != "fixed_k0"
        and value.get("tokens_per_ms") is not None
    ]
    best_fixed = max(positive_fixed, key=lambda item: float(item["tokens_per_ms"])) if positive_fixed else {}
    corrected_timing = [
        row for row in _with_corrected_horizon(
            targets, common_timing_rows, threshold=selected_threshold, max_k=max_k
        )
    ]
    train, dev, test = _split_by_document(corrected_timing)
    test_ladder = replay_oracle_ladder(
        test,
        candidate_ks=candidate_ks,
        require_timing=True,
    )
    test_positive_fixed = [
        value for key, value in test_ladder.get("policies", {}).items()
        if key.startswith("fixed_k") and key != "fixed_k0"
        and value.get("tokens_per_ms") is not None
    ]
    best_fixed_test = (
        max(test_positive_fixed, key=lambda item: float(item["tokens_per_ms"]))
        if test_positive_fixed else {}
    )
    corrected_test = _replay_horizon_policy(test, max_k=max_k, require_timing=True)
    corrected_full = _replay_horizon_policy(corrected_timing, max_k=max_k, require_timing=True)
    corrected_gain = _relative_speed_gain(corrected_test, best_fixed_test) if best_fixed_test else None

    generic_policies: dict[str, Any] = {}
    generic_policies_test: dict[str, Any] = {}
    try:
        from .report import replay_policy

        for policy_name in ("adaptive_entropy", "adaptive_history"):
            generic_policies[policy_name] = replay_policy(
                common_timing_rows, policy=policy_name, max_k=max_k, require_timing=True
            )
            generic_policies_test[policy_name] = replay_policy(
                test, policy=policy_name, max_k=max_k, require_timing=True
            )
    except (ImportError, ValueError, TypeError):
        generic_policies = {}
        generic_policies_test = {}

    best_generic_adaptive_test = _best_timed_policy(generic_policies_test.values())
    best_available_baseline_test = _best_timed_policy(
        (best_fixed_test, best_generic_adaptive_test)
    )
    gain_vs_generic = (
        _relative_speed_gain(corrected_test, best_generic_adaptive_test)
        if best_generic_adaptive_test else None
    )
    gain_vs_available = (
        _relative_speed_gain(corrected_test, best_available_baseline_test)
        if best_available_baseline_test else None
    )
    h4_decision = (
        "PASS" if gain_vs_available is not None and gain_vs_available >= 0.05
        else "FAIL" if gain_vs_available is not None
        else "UNAVAILABLE"
    )

    admission: dict[str, Any] = {}
    true_oracle = ladder.get("policies", {}).get("true_cost_oracle", {})
    true_speed = _finite(true_oracle.get("tokens_per_ms"))
    for k in candidate_ks:
        if int(k) <= 0:
            continue
        result = first_token_admission_oracle(common_timing_rows, candidate_k=int(k), require_timing=True)
        baseline = ladder.get("policies", {}).get(f"fixed_k{k}", {})
        base_speed = _finite(baseline.get("tokens_per_ms"))
        entry_speed = _finite(result.get("tokens_per_ms"))
        denominator = None if base_speed is None or true_speed is None else true_speed - base_speed
        recovery = (
            None if denominator in (None, 0.0) or entry_speed is None
            else (entry_speed - base_speed) / denominator
        )
        result["best_fixed_baseline"] = baseline
        result["oracle_speed"] = true_speed
        result["oracle_gain_recovery"] = recovery
        result["decision"] = (
            "PASS" if recovery is not None and recovery >= 0.40
            else "FAIL" if recovery is not None
            else "UNAVAILABLE"
        )
        admission[str(k)] = result
    best_admission = max(
        admission.values(),
        key=lambda item: float(item.get("oracle_gain_recovery", -math.inf))
        if item.get("oracle_gain_recovery") is not None else -math.inf,
        default=None,
    )

    burst_rows = multistart_rows if multistart_rows else specs
    within = summarize_within_block_burstiness(burst_rows, max_k=max_k)
    across = across_round_persistence(
        burst_rows, bootstrap_samples=bootstrap_samples
    )
    complete_burst_rows = [
        row for row in _ok_rows(burst_rows)
        if not isinstance(row.get("proposal_token_ids"), Sequence)
        or len(row.get("proposal_token_ids", [])) >= int(row.get("max_k", 0) or 0)
    ]
    persistence_values = [
        item.get("excess") for item in across.get("by_delta", {}).values()
        if item.get("excess") is not None
    ]
    persistence_pass = any(
        item.get("excess_ci") is not None
        and float(item["excess_ci"].get("low", 0.0)) > 0.0
        for item in across.get("by_delta", {}).values()
    )
    persistence_measured = any(
        item.get("pair_count", 0) > 0
        for item in across.get("by_delta", {}).values()
    )
    burst_decision = (
        "PASS" if within.get("h1_to_later_hazard_ratio") is not None
        and float(within["h1_to_later_hazard_ratio"]) > 1.0
        and persistence_pass
        else "FAIL" if within.get("status") == "ok" and persistence_measured else "UNAVAILABLE"
    )

    return {
        "schema_version": "groundsync.p0.dataset.v1",
        "coverage": {
            "target_rows": len(targets),
            "speculative_rows": len(specs),
            "timing_rows": len(timings),
            "timing_complete_rows": len(common_timing_rows),
            "timing_complete_documents": len({str(row.get("document_id")) for row in common_timing_rows}),
            "transition_hazard_rows": len(transition_rows),
            "corrected_timing_rows": len(corrected_timing),
            "corrected_test_rows": len(test),
            "corrected_all_rows": len(corrected_all),
            "multistart_rows": len(_ok_rows(burst_rows)),
            "multistart_complete_rows": len(complete_burst_rows),
            "multistart_documents": len({str(row.get("document_id")) for row in _ok_rows(burst_rows)}),
            "target_documents": len({str(row.get("document_id")) for row in targets}),
            "speculative_documents": len({str(row.get("document_id")) for row in specs}),
            "timing_documents": len({str(row.get("document_id")) for row in timings}),
        },
        "protocol": {
            "max_k": max_k,
            "candidate_ks": list(candidate_ks),
            "requested_horizon_threshold": requested_horizon_threshold,
            "selected_horizon_threshold": selected_threshold,
            "timing_basis": "AR one-token + measured draft block + cached target verification",
        },
        "P0-1_corrected_H2": {
            "hazard": hazard,
            "decision": hazard.get("decision", "UNAVAILABLE"),
        },
        "P0-2_corrected_H4_grounding_oracle": {
            "threshold_selection": threshold_selection,
            "train_documents": len({str(row.get("document_id")) for row in train}),
            "dev_documents": len({str(row.get("document_id")) for row in dev}),
            "test_documents": len({str(row.get("document_id")) for row in test}),
            "oracle_test": corrected_test,
            "oracle_full": corrected_full,
            "best_fixed": best_fixed,
            "best_fixed_test": best_fixed_test,
            "test_ladder": test_ladder,
            "generic_adaptive": generic_policies,
            "generic_adaptive_test": generic_policies_test,
            "best_generic_adaptive_test": best_generic_adaptive_test,
            "best_available_baseline_test": best_available_baseline_test,
            "speed_gain_vs_best_generic_adaptive_test": gain_vs_generic,
            "speed_gain_vs_best_fixed_test": corrected_gain,
            "speed_gain_vs_best_available_test": gain_vs_available,
            "decision": h4_decision,
        },
        "P0-3_oracle_ladder": ladder,
        "P0-4_first_token_admission": {
            "by_candidate_k": admission,
            "best_candidate": best_admission,
            "decision": best_admission.get("decision", "UNAVAILABLE") if best_admission else "UNAVAILABLE",
        },
        "P0-5_burstiness": {
            "within_block": within,
            "across_round": across,
            "decision": burst_decision,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _read_trace_source(value: Any) -> list[dict[str, Any]]:
    """Read one trace path or concatenate several non-overlapping trace paths."""

    paths = value if isinstance(value, list) else [value]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path_value in paths:
        if not path_value:
            continue
        for row in _read_jsonl(Path(str(path_value))):
            key = (str(row.get("document_id")), int(row.get("start_position", -1)))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _write_plot(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    width, height = 1000, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), title, fill="black")
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        draw.text((25, 60), "No finite data", fill="black")
        image.save(path)
        return
    lo = min(0.0, min(finite_values))
    hi = max(1.0, max(finite_values))
    left, top, bottom = 90, 70, 490
    scale = (bottom - top) / (hi - lo or 1.0)
    zero_y = top + int((hi - 0.0) * scale)
    draw.line((left, top, left, bottom), fill="black")
    draw.line((left, zero_y, width - 35, zero_y), fill="black")
    slot = (width - left - 45) / max(len(values), 1)
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + int((index + 0.5) * slot)
        y = top + int((hi - float(value)) * scale)
        draw.line((x, zero_y, x, y), fill=(35, 95, 160), width=8)
        draw.text((x - 25, bottom + 15), str(label), fill="black")
    image.save(path)


def write_p0_artifacts(
    output_dir: Path,
    results: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> None:
    """Write co-located machine-readable and human-readable P0 artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "schema_version": "groundsync.p0.report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": dict(results),
        "cross_regime": summarize_cross_regime(results),
    }
    (output_dir / "p0_metrics.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_rows: list[dict[str, Any]] = []
    for name, result in results.items():
        for experiment, key in (
            ("corrected_H2", "P0-1_corrected_H2"),
            ("corrected_H4", "P0-2_corrected_H4_grounding_oracle"),
            ("oracle_ladder", "P0-3_oracle_ladder"),
            ("admission", "P0-4_first_token_admission"),
            ("burstiness", "P0-5_burstiness"),
        ):
            section = result.get(key, {})
            if experiment == "corrected_H2":
                hazard = section.get("hazard", {})
                csv_rows.append({
                    "dataset": name,
                    "experiment": experiment,
                    "metric": "drift_coefficient",
                    "value": hazard.get("drift_coefficient"),
                    "decision": section.get("decision"),
                })
                ci = hazard.get("drift_coefficient_ci") or {}
                csv_rows.extend([
                    {"dataset": name, "experiment": experiment, "metric": "ci_low", "value": ci.get("low"), "decision": section.get("decision")},
                    {"dataset": name, "experiment": experiment, "metric": "ci_high", "value": ci.get("high"), "decision": section.get("decision")},
                ])
            elif experiment == "corrected_H4":
                for metric in (
                    "speed_gain_vs_best_fixed_test",
                    "speed_gain_vs_best_generic_adaptive_test",
                    "speed_gain_vs_best_available_test",
                ):
                    csv_rows.append({
                        "dataset": name,
                        "experiment": experiment,
                        "metric": metric,
                        "value": section.get(metric),
                        "decision": section.get("decision"),
                    })
            elif experiment == "oracle_ladder":
                for policy, policy_result in section.get("policies", {}).items():
                    csv_rows.append({
                        "dataset": name,
                        "experiment": experiment,
                        "metric": f"{policy}.tokens_per_ms",
                        "value": policy_result.get("tokens_per_ms"),
                        "decision": policy_result.get("status"),
                    })
                for level, level_result in section.get("levels", {}).items():
                    csv_rows.append({
                        "dataset": name,
                        "experiment": experiment,
                        "metric": f"{level}.headroom_vs_best_fixed",
                        "value": level_result.get("headroom_vs_best_fixed"),
                        "decision": level_result.get("decision"),
                    })
            elif experiment == "admission":
                for candidate, item in section.get("by_candidate_k", {}).items():
                    csv_rows.append({
                        "dataset": name,
                        "experiment": experiment,
                        "metric": f"k{candidate}.oracle_gain_recovery",
                        "value": item.get("oracle_gain_recovery"),
                        "decision": item.get("decision"),
                    })
            else:
                within = section.get("within_block", {})
                across = section.get("across_round", {})
                csv_rows.append({
                    "dataset": name,
                    "experiment": experiment,
                    "metric": "h1_to_later_hazard_ratio",
                    "value": within.get("h1_to_later_hazard_ratio"),
                    "decision": section.get("decision"),
                })
                for delta, item in across.get("by_delta", {}).items():
                    csv_rows.append({
                        "dataset": name,
                        "experiment": experiment,
                        "metric": f"delta{delta}.excess",
                        "value": item.get("excess"),
                        "decision": section.get("decision"),
                    })
                    ci = item.get("excess_ci") or {}
                    csv_rows.extend([
                        {"dataset": name, "experiment": experiment, "metric": f"delta{delta}.excess_ci_low", "value": ci.get("low"), "decision": section.get("decision")},
                        {"dataset": name, "experiment": experiment, "metric": f"delta{delta}.excess_ci_high", "value": ci.get("high"), "decision": section.get("decision")},
                    ])
    with (output_dir / "p0_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "experiment", "metric", "value", "decision"])
        writer.writeheader()
        writer.writerows(csv_rows)

    ladder_labels: list[str] = []
    ladder_values: list[float] = []
    for name, result in results.items():
        for policy, item in result.get("P0-3_oracle_ladder", {}).get("policies", {}).items():
            value = item.get("tokens_per_ms")
            if value is not None:
                ladder_labels.append(f"{name[:5]}:{policy.replace('fixed_', '')}")
                ladder_values.append(float(value))
    _write_plot(output_dir / "p0_oracle_ladder.png", "P0 oracle ladder tokens/ms", ladder_labels, ladder_values)
    admission_labels: list[str] = []
    admission_values: list[float] = []
    for name, result in results.items():
        for candidate, item in result.get("P0-4_first_token_admission", {}).get("by_candidate_k", {}).items():
            value = item.get("oracle_gain_recovery")
            if value is not None:
                admission_labels.append(f"{name[:5]}:k{candidate}")
                admission_values.append(float(value))
    _write_plot(output_dir / "p0_admission_recovery.png", "P0 first-token admission oracle gain recovery", admission_labels, admission_values)
    hazard_labels: list[str] = []
    hazard_values: list[float] = []
    for name, result in results.items():
        value = result.get("P0-1_corrected_H2", {}).get("hazard", {}).get("drift_coefficient")
        if value is not None:
            hazard_labels.append(name)
            hazard_values.append(float(value))
    _write_plot(output_dir / "p0_transition_hazard.png", "P0 corrected transition drift coefficient", hazard_labels, hazard_values)

    lines = [
        "# Báo cáo P0 — corrected GroundSync và BurstSpec",
        "",
        "Báo cáo này là controlled/discovery evidence. Timing dùng AR một token,",
        "draft block và cached target verification; không phải EAGLE/vLLM production throughput.",
        "",
        "| Dataset | P0-1 corrected H2 | P0-2 corrected H4 | P0-3 O3 ladder | P0-4 admission | P0-5 burstiness |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result.get('P0-1_corrected_H2', {}).get('decision')} | "
            f"{result.get('P0-2_corrected_H4_grounding_oracle', {}).get('decision')} | "
            f"{result.get('P0-3_oracle_ladder', {}).get('levels', {}).get('O3_with_long_burst', {}).get('decision')} | "
            f"{result.get('P0-4_first_token_admission', {}).get('decision')} | "
            f"{result.get('P0-5_burstiness', {}).get('decision')} |"
        )
    lines.extend([
        "",
        "## Quyết định cross-regime",
        "",
        f"`{summarize_cross_regime(results).get('overall_decision')}`",
        "",
        "| Experiment | Decision |",
        "|---|---:|",
    ])
    for experiment, decision in summarize_cross_regime(results).items():
        if experiment != "overall_decision":
            lines.append(f"| {experiment} | {decision} |")
    lines.extend(["", "## Chi tiết", ""])
    for name, result in results.items():
        coverage = result.get("coverage", {})
        h2 = result.get("P0-1_corrected_H2", {}).get("hazard", {})
        h4 = result.get("P0-2_corrected_H4_grounding_oracle", {})
        ladder = result.get("P0-3_oracle_ladder", {}).get("policies", {})
        admission = result.get("P0-4_first_token_admission", {}).get("by_candidate_k", {})
        ladder_levels = result.get("P0-3_oracle_ladder", {}).get("levels", {})
        burst = result.get("P0-5_burstiness", {})
        lines.extend([
            f"### {name}",
            "",
            f"Coverage: target {coverage.get('target_rows')} rows / "
            f"spec {coverage.get('speculative_rows')} / timing-ok {coverage.get('timing_rows')} / "
            f"timing-complete(k=0,2,4,8,16) {coverage.get('timing_complete_rows')} / "
            f"multi-start {coverage.get('multistart_rows')} "
            f"({coverage.get('multistart_complete_rows')} complete proposals).",
            "",
            f"- P0-1: coefficient={h2.get('drift_coefficient')}, "
            f"CI={h2.get('drift_coefficient_ci')}, decision={result.get('P0-1_corrected_H2', {}).get('decision')}.",
            f"- P0-2: threshold selected={h4.get('threshold_selection', {}).get('selected')}, "
            f"gain vs best fixed={h4.get('speed_gain_vs_best_fixed_test')}, "
            f"gain vs best generic adaptive={h4.get('speed_gain_vs_best_generic_adaptive_test')}, "
            f"gain vs best available={h4.get('speed_gain_vs_best_available_test')}, "
            f"decision={h4.get('decision')}.",
            f"- P0-3: " + ", ".join(
                f"{key}={item.get('tokens_per_ms')}" for key, item in ladder.items()
            ) + "; headroom " + ", ".join(
                f"{key}={item.get('headroom_vs_best_fixed')} ({item.get('decision')})"
                for key, item in ladder_levels.items()
            ) + ".",
            f"- P0-4: " + ", ".join(
                f"k{k}: recovery={item.get('oracle_gain_recovery')}"
                for k, item in admission.items()
            ) + ".",
            f"- P0-5: within ratio={burst.get('within_block', {}).get('h1_to_later_hazard_ratio')}, "
            f"across excess/95% CI={{{', '.join(f'{k}: {v.get('excess')} [{(v.get('excess_ci') or {}).get('low')}, {(v.get('excess_ci') or {}).get('high')}]' for k, v in burst.get('across_round', {}).get('by_delta', {}).items())}}}, "
            f"decision={burst.get('decision')}.",
            "",
        ])
    (output_dir / "p0_decision_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": "groundsync.p0.manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config": dict(config),
        "datasets": sorted(results),
        "artifact_files": [
            "p0_metrics.json", "p0_metrics.csv", "p0_decision_report.md",
            "p0_oracle_ladder.png", "p0_admission_recovery.png", "p0_transition_hazard.png",
        ],
    }
    (output_dir / "p0_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON list of dataset trace paths")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-k", type=int, default=16)
    parser.add_argument("--candidate-ks", default="0,2,4,8,16")
    parser.add_argument("--horizon-threshold", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, list) or not config:
        raise ValueError("config must be a non-empty JSON list")
    candidate_ks = tuple(int(item.strip()) for item in args.candidate_ks.split(",") if item.strip())
    results: dict[str, Mapping[str, Any]] = {}
    for item in config:
        name = str(item["name"])
        target = _read_jsonl(Path(item["target"]))
        specs = _read_jsonl(Path(item["speculative"]))
        timing = _read_trace_source(item["timing"]) if item.get("timing") else None
        multi = _read_jsonl(Path(item["multistart"])) if item.get("multistart") else None
        results[name] = analyze_p0_dataset(
            target,
            specs,
            timing_rows=timing,
            multistart_rows=multi,
            max_k=args.max_k,
            candidate_ks=candidate_ks,
            requested_horizon_threshold=args.horizon_threshold,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(f"p0 {name}: target={len(target)} spec={len(specs)} timing={len(timing or [])}", flush=True)
    config_meta = {
        "config_path": str(config_path),
        "max_k": args.max_k,
        "candidate_ks": candidate_ks,
        "horizon_threshold": args.horizon_threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "elapsed_s": time.perf_counter() - started,
    }
    write_p0_artifacts(Path(args.output), results, config=config_meta)
    print(f"p0 artifacts: {args.output}", flush=True)
    return 0


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
