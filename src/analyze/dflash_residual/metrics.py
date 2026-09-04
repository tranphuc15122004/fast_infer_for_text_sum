"""Dependency-light metrics for DFlash/DFlash2 residual-headroom traces."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def _ok(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def prefix_match_length(selected: Sequence[int], target: Sequence[int]) -> int:
    """Return the length of the exact consecutive prefix."""

    length = 0
    for actual, expected in zip(selected, target):
        if int(actual) != int(expected):
            break
        length += 1
    return length


def recall_at_k(rows: Iterable[Mapping[str, Any]], k: int) -> float | None:
    """Compute token-level target-in-Top-K recall without target injection."""

    if k <= 0:
        raise ValueError("k must be positive")
    usable = _ok(rows)
    if not usable:
        return None
    hits = sum(int(int(row["target_token_id"]) in row["candidate_token_ids"][:k]) for row in usable)
    return hits / len(usable)


def survival(accepted_lengths: Sequence[int], points: Sequence[int] = (1, 2, 4, 8)) -> dict[str, float]:
    """Compute ``S(j)=P(A >= j)`` for block-level accepted draft lengths."""

    values = [int(value) for value in accepted_lengths]
    if not values:
        return {}
    if any(value < 0 for value in values):
        raise ValueError("accepted lengths must be non-negative")
    return {str(point): sum(value >= point for value in values) / len(values) for point in points}


def _block_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row["run_id"]),
        str(row["sample_id"]),
        int(row["round_index"]),
        int(row["context_length"]),
        str(row.get("context_cap", "")),
    )


def _blocks(rows: Iterable[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in _ok(rows):
        grouped[_block_key(row)].append(row)
    return [sorted(block, key=lambda row: int(row["draft_position"])) for block in grouped.values()]


def _observed_acceptance(block: Sequence[Mapping[str, Any]]) -> int:
    values = [row.get("accepted_draft_len") for row in block if row.get("accepted_draft_len") is not None]
    if values:
        if len(set(int(value) for value in values)) != 1:
            raise ValueError("accepted_draft_len differs within one block")
        return max(0, int(values[0]))
    selected = [int(row["dflash_selected_token_id"]) for row in block]
    target = [int(row["target_token_id"]) for row in block]
    return prefix_match_length(selected, target)


def _selected_prefix(block: Sequence[Mapping[str, Any]], field: str) -> int | None:
    if any(row.get(field) is None for row in block):
        return None
    selected = [int(row[field]) for row in block]
    target = [int(row["target_token_id"]) for row in block]
    return prefix_match_length(selected, target)


def oracle_prefix_length(block: Sequence[Mapping[str, Any]], k: int = 16) -> int:
    """Return longest prefix whose target token is present in Top-K."""

    if k <= 0:
        raise ValueError("k must be positive")
    length = 0
    for row in sorted(block, key=lambda item: int(item["draft_position"])):
        if int(row["target_token_id"]) not in row["candidate_token_ids"][:k]:
            break
        length += 1
    return length


def _regime_summary(rows: Sequence[Mapping[str, Any]], recall_ks: Sequence[int], survival_points: Sequence[int]) -> dict[str, Any]:
    blocks = _blocks(rows)
    accepted = [_observed_acceptance(block) for block in blocks]
    result: dict[str, Any] = {
        "rows": len(rows),
        "blocks": len(blocks),
        "documents": len({str(row["document_id"]) for row in rows}),
        "mat": (sum(accepted) / len(accepted)) if accepted else None,
        "survival": survival(accepted, survival_points),
        "recall": {str(k): recall_at_k(rows, k) for k in recall_ks},
    }
    return result


def summarize_p1(
    rows: Iterable[Mapping[str, Any]],
    *,
    recall_ks: Sequence[int] = (1, 4, 8, 16),
    survival_points: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """Summarize MAT, survival and candidate recall by workload regime."""

    usable = _ok(rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        grouped[str(row.get("task_regime", "other"))].append(row)
    return {
        "status": "ok" if usable else "unavailable",
        "reason": None if usable else "no_valid_trace_rows",
        "rows": len(usable),
        "regimes": {
            regime: _regime_summary(regime_rows, recall_ks, survival_points)
            for regime, regime_rows in sorted(grouped.items())
        },
    }


def summarize_headroom(rows: Iterable[Mapping[str, Any]], *, oracle_k: int = 16) -> dict[str, Any]:
    """Compare DFlash, DFlash2 and a perfect Top-K candidate oracle."""

    if oracle_k <= 0:
        raise ValueError("oracle_k must be positive")
    blocks = _blocks(rows)
    if not blocks:
        return {"status": "unavailable", "reason": "no_valid_trace_rows"}
    if any(_selected_prefix(block, "dflash2_selected_token_id") is None for block in blocks):
        return {
            "status": "unavailable",
            "reason": "missing_dflash2_selection",
            "blocks": len(blocks),
        }
    mat_d_values = [_observed_acceptance(block) for block in blocks]
    mat_d2_values = [_selected_prefix(block, "dflash2_selected_token_id") or 0 for block in blocks]
    mat_oracle_values = [oracle_prefix_length(block, oracle_k) for block in blocks]
    candidate_miss_rows = sum(
        int(int(row["target_token_id"]) not in row["candidate_token_ids"][:oracle_k])
        for block in blocks for row in block
    )
    selection_error_rows = sum(
        int(
            int(row["target_token_id"]) in row["candidate_token_ids"][:oracle_k]
            and int(row["dflash2_selected_token_id"]) != int(row["target_token_id"])
        )
        for block in blocks for row in block
    )
    mat_d = sum(mat_d_values) / len(blocks)
    mat_d2 = sum(mat_d2_values) / len(blocks)
    mat_oracle = sum(mat_oracle_values) / len(blocks)
    headroom = mat_oracle - mat_d
    result: dict[str, Any] = {
        "status": "ok",
        "blocks": len(blocks),
        "documents": len({str(row["document_id"]) for block in blocks for row in block}),
        "oracle_k": oracle_k,
        "mat_d": mat_d,
        "mat_d2": mat_d2,
        "mat_o16": mat_oracle,
        "oracle_headroom": headroom,
        "candidate_miss_rows": candidate_miss_rows,
        "selection_error_rows": selection_error_rows,
        "candidate_miss_rate": candidate_miss_rows / max(sum(len(block) for block in blocks), 1),
        "selection_error_rate": selection_error_rows / max(sum(len(block) for block in blocks), 1),
        "rho_d2": None,
        "rho_status": "zero_headroom" if headroom <= 0 else "ok",
    }
    if headroom > 0:
        result["rho_d2"] = (mat_d2 - mat_d) / headroom
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(min(value, 700.0))
    return z / (1.0 + z)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense linear system with deterministic pivoting."""

    n = len(vector)
    augmented = [list(row) + [vector[index]] for index, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda index: abs(augmented[index][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular design matrix")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [
                left - factor * right for left, right in zip(augmented[row], augmented[col])
            ]
    return [augmented[index][-1] for index in range(n)]


def _fit_logistic(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    if len(rows) < 5:
        raise ValueError("too_few_rows")
    features: list[list[float]] = []
    labels: list[float] = []
    for row in rows:
        context = float(row["context_length"])
        position = float(row["draft_position"])
        features.append([1.0, math.log1p(context), position, math.log1p(context) * position])
        labels.append(float(int(int(row["target_token_id"]) in row["candidate_token_ids"][:16])))
    beta = [0.0] * 4
    for _ in range(80):
        weights: list[float] = []
        working: list[float] = []
        for feature, label in zip(features, labels):
            eta = sum(left * right for left, right in zip(feature, beta))
            probability = _sigmoid(eta)
            weight = max(probability * (1.0 - probability), 1e-5)
            weights.append(weight)
            working.append(eta + (label - probability) / weight)
        matrix = [[0.0] * 4 for _ in range(4)]
        vector = [0.0] * 4
        for feature, weight, value in zip(features, weights, working):
            for left in range(4):
                vector[left] += feature[left] * weight * value
                for right in range(4):
                    matrix[left][right] += feature[left] * weight * feature[right]
        for diagonal in range(4):
            matrix[diagonal][diagonal] += 1e-6
        updated = _solve(matrix, vector)
        if max(abs(left - right) for left, right in zip(beta, updated)) < 1e-7:
            beta = updated
            break
        beta = updated
    return beta


def fit_context_depth_interaction(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 500,
    seed: int = 42,
    min_documents: int = 5,
) -> dict[str, Any]:
    """Fit ``hit ~ log(L) + j + log(L)*j`` with document bootstrap CI."""

    usable = _ok(rows)
    documents = sorted({str(row["document_id"]) for row in usable})
    if len(documents) < min_documents:
        return {
            "status": "inconclusive",
            "reason": "insufficient_documents",
            "documents": len(documents),
            "rows": len(usable),
        }
    positions = {int(row["draft_position"]) for row in usable}
    contexts = {int(row["context_length"]) for row in usable}
    if len(positions) < 2 or len(contexts) < 2:
        return {
            "status": "inconclusive",
            "reason": "insufficient_context_or_depth_variation",
            "documents": len(documents),
            "rows": len(usable),
        }
    try:
        beta = _fit_logistic(usable)
    except ValueError as exc:
        return {"status": "unavailable", "reason": str(exc), "documents": len(documents), "rows": len(usable)}
    by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        by_document[str(row["document_id"])].append(row)
    rng = random.Random(seed)
    bootstrap_values: list[float] = []
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(documents) for _ in documents]
        sampled_rows = [row for document in sampled for row in by_document[document]]
        try:
            bootstrap_values.append(_fit_logistic(sampled_rows)[3])
        except ValueError:
            continue
    if not bootstrap_values:
        return {"status": "unavailable", "reason": "bootstrap_fit_failed", "documents": len(documents), "rows": len(usable)}
    ordered = sorted(bootstrap_values)
    low = ordered[max(0, int(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, int(0.975 * (len(ordered) - 1)))]
    if high < 0.0:
        decision = "PASS"
    elif low > 0.0:
        decision = "FAIL"
    else:
        decision = "INCONCLUSIVE"
    return {
        "status": "ok",
        "decision": decision,
        "documents": len(documents),
        "rows": len(usable),
        "beta_intercept": beta[0],
        "beta_log_context": beta[1],
        "beta_position": beta[2],
        "beta_log_context_x_position": beta[3],
        "bootstrap_ci": [low, high],
        "bootstrap_successes": len(bootstrap_values),
        "bootstrap_samples": bootstrap_samples,
    }
