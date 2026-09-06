"""Offline prefix-gap analyses for DFlash candidate lattices.

The functions in this module operate only on JSON-safe trace rows.  They do not
load a model and intentionally separate marginal candidate recall from the
joint prefix event required by speculative verification.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .metrics import _blocks, _observed_acceptance, summarize_p1


def _ok(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def _validate_k_values(k_values: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in k_values}))
    if not values or any(value <= 0 for value in values):
        raise ValueError("k_values must contain positive integers")
    return values


def _prefix_hit_flags(block: Sequence[Mapping[str, Any]], k: int) -> list[bool]:
    if k <= 0:
        raise ValueError("k must be positive")
    return [_candidate_hit(row, k) for row in sorted(block, key=lambda item: int(item["draft_position"]))]


def _candidate_hit(row: Mapping[str, Any], k: int) -> bool:
    """Return Top-K membership, including exact logit ties at the boundary.

    ``torch.topk`` has an arbitrary order for equal logits while DFlash greedy
    selection uses ``argmax``.  Treating a target as a hit when its recorded
    logit reaches the K-th recorded logit keeps the oracle an upper bound on
    the actual greedy path without injecting any token.
    """

    candidates = [int(token) for token in row["candidate_token_ids"]]
    target = int(row["target_token_id"])
    if target in candidates[:k]:
        return True
    logits = row.get("candidate_logits")
    if not isinstance(logits, list) or k > len(logits) or target not in candidates:
        return False
    target_index = candidates.index(target)
    if target_index >= len(logits):
        return False
    boundary = float(logits[k - 1])
    return float(logits[target_index]) >= boundary - 1e-6


def prefix_oracle_length(block: Sequence[Mapping[str, Any]], k: int) -> int:
    """Return the longest consecutive Top-K-hit prefix for one block."""

    length = 0
    for hit in _prefix_hit_flags(block, k):
        if not hit:
            break
        length += 1
    return length


def _position_groups(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _ok(rows):
        grouped[int(row["draft_position"])].append(row)
    return dict(grouped)


def _survival_from_lengths(lengths: Sequence[int], max_position: int) -> dict[int, float]:
    if not lengths:
        return {}
    denominator = len(lengths)
    return {
        position: sum(int(length) >= position for length in lengths) / denominator
        for position in range(1, max_position + 1)
    }


def joint_prefix_survival(rows: Iterable[Mapping[str, Any]], k: int) -> dict[int, float]:
    """Compute ``S_K(j)`` over blocks, not independent token rows."""

    blocks = _blocks(_ok(rows))
    if not blocks:
        return {}
    lengths = [prefix_oracle_length(block, k) for block in blocks]
    max_position = max(len(block) for block in blocks)
    return _survival_from_lengths(lengths, max_position)


def marginal_prefix_recall(rows: Iterable[Mapping[str, Any]], k: int) -> dict[int, float]:
    """Compute per-position marginal Top-K recall ``R_K(j)``."""

    grouped = _position_groups(rows)
    return {
        position: sum(
            int(_candidate_hit(row, k))
            for row in position_rows
        ) / len(position_rows)
        for position, position_rows in sorted(grouped.items())
    }


def independent_prefix_survival(rows: Iterable[Mapping[str, Any]], k: int) -> dict[int, float]:
    """Compute ``S_K^ind(j)=prod_i R_K(i)`` from marginal recalls."""

    recalls = marginal_prefix_recall(rows, k)
    product = 1.0
    result: dict[int, float] = {}
    for position in sorted(recalls):
        product *= recalls[position]
        result[position] = product
    return result


def conditional_prefix_survival(rows: Iterable[Mapping[str, Any]], k: int) -> dict[int, float | None]:
    """Compute ``c_K(j)=P(H_j=1 | H_1=...=H_{j-1}=1)``."""

    joint = joint_prefix_survival(rows, k)
    result: dict[int, float | None] = {}
    previous = 1.0
    for position, value in sorted(joint.items()):
        result[position] = value / previous if previous > 0.0 else None
        previous = value
    return result


def _mat_d(rows: Iterable[Mapping[str, Any]]) -> float | None:
    blocks = _blocks(_ok(rows))
    if not blocks:
        return None
    return sum(_observed_acceptance(block) for block in blocks) / len(blocks)


def _prefix_group_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('task_regime', 'other')}|{row.get('context_bin', 'unknown')}"


def analyze_prefix_oracle(
    rows: Iterable[Mapping[str, Any]],
    *,
    k_values: Sequence[int] = (1, 4, 8, 16),
    min_documents: int = 5,
    context_cap: int | None = None,
) -> dict[str, Any]:
    """Report joint prefix oracle survival and independence baselines by group."""

    values = _validate_k_values(k_values)
    usable = _ok(rows)
    if not usable:
        return {"status": "unavailable", "reason": "no_valid_trace_rows", "rows": 0}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        grouped[_prefix_group_key(row)].append(row)
    output: dict[str, Any] = {
        "status": "ok",
        "rows": len(usable),
        "k_values": list(values),
        "context_cap": context_cap,
        "candidate_membership": "tie_aware_logit_boundary",
        "tie_tolerance": 1e-6,
        "groups": {},
    }
    for key, group_rows in sorted(grouped.items()):
        blocks = _blocks(group_rows)
        documents = len({(str(row.get("dataset")), str(row["document_id"])) for row in group_rows})
        mat_d = _mat_d(group_rows)
        group_metrics: dict[str, Any] = {
            "rows": len(group_rows),
            "blocks": len(blocks),
            "documents": documents,
            "min_documents": min_documents,
            "mat_d": mat_d,
            "k_values": {},
        }
        for k in values:
            joint = joint_prefix_survival(group_rows, k)
            independent = independent_prefix_survival(group_rows, k)
            conditional = conditional_prefix_survival(group_rows, k)
            marginal = marginal_prefix_recall(group_rows, k)
            mat_oracle = sum(joint.values()) if joint else None
            positions = sorted(set(marginal) | set(joint) | set(independent))
            group_metrics["k_values"][str(k)] = {
                "marginal_recall": {str(position): value for position, value in marginal.items()},
                "joint_survival": {str(position): value for position, value in joint.items()},
                "independent_survival": {str(position): value for position, value in independent.items()},
                "conditional_survival": {str(position): value for position, value in conditional.items()},
                "prefix_gap": {
                    str(position): marginal.get(position, 0.0) - joint.get(position, 0.0)
                    for position in positions
                },
                "joint_to_independence": {
                    str(position): (
                        joint.get(position, 0.0) / independent[position]
                        if independent.get(position, 0.0) > 0.0 else None
                    )
                    for position in positions
                },
                "mat_oracle": mat_oracle,
                "oracle_headroom_over_dflash": (
                    mat_oracle - mat_d if mat_oracle is not None and mat_d is not None else None
                ),
            }
        k16 = group_metrics["k_values"].get("16")
        if k16 is not None and mat_d is not None and mat_d > 0.0:
            ratio = float(k16["mat_oracle"]) / mat_d if k16.get("mat_oracle") is not None else None
            group_metrics["oracle_ratio_k16"] = ratio
            group_metrics["e4_gate_k16"] = (
                "PASS" if ratio is not None and documents >= min_documents and ratio >= 1.5 else "FAIL"
            )
        else:
            group_metrics["oracle_ratio_k16"] = None
            group_metrics["e4_gate_k16"] = "INCONCLUSIVE"
        output["groups"][key] = group_metrics
    return output


def _document_block_stats(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], tuple[float, int]]:
    stats: dict[tuple[str, str], list[float | int]] = defaultdict(lambda: [0.0, 0])
    for block in _blocks(_ok(rows)):
        first = block[0]
        key = (str(first.get("dataset", "other")), str(first["document_id"]))
        stats[key][0] += float(_observed_acceptance(block))
        stats[key][1] += 1
    return {key: (float(value[0]), int(value[1])) for key, value in stats.items()}


def _mat_from_document_sample(
    stats: Mapping[tuple[str, str], tuple[float, int]],
    sampled_keys: Sequence[tuple[str, str]],
) -> float:
    total_acceptance = sum(stats[key][0] for key in sampled_keys)
    total_blocks = sum(stats[key][1] for key in sampled_keys)
    return total_acceptance / total_blocks if total_blocks else 0.0


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def _bootstrap_relative_drop(
    canonical_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    canonical_stats = _document_block_stats(canonical_rows)
    summary_stats = _document_block_stats(summary_rows)
    canonical_keys = sorted(canonical_stats)
    summary_keys = sorted(summary_stats)
    actual_canonical = _mat_from_document_sample(canonical_stats, canonical_keys)
    actual_summary = _mat_from_document_sample(summary_stats, summary_keys)
    actual_drop = (actual_canonical - actual_summary) / actual_canonical if actual_canonical else None
    if not canonical_keys or not summary_keys or samples <= 0:
        return {
            "canonical_documents": len(canonical_keys),
            "summarization_documents": len(summary_keys),
            "canonical_mat": actual_canonical,
            "summarization_mat": actual_summary,
            "relative_drop": actual_drop,
            "bootstrap_ci": None,
            "bootstrap_samples": samples,
            "bootstrap_successes": 0,
        }
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        sampled_canonical = [rng.choice(canonical_keys) for _ in canonical_keys]
        sampled_summary = [rng.choice(summary_keys) for _ in summary_keys]
        mat_canonical = _mat_from_document_sample(canonical_stats, sampled_canonical)
        mat_summary = _mat_from_document_sample(summary_stats, sampled_summary)
        if mat_canonical:
            values.append((mat_canonical - mat_summary) / mat_canonical)
    ci = [_quantile(values, 0.025), _quantile(values, 0.975)] if values else None
    return {
        "canonical_documents": len(canonical_keys),
        "summarization_documents": len(summary_keys),
        "canonical_mat": actual_canonical,
        "summarization_mat": actual_summary,
        "relative_drop": actual_drop,
        "bootstrap_ci": ci,
        "bootstrap_samples": samples,
        "bootstrap_successes": len(values),
    }


def _decision_from_drop(comparison: Mapping[str, Any], gate: float) -> str:
    relative_drop = comparison.get("relative_drop")
    ci = comparison.get("bootstrap_ci")
    if relative_drop is None or ci is None:
        return "INCONCLUSIVE"
    if float(relative_drop) >= gate and float(ci[0]) > 0.0:
        return "PASS"
    if float(relative_drop) < gate and float(ci[1]) < gate:
        return "FAIL"
    return "INCONCLUSIVE"


def analyze_matched_context(
    rows: Iterable[Mapping[str, Any]],
    *,
    context_cap: int = 1024,
    relative_drop_gate: float = 0.20,
    bootstrap_samples: int = 500,
    seed: int = 42,
    min_documents: int = 5,
) -> dict[str, Any]:
    """Compare canonical and summarization MAT at one matched context cap."""

    cap = int(context_cap)
    usable = [
        row for row in _ok(rows)
        if int(row.get("context_cap", row.get("context_length", -1))) == cap
    ]
    if not usable:
        return {
            "status": "unavailable",
            "reason": "no_rows_for_context_cap",
            "context_cap": cap,
            "rows": 0,
        }
    regimes = summarize_p1(usable)["regimes"]
    canonical_rows = [row for row in usable if str(row.get("task_regime")) == "canonical"]
    summary_rows = [row for row in usable if str(row.get("task_regime")) != "canonical"]
    pairwise: dict[str, Any] = {}
    summary_regimes = sorted({str(row.get("task_regime")) for row in summary_rows})
    for regime in ["summarization", *summary_regimes]:
        comparison_rows = summary_rows if regime == "summarization" else [
            row for row in summary_rows if str(row.get("task_regime")) == regime
        ]
        if not comparison_rows:
            continue
        comparison = _bootstrap_relative_drop(
            canonical_rows,
            comparison_rows,
            samples=bootstrap_samples,
            seed=seed,
        )
        comparison["decision"] = _decision_from_drop(comparison, relative_drop_gate)
        comparison["relative_drop_gate"] = relative_drop_gate
        comparison["min_documents"] = min_documents
        pairwise[regime] = comparison
    overall = pairwise.get("summarization", {})
    h1_gate = {
        **overall,
        "decision": overall.get("decision", "INCONCLUSIVE"),
        "context_cap": cap,
        "status": "ok",
        "reason": "matched_context_bootstrap_comparison",
    }
    return {
        "status": "ok",
        "context_cap": cap,
        "rows": len(usable),
        "regimes": regimes,
        "pairwise": pairwise,
        "h1_gate": h1_gate,
    }


def _softmax_entropy(logits: Sequence[float]) -> float | None:
    values = [float(value) for value in logits]
    if not values:
        return None
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    if denominator <= 0.0:
        return None
    return -sum((value / denominator) * math.log(value / denominator) for value in exponentials if value > 0.0)


def _rank_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rank_histogram: dict[str, int] = defaultdict(int)
    ranks: list[int] = []
    deficits: list[float] = []
    entropies: list[float] = []
    margins: list[float] = []
    for row in rows:
        candidates = [int(token) for token in row["candidate_token_ids"][:16]]
        target = int(row["target_token_id"])
        try:
            rank = candidates.index(target) + 1
        except ValueError:
            continue
        logits = row.get("candidate_logits")
        if not isinstance(logits, list) or len(logits) < rank:
            continue
        logits = [float(value) for value in logits[:len(candidates)]]
        ranks.append(rank)
        rank_histogram[str(rank)] += 1
        deficits.append(logits[0] - logits[rank - 1])
        entropy = _softmax_entropy(logits)
        if entropy is not None:
            entropies.append(entropy)
        if len(logits) >= 2:
            margins.append(logits[0] - logits[1])
    rank_conditioned = len(ranks)
    return {
        "rank_conditioned_rows": rank_conditioned,
        "rank_histogram": {str(rank): rank_histogram.get(str(rank), 0) for rank in range(1, 17)},
        "mrr": sum(1.0 / rank for rank in ranks) / rank_conditioned if rank_conditioned else None,
        "mean_target_rank": sum(ranks) / rank_conditioned if rank_conditioned else None,
        "mean_target_logit_deficit": sum(deficits) / len(deficits) if deficits else None,
        "mean_top16_entropy": sum(entropies) / len(entropies) if entropies else None,
        "mean_top1_top2_margin": sum(margins) / len(margins) if margins else None,
    }


def analyze_rank_ambiguity(
    rows: Iterable[Mapping[str, Any]],
    *,
    context_cap: int | None = None,
) -> dict[str, Any]:
    """Measure target rank and logit ambiguity, conditioning on Top-16 hits."""

    usable = _ok(rows)
    if context_cap is not None:
        cap = int(context_cap)
        usable = [row for row in usable if int(row.get("context_cap", row.get("context_length", -1))) == cap]
    if not usable:
        return {"status": "unavailable", "reason": "no_valid_trace_rows", "rows": 0, "context_cap": context_cap}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        grouped[str(row.get("task_regime", "other"))].append(row)
    regimes: dict[str, Any] = {}
    for regime, regime_rows in sorted(grouped.items()):
        hits = [
            row for row in regime_rows
            if int(row["target_token_id"]) in [int(token) for token in row["candidate_token_ids"][:16]]
        ]
        metrics = _rank_metrics(hits)
        metrics.update({
            "rows": len(regime_rows),
            "top16_hit_rows": len(hits),
            "top16_miss_rows": len(regime_rows) - len(hits),
            "recall_at_16": len(hits) / len(regime_rows) if regime_rows else None,
        })
        regimes[regime] = metrics
    return {
        "status": "ok",
        "context_cap": context_cap,
        "rows": len(usable),
        "regimes": regimes,
    }
