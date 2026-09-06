"""Target–draft alignment and prefix-utility diagnostics for E11/E12."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .metrics import _blocks, _observed_acceptance
from .prefix_gap import prefix_oracle_length


def _ok(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(float(value) for value in values)
    weights = [math.exp(float(value) - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights] if total > 0.0 else [1.0 / len(values)] * len(values)


def _js(left: Sequence[float], right: Sequence[float]) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]
    result = 0.0
    for values in (left, right):
        result += 0.5 * sum(
            value * math.log(value / middle)
            for value, middle in zip(values, midpoint)
            if value > 0.0 and middle > 0.0
        )
    return result


def _rank(values: Sequence[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    result = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        result[index] = rank
    return result


def _rank_corr(left: Sequence[float], right: Sequence[float]) -> tuple[float | None, float | None]:
    if len(left) < 2 or len(left) != len(right):
        return None, None
    left_ranks = _rank(left)
    right_ranks = _rank(right)
    n = len(left_ranks)
    left_mean = sum(left_ranks) / n
    right_mean = sum(right_ranks) / n
    left_var = sum((value - left_mean) ** 2 for value in left_ranks)
    right_var = sum((value - right_mean) ** 2 for value in right_ranks)
    if left_var <= 0.0 or right_var <= 0.0:
        return None, None
    spearman = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks)) / math.sqrt(left_var * right_var)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            product = (float(left[i]) - float(left[j])) * (float(right[i]) - float(right[j]))
            if product > 0.0:
                concordant += 1
            elif product < 0.0:
                discordant += 1
    denominator = n * (n - 1) / 2
    kendall = (concordant - discordant) / denominator if denominator else None
    return kendall, spearman


def _inversion_rate(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    inversions = 0
    denominator = len(left) * (len(left) - 1) // 2
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            if float(left[i]) > float(left[j]) and float(right[i]) < float(right[j]):
                inversions += 1
    return inversions / denominator if denominator else None


def row_alignment(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Compute E11 statistics for one row's recorded candidate lattice."""

    draft = row.get("candidate_logits")
    target = row.get("target_candidate_logits")
    candidates = row.get("candidate_token_ids")
    if not isinstance(draft, list) or not isinstance(target, list) or not isinstance(candidates, list):
        return None
    if len(draft) != len(target) or len(draft) != len(candidates) or len(draft) < 2:
        return None
    draft_values = [float(value) for value in draft]
    target_values = [float(value) for value in target]
    kendall, spearman = _rank_corr(draft_values, target_values)
    target_token = int(row["target_token_id"])
    candidate_ids = [int(value) for value in candidates]
    target_rank = candidate_ids.index(target_token) + 1 if target_token in candidate_ids else None
    return {
        "kendall_tau": kendall,
        "spearman_rho": spearman,
        "pairwise_inversion_rate": _inversion_rate(draft_values, target_values),
        "js_divergence": _js(_softmax(draft_values), _softmax(target_values)),
        "target_in_lattice": target_rank is not None,
        "target_rank_draft": target_rank,
        "draft_top1_is_target": candidate_ids[0] == target_token,
        "dataset": str(row.get("task_regime", row.get("dataset", "other"))),
        "document_id": str(row.get("document_id")),
        "sample_id": str(row.get("sample_id")),
        "round_index": int(row.get("round_index", 0)),
        "draft_position": int(row.get("draft_position", 0)),
    }


def _mean(values: Sequence[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def analyze_alignment(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Run E11 aggregate rank-alignment anatomy."""

    measured = [item for row in _ok(rows) if (item := row_alignment(row)) is not None]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in measured:
        by_dataset[item["dataset"]].append(item)
    output: dict[str, Any] = {
        "status": "ok" if measured else "unavailable",
        "experiment": "E11",
        "rows_with_target_logits": len(measured),
        "datasets": {},
    }
    for dataset, items in sorted(by_dataset.items()):
        output["datasets"][dataset] = {
            "rows": len(items),
            "documents": len({item["document_id"] for item in items}),
            "mean_kendall_tau": _mean([item["kendall_tau"] for item in items]),
            "mean_spearman_rho": _mean([item["spearman_rho"] for item in items]),
            "mean_pairwise_inversion_rate": _mean([item["pairwise_inversion_rate"] for item in items]),
            "mean_js_divergence": _mean([item["js_divergence"] for item in items]),
            "target_in_lattice_rate": sum(item["target_in_lattice"] for item in items) / len(items),
            "draft_top1_target_rate": sum(item["draft_top1_is_target"] for item in items) / len(items),
            "mean_target_rank_draft": _mean([
                item["target_rank_draft"] for item in items if item["target_rank_draft"] is not None
            ]),
        }
    if not measured:
        output["reason"] = "missing_target_candidate_logits"
    output["document_bootstrap"] = bootstrap_alignment_comparison(rows, samples=1000, seed=42)
    return output


def bootstrap_alignment_comparison(
    rows: Iterable[Mapping[str, Any]],
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap document-level alignment differences against canonical.

    The canonical set contains only eight documents, so this is reported as
    uncertainty information rather than a formal population estimate.
    """

    measured = [item for row in _ok(rows) if (item := row_alignment(row)) is not None]
    by_regime_document: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in measured:
        by_regime_document[item["dataset"]][item["document_id"]].append(item)
    canonical = by_regime_document.get("canonical", {})
    metrics = ("kendall_tau", "spearman_rho", "pairwise_inversion_rate", "js_divergence")

    def document_means(document_map: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[float]]:
        return {
            metric: [
                sum(float(item[metric]) for item in items) / len(items)
                for items in document_map.values()
            ]
            for metric in metrics
        }

    canonical_means = document_means(canonical)
    if len(canonical_means.get("kendall_tau", [])) < 2:
        return {"status": "inconclusive", "reason": "insufficient_canonical_documents"}
    rng = random.Random(seed)
    output: dict[str, Any] = {
        "status": "ok",
        "samples": samples,
        "seed": seed,
        "reference": "canonical",
        "regimes": {},
    }
    for regime, document_map in sorted(by_regime_document.items()):
        if regime == "canonical" or len(document_map) < 2:
            continue
        summary_means = document_means(document_map)
        if not summary_means["kendall_tau"]:
            continue
        differences = {metric: [] for metric in metrics}
        canonical_keys = list(range(len(canonical_means[metrics[0]])))
        summary_keys = list(range(len(summary_means[metrics[0]])))
        for _ in range(samples):
            canonical_sample = [rng.choice(canonical_keys) for _ in canonical_keys]
            summary_sample = [rng.choice(summary_keys) for _ in summary_keys]
            for metric in metrics:
                canonical_mean = sum(canonical_means[metric][index] for index in canonical_sample) / len(canonical_sample)
                summary_mean = sum(summary_means[metric][index] for index in summary_sample) / len(summary_sample)
                differences[metric].append(summary_mean - canonical_mean)
        regime_result: dict[str, Any] = {"documents": len(document_map), "metrics": {}}
        for metric, values in differences.items():
            ordered = sorted(values)
            lower = ordered[max(0, int(0.025 * (len(ordered) - 1)))]
            upper = ordered[min(len(ordered) - 1, int(0.975 * (len(ordered) - 1)))]
            regime_result["metrics"][metric] = {
                "mean_difference_summary_minus_canonical": sum(values) / len(values),
                "ci95": [lower, upper],
            }
        output["regimes"][regime] = regime_result
    return output


def _correlation(left: Sequence[float], right: Sequence[float]) -> dict[str, float | None]:
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return {"pearson_r": None, "spearman_rho": None, "n": len(pairs)}
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((value - x_mean) ** 2 for value in xs)
    y_var = sum((value - y_mean) ** 2 for value in ys)
    pearson = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / math.sqrt(x_var * y_var) if x_var and y_var else None
    x_rank = _rank(xs)
    y_rank = _rank(ys)
    xr = sum(x_rank) / len(x_rank)
    yr = sum(y_rank) / len(y_rank)
    xv = sum((value - xr) ** 2 for value in x_rank)
    yv = sum((value - yr) ** 2 for value in y_rank)
    spearman = sum((x - xr) * (y - yr) for x, y in zip(x_rank, y_rank)) / math.sqrt(xv * yv) if xv and yv else None
    return {"pearson_r": pearson, "spearman_rho": spearman, "n": len(pairs)}


def _hazard(lengths: Sequence[int], max_position: int) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for position in range(1, max_position + 1):
        eligible = [length for length in lengths if length >= position - 1]
        result[str(position)] = (
            sum(length < position for length in eligible) / len(eligible) if eligible else None
        )
    return result


def _survival(lengths: Sequence[int], max_position: int) -> dict[str, float]:
    return {
        str(position): sum(length >= position for length in lengths) / len(lengths)
        for position in range(1, max_position + 1)
    }


def analyze_alignment_utility(rows: Iterable[Mapping[str, Any]], *, max_prefix: int = 16) -> dict[str, Any]:
    """Run E12 alignment-to-prefix utility and first-rejection analysis."""

    usable = _ok(rows)
    blocks = [block for block in _blocks(usable) if all(row_alignment(row) is not None for row in block)]
    by_dataset: dict[str, list[list[Mapping[str, Any]]]] = defaultdict(list)
    for block in blocks:
        by_dataset[str(block[0].get("task_regime", block[0].get("dataset", "other")))].append(block)
    output: dict[str, Any] = {
        "status": "ok" if blocks else "unavailable",
        "experiment": "E12",
        "max_prefix": max_prefix,
        "datasets": {},
    }
    for dataset, dataset_blocks in sorted(by_dataset.items()):
        d_lengths = [min(_observed_acceptance(block), max_prefix) for block in dataset_blocks]
        o_lengths = [min(prefix_oracle_length(block, max_prefix), max_prefix) for block in dataset_blocks]
        block_metrics: list[dict[str, Any]] = []
        for block, d_length, o_length in zip(dataset_blocks, d_lengths, o_lengths):
            alignments = [row_alignment(row) for row in block]
            alignment_values = [item["kendall_tau"] for item in alignments if item and item["kendall_tau"] is not None]
            first_m = {}
            for m in (1, 2, 3, 4):
                prefix = alignment_values[:m]
                first_m[str(m)] = _mean(prefix) if len(prefix) == m else None
            block_metrics.append({
                "document_id": str(block[0]["document_id"]),
                "sample_id": str(block[0]["sample_id"]),
                "round_index": int(block[0]["round_index"]),
                "mat_d_block": d_length,
                "mat_o16_block": o_length,
                "oracle_gap_block": o_length - d_length,
                "alignment_mean": _mean(alignment_values),
                "alignment_first_m": first_m,
            })
        dataset_result: dict[str, Any] = {
            "blocks": len(dataset_blocks),
            "documents": len({item["document_id"] for item in block_metrics}),
            "mat_d": _mean(d_lengths),
            "mat_o16": _mean(o_lengths),
            "oracle_gap": _mean([o - d for d, o in zip(d_lengths, o_lengths)]),
            "dflash_survival": _survival(d_lengths, max_prefix),
            "oracle_survival": _survival(o_lengths, max_prefix),
            "dflash_first_rejection_hazard": _hazard(d_lengths, max_prefix),
            "oracle_first_rejection_hazard": _hazard(o_lengths, max_prefix),
            "alignment_vs_mat_d": {},
            "alignment_vs_oracle_gap": {},
            "alignment_vs_mat_d_first_m": {},
            "block_metrics": block_metrics,
        }
        alignment_all = [item["alignment_mean"] for item in block_metrics if item["alignment_mean"] is not None]
        mat_d_all = [item["mat_d_block"] for item in block_metrics if item["alignment_mean"] is not None]
        gap_all = [item["oracle_gap_block"] for item in block_metrics if item["alignment_mean"] is not None]
        dataset_result["alignment_vs_mat_d"] = _correlation(alignment_all, mat_d_all)
        dataset_result["alignment_vs_oracle_gap"] = _correlation(alignment_all, gap_all)
        for m in (1, 2, 3, 4):
            keyed = [item for item in block_metrics if item["alignment_first_m"][str(m)] is not None]
            xs = [item["alignment_first_m"][str(m)] for item in keyed]
            dataset_result["alignment_vs_mat_d_first_m"][str(m)] = _correlation(
                xs, [item["mat_d_block"] for item in keyed]
            )
        dataset_result["utility_loss_by_position"] = {
            str(position): dataset_result["oracle_survival"].get(str(position), 0.0)
            - dataset_result["dflash_survival"].get(str(position), 0.0)
            for position in range(1, max_prefix + 1)
        }
        output["datasets"][dataset] = dataset_result
    if not blocks:
        output["reason"] = "missing_target_candidate_logits_or_empty_blocks"
    return output
