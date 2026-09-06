"""E14/E14b decomposition of marginal and joint Top-K lattice quality."""

from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .metrics import _blocks
from .prefix_gap import _candidate_hit, prefix_oracle_length


def _ok(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _regime(row: Mapping[str, Any]) -> str:
    return str(row.get("task_regime", row.get("dataset", "other")))


def _position_hits(rows: Sequence[Mapping[str, Any]], k: int) -> dict[int, list[bool]]:
    result: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        position = int(row["draft_position"])
        result[position].append(bool(_candidate_hit(row, k)))
    return dict(result)


def lattice_stats(rows: Iterable[Mapping[str, Any]], *, k: int = 16, max_position: int | None = None) -> dict[str, Any]:
    """Return marginal R, joint J and normalized coherence C for one regime."""

    usable = _ok(rows)
    blocks = _blocks(usable)
    if not blocks:
        return {"status": "unavailable", "reason": "no_blocks"}
    max_pos = max_position or max(len(block) for block in blocks)
    max_pos = min(max_pos, max(len(block) for block in blocks))
    position_hits = _position_hits(usable, k)
    marginal = {
        str(position): sum(values) / len(values)
        for position, values in sorted(position_hits.items())
        if position <= max_pos and values
    }
    joint: dict[str, float] = {}
    oracle_lengths: list[int] = []
    for position in range(1, max_pos + 1):
        eligible = [block for block in blocks if len(block) >= position]
        joint[str(position)] = (
            sum(prefix_oracle_length(block, k) >= position for block in eligible) / len(eligible)
            if eligible else 0.0
        )
    products: dict[str, float] = {}
    coherence: dict[str, float | None] = {}
    product = 1.0
    for position in range(1, max_pos + 1):
        product *= float(marginal.get(str(position), 0.0))
        products[str(position)] = product
        coherence[str(position)] = joint[str(position)] / product if product > 0.0 else None
    for block in blocks:
        oracle_lengths.append(min(prefix_oracle_length(block, k), max_pos))
    return {
        "status": "ok",
        "k": k,
        "max_position": max_pos,
        "rows": len(usable),
        "blocks": len(blocks),
        "documents": len({str(row["document_id"]) for row in usable}),
        "marginal_recall": marginal,
        "joint_survival": joint,
        "independent_survival": products,
        "coherence": coherence,
        "mat_o16": sum(joint.values()),
        "oracle_lengths": oracle_lengths,
    }


def marginal_joint_decomposition(
    canonical: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Counterfactually hold canonical coherence while changing summary R."""

    max_position = min(int(canonical["max_position"]), int(summary["max_position"]))
    counterfactual: dict[str, float] = {}
    product = 1.0
    for position in range(1, max_position + 1):
        product *= float(summary["marginal_recall"].get(str(position), 0.0))
        canonical_coherence = canonical["coherence"].get(str(position))
        counterfactual[str(position)] = (
            float(canonical_coherence) * product if canonical_coherence is not None else 0.0
        )
    mat_canonical = float(canonical["mat_o16"])
    mat_summary = float(summary["mat_o16"])
    mat_marginal_cf = sum(counterfactual.values())
    total_degradation = mat_canonical - mat_summary
    marginal_component = mat_canonical - mat_marginal_cf
    joint_component = mat_marginal_cf - mat_summary
    return {
        "max_position": max_position,
        "mat_o16_canonical": mat_canonical,
        "mat_o16_summary": mat_summary,
        "mat_o16_marginal_counterfactual": mat_marginal_cf,
        "total_degradation": total_degradation,
        "marginal_component": marginal_component,
        "joint_component": joint_component,
        "marginal_fraction": marginal_component / total_degradation if total_degradation else None,
        "joint_fraction": joint_component / total_degradation if total_degradation else None,
        "counterfactual_joint_survival": counterfactual,
    }


def analyze_decomposition(
    rows: Iterable[Mapping[str, Any]],
    *,
    k: int = 16,
    canonical_name: str = "canonical",
    max_position: int | None = None,
) -> dict[str, Any]:
    usable = _ok(rows)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        groups[_regime(row)].append(row)
    stats = {name: lattice_stats(group, k=k, max_position=max_position) for name, group in sorted(groups.items())}
    canonical = stats.get(canonical_name)
    output: dict[str, Any] = {
        "status": "ok" if canonical and canonical.get("status") == "ok" else "unavailable",
        "experiment": "E14",
        "candidate_k": k,
        "groups": stats,
        "decomposition": {},
    }
    if canonical and canonical.get("status") == "ok":
        for name, summary in stats.items():
            if name == canonical_name or summary.get("status") != "ok":
                continue
            output["decomposition"][name] = marginal_joint_decomposition(canonical, summary)
    else:
        output["reason"] = "canonical_group_missing"
    return output


def _quantile_edges(values: Sequence[float], bins: int) -> list[float]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered or bins < 1:
        return []
    edges: list[float] = []
    for index in range(1, bins):
        position = int(round(index * (len(ordered) - 1) / bins))
        edge = ordered[position]
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return edges


def _entropy_prefix_records(rows: Sequence[Mapping[str, Any]], k: int, max_position: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block in _blocks(rows):
        ordered = sorted(block, key=lambda row: int(row["draft_position"]))
        hits: list[bool] = []
        entropies: list[float] = []
        for row in ordered[:max_position]:
            entropy = row.get("target_entropy")
            if entropy is None or not math.isfinite(float(entropy)):
                break
            hits.append(bool(_candidate_hit(row, k)))
            entropies.append(float(entropy))
        for position in range(1, len(hits) + 1):
            records.append({
                "position": position,
                "hit": all(hits[:position]),
                # Match the intrinsic target difficulty at the same draft
                # position.  Cumulative-prefix entropy would partly encode
                # the outcome we are trying to control for and can produce
                # non-comparable bins across regimes.
                "entropy": entropies[position - 1],
                "document_id": str(ordered[0]["document_id"]),
            })
    return records


def _bin(value: float, edges: Sequence[float]) -> int:
    return bisect.bisect_right(list(edges), float(value))


def entropy_standardized_stats(
    rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
    *,
    k: int = 16,
    bins: int = 5,
    max_position: int = 15,
) -> dict[str, Any]:
    """Compare prefix survival after equal-weight entropy-bin standardization."""

    current = _entropy_prefix_records(_ok(rows), k, max_position)
    reference = _entropy_prefix_records(_ok(reference_rows), k, max_position)
    edges = _quantile_edges([record["entropy"] for record in current + reference], bins)
    current_by: dict[tuple[int, int], list[bool]] = defaultdict(list)
    reference_by: dict[tuple[int, int], list[bool]] = defaultdict(list)
    for record in current:
        current_by[(int(record["position"]), _bin(record["entropy"], edges))].append(bool(record["hit"]))
    for record in reference:
        reference_by[(int(record["position"]), _bin(record["entropy"], edges))].append(bool(record["hit"]))
    current_survival: dict[str, float] = {}
    reference_survival: dict[str, float] = {}
    shared_bins: dict[str, int] = {}
    for position in range(1, max_position + 1):
        current_bins = {bin_id for (pos, bin_id) in current_by if pos == position}
        reference_bins = {bin_id for (pos, bin_id) in reference_by if pos == position}
        common = sorted(current_bins & reference_bins)
        shared_bins[str(position)] = len(common)
        # Standardize both regimes to the canonical entropy-bin distribution.
        # This is a descriptive control; bins missing in either regime are
        # omitted and the remaining reference weights are renormalized.
        reference_total = sum(len(reference_by[(position, bin_id)]) for bin_id in common)
        weights = {
            bin_id: len(reference_by[(position, bin_id)]) / reference_total
            for bin_id in common
        } if reference_total else {}
        current_survival[str(position)] = (
            sum(
                weights[bin_id]
                * (sum(current_by[(position, bin_id)]) / len(current_by[(position, bin_id)]))
                for bin_id in common
            )
            if common and weights else None
        )
        reference_survival[str(position)] = (
            sum(
                weights[bin_id]
                * (sum(reference_by[(position, bin_id)]) / len(reference_by[(position, bin_id)]))
                for bin_id in common
            )
            if common and weights else None
        )
    current_values = [value for value in current_survival.values() if value is not None]
    reference_values = [value for value in reference_survival.values() if value is not None]
    return {
        "status": "ok" if current and reference and edges else "unavailable",
        "bins": bins,
        "entropy_edges": edges,
        "current_prefix_survival": current_survival,
        "reference_prefix_survival": reference_survival,
        "shared_bins_by_position": shared_bins,
        "mat_entropy_standardized": sum(current_values),
        "mat_reference_entropy_standardized": sum(reference_values),
        "entropy_standardized_gap": sum(reference_values) - sum(current_values),
        "current_records": len(current),
        "reference_records": len(reference),
    }


def bootstrap_decomposition(
    rows: Iterable[Mapping[str, Any]],
    *,
    samples: int = 300,
    seed: int = 42,
    k: int = 16,
) -> dict[str, Any]:
    """Document bootstrap for E14 decomposition components."""

    usable = _ok(rows)
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in usable:
        grouped[_regime(row)][str(row["document_id"])].append(row)
    canonical_docs = grouped.get("canonical", {})
    if len(canonical_docs) < 2:
        return {"status": "inconclusive", "reason": "insufficient_canonical_documents"}
    rng = random.Random(seed)
    output: dict[str, Any] = {"status": "ok", "samples": samples, "seed": seed, "regimes": {}}
    for regime, docs in grouped.items():
        if regime == "canonical" or len(docs) < 2:
            continue
        component_values: dict[str, list[float]] = defaultdict(list)
        canonical_keys = list(canonical_docs)
        summary_keys = list(docs)
        for _ in range(samples):
            sampled_canonical = [key for key in (rng.choice(canonical_keys) for _ in canonical_keys)]
            sampled_summary = [key for key in (rng.choice(summary_keys) for _ in summary_keys)]
            canonical_rows = [row for key in sampled_canonical for row in canonical_docs[key]]
            summary_rows = [row for key in sampled_summary for row in docs[key]]
            canonical_stats = lattice_stats(canonical_rows, k=k)
            summary_stats = lattice_stats(summary_rows, k=k)
            decomposition = marginal_joint_decomposition(canonical_stats, summary_stats)
            for key in ("marginal_component", "joint_component", "marginal_fraction", "joint_fraction"):
                value = decomposition.get(key)
                if value is not None and math.isfinite(float(value)):
                    component_values[key].append(float(value))
        regime_result: dict[str, Any] = {"documents": len(docs), "components": {}}
        for key, values in component_values.items():
            ordered = sorted(values)
            regime_result["components"][key] = {
                "mean": sum(values) / len(values),
                "ci95": [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]],
            }
        output["regimes"][regime] = regime_result
    return output
