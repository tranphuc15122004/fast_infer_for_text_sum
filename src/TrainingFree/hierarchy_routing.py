"""Coarse-to-fine source routing and offline audit metrics for RECAP-KV V3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from .hierarchy import (
    Cluster,
    SourceHierarchy,
    cluster_log_bounds,
    cluster_log_bounds_batch,
)


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


@dataclass(frozen=True)
class RoutingDecision:
    active_block_indices: tuple[int, ...]
    candidate_region_indices: tuple[int, ...]
    upper_missed_mass_bound: float
    routing_representatives: int


def _validate_budget(mass_budget: float) -> float:
    budget = float(mass_budget)
    if not math.isfinite(budget) or not 0.0 <= budget <= 1.0:
        raise ValueError("mass_budget must be finite and in [0, 1]")
    return budget


def _ordered_bounds(
    query: Any, keys: Any, clusters: Sequence[Cluster]
) -> tuple[list[tuple[int, float, float]], float]:
    uppers, lowers = cluster_log_bounds_batch(query, keys, clusters)
    values = [(index, uppers[index], lowers[index]) for index in range(len(clusters))]
    values.sort(key=lambda item: (-item[1], item[0]))
    return values, _logsumexp([item[2] for item in values])


def _ratio(omitted_upper: Sequence[float], lower_total: float) -> float:
    if not omitted_upper:
        return 0.0
    if lower_total == float("-inf"):
        return 1.0
    value = math.exp(min(0.0, _logsumexp(omitted_upper) - lower_total))
    return min(1.0, max(0.0, value))


def route_query(
    hierarchy: SourceHierarchy,
    query: Any,
    *,
    mass_budget: float,
) -> RoutingDecision:
    """Route a query using only representatives and geometric bounds."""

    budget = _validate_budget(mass_budget)
    if budget == 0.0:
        return RoutingDecision(
            active_block_indices=tuple(range(len(hierarchy.blocks))),
            candidate_region_indices=tuple(range(len(hierarchy.regions))),
            upper_missed_mass_bound=0.0,
            routing_representatives=hierarchy.total_representatives,
        )
    region_bounds, lower_total = _ordered_bounds(query, hierarchy.keys, hierarchy.regions)
    block_uppers, _ = cluster_log_bounds_batch(query, hierarchy.keys, hierarchy.blocks)
    best: tuple[int, int, tuple[int, ...], tuple[int, ...], float] | None = None

    for region_prefix in range(1, len(region_bounds) + 1):
        candidate_regions = tuple(item[0] for item in region_bounds[:region_prefix])
        candidate_set = set(candidate_regions)
        candidate_blocks = [
            index
            for index, block in enumerate(hierarchy.blocks)
            if any(block_index == index for region_index in candidate_set for block_index in hierarchy.regions[region_index].children)
        ]
        block_bounds = [(block_index, block_uppers[block_index]) for block_index in candidate_blocks]
        block_bounds.sort(key=lambda item: (-item[1], item[0]))

        for active_count in range(len(block_bounds) + 1):
            active = tuple(sorted(index for index, _ in block_bounds[:active_count]))
            active_set = set(active)
            omitted_upper = [
                upper for index, upper, _ in region_bounds if index not in candidate_set
            ]
            omitted_upper.extend(
                upper for index, upper in block_bounds if index not in active_set
            )
            bound = _ratio(omitted_upper, lower_total)
            if bound <= budget + 1e-12:
                candidate = (
                    sum(hierarchy.blocks[index].size for index in active),
                    region_prefix,
                    candidate_regions,
                    active,
                    bound,
                )
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
                break

    if best is None:
        active = tuple(range(len(hierarchy.blocks)))
        candidate_regions = tuple(range(len(hierarchy.regions)))
        bound = 0.0
        region_prefix = len(candidate_regions)
    else:
        _, region_prefix, candidate_regions, active, bound = best
        del region_prefix

    refined_representatives = sum(
        len(hierarchy.blocks[index].representatives)
        for region_index in candidate_regions
        for index in hierarchy.regions[region_index].children
    )
    routing_representatives = sum(len(region.representatives) for region in hierarchy.regions)
    routing_representatives += refined_representatives
    return RoutingDecision(
        active_block_indices=tuple(active),
        candidate_region_indices=tuple(candidate_regions),
        upper_missed_mass_bound=float(bound),
        routing_representatives=int(routing_representatives),
    )


def evaluate_routing_step(
    hierarchy: SourceHierarchy,
    query: Any,
    decision: RoutingDecision,
    attention_block_masses: Sequence[float],
) -> dict[str, float | int]:
    """Audit a routing decision against exact attention supplied by the caller."""

    masses = [float(value) for value in attention_block_masses]
    if len(masses) != len(hierarchy.blocks):
        raise ValueError("attention_block_masses width must match hierarchy blocks")
    if any(not math.isfinite(value) or value < 0.0 for value in masses):
        raise ValueError("attention_block_masses must be finite and non-negative")
    active = set(decision.active_block_indices)
    if any(index < 0 or index >= len(hierarchy.blocks) for index in active):
        raise ValueError("active block index is outside hierarchy")
    missed = sum(value for index, value in enumerate(masses) if index not in active)
    import torch

    upper_values, _ = cluster_log_bounds_batch(query, hierarchy.keys, hierarchy.blocks)
    vector = query.detach().float() if hasattr(query, "detach") else torch.as_tensor(query).float()
    keys = hierarchy.keys.detach().float() if hasattr(hierarchy.keys, "detach") else torch.as_tensor(hierarchy.keys).float()
    logits = keys @ vector / math.sqrt(float(keys.shape[1]))
    exact_values = torch.stack(
        [torch.logsumexp(logits[block.start : block.end], dim=0) for block in hierarchy.blocks]
    )
    violations = int(
        (exact_values > torch.tensor(upper_values, device=exact_values.device) + 1e-5).sum().item()
    )
    active_tokens = sum(hierarchy.blocks[index].size for index in active)
    return {
        "missed_attention_mass": min(1.0, max(0.0, missed)),
        "exact_expansion_fraction": active_tokens / hierarchy.source_tokens,
        "upper_missed_mass_bound": float(decision.upper_missed_mass_bound),
        "upper_bound_violations": violations,
        "routing_representatives": decision.routing_representatives,
        "routing_fraction": decision.routing_representatives / hierarchy.source_tokens,
        "index_overhead": hierarchy.index_overhead,
        "full_qk_tokens": hierarchy.source_tokens,
        "active_qk_tokens": active_tokens,
    }
