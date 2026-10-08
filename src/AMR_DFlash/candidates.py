"""Deterministic candidate context supports for acceptance label generation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSet:
    candidate_id: str
    method: str
    positions: tuple[int, ...]


def _stable_id(positions: tuple[int, ...]) -> str:
    payload = ",".join(str(value) for value in positions).encode("ascii")
    return hashlib.sha1(payload).hexdigest()[:12]


def generate_candidate_sets(
    *,
    context_length: int,
    raw_budget: int,
    local_window: int,
    seed: int,
    max_candidates: int = 12,
    chunk_size: int = 32,
    max_swaps: int = 8,
) -> list[CandidateSet]:
    """Create fixed-cardinality, guard-preserving deployable candidate sets.

    The methods use only positions in the observed prefix.  The local recent
    guard is included in the raw budget.  A context shorter than the budget
    yields one full-context candidate because it has no preference headroom.
    """
    if context_length < 0 or raw_budget < 0 or local_window < 0:
        raise ValueError("context length and memory budgets must be non-negative")
    if local_window > raw_budget:
        raise ValueError("local_window must fit inside raw_budget")
    if max_candidates < 1 or chunk_size < 1 or max_swaps < 0:
        raise ValueError("candidate limits must be positive (swaps may be zero)")
    if context_length == 0 or raw_budget == 0:
        return []
    budget = min(context_length, raw_budget)
    if context_length <= raw_budget:
        positions = tuple(range(context_length))
        return [CandidateSet("full", "full", positions)]

    guard_count = min(local_window, budget)
    guard_start = context_length - guard_count
    guard = tuple(range(guard_start, context_length))
    capacity = budget - guard_count
    historical = list(range(guard_start))
    rng = random.Random(f"amr-candidates:{seed}:{context_length}:{raw_budget}")
    proposals: list[tuple[str, tuple[int, ...]]] = []

    def append(method: str, history: list[int] | tuple[int, ...]) -> None:
        chosen = tuple(sorted(set(int(value) for value in history).union(guard)))
        if len(chosen) != budget:
            return
        proposals.append((method, chosen))

    recent_history = historical[-capacity:] if capacity else []
    append("recent", recent_history)
    append("head", historical[:capacity])
    middle_start = max(0, (len(historical) - capacity) // 2)
    append("middle", historical[middle_start : middle_start + capacity])
    if capacity:
        starts = max(1, len(historical) - capacity + 1)
        window_start = rng.randrange(starts)
        append("random_window", historical[window_start : window_start + capacity])
        append("random_scattered", rng.sample(historical, capacity))

    if capacity and max_swaps:
        base = set(recent_history)
        available_outside = [value for value in historical if value not in base]
        for swap_index in range(max_swaps):
            if not available_outside:
                break
            selected = sorted(base)
            selected_chunks: dict[int, list[int]] = {}
            for value in selected:
                selected_chunks.setdefault(value // chunk_size, []).append(value)
            removable = [chunk for chunk in selected_chunks.values() if chunk]
            if not removable:
                break
            remove_chunk = removable[rng.randrange(len(removable))]
            remove_count = min(len(remove_chunk), len(available_outside))
            add_values = rng.sample(available_outside, remove_count)
            next_base = base.difference(remove_chunk[:remove_count])
            next_base.update(add_values)
            append(f"chunk_swap_{swap_index:02d}", sorted(next_base))

    seen: set[tuple[int, ...]] = set()
    candidates: list[CandidateSet] = []
    for method, positions in proposals:
        if positions in seen:
            continue
        seen.add(positions)
        candidates.append(CandidateSet(_stable_id(positions), method, positions))
        if len(candidates) >= max_candidates:
            break
    return candidates
