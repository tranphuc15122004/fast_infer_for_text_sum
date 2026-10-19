"""Token-exact context masks used by A1/A2/A3.

Positions always refer to the original target sequence. The returned tuples are
sorted, unique absolute logical positions so DraftContextBank can gather KV
without changing RoPE coordinates.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

from TrainingFree.context_adaptive.types import PromptLayout, Selection


def _position_set(bank_length: int, positions: Sequence[int], *, name: str) -> set[int]:
    values = {int(position) for position in positions}
    invalid = sorted(position for position in values if position < 0 or position >= bank_length)
    if invalid:
        raise ValueError(f"{name} contains positions outside [0, {bank_length}): {invalid[:8]}")
    return values


def global_protected_positions(
    layout: PromptLayout,
    *,
    bank_length: int,
    source_anchors: int = 64,
    processed_output_start: int | None = None,
    recent_output: int = 256,
    protect_recent: bool = False,
) -> tuple[int, ...]:
    """Return G0 plus the optional recent-output protection used by A2/A3."""
    if source_anchors < 0 or recent_output < 0:
        raise ValueError("source_anchors and recent_output must be nonnegative")
    protected = _position_set(bank_length, layout.global_positions, name="global positions")
    source = _position_set(bank_length, layout.source_positions, name="source positions")
    protected.update(sorted(source)[:source_anchors])
    if protect_recent and recent_output:
        if processed_output_start is None:
            raise ValueError("processed_output_start is required when protecting recent output")
        start = max(int(processed_output_start), bank_length - recent_output)
        protected.update(range(start, bank_length))
    return tuple(sorted(protected))


def build_region_positions(
    region: str,
    *,
    layout: PromptLayout,
    bank_length: int,
    processed_output_start: int,
    source_anchors: int = 64,
    recent_output: int = 256,
) -> tuple[int, ...]:
    """Build the four A1 control masks using G0 and processed output only."""
    if not 0 <= processed_output_start <= bank_length:
        raise ValueError("processed_output_start must be within the draft bank")
    global_set = set(global_protected_positions(
        layout,
        bank_length=bank_length,
        source_anchors=source_anchors,
    ))
    source = _position_set(bank_length, layout.source_positions, name="source positions")
    recent_start = max(processed_output_start, bank_length - recent_output)
    recent = set(range(recent_start, bank_length))
    if region == "full":
        keep = set(range(bank_length))
    elif region == "source_only":
        keep = global_set | source
    elif region == "recent_only":
        keep = global_set | recent
    elif region == "source_recent":
        keep = global_set | source | recent
    else:
        raise ValueError(f"unknown A1 region control: {region}")
    return tuple(sorted(keep))


def _ordered_candidates(
    candidates: set[int],
    *,
    strategy: str,
    scores: Mapping[int, float] | None,
    seed: int,
) -> list[int]:
    if strategy in {"score_high", "score_low"}:
        if scores is None:
            raise ValueError(f"{strategy} requires a score for every candidate position")
        missing = candidates - {int(position) for position in scores}
        if missing:
            raise ValueError(f"attention scores missing {len(missing)} candidate positions")
        numeric: dict[int, float] = {}
        for position in candidates:
            value = float(scores[position])
            if not math.isfinite(value):
                raise ValueError(f"non-finite attention score at position {position}")
            numeric[position] = value
        if strategy == "score_high":
            return sorted(candidates, key=lambda position: (-numeric[position], position))
        return sorted(candidates, key=lambda position: (numeric[position], position))
    if strategy == "recency":
        return sorted(candidates, key=lambda position: (-position, position))
    if strategy == "random":
        return sorted(
            candidates,
            key=lambda position: (
                hashlib.sha256(f"{seed}:{position}".encode("ascii")).hexdigest(),
                position,
            ),
        )
    if strategy == "position_low":
        return sorted(candidates)
    raise ValueError(f"unsupported selection strategy: {strategy}")


def exact_budget_positions(
    *,
    bank_length: int,
    budget: int,
    protected_positions: Sequence[int],
    strategy: str,
    scores: Mapping[int, float] | None = None,
    seed: int = 42,
) -> tuple[int, ...]:
    """Select exactly B context keys, retaining every protected key."""
    if not 0 <= budget <= bank_length:
        raise ValueError(f"budget must be in [0, {bank_length}]")
    protected = _position_set(bank_length, protected_positions, name="protected positions")
    if len(protected) > budget:
        raise ValueError(
            f"protected set has {len(protected)} keys, exceeding exact budget {budget}"
        )
    candidates = set(range(bank_length)) - protected
    selected = set(protected)
    ordered = _ordered_candidates(candidates, strategy=strategy, scores=scores, seed=seed)
    selected.update(ordered[: budget - len(selected)])
    if len(selected) != budget:
        raise AssertionError("exact-budget selector failed to produce the requested cardinality")
    return tuple(sorted(selected))


def remove_attention_positions(
    *,
    bank_length: int,
    protected_positions: Sequence[int],
    scores: Mapping[int, float],
    count: int,
    side: str,
) -> tuple[int, ...]:
    """Keep the bank after deleting exactly D lowest or highest unprotected keys."""
    if count < 0:
        raise ValueError("deletion count must be nonnegative")
    if side not in {"low", "high"}:
        raise ValueError("side must be 'low' or 'high'")
    protected = _position_set(bank_length, protected_positions, name="protected positions")
    candidates = set(range(bank_length)) - protected
    if count > len(candidates):
        raise ValueError(f"cannot delete {count} keys from {len(candidates)} eligible keys")
    strategy = "score_low" if side == "low" else "score_high"
    deleted = set(_ordered_candidates(candidates, strategy=strategy, scores=scores, seed=0)[:count])
    return tuple(position for position in range(bank_length) if position not in deleted)


def matched_deletion_positions(
    *,
    bank_length: int,
    region_positions: Sequence[int],
    protected_positions: Sequence[int],
    count: int,
    seed: int = 42,
) -> tuple[int, ...]:
    """Delete exactly D hash-sampled positions from one eligible region."""
    if count < 0:
        raise ValueError("deletion count must be nonnegative")
    protected = _position_set(bank_length, protected_positions, name="protected positions")
    region = _position_set(bank_length, region_positions, name="region positions")
    eligible = region - protected
    if count > len(eligible):
        raise ValueError(f"region has {len(eligible)} eligible keys; cannot delete {count}")
    ranked = _ordered_candidates(eligible, strategy="random", scores=None, seed=seed)
    deleted = set(ranked[:count])
    return tuple(position for position in range(bank_length) if position not in deleted)


def make_selection(
    per_layer_positions: Sequence[Sequence[int]],
    *,
    requested_budget: int | str,
    selector_id: str,
    ranking_origin_round: int | None = None,
) -> Selection:
    """Adapt exact masks to the shared DFlash bank gather contract."""
    positions = tuple(tuple(int(position) for position in layer) for layer in per_layer_positions)
    if not positions:
        raise ValueError("selection requires at least one draft layer")
    for layer in positions:
        if tuple(sorted(set(layer))) != layer:
            raise ValueError("each per-layer mask must be sorted and unique")
    unused = tuple(
        max(0, int(requested_budget) - len(layer))
        if requested_budget != "full"
        else 0
        for layer in positions
    )
    return Selection(positions, requested_budget, selector_id, ranking_origin_round, unused)


def build_online_selection_policy(
    selector: str,
    *,
    budget: int,
    source_anchors: int = 64,
    recent_output: int = 256,
    random_seed: int = 42,
):
    """Create the fixed-budget selection callback used by A3 rollouts.

    Supported live signals are recency, target-parent, and previous dense
    DFlash attention. Oracle/current-draft ranking is intentionally omitted.
    """
    if selector not in {
        "recency",
        "target_parent",
        "multi_query_max_pooling",
        "duo_ensemble",
        "target_multi_query",
        "m1_multi_query",
        "m4_duo_ensemble",
        "multi_query_mean_pooling",
        "m2_chunk_mean",
        "m2_p1_chunk128",
        "previous_dflash",
    }:
        raise ValueError(f"unsupported deployable A3 selector: {selector}")
    if budget < 1:
        raise ValueError("A3 context budget must be positive")

    def select(
        *,
        state,
        layout,
        bank_length: int,
        processed_output_start: int,
        requested_budget: int | str,
        round_index: int,
        refresh_required: bool,
        target_parent_scores: Mapping[int, float] | None,
        previous_dflash_scores: Mapping[int, Mapping[int, float]],
        num_draft_layers: int,
    ) -> Selection:
        if requested_budget == "full":
            positions = tuple(tuple(range(bank_length)) for _ in range(num_draft_layers))
            return make_selection(
                positions, requested_budget="full", selector_id=selector,
                ranking_origin_round=round_index if refresh_required else None,
            )
        requested = int(requested_budget)
        if requested != budget:
            raise ValueError(f"A3 policy budget {budget} received action budget {requested}")
        protected = global_protected_positions(
            layout,
            bank_length=bank_length,
            source_anchors=source_anchors,
            processed_output_start=processed_output_start,
            recent_output=recent_output,
            protect_recent=True,
        )
        masks: list[tuple[int, ...]] = []
        if selector == "recency":
            masks = [
                exact_budget_positions(
                    bank_length=bank_length,
                    budget=requested,
                    protected_positions=protected,
                    strategy="recency",
                )
                for _ in range(num_draft_layers)
            ]
        elif selector in {
            "target_parent",
            "multi_query_max_pooling",
            "duo_ensemble",
            "target_multi_query",
            "m1_multi_query",
            "m4_duo_ensemble",
            "multi_query_mean_pooling",
            "m2_chunk_mean",
            "m2_p1_chunk128",
        }:
            if target_parent_scores is None:
                raise RuntimeError(f"{selector} signal unavailable at this prefix")
            masks = [
                exact_budget_positions(
                    bank_length=bank_length,
                    budget=requested,
                    protected_positions=protected,
                    scores=target_parent_scores,
                    strategy="score_high",
                )
                for _ in range(num_draft_layers)
            ]
        else:
            if not previous_dflash_scores:
                raise RuntimeError("previous DFlash signal unavailable before first dense refresh")
            for layer in range(num_draft_layers):
                layer_scores = previous_dflash_scores.get(layer)
                if layer_scores is None:
                    raise RuntimeError(f"previous DFlash scores missing draft layer {layer}")
                masks.append(exact_budget_positions(
                    bank_length=bank_length,
                    budget=requested,
                    protected_positions=protected,
                    scores=layer_scores,
                    strategy="score_high",
                ))
        origin = None
        if selector == "previous_dflash" and state.ranking_age < 10**9:
            origin = round_index - state.ranking_age
        return make_selection(
            masks,
            requested_budget=requested,
            selector_id=selector,
            ranking_origin_round=origin,
        )

    return select
