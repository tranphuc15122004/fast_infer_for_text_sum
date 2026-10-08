"""Causal source-chunk ranking and deterministic per-layer budget selection."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .types import PromptLayout, RoundState, Selection


def _chunk_score(chunk: Sequence[int], scores: Mapping[int, float] | None) -> float:
    if not scores:
        return 0.0
    return sum(float(scores.get(position, 0.0)) for position in chunk)


def rank_chunks(
    layout: PromptLayout,
    *,
    selector: str,
    scores_by_layer: Sequence[Mapping[int, float]] | None = None,
    features: torch.Tensor | None = None,
    output_ids: Sequence[int] = (),
    last_processed_position: int | None = None,
    num_draft_layers: int = 1,
    tokenizer: Any = None,
    seed_key: str = "",
    source_token_ids: Sequence[int] = (),
) -> tuple[tuple[int, ...], ...]:
    if selector == "current_draft_oracle":
        raise ValueError("current_draft_oracle is an offline diagnostic, not a live selector")
    if not layout.source_chunks:
        layer_count = len(scores_by_layer) if selector == "draft_refresh" and scores_by_layer else num_draft_layers
        return tuple(() for _ in range(layer_count))
    layer_count = (
        len(scores_by_layer)
        if selector == "draft_refresh" and scores_by_layer
        else num_draft_layers
    )
    ranked_layers: list[tuple[int, ...]] = []
    for layer_index in range(layer_count):
        if selector == "draft_refresh":
            scores = scores_by_layer[layer_index] if scores_by_layer and layer_index < len(scores_by_layer) else None
        else:
            scores = scores_by_layer[0] if scores_by_layer else None
        if selector in {"target_parent", "draft_refresh"}:
            key = lambda index: (-_chunk_score(layout.source_chunks[index], scores), layout.source_chunks[index][0])
        elif selector == "recent_only":
            key = lambda index: (-layout.source_chunks[index][-1], layout.source_chunks[index][0])
        elif selector == "random":
            def key(index: int) -> tuple[str, int]:
                chunk = layout.source_chunks[index]
                digest = hashlib.sha256(f"{seed_key}:{layer_index}:{chunk[0]}".encode()).hexdigest()
                return digest, chunk[0]
        elif selector == "feature_cosine":
            if features is None or last_processed_position is None or last_processed_position >= features.shape[1]:
                key = lambda index: (-layout.source_chunks[index][-1], layout.source_chunks[index][0])
            else:
                recent = torch.nn.functional.normalize(features[:, last_processed_position : last_processed_position + 1].float(), dim=-1)
                chunk_scores: dict[int, float] = {}
                for index, chunk in enumerate(layout.source_chunks):
                    values = features.index_select(1, torch.tensor(chunk, device=features.device)).float().mean(dim=1, keepdim=True)
                    chunk_scores[index] = float(torch.nn.functional.cosine_similarity(values, recent, dim=-1).item())
                key = lambda index: (-chunk_scores[index], layout.source_chunks[index][0])
        elif selector == "lexical":
            if tokenizer is None or not output_ids:
                key = lambda index: (-layout.source_chunks[index][-1], layout.source_chunks[index][0])
            else:
                query = set(tokenizer.convert_ids_to_tokens(list(output_ids[-64:])))
                chunk_scores = {}
                for index, chunk in enumerate(layout.source_chunks):
                    text = set(tokenizer.convert_ids_to_tokens([source_token_ids[p] for p in chunk if p < len(source_token_ids)]))
                    chunk_scores[index] = len(query & text) / math.sqrt(max(len(query) * len(text), 1))
                key = lambda index: (-chunk_scores[index], layout.source_chunks[index][0])
        else:
            raise ValueError(f"unsupported selector: {selector}")
        ranked_layers.append(tuple(sorted(range(len(layout.source_chunks)), key=key)))
    return tuple(ranked_layers)


def select_context(
    state: RoundState,
    layout: PromptLayout,
    rankings: Sequence[Sequence[int]],
    budget: int | str,
    *,
    processed_output_start: int,
    bank_length: int,
    source_anchors: int = 64,
    recent_output: int = 256,
    selector_id: str = "target_parent",
    ranking_origin_round: int | None = None,
) -> Selection | None:
    layer_count = len(rankings)
    if layer_count < 1:
        raise ValueError("at least one per-layer ranking is required")
    if budget == "full":
        positions = tuple(tuple(range(bank_length)) for _ in range(layer_count))
        return Selection(positions, "full", selector_id, ranking_origin_round, tuple(0 for _ in positions))
    budget = int(budget)
    if budget >= bank_length:
        positions = tuple(tuple(range(bank_length)) for _ in range(layer_count))
        return Selection(positions, "full", selector_id, ranking_origin_round, tuple(0 for _ in positions))
    source = set(layout.source_positions)
    protected = set(layout.global_positions)
    protected.update(layout.source_positions[:source_anchors])
    recent_start = max(processed_output_start, bank_length - recent_output)
    protected.update(range(recent_start, bank_length))
    protected = {position for position in protected if 0 <= position < bank_length}
    if len(protected) > budget:
        return None
    per_layer: list[tuple[int, ...]] = []
    unused: list[int] = []
    for ranking in rankings:
        selected = set(protected)
        for chunk_index in ranking:
            chunk = layout.source_chunks[int(chunk_index)]
            additions = [position for position in chunk if position in source and position not in selected]
            if len(selected) + len(additions) <= budget:
                selected.update(additions)
        ordered = tuple(sorted(selected))
        per_layer.append(ordered)
        unused.append(max(0, budget - len(ordered)))
    return Selection(tuple(per_layer), budget, selector_id, ranking_origin_round, tuple(unused))
