"""Qwen3 boundary adapter for RECAP-KV V3 hierarchy traces."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from .hierarchy import SourceHierarchy, build_source_hierarchy
from .hierarchy_routing import evaluate_routing_step, route_query
from .collector import _prefill_with_hidden
from .lease_collector import (
    QueryCapture,
    _as_key_tensor,
    _as_query_tensor,
    extract_cache_keys,
    map_query_heads_to_kv,
    source_block_attention,
)


def build_head_hierarchies(
    key_states: Any,
    *,
    source_start: int,
    source_end: int,
    region_size: int,
    block_size: int,
    reps_per_block: int,
    reps_per_region: int,
) -> dict[int, SourceHierarchy]:
    """Build one hierarchy per KV head from a cache key tensor."""

    keys = _as_key_tensor(key_states)
    if keys.shape[0] != 1:
        raise ValueError("key_states must have batch size one")
    if source_start < 0 or source_end <= source_start or source_end > keys.shape[2]:
        raise ValueError("source span is invalid")
    return {
        head: build_source_hierarchy(
            keys[0, head, source_start:source_end, :],
            region_size=region_size,
            block_size=block_size,
            reps_per_block=reps_per_block,
            reps_per_region=reps_per_region,
        )
        for head in range(int(keys.shape[1]))
    }


def collect_hierarchy_trace(
    model: Any,
    tokenizer: Any,
    rendered: Any,
    *,
    sample_id: str,
    dataset: str,
    max_new_tokens: int,
    region_size: int,
    block_size: int,
    reps_per_block: int,
    reps_per_region: int,
    mass_budget: float,
    prefill_chunk_size: int,
    device: Any,
    layer_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Collect routing/audit metrics without changing the model KV cache."""

    import torch
    from src.analyze.groundsync.trace_target import _model_call

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    input_ids = rendered.input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("rendered prompt must contain at least two tokens")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("model must expose model.layers")
    selected_layers = (
        [len(layers) - 1]
        if layer_ids is None
        else [int(layer_id) for layer_id in layer_ids]
    )
    if not selected_layers or any(layer_id < 0 or layer_id >= len(layers) for layer_id in selected_layers):
        raise ValueError("layer_ids must point to model layers")

    with torch.inference_mode():
        past, _ = _prefill_with_hidden(
            model, input_ids[:, :-1], chunk_size=prefill_chunk_size
        )
        hierarchies: dict[int, dict[int, SourceHierarchy]] = {
            layer_id: build_head_hierarchies(
                extract_cache_keys(past, layer_id),
                source_start=rendered.source_start,
                source_end=rendered.source_end,
                region_size=region_size,
                block_size=block_size,
                reps_per_block=reps_per_block,
                reps_per_region=reps_per_region,
            )
            for layer_id in selected_layers
        }
        source_tokens = rendered.source_end - rendered.source_start
        source_block_count = len(next(iter(next(iter(hierarchies.values())).values())).blocks)
        current = input_ids[:, -1:]
        generated: list[int] = []
        steps: list[dict[str, Any]] = []
        eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()

        with QueryCapture(model, layer_ids=selected_layers) as query_capture:
            for step_index in range(max_new_tokens):
                started = time.perf_counter()
                outputs = _model_call(
                    model,
                    input_ids=current,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=True,
                )
                active_blocks: set[int] = set()
                attention_rows: list[list[float]] = []
                decisions = []
                upper_violations = 0
                upper_bounds: list[float] = []
                routing_representatives = 0
                index_overheads: list[float] = []
                for layer_id in selected_layers:
                    query_states = _as_query_tensor(query_capture.latest[layer_id])
                    key_hierarchies = hierarchies[layer_id]
                    mapping = map_query_heads_to_kv(
                        int(query_states.shape[0]), len(key_hierarchies)
                    )
                    attention_rows.extend(
                        source_block_attention(
                            outputs.attentions[layer_id],
                            source_start=rendered.source_start,
                            source_end=rendered.source_end,
                            block_size=block_size,
                        )
                    )
                    for head in range(int(query_states.shape[0])):
                        hierarchy = key_hierarchies[mapping[head]]
                        decision = route_query(
                            hierarchy,
                            query_states[head],
                            mass_budget=mass_budget,
                        )
                        decisions.append((hierarchy, query_states[head], decision))
                        active_blocks.update(decision.active_block_indices)
                        upper_bounds.append(decision.upper_missed_mass_bound)
                        routing_representatives += decision.routing_representatives
                        index_overheads.append(hierarchy.index_overhead)
                        audit = evaluate_routing_step(
                            hierarchy,
                            query_states[head],
                            decision,
                            [0.0] * len(hierarchy.blocks),
                        )
                        upper_violations += int(audit["upper_bound_violations"])
                missed_values = []
                for masses in attention_rows:
                    if len(masses) != source_block_count:
                        raise ValueError("attention block width does not match hierarchy")
                    missed_values.append(
                        sum(value for index, value in enumerate(masses) if index not in active_blocks)
                    )
                active_tokens = sum(
                    hierarchies[selected_layers[0]][
                        map_query_heads_to_kv(
                            int(_as_query_tensor(query_capture.latest[selected_layers[0]]).shape[0]),
                            len(hierarchies[selected_layers[0]]),
                        )[0]
                    ].blocks[index].size
                    for index in active_blocks
                )
                steps.append({
                    "step": step_index,
                    "missed_attention_mass": sum(missed_values) / len(missed_values),
                    "max_missed_attention_mass": max(missed_values),
                    "exact_expansion_fraction": active_tokens / source_tokens,
                    "upper_bound_violations": upper_violations,
                    "index_overhead": sum(index_overheads) / len(index_overheads),
                    "routing_fraction": routing_representatives / (source_tokens * len(decisions)),
                    "active_qk_tokens": active_tokens,
                    "full_qk_tokens": source_tokens,
                    "routing_representatives": routing_representatives,
                    "upper_missed_mass_bound": max(upper_bounds),
                    "active_block_count": len(active_blocks),
                    "routing_time_ms": (time.perf_counter() - started) * 1000.0,
                })
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                token_id = int(next_token[0, 0].item())
                generated.append(token_id)
                past = outputs.past_key_values
                current = next_token
                if token_id in eos_ids:
                    break

    return {
        "schema_version": "recap.hierarchy.trace.v1",
        "status": "ok",
        "sample_id": str(sample_id),
        "dataset": str(dataset),
        "input_tokens": int(input_ids.shape[1]),
        "source_tokens": int(source_tokens),
        "source_block_count": int(source_block_count),
        "output_tokens": len(generated),
        "generated_token_ids": generated,
        "region_size": int(region_size),
        "block_size": int(block_size),
        "reps_per_block": int(reps_per_block),
        "reps_per_region": int(reps_per_region),
        "mass_budget": float(mass_budget),
        "layer_ids": [int(layer_id) for layer_id in selected_layers],
        "steps": steps,
    }
