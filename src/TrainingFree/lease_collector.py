"""Qwen3 model-boundary helpers for RECAP-KV V2 Source-State Leases."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .lease import LeaseState, select_hot_cold


def _as_query_tensor(value: Any) -> Any:
    import torch

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.detach().float()
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError("query tensor batch dimension must be one")
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError("query tensor must have shape [heads, dimension]")
    return tensor


def _as_key_tensor(value: Any) -> Any:
    import torch

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.detach().float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("key tensor must have shape [1, kv_heads, tokens, dimension]")
    if tensor.shape[1] <= 0 or tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
        raise ValueError("key tensor dimensions must be positive")
    return tensor


def map_query_heads_to_kv(query_heads: int, kv_heads: int) -> list[int]:
    """Map GQA/MQA query heads to their repeated KV head indices."""

    query_heads = int(query_heads)
    kv_heads = int(kv_heads)
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query_heads must be positive and divisible by kv_heads")
    groups = query_heads // kv_heads
    return [index // groups for index in range(query_heads)]


def extract_cache_keys(cache: Any, layer_index: int) -> Any:
    """Extract a ``[batch, kv_heads, sequence, head_dim]`` key tensor."""

    layer_index = int(layer_index)
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    if hasattr(cache, "layers"):
        layers = cache.layers
        if layer_index >= len(layers):
            raise IndexError("layer_index is outside cache layers")
        layer = layers[layer_index]
        keys = getattr(layer, "keys", None)
    elif hasattr(cache, "key_cache"):
        keys = cache.key_cache[layer_index]
    elif isinstance(cache, (tuple, list)):
        if layer_index >= len(cache):
            raise IndexError("layer_index is outside tuple cache")
        pair = cache[layer_index]
        keys = pair[0] if isinstance(pair, (tuple, list)) else getattr(pair, "keys", None)
    else:
        keys = None
    if keys is None:
        raise TypeError("cache does not expose layer keys")
    return _as_key_tensor(keys)


def extract_cache_values(cache: Any, layer_index: int) -> Any:
    """Extract a ``[batch, kv_heads, sequence, head_dim]`` value tensor."""

    layer_index = int(layer_index)
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    if hasattr(cache, "layers"):
        layers = cache.layers
        if layer_index >= len(layers):
            raise IndexError("layer_index is outside cache layers")
        values = getattr(layers[layer_index], "values", None)
    elif hasattr(cache, "value_cache"):
        values = cache.value_cache[layer_index]
    elif isinstance(cache, (tuple, list)):
        if layer_index >= len(cache):
            raise IndexError("layer_index is outside tuple cache")
        pair = cache[layer_index]
        values = pair[1] if isinstance(pair, (tuple, list)) else getattr(pair, "values", None)
    else:
        values = None
    if values is None:
        raise TypeError("cache does not expose layer values")
    return _as_key_tensor(values)


def _normalize_source_span(source_start: int, source_end: int, tokens: int) -> tuple[int, int]:
    start, end = int(source_start), int(source_end)
    if start < 0 or end <= start or end > tokens:
        raise ValueError("source span is invalid")
    return start, end


def block_log_geometry(
    query_states: Any,
    key_states: Any,
    *,
    source_start: int,
    source_end: int,
    block_size: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Return anchor ``log Z`` and per-block ``kappa`` for every query head."""

    import torch

    query = _as_query_tensor(query_states)
    keys = _as_key_tensor(key_states)[0]
    if query.shape[1] != keys.shape[2]:
        raise ValueError("query and key head dimensions must match")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    source_start, source_end = _normalize_source_span(source_start, source_end, keys.shape[1])
    mapping = map_query_heads_to_kv(query.shape[0], keys.shape[0])
    grouped_keys = keys[mapping]
    logits = torch.einsum("hd,hsd->hs", query, grouped_keys) / math.sqrt(query.shape[1])
    source_logits = logits[:, source_start:source_end]
    source_keys = grouped_keys[:, source_start:source_end, :]
    source_length = int(source_logits.shape[1])
    block_count = math.ceil(source_length / block_size)
    padding = block_count * block_size - source_length
    if padding:
        source_logits = torch.nn.functional.pad(
            source_logits, (0, padding), value=float("-inf")
        )
        source_keys = torch.nn.functional.pad(source_keys, (0, 0, 0, padding), value=0.0)
    block_logits = source_logits.reshape(query.shape[0], block_count, block_size)
    block_log_z = torch.logsumexp(block_logits, dim=-1)
    key_norms = torch.linalg.vector_norm(source_keys, dim=-1) / math.sqrt(query.shape[1])
    if padding:
        key_norms[:, -padding:] = float("-inf")
    block_kappa = key_norms.reshape(query.shape[0], block_count, block_size).max(dim=-1).values
    return block_log_z.cpu().tolist(), block_kappa.cpu().tolist()


def compute_live_log_z(
    query_states: Any,
    key_states: Any,
    *,
    source_start: int,
    source_end: int,
) -> list[float]:
    """Compute exact log contribution of non-source keys for each query head."""

    import torch

    query = _as_query_tensor(query_states)
    keys = _as_key_tensor(key_states)[0]
    if query.shape[1] != keys.shape[2]:
        raise ValueError("query and key head dimensions must match")
    source_start, source_end = _normalize_source_span(source_start, source_end, keys.shape[1])
    mapping = map_query_heads_to_kv(query.shape[0], keys.shape[0])
    grouped_keys = keys[mapping]
    live_positions = list(range(0, source_start)) + list(range(source_end, keys.shape[1]))
    if not live_positions:
        return [float("-inf")] * query.shape[0]
    logits = torch.einsum(
        "hd,hsd->hs", query, grouped_keys[:, live_positions, :]
    ) / math.sqrt(query.shape[1])
    return [float(torch.logsumexp(logits[head], dim=0).item()) for head in range(query.shape[0])]


class QueryCapture:
    """Temporary Qwen3 self-attention hooks capturing rotated final-token Q."""

    def __init__(self, model: Any, layer_ids: Sequence[int] | None = None) -> None:
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise ValueError("model must expose model.layers for Qwen3 query capture")
        selected = list(range(len(layers))) if layer_ids is None else [int(item) for item in layer_ids]
        if not selected or any(item < 0 or item >= len(layers) for item in selected):
            raise ValueError("layer_ids must point to model layers")
        self.latest: dict[int, Any] = {}
        self._handles = []
        for layer_id in selected:
            attention = getattr(layers[layer_id], "self_attn", None)
            if attention is None:
                raise ValueError(f"layer {layer_id} has no self_attn module")
            self._handles.append(
                attention.register_forward_pre_hook(
                    self._make_hook(layer_id), with_kwargs=True
                )
            )

    def _make_hook(self, layer_id: int):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            import torch
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                raise ValueError("Qwen3 hook did not receive hidden_states/position_embeddings")
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            cos, sin = position_embeddings
            query, _ = apply_rotary_pos_emb(query, query, cos, sin)
            if query.shape[2] <= 0:
                raise ValueError("Qwen3 query sequence is empty")
            self.latest[layer_id] = query[0, :, -1, :].detach()

        return hook

    def clear(self) -> None:
        self.latest.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "QueryCapture":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _last_query_attention(attention: Any) -> Any:
    import torch

    tensor = attention.detach().float() if hasattr(attention, "detach") else torch.as_tensor(attention).float()
    if tensor.ndim == 4:
        return tensor[0, :, -1, :]
    if tensor.ndim == 3:
        return tensor[:, -1, :]
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"unsupported attention rank: {tensor.ndim}")


def source_block_attention(
    attention: Any,
    *,
    source_start: int,
    source_end: int,
    block_size: int,
) -> list[list[float]]:
    """Return absolute source attention mass per head and source block."""

    tensor = _last_query_attention(attention)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    source_start, source_end = _normalize_source_span(source_start, source_end, tensor.shape[-1])
    source = tensor[:, source_start:source_end]
    values: list[list[float]] = []
    for head in range(source.shape[0]):
        row = [
            float(source[head, start : min(start + block_size, source.shape[1])].sum().item())
            for start in range(0, int(source.shape[1]), block_size)
        ]
        if any(not math.isfinite(value) or value < 0.0 for value in row):
            raise ValueError("source attention contains invalid values")
        values.append(row)
    return values


def _mean_block_mass(masses: Sequence[Sequence[Sequence[float]]]) -> list[float]:
    if not masses:
        raise ValueError("masses must not be empty")
    width = len(masses[0][0]) if masses[0] else 0
    if width <= 0:
        raise ValueError("masses must contain source blocks")
    rows = [row for layer in masses for row in layer]
    if not rows or any(len(row) != width for row in rows):
        raise ValueError("source mass rows must be equally wide")
    return [sum(row[index] for row in rows) / len(rows) for index in range(width)]


def _actual_cold_mass(
    masses: Sequence[Sequence[float]], cold: Sequence[int]
) -> list[float]:
    return [sum(row[index] for index in cold) for row in masses]


def _build_head_states(
    query_by_layer: Mapping[int, Any],
    key_by_layer: Mapping[int, Any],
    attention_by_layer: Mapping[int, Any],
    *,
    source_start: int,
    source_end: int,
    block_size: int,
    delta_anchor: float,
    delta_cert: float,
    anchor_step: int,
) -> tuple[dict[int, list[LeaseState]], list[int], list[int], float]:
    masses_by_layer = [
        source_block_attention(
            attention_by_layer[layer_id],
            source_start=source_start,
            source_end=source_end,
            block_size=block_size,
        )
        for layer_id in query_by_layer
    ]
    mean_mass = _mean_block_mass(masses_by_layer)
    hot, cold = select_hot_cold(mean_mass, delta_anchor=delta_anchor)
    states: dict[int, list[LeaseState]] = {}
    for layer_id in query_by_layer:
        log_z0, kappa = block_log_geometry(
            query_by_layer[layer_id],
            key_by_layer[layer_id],
            source_start=source_start,
            source_end=source_end,
            block_size=block_size,
        )
        query = _as_query_tensor(query_by_layer[layer_id])
        states[layer_id] = [
            LeaseState(
                anchor_query=query[head].tolist(),
                log_z0=log_z0[head],
                kappa=kappa[head],
                hot=hot,
                cold=cold,
                delta_cert=delta_cert,
                anchor_step=anchor_step,
            )
            for head in range(query.shape[0])
        ]
    cold_fraction = len(cold) / len(mean_mass)
    return states, hot, cold, cold_fraction


def collect_lease_trace(
    model: Any,
    tokenizer: Any,
    rendered: Any,
    *,
    sample_id: str,
    dataset: str,
    max_new_tokens: int,
    block_size: int,
    prefill_chunk_size: int,
    delta_anchor: float,
    delta_cert: float,
    device: Any,
    layer_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Collect an event-driven lease trace without evicting or mutating KV."""

    import torch
    from src.analyze.groundsync.trace_target import _model_call, _set_attention_implementation
    from src.TrainingFree.collector import _prefill_with_hidden

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    input_ids = rendered.input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("rendered prompt must contain at least two tokens")
    model_layers = getattr(getattr(model, "model", None), "layers", None)
    if model_layers is None:
        raise ValueError("model must expose model.layers")
    selected_layers = list(range(len(model_layers))) if layer_ids is None else [int(item) for item in layer_ids]
    if not selected_layers:
        raise ValueError("layer_ids must not be empty")

    with torch.inference_mode():
        past, _ = _prefill_with_hidden(
            model, input_ids[:, :-1], chunk_size=prefill_chunk_size
        )
        current = input_ids[:, -1:]
        generated: list[int] = []
        token_texts: list[str] = []
        steps: list[dict[str, Any]] = []
        eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
        states: dict[int, list[LeaseState]] | None = None
        hot: list[int] = []
        cold: list[int] = []
        cold_fraction = 0.0
        source_block_count = 0
        anchor_step = 0
        _set_attention_implementation(model, "eager")
        try:
            with QueryCapture(model, selected_layers) as query_capture:
                for step_index in range(max_new_tokens):
                    outputs = _model_call(
                        model,
                        input_ids=current,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                        output_hidden_states=False,
                        output_attentions=True,
                    )
                    if not query_capture.latest:
                        raise RuntimeError("query capture returned no layer states")
                    query_by_layer = {
                        layer_id: query_capture.latest[layer_id]
                        for layer_id in selected_layers
                    }
                    key_by_layer = {
                        layer_id: extract_cache_keys(outputs.past_key_values, layer_id)
                        for layer_id in selected_layers
                    }
                    attention_by_layer = {
                        layer_id: outputs.attentions[layer_id]
                        for layer_id in selected_layers
                    }
                    if states is None:
                        states, hot, cold, cold_fraction = _build_head_states(
                            query_by_layer,
                            key_by_layer,
                            attention_by_layer,
                            source_start=rendered.source_start,
                            source_end=rendered.source_end,
                            block_size=block_size,
                            delta_anchor=delta_anchor,
                            delta_cert=delta_cert,
                            anchor_step=step_index,
                        )
                        full_source_score = True
                        source_block_count = len(hot) + len(cold)
                    else:
                        full_source_score = False

                    masses_by_layer = {
                        layer_id: source_block_attention(
                            attention_by_layer[layer_id],
                            source_start=rendered.source_start,
                            source_end=rendered.source_end,
                            block_size=block_size,
                        )
                        for layer_id in selected_layers
                    }
                    bounds: list[float] = []
                    actual_values: list[float] = []
                    drifts: list[float] = []
                    for layer_id in selected_layers:
                        live_values = compute_live_log_z(
                            query_by_layer[layer_id],
                            key_by_layer[layer_id],
                            source_start=rendered.source_start,
                            source_end=rendered.source_end,
                        )
                        actual_values_for_heads = _actual_cold_mass(
                            masses_by_layer[layer_id], cold
                        )
                        for head, state in enumerate(states[layer_id]):
                            result = state.step(
                                query_by_layer[layer_id][head].tolist(),
                                live_values[head],
                                actual_cold_mass=actual_values_for_heads[head],
                                step_index=step_index,
                            )
                            bounds.append(float(result["bound"]))
                            actual_values.append(float(result["actual_cold_mass"]))
                            drifts.append(float(result["drift"]))
                    mean_bound = sum(bounds) / len(bounds)
                    mean_actual = sum(actual_values) / len(actual_values)
                    valid = mean_bound <= delta_cert
                    expired = not valid
                    if expired:
                        full_source_score = True
                    steps.append({
                        "step": step_index,
                        "anchor_step": anchor_step,
                        "drift": sum(drifts) / len(drifts),
                        "bound": mean_bound,
                        "actual_cold_mass": mean_actual,
                        "slack": mean_bound - mean_actual,
                        "valid": valid,
                        "expired": expired,
                        "cold_fraction": cold_fraction,
                        "hot_blocks": len(hot),
                        "cold_blocks": len(cold),
                        "full_source_score": full_source_score,
                    })
                    if expired:
                        states, hot, cold, cold_fraction = _build_head_states(
                            query_by_layer,
                            key_by_layer,
                            attention_by_layer,
                            source_start=rendered.source_start,
                            source_end=rendered.source_end,
                            block_size=block_size,
                            delta_anchor=delta_anchor,
                            delta_cert=delta_cert,
                            anchor_step=step_index,
                        )
                        anchor_step = step_index
                        source_block_count = len(hot) + len(cold)
                    logits = outputs.logits[:, -1, :]
                    next_token = logits.argmax(dim=-1, keepdim=True)
                    token_id = int(next_token[0, 0].item())
                    generated.append(token_id)
                    try:
                        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                    except (AttributeError, TypeError):
                        text = ""
                    token_texts.append(text)
                    past = outputs.past_key_values
                    current = next_token
                    query_capture.clear()
                    if token_id in eos_ids:
                        break
        finally:
            _set_attention_implementation(model, "eager")

    return {
        "schema_version": "recap.lease.trace.v1",
        "status": "ok",
        "sample_id": str(sample_id),
        "dataset": str(dataset),
        "input_tokens": int(input_ids.shape[1]),
        "source_start": int(rendered.source_start),
        "source_end": int(rendered.source_end),
        "source_block_count": int(source_block_count),
        "block_size": int(block_size),
        "output_tokens": len(generated),
        "generated_token_ids": generated,
        "token_texts": token_texts,
        "layer_ids": selected_layers,
        "delta_anchor": float(delta_anchor),
        "delta_cert": float(delta_cert),
        "steps": steps,
    }
