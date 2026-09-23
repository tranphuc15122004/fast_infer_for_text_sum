"""Dense-reference collector for E43 temporal support reuse audits."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from .collector import _prefill_with_hidden
from .lease_collector import (
    _as_key_tensor,
    extract_cache_keys,
    map_query_heads_to_kv,
)
from .temporal import TemporalAnalyzer, TemporalConfig


def collect_temporal_trace(
    model: Any,
    tokenizer: Any,
    rendered: Any,
    *,
    sample_id: str,
    dataset: str,
    max_new_tokens: int,
    prefill_chunk_size: int,
    device: Any,
    layer_ids: Sequence[int] | None = None,
    temporal_config: TemporalConfig | None = None,
) -> dict[str, Any]:
    """Collect exact attention and audit temporal policies without KV mutation."""

    import torch
    from src.analyze.groundsync.trace_target import _model_call
    from .concentration import _last_query_attention

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
    if len(selected_layers) != 1:
        raise ValueError("temporal audit currently requires exactly one layer")
    if any(layer_id < 0 or layer_id >= len(layers) for layer_id in selected_layers):
        raise ValueError("layer_ids must point to model layers")
    config = temporal_config or TemporalConfig()

    with torch.inference_mode():
        past, _ = _prefill_with_hidden(
            model, input_ids[:, :-1], chunk_size=prefill_chunk_size
        )
        source_tokens = int(rendered.source_end - rendered.source_start)
        current = input_ids[:, -1:]
        generated: list[int] = []
        steps: list[dict[str, Any]] = []
        eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
        analyzer: TemporalAnalyzer | None = None

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
            layer_id = selected_layers[0]
            attention = _last_query_attention(outputs.attentions[layer_id])
            source_attention = attention[:, rendered.source_start : rendered.source_end]
            if analyzer is None:
                key_states = _as_key_tensor(extract_cache_keys(outputs.past_key_values, layer_id))
                mapping = map_query_heads_to_kv(
                    int(source_attention.shape[0]), int(key_states.shape[1])
                )
                analyzer = TemporalAnalyzer(
                    query_to_kv=mapping,
                    source_tokens=source_tokens,
                    config=config,
                )
            temporal = analyzer.observe(source_attention)
            temporal["step"] = step_index
            steps.append(temporal)

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = int(next_token[0, 0].item())
            generated.append(token_id)
            past = outputs.past_key_values
            current = next_token
            if token_id in eos_ids:
                break

    return {
        "schema_version": "recap.e43.temporal.trace.v1",
        "status": "ok",
        "sample_id": str(sample_id),
        "dataset": str(dataset),
        "input_tokens": int(input_ids.shape[1]),
        "source_tokens": source_tokens,
        "output_tokens": len(generated),
        "generated_token_ids": generated,
        "layer_ids": [int(layer_id) for layer_id in selected_layers],
        "temporal_config": asdict(config),
        "steps": steps,
    }
