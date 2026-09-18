"""Frozen Hugging Face target collector for RECAP-KV traces.

Model-specific code is intentionally isolated here.  The policy/evaluator
modules consume only JSON-compatible vectors and do not import Transformers.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _as_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    return [float(item) for item in value]


def build_source_index(
    hidden_states: Any,
    *,
    source_start: int,
    source_end: int,
    block_size: int,
    num_prototypes: int = 2,
) -> list[dict[str, Any]]:
    """Pool source hidden states into block embeddings and prototypes."""

    if getattr(hidden_states, "ndim", None) != 3 or hidden_states.shape[0] != 1:
        raise ValueError("hidden_states must have shape [1, tokens, dimension]")
    if source_start < 0 or source_end <= source_start or source_end > hidden_states.shape[1]:
        raise ValueError("source span is invalid")
    if block_size <= 0 or num_prototypes <= 0:
        raise ValueError("block_size and num_prototypes must be positive")
    source = hidden_states[0, source_start:source_end].float()
    blocks: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, source.shape[0], block_size)):
        end = min(start + block_size, int(source.shape[0]))
        block = source[start:end]
        embedding = block.mean(dim=0)
        prototypes: list[list[float]] = []
        for prototype_index in range(num_prototypes):
            left = math.floor(prototype_index * block.shape[0] / num_prototypes)
            right = max(left + 1, math.floor((prototype_index + 1) * block.shape[0] / num_prototypes))
            right = min(right, int(block.shape[0]))
            prototypes.append(_as_float_list(block[left:right].mean(dim=0)))
        blocks.append({
            "index": index,
            "start": source_start + start,
            "end": source_start + end,
            "embedding": _as_float_list(embedding),
            "prototypes": prototypes,
        })
    return blocks


def segment_token_indices(token_texts: Sequence[str], *, max_tokens: int) -> list[tuple[int, int]]:
    """Split generated token text at sentence punctuation or a fixed cap."""

    if not token_texts or max_tokens <= 0:
        raise ValueError("token_texts must be non-empty and max_tokens positive")
    segments: list[tuple[int, int]] = []
    start = 0
    for index, token_text in enumerate(token_texts, 1):
        punctuation = any(mark in str(token_text) for mark in (".", "!", "?"))
        if punctuation or index - start >= max_tokens:
            segments.append((start, index))
            start = index
    if start < len(token_texts):
        segments.append((start, len(token_texts)))
    return segments


def _collapse_last_query(attentions: Sequence[Any]) -> Any:
    import torch

    vectors = []
    for attention in attentions:
        if attention is None:
            continue
        tensor = attention.detach().float() if hasattr(attention, "detach") else torch.as_tensor(attention).float()
        if tensor.ndim == 4:
            vectors.append(tensor[0, :, -1, :].mean(dim=0))
        elif tensor.ndim == 3:
            vectors.append(tensor[:, -1, :].mean(dim=0))
        elif tensor.ndim == 2:
            vectors.append(tensor[-1])
        elif tensor.ndim == 1:
            vectors.append(tensor)
        else:
            raise ValueError(f"unsupported attention rank: {tensor.ndim}")
    if not vectors:
        raise ValueError("model returned no attention tensors")
    return torch.stack(vectors, dim=0).mean(dim=0)


def _attention_to_blocks(
    attentions: Sequence[Any], *, source_start: int, source_end: int, block_size: int
) -> list[float]:
    source = _collapse_last_query(attentions)[source_start:source_end]
    values = [
        float(source[start : min(start + block_size, source.shape[0])].sum().item())
        for start in range(0, int(source.shape[0]), block_size)
    ]
    total = sum(values)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / len(values)] * len(values)
    normalized = [value / total for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("source attention contains non-finite values")
    return normalized


def _prefill_with_hidden(model: Any, input_ids: Any, *, chunk_size: int) -> tuple[Any, Any]:
    """Chunked prefill returning the cache and concatenated last-layer hiddens."""

    import torch
    from src.analyze.groundsync.trace_target import (
        _bottom_right_causal_mask,
        _model_call,
        _set_attention_implementation,
    )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    dtype = getattr(model, "dtype", torch.float32)
    if not isinstance(dtype, torch.dtype):
        dtype = torch.float32
    _set_attention_implementation(model, "sdpa")
    past = None
    hidden_parts = []
    for start in range(0, int(input_ids.shape[1]), chunk_size):
        end = min(start + chunk_size, int(input_ids.shape[1]))
        chunk = input_ids[:, start:end]
        mask = _bottom_right_causal_mask(end - start, start, dtype=dtype, device=input_ids.device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": True,
            "output_attentions": False,
            "attention_mask": mask,
        }
        if past is not None:
            kwargs["past_key_values"] = past
        outputs = _model_call(model, **kwargs)
        hidden_parts.append(outputs.hidden_states[-1].detach())
        past = outputs.past_key_values
    _set_attention_implementation(model, "eager")
    return past, torch.cat(hidden_parts, dim=1)


def collect_trace(
    model: Any,
    tokenizer: Any,
    rendered: Any,
    *,
    sample_id: str,
    dataset: str,
    max_new_tokens: int,
    block_size: int,
    segment_token_cap: int,
    prefill_chunk_size: int,
    device: Any,
) -> dict[str, Any]:
    """Generate one greedy target trace with compact source evidence records."""

    import torch
    from src.analyze.groundsync.trace_target import _model_call

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    input_ids = rendered.input_ids.to(device)
    if input_ids.shape[1] < 2:
        raise ValueError("rendered prompt must contain at least two tokens")
    with torch.inference_mode():
        past, hidden = _prefill_with_hidden(
            model, input_ids[:, :-1], chunk_size=prefill_chunk_size
        )
        source_blocks = build_source_index(
            hidden,
            source_start=rendered.source_start,
            source_end=rendered.source_end,
            block_size=block_size,
        )
        current = input_ids[:, -1:]
        query = hidden[:, -1, :].detach()
        segment_hiddens: list[Any] = []
        segment_attention: list[list[float]] = []
        current_segment_tokens: list[int] = []
        segments: list[dict[str, Any]] = []
        generated: list[int] = []
        token_texts: list[str] = []
        eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()

        for _ in range(max_new_tokens):
            outputs = _model_call(
                model,
                input_ids=current,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            token_id = int(next_token[0, 0].item())
            try:
                text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            except (AttributeError, TypeError):
                text = ""
            token_hidden = outputs.hidden_states[-1][:, -1, :].detach()
            segment_hiddens.append(token_hidden[0])
            segment_attention.append(
                _attention_to_blocks(
                    outputs.attentions,
                    source_start=rendered.source_start,
                    source_end=rendered.source_end,
                    block_size=block_size,
                )
            )
            current_segment_tokens.append(token_id)
            generated.append(token_id)
            token_texts.append(text)
            boundary = (
                any(mark in text for mark in (".", "!", "?"))
                or len(current_segment_tokens) >= segment_token_cap
                or token_id in eos_ids
            )
            if boundary:
                attention = [
                    sum(row[index] for row in segment_attention) / len(segment_attention)
                    for index in range(len(source_blocks))
                ]
                segment_vector = torch.stack(segment_hiddens, dim=0).float().mean(dim=0)
                segments.append({
                    "index": len(segments),
                    "attention": attention,
                    "query_vectors": [_as_float_list(query[0])],
                    "segment_vector": _as_float_list(segment_vector),
                    "token_count": len(current_segment_tokens),
                    "generated_token_ids": list(current_segment_tokens),
                })
                query = token_hidden
                segment_hiddens.clear()
                segment_attention.clear()
                current_segment_tokens.clear()
            past = outputs.past_key_values
            current = next_token
            if token_id in eos_ids:
                break
        if current_segment_tokens:
            attention = [
                sum(row[index] for row in segment_attention) / len(segment_attention)
                for index in range(len(source_blocks))
            ]
            segment_vector = torch.stack(segment_hiddens, dim=0).float().mean(dim=0)
            segments.append({
                "index": len(segments),
                "attention": attention,
                "query_vectors": [_as_float_list(query[0])],
                "segment_vector": _as_float_list(segment_vector),
                "token_count": len(current_segment_tokens),
                "generated_token_ids": list(current_segment_tokens),
            })
    return {
        "schema_version": "recap.trace.v1",
        "status": "ok",
        "sample_id": str(sample_id),
        "dataset": str(dataset),
        "input_tokens": int(input_ids.shape[1]),
        "source_start": int(rendered.source_start),
        "source_end": int(rendered.source_end),
        "source_tokens": int(rendered.source_end - rendered.source_start),
        "block_size": int(block_size),
        "output_tokens": len(generated),
        "generated_token_ids": generated,
        "token_texts": token_texts,
        "source_blocks": source_blocks,
        "segments": segments,
    }


def record_document(record: Mapping[str, Any]) -> str:
    for key in ("document", "context", "input", "prompt", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    turns = record.get("turns")
    if isinstance(turns, list) and turns and isinstance(turns[0], str):
        return turns[0]
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                if message["content"].strip():
                    return message["content"]
    raise ValueError("record has no usable document/prompt field")
