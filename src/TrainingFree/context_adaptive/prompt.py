"""Baseline-compatible prompt rendering and exact source-token mapping."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .types import PromptLayout


_CANONICAL = {"gov_report", "qmsum", "multi_news", "lcc", "repobench-p"}


def render_prompt(sample: Mapping[str, Any], tokenizer: Any) -> str:
    raw = sample.get("raw") or {}
    if raw.get("dataset") in _CANONICAL:
        return str(sample["prompt"])
    prompt = str(sample["prompt"])
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


def prepare_prompt(
    sample: Mapping[str, Any],
    tokenizer: Any,
    *,
    chunk_size: int = 128,
    max_tokens: int = 0,
    suffix_tokens: int = 256,
) -> tuple[Any, PromptLayout]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    rendered = render_prompt(sample, tokenizer)
    try:
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"]
    token_ids = input_ids[0].tolist()
    try:
        offsets = encoded["offset_mapping"][0].tolist()
    except (KeyError, TypeError, AttributeError, NotImplementedError) as exc:
        offsets = []
        source_reason = f"offset_mapping_unavailable:{type(exc).__name__}"
    else:
        source_reason = None
    retained_indices = list(range(len(token_ids)))
    if max_tokens > 0 and len(token_ids) > max_tokens:
        if max_tokens < 2:
            retained_indices = retained_indices[-max_tokens:]
        else:
            tail = min(max(int(suffix_tokens), 1), max(max_tokens // 2, 1))
            head = max_tokens - tail
            retained_indices = retained_indices[:head] + retained_indices[-tail:]
        token_ids = [token_ids[index] for index in retained_indices]
        if offsets:
            offsets = [offsets[index] for index in retained_indices]
        input_ids = input_ids[:, retained_indices]
    prompt_hash = hashlib.sha256(
        ",".join(str(int(token)) for token in token_ids).encode("ascii")
    ).hexdigest()
    raw = sample.get("raw") or {}
    source = raw.get("context")
    source_positions: list[int] = []
    crossing_positions: set[int] = set()
    if not isinstance(source, str) or not source:
        source_reason = "source_context_unavailable"
    elif not offsets:
        source_reason = source_reason or "tokenizer_offsets_unavailable"
    else:
        first = rendered.find(source)
        if first < 0:
            source_reason = "source_span_not_found"
        elif rendered.find(source, first + 1) >= 0:
            source_reason = "source_span_ambiguous"
        else:
            last = first + len(source)
            source_positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if int(end) > first and int(start) < last and int(end) > int(start)
            ]
            crossing_positions = {
                index
                for index, (start, end) in enumerate(offsets)
                if int(end) > first
                and int(start) < last
                and int(end) > int(start)
                and (int(start) < first or int(end) > last)
            }
            if not source_positions:
                source_reason = "source_span_has_no_tokens"
    source_set = set(source_positions)
    global_positions = tuple(
        index
        for index in range(len(token_ids))
        if index not in source_set or index in crossing_positions
    )
    chunks = tuple(
        tuple(source_positions[start : start + chunk_size])
        for start in range(0, len(source_positions), chunk_size)
    )
    layout = PromptLayout(
        prompt_tokens=len(token_ids),
        source_positions=tuple(source_positions),
        global_positions=global_positions,
        source_chunks=chunks,
        prompt_hash=prompt_hash,
        source_available=bool(source_positions) and source_reason is None,
        source_reason=source_reason,
        dataset=raw.get("dataset"),
        source_group_id=raw.get("source_group_id"),
    )
    return input_ids, layout
