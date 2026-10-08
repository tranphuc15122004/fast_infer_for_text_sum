"""DFlash block execution over gathered target-derived context KV."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .cache import _rotate_half
from .types import Action, DraftProposal, LayerContext, PromptLayout, Selection


def _attention_interface(config: Any) -> Any:
    from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward

    registry = ALL_ATTENTION_FUNCTIONS
    if hasattr(registry, "get_interface"):
        return registry.get_interface(config._attn_implementation, eager_attention_forward)
    return eager_attention_forward if config._attn_implementation == "eager" else registry[config._attn_implementation]


def _draft_attention(
    layer: Any,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    context: LayerContext,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    collect_source_scores: bool,
    source_chunks: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, dict[int, float]]:
    module = layer.self_attn
    batch, query_length, _ = hidden_states.shape
    head_dim = int(module.head_dim)
    query = module.q_norm(module.q_proj(hidden_states).view(batch, query_length, -1, head_dim)).transpose(1, 2)
    block_key = module.k_norm(module.k_proj(hidden_states).view(batch, query_length, -1, head_dim)).transpose(1, 2)
    block_value = module.v_proj(hidden_states).view(batch, query_length, -1, head_dim).transpose(1, 2)
    cosine, sine = position_embeddings
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    query = query * cosine + _rotate_half(query) * sine
    block_key = block_key * cosine + _rotate_half(block_key) * sine

    context_key = context.key
    context_value = context.value
    context_positions = context.positions
    if module.sliding_window is not None:
        query_positions = position_ids.reshape(-1)
        radius = int(module.sliding_window) - 1
        visible = (context_positions >= query_positions.min() - radius) & (
            context_positions <= query_positions.max() + radius
        )
        visible_indices = visible.nonzero(as_tuple=True)[0]
        context_key = context_key.index_select(2, visible_indices)
        context_value = context_value.index_select(2, visible_indices)
        context_positions = context_positions.index_select(0, visible_indices)
    key = torch.cat((context_key, block_key), dim=-2)
    value = torch.cat((context_value, block_value), dim=-2)
    attention_weights: dict[int, float] = {}
    # FlashAttention's sliding-window implementation indexes the window in
    # physical KV order. After sparse gathering, physical neighbors need not
    # be logical neighbors, so applying its window to the gathered sequence
    # changes the DFlash mask. Sliding layers use an explicit mask in original
    # token coordinates; dense layers retain the configured fast backend.
    if module.sliding_window is not None:
        groups = int(module.num_key_value_groups)
        expanded_key = key.float().repeat_interleave(groups, dim=1)
        expanded_value = value.repeat_interleave(groups, dim=1)
        raw_scores = torch.matmul(query.float(), expanded_key.float().transpose(-1, -2)) * float(module.scaling)
        all_positions = torch.cat((context_positions, position_ids.reshape(-1)))
        query_positions = position_ids.reshape(-1)
        distance = query_positions[:, None] - all_positions[None, :]
        allowed = distance.abs() < int(module.sliding_window)
        raw_scores = raw_scores.masked_fill(~allowed[None, None, :, :], torch.finfo(raw_scores.dtype).min)
        probabilities = torch.softmax(raw_scores, dim=-1)
        attention_output = torch.matmul(probabilities.to(expanded_value.dtype), expanded_value)
        attention_output = attention_output.transpose(1, 2).contiguous()
    else:
        if collect_source_scores and source_chunks:
            groups = int(module.num_key_value_groups)
            expanded_key = key.float().repeat_interleave(groups, dim=1)
            raw_scores = torch.matmul(query.float(), expanded_key.transpose(-1, -2)) * float(module.scaling)
            probabilities = torch.softmax(raw_scores, dim=-1)
        interface = _attention_interface(module.config)
        attention_output, _ = interface(
            module,
            query,
            key,
            value,
            None,
            dropout=0.0,
            scaling=module.scaling,
            sliding_window=None,
        )

    if collect_source_scores and source_chunks:
        probs = probabilities.mean(dim=(0, 1))[1:, :].mean(dim=0)
        context_probabilities = probs[: context_positions.numel()].detach().cpu().tolist()
        lookup = {int(position): index for index, position in enumerate(context_positions.tolist())}
        for chunk in source_chunks:
            for position in chunk:
                if position in lookup:
                    attention_weights[position] = float(context_probabilities[lookup[position]])
    attention_output = attention_output.reshape(batch, query_length, -1).contiguous()
    return module.o_proj(attention_output), attention_weights


def draft_block(
    draft_model: Any,
    target_model: Any,
    bank: Any,
    selection: Selection,
    anchor_and_masks: torch.Tensor,
    position_ids: torch.Tensor,
    action: Action,
    *,
    layout: PromptLayout,
    collect_refresh_scores: bool = False,
) -> DraftProposal:
    if anchor_and_masks.shape[1] != action.block_size:
        raise ValueError("draft input shape must equal gamma + 1")
    if position_ids.numel() != action.block_size:
        raise ValueError("absolute position ids must match the draft block")
    started = time.perf_counter()
    contexts = bank.gather(selection)
    gather_bytes = sum(ctx.key.numel() * ctx.key.element_size() + ctx.value.numel() * ctx.value.element_size() for ctx in contexts)
    hidden_states = target_model.model.embed_tokens(anchor_and_masks)
    position_embeddings = draft_model.rotary_emb(hidden_states, position_ids.reshape(1, -1))
    # The bank already stores context K in RoPE space; only the current noise
    # block receives new position embeddings in this forward.
    per_layer_scores: dict[int, dict[int, float]] = {}
    for layer_index, (layer, context) in enumerate(zip(draft_model.layers, contexts)):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attended, scores = _draft_attention(
            layer,
            normalized,
            position_ids.reshape(1, -1),
            context,
            position_embeddings,
            collect_source_scores=collect_refresh_scores,
            source_chunks=layout.source_chunks,
        )
        hidden_states = residual + attended
        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
        if collect_refresh_scores:
            per_layer_scores[layer_index] = scores
    hidden_states = draft_model.norm(hidden_states)
    logits = target_model.lm_head(hidden_states[:, 1:, :])
    candidates = torch.argmax(logits, dim=-1)
    return DraftProposal(
        token_ids=candidates,
        attention_scores_by_layer=per_layer_scores,
        draft_latency_ms=(time.perf_counter() - started) * 1000.0,
        context_tokens_by_layer=tuple(len(positions) for positions in selection.positions_by_layer),
        gather_bytes=gather_bytes,
        draft_tokens_proposed=action.gamma,
    )
