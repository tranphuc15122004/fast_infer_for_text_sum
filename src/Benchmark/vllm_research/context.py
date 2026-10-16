"""Toạ độ logic và attention drafter; không import binary của vLLM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


def prompt_key(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(',', ':')).encode()).hexdigest()


@dataclass
class ContextState:
    prompt_tokens: int
    source_positions: tuple[int, ...] = ()
    global_positions: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    round_index: int = 0
    rankings: dict[int, torch.Tensor] = field(default_factory=dict)
    ranking_length: int = 0
    ranking_round: int | None = None
    history_acceptance: float | None = None
    previous_uncertainty: dict[str, Any] | None = None
    previous_gamma: int | None = None
    pending_trace: dict[str, Any] | None = None


def gather_flash_kv(cache, block_table, positions, block_size: int, head_dim: int):
    """vLLM 0.30 FLASH_ATTN cache: [page, kv_head, slot, 2*head_dim]."""
    if cache.ndim != 4 or cache.shape[2] != block_size or cache.shape[3] != 2 * head_dim:
        raise RuntimeError('research adapter requires vLLM 0.30 FLASH_ATTN B,H,N,2D KV layout')
    pages = block_table.index_select(0, positions // block_size).long()
    offsets = positions % block_size
    values = cache[pages, :, offsets, :]
    return values[..., :head_dim], values[..., head_dim:]


def select_positions(state: ContextState, *, layer: int, bank_length: int, budget: int,
                     selector: str, source_anchors: int = 64, recent_output: int = 256):
    """Exact B bao gồm G0, source anchors, recent output và key mới sau refresh."""
    ranking = state.rankings.get(layer)
    device = ranking.device if ranking is not None else 'cpu'
    if min(bank_length, budget) < 0 or budget < 1:
        raise ValueError('bank length/budget must be nonnegative/positive')
    if bank_length <= budget:
        return torch.arange(bank_length, device=device)
    protected = set(state.global_positions)
    protected.update(sorted(state.source_positions)[:source_anchors])
    protected.update(range(max(state.prompt_tokens, bank_length - recent_output), bank_length))
    if selector == 'previous_dflash' and state.ranking_length:
        protected.update(range(state.ranking_length, bank_length))
    if any(p < 0 or p >= bank_length for p in protected):
        raise ValueError('prompt layout contains a protected key outside committed context')
    if len(protected) > budget:
        raise ValueError(f'protected keys ({len(protected)}) exceed exact context budget ({budget})')
    if selector == 'recency':
        order = torch.arange(bank_length - 1, -1, -1, device=device)
    elif selector == 'previous_dflash':
        if ranking is None or ranking.numel() != state.ranking_length:
            raise RuntimeError('previous_dflash requires a real earlier dense ranking for this layer')
        # Stable sorting makes ties deterministic in original logical coordinates.
        order = torch.argsort(ranking, descending=True, stable=True)
    else:
        raise ValueError(f'unsupported live context selector: {selector}')
    selected = torch.zeros(bank_length, dtype=torch.bool, device=device)
    if protected:
        selected[torch.tensor(sorted(protected), device=device)] = True
    candidates = order[~selected[order]]
    selected[candidates[:budget - len(protected)]] = True
    return selected.nonzero(as_tuple=True)[0]


def selected_attention(query, context_key, context_value, block_key, block_value,
                       context_positions, query_positions, *, causal: bool,
                       scaling: float, collect_scores: bool, sliding_window: int | None = None):
    """Q/K đã RoPE ở vị trí gốc. Mask luôn dùng logical positions, kể cả sparse."""
    groups = query.shape[1] // block_key.shape[1]
    if groups < 1 or query.shape[1] != groups * block_key.shape[1]:
        raise ValueError('query and KV head counts must support grouped query attention')
    key = torch.cat((context_key, block_key), dim=0).transpose(0, 1).unsqueeze(0)
    value = torch.cat((context_value, block_value), dim=0).transpose(0, 1).unsqueeze(0)
    q = query.transpose(0, 1).unsqueeze(0)
    positions = torch.cat((context_positions, query_positions))
    distance = query_positions[:, None] - positions[None, :]
    allowed = torch.ones_like(distance, dtype=torch.bool)
    if causal:
        allowed &= distance >= 0
    if sliding_window is not None:
        allowed &= distance.abs() < sliding_window
    mask = torch.zeros_like(distance, dtype=q.dtype).masked_fill(~allowed, float('-inf'))
    result = F.scaled_dot_product_attention(q, key, value, attn_mask=mask,
                                          scale=scaling, enable_gqa=True)
    scores = None
    if collect_scores:
        logits = torch.matmul(q.float(), key.float().repeat_interleave(groups, dim=1).transpose(-1, -2)) * scaling
        probabilities = (logits + mask.float()).softmax(dim=-1)
        # Exclude anchor query, include all block keys in the softmax denominator.
        scores = probabilities[0, :, 1:, :context_positions.numel()].mean(dim=(0, 1))
        if not torch.isfinite(scores).all():
            raise RuntimeError('nonfinite DFlash dense attention ranking')
    return result.squeeze(0).transpose(0, 1), scores
