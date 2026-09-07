"""MR-DFlash drafter: DFlash joint attention + HCA/CSA target views."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .memory import MRMemoryState, MRTargetMemory
from .model import (
    DraftSpec,
    RMSNorm,
    SwiGLUMLP,
    _compute_rope_cache,
    _rotate_half,
)


def _apply_rope(
    states: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
) -> torch.Tensor:
    cos, sin = _compute_rope_cache(
        position_ids, head_dim, rope_theta, states.dtype
    )
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return states * cos + _rotate_half(states) * sin


@dataclass(frozen=True)
class MRProjectedContextPart:
    """Một đoạn context đã chiếu, có thể dùng chung cho nhiều block."""

    key: torch.Tensor
    value: torch.Tensor
    # Với key/value [B,C,H,D], ánh xạ query batch [B*N] về B gốc. None nghĩa
    # batch projected đã trùng trực tiếp với query batch.
    batch_indices: Optional[torch.Tensor] = None


class MRBlockAttention(nn.Module):
    """Self attention của draft block, không cho phép cross-block leakage."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        rope_theta: float,
        use_qk_norm: bool,
        rms_norm_eps: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_attention_heads phải chia hết cho num_key_value_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scaling = head_dim ** -0.5
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            batch, length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            batch, length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(_apply_rope(q, position_ids, self.head_dim, self.rope_theta))
        k = self.k_norm(_apply_rope(k, position_ids, self.head_dim, self.rope_theta))
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, length, -1))


class MRTargetAttention(nn.Module):
    """Cross attention, hỗ trợ context chung hoặc context riêng từng query."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        use_qk_norm: bool,
        rms_norm_eps: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_attention_heads phải chia hết cho num_key_value_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scaling = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context.ndim == 3:
            return self._shared_context(query, context, attention_mask)
        if context.ndim == 4:
            return self._per_query_context(query, context, attention_mask)
        raise ValueError("context phải có dạng [B,S,H] hoặc [B,Q,K,H]")

    def _query(self, query: torch.Tensor) -> torch.Tensor:
        batch, query_len, _ = query.shape
        q = self.q_proj(query).view(
            batch, query_len, self.num_heads, self.head_dim
        )
        return self.q_norm(q)

    def _shared_context(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, query_len, _ = query.shape
        context_len = context.shape[1]
        q = self._query(query)
        k = self.k_proj(context).view(
            batch, context_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(context).view(
            batch, context_len, self.num_kv_heads, self.head_dim
        )
        k = self.k_norm(k)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=2)
            v = v.repeat_interleave(self.num_key_value_groups, dim=2)
        scores = torch.einsum("bqhd,bkhd->bqhk", q, k) * self.scaling
        weights_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, query_len, context_len):
                raise ValueError("shared attention_mask phải có dạng [B,Q,S]")
            weights_mask = attention_mask.to(dtype=torch.bool).unsqueeze(2)
            scores = scores.masked_fill(~weights_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        if weights_mask is not None:
            weights = weights * weights_mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        output = torch.einsum("bqhk,bkhd->bqhd", weights, v)
        return self.o_proj(output.reshape(batch, query_len, -1))

    def _per_query_context(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, query_len, _ = query.shape
        if context.shape[:2] != (batch, query_len):
            raise ValueError("per-query context phải cùng batch/query length với query")
        top_k = context.shape[2]
        q = self._query(query)
        k = self.k_proj(context).view(
            batch, query_len, top_k, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(context).view(
            batch, query_len, top_k, self.num_kv_heads, self.head_dim
        )
        k = self.k_norm(k)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=3)
            v = v.repeat_interleave(self.num_key_value_groups, dim=3)
        scores = torch.einsum("bqhd,bqkhd->bqhk", q, k) * self.scaling
        weights_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, query_len, top_k):
                raise ValueError("per-query attention_mask phải có dạng [B,Q,K]")
            weights_mask = attention_mask.to(dtype=torch.bool).unsqueeze(2)
            scores = scores.masked_fill(~weights_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        if weights_mask is not None:
            weights = weights * weights_mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        output = torch.einsum("bqhk,bqkhd->bqhd", weights, v)
        return self.o_proj(output.reshape(batch, query_len, -1))


class MRDFlashJointAttention(nn.Module):
    """DFlash attention với memory MR làm context.

    KV được tạo một lần từ ``[MR target memory ; draft block]`` và chạy qua
    cùng một softmax. Điều này giữ đúng primitive của DFlash, đồng thời cho
    HCA/CSA điều chỉnh attention mass giữa target và draft.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        rope_theta: float,
        use_qk_norm: bool,
        rms_norm_eps: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError("num_attention_heads phải chia hết cho num_key_value_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scaling = head_dim ** -0.5
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _rope(self, states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos, sin = _compute_rope_cache(
            positions.reshape(positions.shape[0], -1),
            self.head_dim,
            self.rope_theta,
            states.dtype,
        )
        cos = cos.reshape(*positions.shape, self.head_dim)
        sin = sin.reshape(*positions.shape, self.head_dim)
        # Head is immediately before the rotary dimension for all layouts in
        # this module: [B,Q,H,D] or [B,Q,K,H,D].
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        return states * cos + _rotate_half(states) * sin

    def project_context(
        self,
        context: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chiếu context một lần, không mở rộng hidden theo query.

        Shared context có layout ``[B,C,H]`` và projected KV có layout
        ``[B,C,n_heads,D]``. Chỉ CSA Top-k cần layout per-query sau khi đã
        gather projected KV: ``[B,Q,C,n_heads,D]``.
        """
        if context.ndim == 3:
            batch, context_len, _ = context.shape
            if context_positions.shape != (batch, context_len):
                raise ValueError("context_positions không khớp shared context")
            k = self.k_proj(context).view(
                batch, context_len, self.num_kv_heads, self.head_dim
            )
            v = self.v_proj(context).view(
                batch, context_len, self.num_kv_heads, self.head_dim
            )
            k = self._rope(self.k_norm(k), context_positions)
            if self.num_key_value_groups > 1:
                k = k.repeat_interleave(self.num_key_value_groups, dim=2)
                v = v.repeat_interleave(self.num_key_value_groups, dim=2)
            return k, v
        if context.ndim == 4:
            batch, query_len, context_len, _ = context.shape
            if context_positions.shape != (batch, query_len, context_len):
                raise ValueError("context_positions không khớp per-query context")
            k = self.k_proj(context).view(
                batch, query_len, context_len, self.num_kv_heads, self.head_dim
            )
            v = self.v_proj(context).view(
                batch, query_len, context_len, self.num_kv_heads, self.head_dim
            )
            k = self._rope(self.k_norm(k), context_positions)
            if self.num_key_value_groups > 1:
                k = k.repeat_interleave(self.num_key_value_groups, dim=3)
                v = v.repeat_interleave(self.num_key_value_groups, dim=3)
            return k, v
        raise ValueError("context phải có dạng [B,C,H] hoặc [B,Q,C,H]")

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor],
        query_positions: torch.Tensor,
        context_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_bias: Optional[torch.Tensor] = None,
        projected_context: Optional[tuple[MRProjectedContextPart, ...]] = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or query_positions.ndim != 2:
            raise ValueError("hidden_states/query_positions phải có dạng [B,Q,...]")
        batch, query_len, _ = hidden_states.shape
        if query_positions.shape != (batch, query_len):
            raise ValueError("query_positions không khớp hidden_states")
        q = self.q_proj(hidden_states).view(
            batch, query_len, self.num_heads, self.head_dim
        )
        # DFlash convention: Q/K norm trước RoPE. RMSNorm có learned
        # per-dimension scale nên đổi thứ tự không phải phép biến đổi vô hại.
        q = self._rope(self.q_norm(q), query_positions)

        if projected_context is None:
            if context is None:
                raise ValueError("context hoặc projected_context là bắt buộc")
            k_context, v_context = self.project_context(context, context_positions)
            context_parts = (MRProjectedContextPart(k_context, v_context),)
        else:
            context_parts = projected_context
        context_lengths = []
        for part in context_parts:
            key = part.key
            if key.ndim == 4:
                if part.batch_indices is None and key.shape[0] != batch:
                    raise ValueError("shared projected context không khớp batch")
                if part.batch_indices is not None and part.batch_indices.shape != (batch,):
                    raise ValueError("batch_indices của projected context không hợp lệ")
                context_lengths.append(key.shape[1])
            elif key.ndim == 5:
                if key.shape[:2] != (batch, query_len):
                    raise ValueError("per-query projected context không khớp query")
                context_lengths.append(key.shape[2])
            else:
                raise ValueError(
                    "projected context phải có dạng [B,C,H,D] hoặc [B,Q,C,H,D]"
                )
        context_len = sum(context_lengths)
        expected_positions = (
            (batch, context_len)
            if context_positions.ndim == 2
            else (batch, query_len, context_len)
        )
        if context_positions.shape != expected_positions:
            raise ValueError("context_positions không khớp projected context")

        k_draft = self.k_proj(hidden_states).view(
            batch, query_len, self.num_kv_heads, self.head_dim
        )
        v_draft = self.v_proj(hidden_states).view(
            batch, query_len, self.num_kv_heads, self.head_dim
        )
        k_draft = self._rope(self.k_norm(k_draft), query_positions)

        if self.num_key_value_groups > 1:
            # project_context() đã trả context theo num_attention_heads;
            # draft KV vẫn bắt đầu ở num_key_value_heads.
            k_draft = k_draft.repeat_interleave(self.num_key_value_groups, dim=2)
            v_draft = v_draft.repeat_interleave(self.num_key_value_groups, dim=2)

        # Draft KV được broadcast theo query trong *một block*. Training đã
        # đưa mỗi block vào batch dimension, nên query_len ở đây chỉ là K,
        # không còn là toàn bộ N*K của sample.
        k_draft = k_draft.unsqueeze(1).expand(-1, query_len, -1, -1, -1)
        v_draft = v_draft.unsqueeze(1).expand(-1, query_len, -1, -1, -1)

        context_score_parts = []
        for part in context_parts:
            key = part.key
            if key.ndim == 4:
                if part.batch_indices is None:
                    part_scores = torch.einsum("bqhd,bchd->bqhc", q, key)
                else:
                    base_batch = key.shape[0]
                    if batch % base_batch:
                        raise ValueError("projected context batch không chia hết query batch")
                    num_blocks = batch // base_batch
                    expected_indices = torch.arange(
                        base_batch, device=q.device, dtype=torch.long
                    ).repeat_interleave(num_blocks)
                    if not torch.equal(part.batch_indices, expected_indices):
                        raise ValueError("projected context batch_indices không tuần tự")
                    grouped_q = q.reshape(
                        base_batch, num_blocks, query_len, self.num_heads, self.head_dim
                    )
                    part_scores = torch.einsum(
                        "bnqhd,bchd->bnqhc", grouped_q, key
                    ).reshape(batch, query_len, self.num_heads, key.shape[1])
            else:
                part_scores = torch.einsum("bqhd,bqchd->bqhc", q, key)
            context_score_parts.append(part_scores)
        draft_scores = torch.einsum("bqhd,bqkhd->bqhk", q, k_draft)
        scores = torch.cat([*context_score_parts, draft_scores], dim=-1) * self.scaling

        if context_bias is not None:
            if context_bias.shape != (batch, query_len, context_len):
                raise ValueError("context_bias phải có dạng [B,Q,K_context]")
            scores = torch.cat(
                [
                    scores[..., :context_len] + context_bias.unsqueeze(2),
                    scores[..., context_len:],
                ],
                dim=-1,
            )
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)
            expected = (batch, 1, query_len, context_len + query_len)
            if attention_mask.shape != expected:
                raise ValueError(
                    f"attention_mask phải có shape {expected}, got {tuple(attention_mask.shape)}"
                )
            scores = scores + attention_mask.squeeze(1).unsqueeze(2)

        weights = scores.softmax(dim=-1)
        draft_weights = weights[..., context_len:]
        context_output = torch.zeros_like(q)
        offset = 0
        for part, part_len in zip(context_parts, context_lengths):
            part_weights = weights[..., offset : offset + part_len]
            value = part.value
            if value.ndim == 4:
                if part.batch_indices is None:
                    part_output = torch.einsum("bqhc,bchd->bqhd", part_weights, value)
                else:
                    base_batch = value.shape[0]
                    num_blocks = batch // base_batch
                    grouped_weights = part_weights.reshape(
                        base_batch, num_blocks, query_len, self.num_heads, part_len
                    )
                    part_output = torch.einsum(
                        "bnqhc,bchd->bnqhd", grouped_weights, value
                    ).reshape(batch, query_len, self.num_heads, self.head_dim)
            else:
                part_output = torch.einsum(
                    "bqhc,bqchd->bqhd", part_weights, value
                )
            context_output = context_output + part_output
            offset += part_len
        draft_output = torch.einsum("bqhk,bqkhd->bqhd", draft_weights, v_draft)
        output = context_output + draft_output
        return self.o_proj(output.reshape(batch, query_len, -1))


class MRDraftStage(nn.Module):
    """Một stage DFlash joint attention → FFN, route HCA hoặc CSA."""

    def __init__(self, spec: "MRDraftSpec", route: str) -> None:
        super().__init__()
        if route not in {"hca", "csa"}:
            raise ValueError(f"route không hợp lệ: {route}")
        self.route = route
        self.input_layernorm = RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.joint_attn = MRDFlashJointAttention(
            spec.hidden_size,
            spec.num_attention_heads,
            spec.num_key_value_heads,
            spec.head_dim or spec.hidden_size // spec.num_attention_heads,
            rope_theta=spec.rope_theta,
            use_qk_norm=spec.use_qk_norm,
            rms_norm_eps=spec.rms_norm_eps,
            bias=spec.attention_bias,
        )
        self.post_attention_layernorm = RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.mlp = SwiGLUMLP(spec.hidden_size, spec.intermediate_size)

    @staticmethod
    def _query_view(
        values: torch.Tensor,
        positions: torch.Tensor,
        query_len: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del block_size
        if values.ndim == 3 and positions.ndim == 2:
            return values, positions
        if values.ndim == 4 and positions.ndim == 3:
            if values.shape[:2] != (values.shape[0], query_len):
                raise ValueError("per-query memory không khớp số draft query")
            return values, positions
        raise ValueError(
            "memory view phải có dạng [B,K,H] hoặc [B,Q,K,H] và positions tương ứng"
        )

    @staticmethod
    def _gather_projected(
        values: torch.Tensor,
        indices: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather ``[B,C,H,D]`` projected KV thành ``[B,Q,K,H,D]``."""
        batch, query_len, top_k = indices.shape
        if values.ndim != 4:
            raise ValueError("projected memory không khớp indices")
        if values.shape[0] == batch and batch_indices is None:
            return torch.gather(
                values.unsqueeze(1).expand(-1, query_len, -1, -1, -1),
                2,
                indices.unsqueeze(-1).unsqueeze(-1).expand(
                    batch, query_len, top_k, values.shape[2], values.shape[3]
                ),
            )
        if batch_indices is None:
            raise ValueError("projected memory cần batch_indices khi batch khác nhau")
        base_batch = values.shape[0]
        if batch % base_batch or batch_indices.shape != (batch,):
            raise ValueError("batch_indices không khớp projected memory")
        num_blocks = batch // base_batch
        expected_indices = torch.arange(
            base_batch, device=indices.device, dtype=torch.long
        ).repeat_interleave(num_blocks)
        if not torch.equal(batch_indices, expected_indices):
            raise ValueError("batch_indices phải ánh xạ tuần tự các block")
        grouped_values = values.unsqueeze(1).unsqueeze(2).expand(
            -1, num_blocks, query_len, -1, -1, -1
        )
        grouped_indices = indices.reshape(
            base_batch, num_blocks, query_len, top_k
        )
        selected = torch.gather(
            grouped_values,
            3,
            grouped_indices.unsqueeze(-1).unsqueeze(-1).expand(
                base_batch,
                num_blocks,
                query_len,
                top_k,
                values.shape[2],
                values.shape[3],
            ),
        )
        return selected.reshape(batch, query_len, top_k, values.shape[2], values.shape[3])

    @staticmethod
    def _query_batch_positions(
        positions: torch.Tensor,
        batch_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if batch_indices is None:
            return positions
        return positions.index_select(0, batch_indices)

    def _context(
        self,
        hidden: torch.Tensor,
        memory: MRMemoryState,
        anchor_positions: torch.Tensor,
        indexer: nn.Module,
        csa_top_k: int,
        block_size: int,
        indexer_mode: str,
    ) -> tuple[
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[tuple[MRProjectedContextPart, ...]],
    ]:
        batch, query_len, _ = hidden.shape
        if self.route == "hca":
            global_values, global_pos = self._query_view(
                memory.hca, memory.hca_positions, query_len, block_size
            )
            local_values, local_pos = self._query_view(
                memory.local_hca, memory.local_positions, query_len, block_size
            )
            global_k, global_v = self.joint_attn.project_context(
                memory.hca, memory.hca_positions
            )
            local_k, local_v = self.joint_attn.project_context(
                local_values, local_pos
            )
            global_pos = self._query_batch_positions(
                memory.hca_positions, memory.block_batch_indices
            )
            return (
                None,
                torch.cat([global_pos, local_pos], dim=1),
                None,
                (
                    MRProjectedContextPart(
                        global_k, global_v, memory.block_batch_indices
                    ),
                    MRProjectedContextPart(local_k, local_v),
                ),
            )

        local_values, local_pos = self._query_view(
            memory.local_csa, memory.local_positions, query_len, block_size
        )
        if memory.csa.shape[1] == 0:
            return local_values, local_pos, None, None

        global_values = memory.csa
        global_base_pos = memory.csa_positions
        global_pos = self._query_batch_positions(
            global_base_pos, memory.block_batch_indices
        )
        allowed = global_pos.unsqueeze(1) < anchor_positions.unsqueeze(-1)
        raw_scores = indexer.score(
            hidden,
            global_values,
            batch_indices=memory.block_batch_indices,
        )
        scores = raw_scores.masked_fill(~allowed, 0.0)
        if indexer_mode == "dense":
            selected_values, selected_pos = global_values, global_pos
            bias = torch.cat(
                [
                    torch.zeros(
                        (batch, query_len, local_pos.shape[-1]),
                        device=hidden.device,
                        dtype=hidden.dtype,
                    ),
                    scores,
                ],
                dim=-1,
            )
            global_k, global_v = self.joint_attn.project_context(
                global_values, memory.csa_positions
            )
            local_k, local_v = self.joint_attn.project_context(
                local_values, local_pos
            )
            return (
                None,
                torch.cat([local_pos, global_pos], dim=1),
                bias,
                (
                    MRProjectedContextPart(local_k, local_v),
                    MRProjectedContextPart(
                        global_k, global_v, memory.block_batch_indices
                    ),
                ),
            )
        elif indexer_mode == "topk":
            masked_scores = raw_scores.masked_fill(
                ~allowed, torch.finfo(hidden.dtype).min
            )
            k = min(csa_top_k, global_values.shape[1])
            top_scores, indices = masked_scores.topk(k=k, dim=-1)
            selected_pos = torch.gather(
                global_pos.unsqueeze(1).expand(-1, query_len, -1), 2, indices
            )
            bias = torch.cat(
                [
                    torch.zeros(
                        (batch, query_len, local_pos.shape[-1]),
                        device=hidden.device,
                        dtype=hidden.dtype,
                    ),
                    top_scores,
                ],
                dim=-1,
            )
            local_k, local_v = self.joint_attn.project_context(
                local_values, local_pos
            )
            global_k, global_v = self.joint_attn.project_context(
                global_values, global_base_pos
            )
            selected_k = self._gather_projected(
                global_k, indices, memory.block_batch_indices
            )
            selected_v = self._gather_projected(
                global_v, indices, memory.block_batch_indices
            )
            local_k = local_k.unsqueeze(1).expand(-1, query_len, -1, -1, -1)
            local_v = local_v.unsqueeze(1).expand(-1, query_len, -1, -1, -1)
            projected_context = (
                MRProjectedContextPart(local_k, local_v),
                MRProjectedContextPart(selected_k, selected_v),
            )
            context_positions = torch.cat([local_pos.unsqueeze(1).expand(-1, query_len, -1), selected_pos], dim=2)
            return None, context_positions, bias, projected_context
        else:
            raise ValueError("indexer_mode phải là 'dense' hoặc 'topk'")

    def forward(
        self,
        hidden: torch.Tensor,
        memory: MRMemoryState,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        anchor_positions: torch.Tensor,
        indexer: nn.Module,
        csa_top_k: int,
        block_size: int,
        indexer_mode: str,
    ) -> torch.Tensor:
        normalized = self.input_layernorm(hidden)
        context, context_positions, context_bias, projected_context = self._context(
            normalized,
            memory,
            anchor_positions,
            indexer,
            csa_top_k,
            block_size,
            indexer_mode,
        )
        draft_mask = attention_mask
        if draft_mask is not None:
            if draft_mask.ndim != 4:
                raise ValueError("attention_mask phải có dạng [B,1,Q,Q]")
            if draft_mask.shape[-1] != hidden.shape[1]:
                draft_mask = draft_mask[..., -hidden.shape[1] :]
        if context_positions.ndim == 2:
            context_allow = context_positions.unsqueeze(1) < anchor_positions.unsqueeze(-1)
        else:
            context_allow = context_positions < anchor_positions.unsqueeze(-1)
        context_mask = torch.zeros(
            (hidden.shape[0], 1, hidden.shape[1], context_allow.shape[-1]),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        context_mask = context_mask.masked_fill(
            ~context_allow.unsqueeze(1), torch.finfo(hidden.dtype).min
        )
        if draft_mask is None:
            draft_mask = torch.zeros(
                (hidden.shape[0], 1, hidden.shape[1], hidden.shape[1]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        joint_mask = torch.cat([context_mask, draft_mask], dim=-1)
        residual = hidden
        hidden = residual + self.joint_attn(
            normalized,
            context,
            position_ids,
            context_positions,
            attention_mask=joint_mask,
            context_bias=context_bias,
            projected_context=projected_context,
        )
        residual = hidden
        hidden = residual + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


@dataclass
class MRDraftSpec(DraftSpec):
    """DraftSpec mở rộng, giữ nguyên các tham số DFlash và thêm MR knobs."""

    num_stages: int = 2
    hca_compression_ratio: int = 128
    csa_compression_ratio: int = 4
    local_window: int = 128
    csa_top_k: int = 64
    indexer_dim: Optional[int] = None
    indexer_num_heads: int = 1

    def __post_init__(self) -> None:
        if self.block_size < 2:
            raise ValueError("block_size phải >= 2")
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers phải >= 1")
        if self.num_stages < 2:
            raise ValueError("MR-DFlash cần ít nhất 2 stages: HCA và CSA")
        for name in (
            "hca_compression_ratio",
            "csa_compression_ratio",
            "local_window",
            "csa_top_k",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} phải >= 1")
        if self.indexer_dim is not None and self.indexer_dim < 1:
            raise ValueError("indexer_dim phải dương hoặc null")
        if self.indexer_num_heads < 1:
            raise ValueError("indexer_num_heads phải >= 1")

    @classmethod
    def from_dflash(cls, spec: DraftSpec, **kwargs) -> "MRDraftSpec":
        values = {field.name: getattr(spec, field.name) for field in fields(DraftSpec)}
        values.update(kwargs)
        return cls(**values)


class MRDFlashDraftModel(nn.Module):
    """MR-DFlash draft model nhận memory đã build từ target hidden features."""

    def __init__(self, spec: MRDraftSpec) -> None:
        super().__init__()
        self.spec = spec
        self.block_size = spec.block_size
        self.memory = MRTargetMemory(
            input_dim=spec.context_feature_dim,
            hidden_size=spec.hidden_size,
            hca_compression_ratio=spec.hca_compression_ratio,
            csa_compression_ratio=spec.csa_compression_ratio,
            local_window=spec.local_window,
            csa_top_k=spec.csa_top_k,
            indexer_dim=spec.indexer_dim,
            indexer_num_heads=spec.indexer_num_heads,
        )
        routes = ["hca" if index % 2 == 0 else "csa" for index in range(spec.num_stages)]
        self.stages = nn.ModuleList([MRDraftStage(spec, route) for route in routes])
        self.norm = RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)

    def build_memory(
        self,
        target_hidden: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        query_positions: Optional[torch.Tensor] = None,
    ) -> MRMemoryState:
        return self.memory.build(
            target_hidden, positions=positions, query_positions=query_positions
        )

    def append_memory(
        self,
        state: MRMemoryState,
        target_hidden: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> MRMemoryState:
        return self.memory.append(state, target_hidden, positions=positions)

    def forward(
        self,
        *,
        noise_embedding: torch.Tensor,
        memory: MRMemoryState,
        position_ids: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        indexer_mode: str = "topk",
    ) -> torch.Tensor:
        if noise_embedding.ndim != 3:
            raise ValueError("noise_embedding phải có dạng [B,L,H]")
        batch, length, hidden_size = noise_embedding.shape
        if hidden_size != self.spec.hidden_size:
            raise ValueError("noise_embedding không khớp hidden_size")
        if position_ids.ndim != 2 or position_ids.shape[:1] != (batch,):
            raise ValueError("position_ids phải có dạng [B,L] hoặc [B,S+L]")
        draft_positions = position_ids[:, -length:]
        if length % self.block_size:
            raise ValueError("noise_embedding length phải chia hết cho block_size")
        restore_shape = None
        if length > self.block_size:
            # Backward-compatible path cho callers cũ truyền toàn bộ anchor
            # blocks trong một sequence. MR training dùng path mới trực tiếp
            # với [B*N,K,H]; inference thường chỉ có một block.
            num_blocks = length // self.block_size
            restore_shape = (batch, length)
            memory = memory.flatten_blocks(num_blocks)
            noise_embedding = noise_embedding.reshape(
                batch * num_blocks, self.block_size, hidden_size
            )
            draft_positions = draft_positions.reshape(
                batch * num_blocks, self.block_size
            )
            if isinstance(attention_mask, dict):
                attention_mask = attention_mask.get("full_attention")
            if attention_mask is not None:
                if attention_mask.ndim != 4:
                    raise ValueError("attention_mask phải có dạng [B,1,L,L]")
                # Nếu caller truyền mask có phần context, chỉ giữ draft ×
                # draft trước khi lấy các block diagonal.
                attention_mask = attention_mask[..., -length:, -length:]
                attention_mask = torch.stack(
                    [
                        attention_mask[
                            :, :, start : start + self.block_size,
                            start : start + self.block_size,
                        ]
                        for start in range(0, length, self.block_size)
                    ],
                    dim=1,
                ).reshape(batch * num_blocks, 1, self.block_size, self.block_size)
            batch = batch * num_blocks
            length = self.block_size
        anchor_positions = draft_positions.view(
            batch, length // self.block_size, self.block_size
        )[:, :, 0].repeat_interleave(self.block_size, dim=1)
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask.get("full_attention")
        if attention_mask is not None:
            if attention_mask.ndim != 4 or attention_mask.shape[0] != batch:
                raise ValueError("attention_mask phải có dạng [B,1,L,L]")
            if attention_mask.shape[-1] != length:
                attention_mask = attention_mask[..., -length:]
        hidden = noise_embedding
        for stage in self.stages:
            hidden = stage(
                hidden,
                memory,
                draft_positions,
                attention_mask,
                anchor_positions,
                self.memory.indexer,
                self.spec.csa_top_k,
                self.block_size,
                indexer_mode,
            )
        hidden = self.norm(hidden)
        if restore_shape is not None:
            original_batch, original_length = restore_shape
            hidden = hidden.reshape(original_batch, original_length, hidden_size)
        return hidden

    def init_from_target(
        self,
        target_model: nn.Module,
        target_layer_ids: Optional[List[int]] = None,
    ) -> List[str]:
        """Copy common attention/FFN weights vào các stage MR."""
        target_layers = getattr(getattr(target_model, "model", target_model), "layers", None)
        if target_layers is None:
            raise ValueError("target model không có .model.layers")
        copied: List[str] = []
        current = self.state_dict()
        source_layer_ids = target_layer_ids or self.spec.target_layer_ids[: len(self.stages)]
        if len(source_layer_ids) != len(self.stages):
            raise ValueError(
                "target_layer_ids cho init phải có đúng số MR stage: "
                f"{len(source_layer_ids)} != {len(self.stages)}"
            )
        for stage_idx, stage in enumerate(self.stages):
            target_id = source_layer_ids[stage_idx]
            if target_id >= len(target_layers):
                raise ValueError(
                    f"target layer id {target_id} vượt số layer target {len(target_layers)}"
                )
            source = target_layers[target_id].state_dict()
            prefix = f"stages.{stage_idx}."
            mapping = {
                "input_layernorm": "input_layernorm",
                "joint_attn": "self_attn",
                "post_attention_layernorm": "post_attention_layernorm",
                "mlp": "mlp",
            }
            for destination_group, source_group in mapping.items():
                destination_prefix = prefix + destination_group + "."
                for destination_key in list(current):
                    if not destination_key.startswith(destination_prefix):
                        continue
                    suffix = destination_key[len(destination_prefix) :]
                    source_key = source_group + "." + suffix
                    if source_key not in source or source[source_key].shape != current[destination_key].shape:
                        continue
                    with torch.no_grad():
                        current[destination_key].copy_(source[source_key].to(current[destination_key]))
                    copied.append(destination_key)
        self.load_state_dict(current, strict=False)
        return copied


__all__ = [
    "MRBlockAttention",
    "MRDFlashDraftModel",
    "MRDFlashJointAttention",
    "MRDraftSpec",
    "MRTargetAttention",
]
