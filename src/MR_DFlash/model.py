"""Draft model DFlash — bản tự đóng gói từ ``specforge/modeling/draft/dflash.py``.

Kiến trúc giữ nguyên logic SpecForge nhưng chỉ dùng ``torch`` (không phụ thuộc
``transformers.models.qwen3``): RMSNorm, SwiGLU MLP, RoPE và attention kiểu
Qwen3 đều tự triển khai, nên model chạy được trên CPU cho contract test và dễ
chỉnh sửa cho MR-DFlash. Attention dùng SDPA với additive mask dày đặc; mask
được dựng ở tầng huấn luyện (block-parallel) theo đúng ngữ nghĩa DFlash:
query draft chỉ attend context thật ``< anchor`` và các vị trí draft trước nó
trong cùng block.

Khác SpecForge ở chỗ:
- Không kế thừa ``Qwen3PreTrainedModel``; config truyền qua ``DraftSpec``.
- Có hook ``init_from_target`` (copy weight draft layer i từ target layer
  ``target_layer_ids[i]``) — đặt sẵn cho MR-DFlash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import math

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Các khối cơ bản
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    """RMSNorm không bias, giống transformers (variance không chia N)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class SwiGLUMLP(nn.Module):
    """MLP gate SiLU kiểu Llama/Qwen3: down(gate(x) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Áp RoPE. q/k: (B, H, T, D); cos/sin: (B, T, D) (cùng batch)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _compute_rope_cache(
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tính cos/sin (fp32 rồi cast) cho từng vị trí tuyệt đối.

    ``position_ids``: (B, T) int. Trả về cos/sin (B, T, head_dim).
    """
    device = position_ids.device
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    # angles: (B, T, head_dim/2)
    angles = position_ids.unsqueeze(-1).float() * inv_freq
    emb = torch.cat((angles, angles), dim=-1)  # (B, T, head_dim)
    cos = torch.cos(emb).to(dtype)
    sin = torch.sin(emb).to(dtype)
    return cos, sin


# --------------------------------------------------------------------------- #
# Attention DFlash
# --------------------------------------------------------------------------- #

class DFlashAttention(nn.Module):
    """Attention 1 draft layer.

    Query lấy từ hidden của draft block; KV = concat(context target đã chiếu,
    hidden draft). Rotary: context dùng vị trí 0..S-1, draft dùng vị trí tuyệt
    đối của nó. ``cos/sin`` đầu vào đã ứng với combined position ids
    [context | noise]; ``ctx_len = S`` dùng để cắt phần context cho K.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,      # (B, L, H) đã qua input_layernorm
        target_hidden: torch.Tensor,      # (B, S, H) context đã chiếu
        attention_mask: Optional[torch.Tensor],  # (B,1,L,S+L) additive
        cos: torch.Tensor,                # (B, S+L, head_dim)
        sin: torch.Tensor,
        ctx_len: int,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        ctx_len_k = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1)  # (B, S+L, H)
        k = k.view(bsz, ctx_len_k + q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(k)

        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        v = torch.cat([v_ctx, v_noise], dim=1)
        v = v.view(bsz, ctx_len_k + q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # SDPA không broadcast GQA khi có attn_mask → repeat KV lên số query heads.
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # RoPE: cos/sin tương ứng [context(0..S-1) | noise]
        cos_k = cos.unsqueeze(1)  # (B,1,S+L,D)
        sin_k = sin.unsqueeze(1)
        q_cos = cos[:, ctx_len:, :].unsqueeze(1)  # (B,1,L,D)
        q_sin = sin[:, ctx_len:, :].unsqueeze(1)

        q = (q * q_cos) + (_rotate_half(q) * q_sin)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

        q = q * self.scaling
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class DFlashDecoderLayer(nn.Module):
    """Decoder layer: input_layernorm → attention → residual → MLP → residual."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
        sliding_window: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.sliding_window = sliding_window
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DFlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
            rms_norm_eps=rms_norm_eps,
            bias=bias,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx_len: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn = self.self_attn(
            hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
            ctx_len=ctx_len,
        )
        hidden_states = residual + attn
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------- #
# DraftSpec + DFlashDraftModel
# --------------------------------------------------------------------------- #

@dataclass
class DraftSpec:
    """Toàn bộ tham số kiến trúc draft DFlash (tự chứa, không cần HF config)."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    head_dim: Optional[int] = None
    use_qk_norm: bool = True
    #: số draft layer.
    num_hidden_layers: int = 1
    #: các layer target dùng làm context feature (concat) — cũng là vị trí copy
    #: weight khi init_draft_from_target.
    target_layer_ids: List[int] = field(default_factory=lambda: [0])
    layer_types: List[str] = field(default_factory=lambda: ["full_attention"])
    sliding_window: Optional[int] = None
    block_size: int = 16
    mask_token_id: Optional[int] = None
    attention_bias: bool = False

    @property
    def num_target_layers(self) -> int:
        return max(self.target_layer_ids) + 1

    @property
    def context_feature_dim(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size


def build_draft_spec(
    *,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int,
    num_target_layers: int,
    draft_num_hidden_layers: int,
    block_size: int,
    target_layer_ids: Optional[List[int]] = None,
    layer_types: Optional[List[str]] = None,
    sliding_window: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 1_000_000.0,
    head_dim: Optional[int] = None,
    use_qk_norm: bool = True,
    mask_token_id: Optional[int] = None,
    max_position_embeddings: int = 32768,
) -> DraftSpec:
    """Dựng DraftSpec từ tham số target; tự sinh target_layer_ids/layer_types."""
    from .config import build_target_layer_ids, resolve_dflash_attention_layout

    if target_layer_ids is None:
        target_layer_ids = build_target_layer_ids(
            num_target_layers, draft_num_hidden_layers
        )
    if layer_types is None:
        layer_types = ["full_attention"] * draft_num_hidden_layers
    resolve_dflash_attention_layout(layer_types, draft_num_hidden_layers, sliding_window)
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    return DraftSpec(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        head_dim=head_dim,
        use_qk_norm=use_qk_norm,
        num_hidden_layers=draft_num_hidden_layers,
        target_layer_ids=list(target_layer_ids),
        layer_types=list(layer_types),
        sliding_window=sliding_window,
        block_size=block_size,
        mask_token_id=mask_token_id,
        attention_bias=False,
    )


class DFlashDraftModel(nn.Module):
    """Draft model: projector + N decoder layer + final norm.

    Forward nhận ``noise_embedding`` (block đang dự đoán) + ``target_hidden``
    (feature concat các target layer, trước chiếu) + ``position_ids`` cho
    combined [context | noise] + additive ``attention_mask``.
    """

    def __init__(self, spec: DraftSpec) -> None:
        super().__init__()
        self.spec = spec
        self.block_size = spec.block_size
        self.mask_token_id = spec.mask_token_id
        self.target_layer_ids = list(spec.target_layer_ids)
        self.sliding_window = spec.sliding_window

        self.fc = nn.Linear(spec.context_feature_dim, spec.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)

        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    hidden_size=spec.hidden_size,
                    num_heads=spec.num_attention_heads,
                    num_kv_heads=spec.num_key_value_heads,
                    intermediate_size=spec.intermediate_size,
                    head_dim=spec.head_dim,
                    rms_norm_eps=spec.rms_norm_eps,
                    use_qk_norm=spec.use_qk_norm,
                    sliding_window=(
                        spec.sliding_window
                        if spec.layer_types[i] == "sliding_attention"
                        else None
                    ),
                    bias=spec.attention_bias,
                )
                for i in range(spec.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)

    def forward(
        self,
        *,
        noise_embedding: torch.Tensor,   # (B, L, H) L = N*block_size
        target_hidden: torch.Tensor,     # (B, S, feat_dim) feature thô của target
        position_ids: torch.Tensor,      # (B, S+L) vị trí tuyệt đối [ctx | noise]
        attention_mask: Union[
            Optional[torch.Tensor],
            Dict[str, Optional[torch.Tensor]],
        ] = None,  # additive (B,1,L,S+L) hoặc dict theo layer type
    ) -> torch.Tensor:
        spec = self.spec
        bsz = noise_embedding.shape[0]
        ctx_len = target_hidden.shape[1]
        device = noise_embedding.device

        # Chiếu feature concat của target → context hidden.
        ctx = self.hidden_norm(self.fc(target_hidden))  # (B,S,H)

        cos, sin = _compute_rope_cache(
            position_ids,
            spec.head_dim,
            spec.rope_theta,
            noise_embedding.dtype,
        )  # (B, S+L, head_dim)

        hidden = noise_embedding
        for layer_type, layer in zip(spec.layer_types, self.layers):
            layer_mask = (
                attention_mask[layer_type]
                if isinstance(attention_mask, dict)
                else attention_mask
            )
            hidden = layer(
                hidden,
                target_hidden=ctx,
                attention_mask=layer_mask,
                cos=cos,
                sin=sin,
                ctx_len=ctx_len,
            )
        return self.norm(hidden)

    def init_from_target(self, target_model: nn.Module) -> List[str]:
        """[MR hook] Copy weight draft layer i từ target layer target_layer_ids[i].

        Chỉ copy các key trùng tên (q/k/v/o_proj, gate/up/down_proj,
        input_layernorm, post_attention_layernorm). fc/hidden_norm/norm vẫn
        random. Trả về danh sách key đã copy.
        """
        copied: List[str] = []
        try:
            target_layers = target_model.model.layers
        except AttributeError:
            target_layers = getattr(target_model, "layers", None)
        if target_layers is None:
            raise ValueError(
                "target model không có .model.layers — không init_from_target được"
            )
        for i, (layer, target_id) in enumerate(
            zip(self.layers, self.target_layer_ids)
        ):
            if target_id >= len(target_layers):
                raise ValueError(
                    f"target_layer_ids[{i}]={target_id} vượt số layer target "
                    f"({len(target_layers)})"
                )
            src = target_layers[target_id].state_dict()
            dst = layer.state_dict()
            for key, value in src.items():
                if key in dst:
                    with torch.no_grad():
                        dst[key].copy_(value.to(dst[key].device).to(dst[key].dtype))
                    copied.append(f"layers.{i}.{key}")
        return copied


__all__ = [
    "RMSNorm",
    "SwiGLUMLP",
    "DFlashAttention",
    "DFlashDecoderLayer",
    "DraftSpec",
    "DFlashDraftModel",
    "build_draft_spec",
    "build_target_layer_ids",
]
