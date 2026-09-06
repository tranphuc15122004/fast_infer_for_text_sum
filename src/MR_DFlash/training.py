"""Wrapper huấn luyện OnlineDFlashModel + chiến lược DFlash.

Port tự đóng gói từ ``specforge/algorithms/common/dflash_family_model.py`` và
``specforge/training/strategies/base.py`` (DFlashTrainStrategy). Giữ nguyên
thuật toán block-parallel của DFlash:

1. Sample tối đa ``num_anchors`` anchor trên mỗi chuỗi, tại vị trí mà cả
   ``loss_mask[t]`` và ``loss_mask[t+1]`` đều được supervise.
2. Với mỗi anchor dựng một block ``block_size``: vị trí 0 = embedding token
   anchor, các vị trí còn lại = embedding ``mask_token``.
3. Mọi block chạy song song qua draft model; query chỉ attend context thật
   ``< anchor`` và các vị trí draft trước nó trong cùng block (không
   cross-block) → độ dài huấn luyện hiệu quả O(block) thay vì O(seq).
4. Label "same-position": vị trí k trong block dự đoán token thật tại
   anchor + k; loss = cross-entropy với label hard (không dùng target
   distribution), weight = keep * (k>0) * bounds * loss_mask[label], có thể
   nhân positional decay theo ``loss_decay_gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .chunking import checkpointed_chunk_reduce
from .model import DFlashDraftModel, DraftSpec
from .mr_model import MRDFlashDraftModel, MRDraftSpec

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:  # pragma: no cover - phụ thuộc phiên bản torch
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None  # type: ignore
    create_block_mask = None  # type: ignore

if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False

_VALID_LOSS_TYPES = {
    "dflash",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_DPACE_LOSS_TYPES = _VALID_LOSS_TYPES - {"dflash"}


# --------------------------------------------------------------------------- #
# Mask DFlash
# --------------------------------------------------------------------------- #

def build_dflash_additive_mask(
    anchor_positions: torch.Tensor,      # (B, N) long
    block_keep_mask: torch.Tensor,       # (B, N) bool
    S: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Additive mask dày đặc (B,1,Q,KV) cho SDPA; Q=N*bs, KV=S+Q.

    Ngữ nghĩa giống ``create_dflash_sdpa_mask`` của SpecForge:
    - draft query tại block q, offset k attend context kv < anchor (strict) và
      draft trong cùng block offset <= k;
    - chỉ các block được giữ (block_keep_mask) là hợp lệ.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + Q_LEN

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)

    q_block_ids = q_indices // block_size
    q_block_offsets = q_indices % block_size

    # anchor cho từng vị trí query (lặp lại theo offset trong block).
    anchor_expanded = (
        anchor_positions.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    )  # (B,1,Q,1)

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    if sliding_window is not None:
        context_lower_bound = (
            anchor_expanded + q_block_offsets - (sliding_window - 1)
        )
        mask_context = mask_context & (kv_indices >= context_lower_bound)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)
    if sliding_window is not None:
        kv_block_offsets = (kv_indices - S) % block_size
        mask_draft = mask_draft & (kv_block_offsets <= q_block_offsets)

    keep_expanded = (
        block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    )  # (B,1,Q,1)
    allow = (mask_context | mask_draft) & keep_expanded  # (B,1,Q,KV) bool

    additive = torch.zeros_like(allow, dtype=dtype)
    additive.masked_fill_(~allow, torch.finfo(dtype).min)
    return additive


def build_dflash_flex_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    sliding_window: Optional[int] = None,
):
    """BlockMask Flex Attention (bản GPU hiệu quả, không materialize mask dày).

    Port ``create_dflash_block_mask`` của SpecForge.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise ValueError("attention_backend=flex yêu cầu flex_attention khả dụng")
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + Q_LEN

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        q_block_offset = q_idx % block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        mask_context = is_context & (kv_idx < anchor_pos)
        if sliding_window is not None:
            context_lower_bound = (
                anchor_pos + q_block_offset - (sliding_window - 1)
            )
            mask_context = mask_context & (kv_idx >= context_lower_bound)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        if sliding_window is not None:
            kv_block_offset = (kv_idx - S) % block_size
            mask_draft = mask_draft & (kv_block_offset <= q_block_offset)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    return create_block_mask(
        dflash_mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )


# --------------------------------------------------------------------------- #
# OnlineDFlashModel
# --------------------------------------------------------------------------- #

class OnlineDFlashModel(nn.Module):
    """Draft + frozen target lm_head/embed_tokens + loss DFlash (online training).

    ``trainable_module()`` chỉ trả draft model; lm_head/embed_tokens là module
    frozen của target.
    """

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        *,
        block_size: Optional[int] = None,
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        loss_type: str = "dflash",
        attention_backend: str = "sdpa",
        dpace_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; phải thuộc {sorted(_VALID_LOSS_TYPES)}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha phải thuộc [0,1], got {dpace_alpha}")
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks phải >= 0")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = int(block_size or draft_model.block_size)
        self.mask_token_id = int(mask_token_id)
        self.num_anchors = int(num_anchors)
        self.loss_decay_gamma = loss_decay_gamma
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.loss_type = loss_type
        self.attention_backend = attention_backend
        self.dpace_alpha = dpace_alpha

        self._freeze_target()

    def _freeze_target(self) -> None:
        for module in (self.lm_head, self.embed_tokens):
            for param in module.parameters():
                param.requires_grad_(False)

    def trainable_module(self) -> nn.Module:
        return self.draft_model

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample anchor có cả token sạch (t) và token label (t+1) supervised."""
        num_candidates = max(seq_len - 1, 0)
        valid = (loss_mask[:, :num_candidates] > 0.5) & (
            loss_mask[:, 1 : num_candidates + 1] > 0.5
        )
        valid_counts = valid.sum(dim=1)
        width = min(self.num_anchors, int(valid_counts.max().item()))
        if width == 0:
            raise ValueError(
                "DFlash-family cần hai token được supervise liên tiếp"
            )

        random_values = torch.rand(valid.shape, device=device)
        random_values.masked_fill_(~valid, 2.0)
        candidates = random_values.argsort(dim=1)[:, :width]
        keep_mask = (
            torch.arange(width, device=device).unsqueeze(0)
            < valid_counts.clamp(max=width).unsqueeze(1)
        )

        sentinel = valid.shape[1]
        anchors = torch.where(
            keep_mask,
            candidates,
            torch.full_like(candidates, sentinel),
        )
        anchors = anchors.sort(dim=1).values
        keep_mask = anchors < sentinel
        return torch.where(keep_mask, anchors, 0), keep_mask

    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Block embedding: offset 0 = token anchor (nếu block được giữ)."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )
        block_starts = (
            torch.arange(n, device=device) * bs
        ).unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.full_like(anchor_tokens, self.mask_token_id),
        )
        return self.embed_tokens(noise_ids)

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = (
            anchor_positions.unsqueeze(-1)
            + torch.arange(self.block_size, device=device).view(1, 1, -1)
        ).view(bsz, -1)
        full_position_ids = torch.cat(
            [context_position_ids, draft_position_ids], dim=1
        )

        spec = self.draft_model.spec
        sliding_window = spec.sliding_window
        if self.attention_backend == "flex":
            full_mask = build_dflash_flex_block_mask(
                anchor_positions,
                block_keep_mask,
                seq_len,
                self.block_size,
                device,
            )
            attn_mask: Any = full_mask
            if sliding_window is not None:
                attn_mask = {
                    "full_attention": full_mask,
                    "sliding_attention": build_dflash_flex_block_mask(
                        anchor_positions,
                        block_keep_mask,
                        seq_len,
                        self.block_size,
                        device,
                        sliding_window=sliding_window,
                    ),
                }
        else:  # sdpa
            dtype = next(self.draft_model.parameters()).dtype
            full_mask = build_dflash_additive_mask(
                anchor_positions,
                block_keep_mask,
                seq_len,
                self.block_size,
                device,
                dtype,
            )
            attn_mask = full_mask
            if sliding_window is not None:
                attn_mask = {
                    "full_attention": full_mask,
                    "sliding_attention": build_dflash_additive_mask(
                        anchor_positions,
                        block_keep_mask,
                        seq_len,
                        self.block_size,
                        device,
                        dtype,
                        sliding_window=sliding_window,
                    ),
                }

        output_hidden = self.draft_model(
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            position_ids=full_position_ids,
            attention_mask=attn_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden, n_blocks

    def _objective_chunk_terms(
        self,
        hidden: torch.Tensor,     # (B, n_chunk, bs, H)
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Trả (loss_num, loss_den, correct_num, accuracy_den) cho 1 lát block."""
        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        neg_log_q = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)

        if self.loss_type == "dflash":
            loss_weights = weight_mask
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                positions = torch.arange(
                    block_size, device=hidden.device
                ).view(1, 1, -1)
                decay_weights = torch.exp(
                    -(positions - 1).clamp(min=0).float() / self.loss_decay_gamma
                )
                loss_weights = loss_weights * decay_weights
            loss_num = (neg_log_q * loss_weights).sum()
            loss_den = loss_weights.sum()
        elif self.loss_type in _DPACE_LOSS_TYPES:
            with torch.no_grad():
                target_probability = torch.exp(-neg_log_q)
                dpace_weights = self._dpace_weight(
                    target_probability,
                    weight_mask,
                    weight_mask > 0,
                    self.loss_type,
                )
            loss_num = (neg_log_q * weight_mask * dpace_weights).sum()
            loss_den = loss_num.new_zeros(())
        else:  # defensive
            raise ValueError(f"unknown loss_type {self.loss_type!r}")

        with torch.no_grad():
            predicted_ids = logits.argmax(dim=-1)
            correct_num = (
                ((predicted_ids == target_ids) & (weight_mask > 0.5)).sum().float()
            )
            accuracy_den = weight_mask.sum()
        return loss_num, loss_den, correct_num, accuracy_den

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Detached D-PACE position weights (xem SpecForge dflash_family_model)."""
        smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
        smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)
        if loss_type == "dpace-cumulative-confidence-only":
            return prefix
        suffix = torch.flip(
            torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
            dims=[-1],
        )
        if loss_type == "dpace":
            return suffix
        if loss_type == "dpace-continuation-value-only":
            return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Forward block-wise song song; trả (loss, accuracy, metrics)."""
        if self.attention_backend == "flex" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention không khả dụng trên device này; dùng sdpa."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden, n_blocks = (
            self._forward_draft_blocks(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )
        )

        # --- Labels: vị trí k dự đoán token thật tại anchor+k ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )

        # --- Weight: block_keep * bounds * (k>0) * loss_mask tại vị trí label ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()
        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        original_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask

        hidden_4d = output_hidden.reshape(bsz, n_blocks, self.block_size, -1)
        loss_num, loss_den, correct_num, accuracy_denom = checkpointed_chunk_reduce(
            self._objective_chunk_terms,
            hidden_4d,
            target_ids,
            weight_mask,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )

        ratio_metrics = {
            "acc": (correct_num.detach(), accuracy_denom.detach()),
        }
        metrics: Dict[str, object] = {
            "accuracy_denom": accuracy_denom.detach(),
            "ratio_metrics": ratio_metrics,
        }
        loss_denominator = (
            loss_den if self.loss_type == "dflash" else loss_num.new_tensor(float(bsz))
        )
        loss = loss_num / loss_denominator
        metrics["loss_terms"] = (loss_num, loss_denominator.detach())
        accuracy = correct_num / accuracy_denom
        return loss, accuracy, metrics


class OnlineMRDFlashModel(OnlineDFlashModel):
    """Online wrapper MR-DFlash, giữ nguyên anchor và DFlash objective.

    Khác biệt duy nhất với ``OnlineDFlashModel`` là target feature được build
    thành ``MRMemoryState`` trước khi chạy draft. Noise embedding, label,
    weight mask và loss vẫn dùng chính implementation DFlash bên trên.
    """

    draft_model: MRDFlashDraftModel

    def __init__(
        self,
        draft_model: MRDFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            draft_model,
            target_lm_head,
            target_embed_tokens,
            mask_token_id,
            **kwargs,
        )

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = (
            anchor_positions.unsqueeze(-1)
            + torch.arange(self.block_size, device=device).view(1, 1, -1)
        ).reshape(bsz, -1)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        memory = self.draft_model.build_memory(hidden_states)

        dtype = next(self.draft_model.parameters()).dtype
        full_mask = build_dflash_additive_mask(
            anchor_positions,
            block_keep_mask,
            seq_len,
            self.block_size,
            device,
            dtype,
        )
        output_hidden = self.draft_model(
            noise_embedding=noise_embedding,
            memory=memory,
            position_ids=full_position_ids,
            attention_mask=full_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden, n_blocks


# --------------------------------------------------------------------------- #
# Batch + StepOutput + Strategy (cầu nối vào trainer spine)
# --------------------------------------------------------------------------- #

@dataclass
class TrainBatch:
    """Một micro-batch chuẩn hoá; tensor lưu trong ``tensors``."""

    tensors: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepOutput:
    """Kết quả một bước: loss + metrics (giữ generic cho trainer)."""

    loss: torch.Tensor
    metrics: Dict[str, Any]
    ratio_metrics: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    loss_terms: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


@dataclass(frozen=True)
class StepContext:
    """Trạng thái lịch truyền vào forward_loss (Domino dùng; DFlash bỏ qua)."""

    global_step: int = 0
    total_steps: Optional[int] = None


class DFlashTrainStrategy:
    """Chiến lược DFlash: biến TrainBatch thành loss.

    Tương đương ``DFlashTrainStrategy`` trong SpecForge: required_features =
    {input_ids, hidden_states, loss_mask}; chỉ weight dưới ``draft_model``
    được persist làm draft weights.
    """

    name = "dflash"
    required_features = {"input_ids", "hidden_states", "loss_mask"}

    def __init__(self, model: OnlineDFlashModel) -> None:
        self.model = model
        self._forward_model: nn.Module = model

    def trainable_module(self) -> nn.Module:
        return self.model.trainable_module()

    def set_forward_model(self, model: nn.Module) -> None:
        """Đặt wrapper forward (DDP) nhưng giữ draft làm optimizer module."""
        self._forward_model = model

    def validate_batch(self, batch: TrainBatch) -> None:
        missing = {
            f for f in self.required_features if f not in batch.tensors
        }
        if missing:
            raise ValueError(
                f"batch thiếu feature {sorted(missing)}; "
                f"có={sorted(batch.tensors)}"
            )

    def forward_loss(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepOutput:
        del ctx
        self.validate_batch(batch)
        t = batch.tensors
        trainable = self.trainable_module()
        device = next(trainable.parameters()).device
        dtype = next(trainable.parameters()).dtype
        loss, accuracy, model_metrics = self._forward_model(
            input_ids=t["input_ids"].to(device=device, dtype=torch.long),
            hidden_states=t["hidden_states"].to(device=device, dtype=dtype),
            loss_mask=t["loss_mask"].to(device=device, dtype=dtype),
        )
        metrics = {"accuracy": accuracy.detach()}
        if "accuracy_denom" in model_metrics:
            metrics["accuracy_denom"] = model_metrics["accuracy_denom"]
        return StepOutput(
            loss=loss,
            metrics=metrics,
            ratio_metrics=model_metrics.get("ratio_metrics", {}),
            loss_terms=model_metrics.get("loss_terms"),
        )

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Chỉ giữ weight dưới ``draft_model`` (không persist target head)."""
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("draft_model.")
        }


def build_draft_spec_from_target_config(
    target_config: Any,
    *,
    draft_num_hidden_layers: int,
    block_size: int,
    target_layer_ids: Optional[list] = None,
    layer_types: Optional[list] = None,
    sliding_window: Optional[int] = None,
    mask_token_id: Optional[int] = None,
) -> DraftSpec:
    """Dựng DraftSpec từ HF target config (Qwen3/Llama tương thích)."""
    from .config import build_target_layer_ids
    from .model import build_draft_spec

    tc = getattr(target_config, "text_config", target_config)
    hidden_size = int(tc.hidden_size)
    num_heads = int(tc.num_attention_heads)
    kv_heads = int(getattr(tc, "num_key_value_heads", num_heads))
    intermediate = int(tc.intermediate_size)
    head_dim = getattr(tc, "head_dim", None)
    num_layers = int(tc.num_hidden_layers)

    resolved_layer_ids = target_layer_ids or build_target_layer_ids(
        num_layers, draft_num_hidden_layers
    )
    resolved_layer_types = layer_types or ["full_attention"] * draft_num_hidden_layers
    resolved_sliding = sliding_window
    if "sliding_attention" in resolved_layer_types:
        resolved_sliding = resolved_sliding or int(
            getattr(tc, "sliding_window", None)
        )

    use_qk_norm = bool(
        getattr(tc, "use_qk_norm", True)
        or getattr(tc, "qk_norm", False)
    )
    return build_draft_spec(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=kv_heads,
        intermediate_size=intermediate,
        num_target_layers=num_layers,
        draft_num_hidden_layers=draft_num_hidden_layers,
        block_size=block_size,
        target_layer_ids=resolved_layer_ids,
        layer_types=resolved_layer_types,
        sliding_window=resolved_sliding,
        rms_norm_eps=float(tc.rms_norm_eps),
        rope_theta=float(getattr(tc, "rope_theta", 1_000_000.0)),
        head_dim=head_dim,
        use_qk_norm=use_qk_norm,
        mask_token_id=mask_token_id,
        max_position_embeddings=int(tc.max_position_embeddings),
    )


class MRDFlashTrainStrategy(DFlashTrainStrategy):
    """Strategy MR-DFlash dùng cùng feature contract với DFlash."""

    name = "mr_dflash"

    def __init__(self, model: OnlineMRDFlashModel) -> None:
        super().__init__(model)


def build_mr_draft_spec_from_target_config(
    target_config: Any,
    *,
    draft_num_hidden_layers: int,
    block_size: int,
    target_layer_ids: Optional[list] = None,
    layer_types: Optional[list] = None,
    sliding_window: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    num_stages: int = 2,
    hca_compression_ratio: int = 128,
    csa_compression_ratio: int = 4,
    local_window: int = 128,
    csa_top_k: int = 64,
    indexer_dim: Optional[int] = None,
) -> MRDraftSpec:
    """Dựng MRDraftSpec từ HF config, kế thừa toàn bộ DFlash knobs."""
    base = build_draft_spec_from_target_config(
        target_config,
        draft_num_hidden_layers=draft_num_hidden_layers,
        block_size=block_size,
        target_layer_ids=target_layer_ids,
        layer_types=layer_types,
        sliding_window=sliding_window,
        mask_token_id=mask_token_id,
    )
    return MRDraftSpec.from_dflash(
        base,
        num_stages=num_stages,
        hca_compression_ratio=hca_compression_ratio,
        csa_compression_ratio=csa_compression_ratio,
        local_window=local_window,
        csa_top_k=csa_top_k,
        indexer_dim=indexer_dim,
    )


__all__ = [
    "OnlineDFlashModel",
    "OnlineMRDFlashModel",
    "DFlashTrainStrategy",
    "MRDFlashTrainStrategy",
    "TrainBatch",
    "StepOutput",
    "StepContext",
    "build_dflash_additive_mask",
    "build_dflash_flex_block_mask",
    "build_draft_spec_from_target_config",
    "build_mr_draft_spec_from_target_config",
]
