"""Self-contained SpecForge-compatible DFlash training components."""

from .dflash_kernels import DEFAULT_DFLASH_KERNELS, DFlashKernels
from .model import (
    DFlashDraftModel,
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
    apply_rotary_pos_emb,
    build_target_layer_ids,
    extract_context_feature,
    normalize_draft_head_checkpoint_keys,
    resolve_dflash_attention_layout,
    sample,
)

__all__ = [
    "DEFAULT_DFLASH_KERNELS",
    "DFlashKernels",
    "DFlashDraftModel",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "apply_rotary_pos_emb",
    "build_target_layer_ids",
    "extract_context_feature",
    "normalize_draft_head_checkpoint_keys",
    "resolve_dflash_attention_layout",
    "sample",
]
