"""Adapter around the repository's pretrained five-layer DFlash checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
DFLASH_ROOT = ROOT / "externals" / "dflash"
if str(DFLASH_ROOT) not in sys.path:
    sys.path.insert(0, str(DFLASH_ROOT))


def extract_target_features(outputs: Any, layer_ids: list[int]) -> torch.Tensor:
    """Concatenate target hidden states using DFlash's hidden-state +1 offset."""
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("target output is missing hidden_states")
    if not layer_ids:
        raise ValueError("DFlash checkpoint has no target feature layers")
    try:
        return torch.cat([hidden_states[int(layer_id) + 1] for layer_id in layer_ids], dim=-1)
    except IndexError as exc:
        raise ValueError("DFlash feature layer id exceeds target hidden states") from exc


class AMRDFlashDraft(nn.Module):
    """Run original DFlash layers over a compact projected context.

    ``project_context`` applies the checkpoint's original ``fc`` and
    ``hidden_norm`` once to the full feature bank.  Selection and slots then
    gather/aggregate in that projected space.  The DFlash decoder layers,
    RoPE, live-block semantics and final normalization remain the pretrained
    modules, and callers provide original logical positions for every key.
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base = base_model
        if int(getattr(base_model, "block_size", 0)) != 16:
            raise ValueError(
                f"AMR-DFlash V0 requires the pretrained 16-position block; got {getattr(base_model, 'block_size', None)}"
            )
        if len(base_model.layers) != 5:
            raise ValueError(
                f"AMR-DFlash V0 requires all 5 pretrained draft layers; got {len(base_model.layers)}"
            )

    @property
    def block_size(self) -> int:
        return int(self.base.block_size)

    @property
    def target_layer_ids(self) -> list[int]:
        return [int(value) for value in self.base.target_layer_ids]

    @property
    def mask_token_id(self) -> int:
        value = self.base.mask_token_id
        if value is None:
            raise ValueError("DFlash checkpoint does not define mask_token_id")
        return int(value)

    @property
    def hidden_size(self) -> int:
        return int(self.base.config.hidden_size)

    def project_context(self, target_features: torch.Tensor) -> torch.Tensor:
        return self.base.hidden_norm(self.base.fc(target_features))

    def forward_projected(
        self,
        *,
        projected_context: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if projected_context.ndim != 3 or noise_embedding.ndim != 3:
            raise ValueError("projected context and noise embeddings must be [B,L,H]")
        batch, query_length, hidden_size = noise_embedding.shape
        if projected_context.shape[0] != batch or projected_context.shape[2] != hidden_size:
            raise ValueError("context/noise batch and hidden dimensions must match")
        expected_positions = projected_context.shape[1] + query_length
        if position_ids.shape != (batch, expected_positions):
            raise ValueError(
                f"position_ids must be [B,{expected_positions}] for compact context + live block"
            )
        if attention_mask.shape != (
            batch,
            1,
            query_length,
            expected_positions,
        ):
            raise ValueError("attention mask must align with compact context and live block")

        hidden_states = noise_embedding
        position_embeddings = self.base.rotary_emb(hidden_states, position_ids)
        for layer in self.base.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=projected_context,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        return self.base.norm(hidden_states)


def load_pretrained_models(
    target_path: str | Path,
    draft_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attention_backend: str = "sdpa",
) -> tuple[Any, Any, AMRDFlashDraft]:
    """Load only local model/tokenizer assets and validate the DFlash contract."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel

    target_path = str(target_path)
    draft_path = str(draft_path)
    if not Path(target_path).exists() or not Path(draft_path).exists():
        raise FileNotFoundError(
            "target and DFlash draft must be local paths when the server is offline"
        )
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    target_config = AutoConfig.from_pretrained(target_path, local_files_only=True)
    draft_config = AutoConfig.from_pretrained(draft_path, local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        config=target_config,
        dtype=dtype,
        attn_implementation=attention_backend,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    base_draft = DFlashDraftModel.from_pretrained(
        draft_path,
        config=draft_config,
        dtype=dtype,
        attn_implementation=attention_backend,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    draft = AMRDFlashDraft(base_draft).eval()
    if target.get_input_embeddings() is None or target.get_output_embeddings() is None:
        raise ValueError("target model must expose input embeddings and an output head")
    if int(base_draft.config.hidden_size) != int(target.config.hidden_size):
        raise ValueError("DFlash projected hidden size must match target hidden size")
    vocab_size = int(target.get_output_embeddings().weight.shape[0])
    if not 0 <= draft.mask_token_id < vocab_size:
        raise ValueError("DFlash mask token id is outside the target vocabulary")
    return target, tokenizer, draft


def freeze_backbones(target: nn.Module, draft: AMRDFlashDraft) -> None:
    """Freeze target and original DFlash; input memory remains differentiable."""
    target.eval()
    draft.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    for parameter in draft.parameters():
        parameter.requires_grad_(False)
