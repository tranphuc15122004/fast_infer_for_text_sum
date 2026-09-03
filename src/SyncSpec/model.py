"""Native shallow block-diffusion drafter used by SyncSpec-v1.

The implementation intentionally keeps the model independent of a particular
target architecture. A real target can tie its embedding/LM head, while the
same module remains small enough for CPU contract tests and B200 profiling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SyncSpecDrafterConfig:
    vocab_size: int
    hidden_size: int
    layers: int = 3
    heads: int = 8
    groups: int = 16
    kernel_size: int = 2
    max_positions: int = 4096
    top_m: int = 16
    mask_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        if self.top_m <= 0:
            raise ValueError("top_m must be positive")
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        if self.groups <= 0:
            raise ValueError("groups must be positive")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if self.heads > self.hidden_size or self.groups > self.hidden_size:
            raise ValueError("heads and groups must not exceed hidden_size")
        if self.hidden_size % self.heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        if self.hidden_size % self.groups != 0:
            raise ValueError("hidden_size must be divisible by groups")
        if self.layers <= 0 or self.kernel_size <= 0:
            raise ValueError("layers and kernel_size must be positive")


@dataclass
class DrafterOutput:
    logits: torch.Tensor
    hidden: torch.Tensor


def build_masked_block(
    anchor_ids: torch.Tensor, kd: int, mask_token_id: int, sample_from_anchor: bool = False
) -> torch.Tensor:
    """Build `[B,K_d]` masked future slots from the current target anchor."""
    anchors = anchor_ids.to(torch.long).flatten()
    if kd <= 0:
        return torch.empty((anchors.shape[0], 0), dtype=torch.long, device=anchors.device)
    block = torch.full((anchors.shape[0], kd), int(mask_token_id), dtype=torch.long, device=anchors.device)
    if sample_from_anchor:
        block[:, 0] = anchors
    return block


def top_m_candidates(logits: torch.Tensor, top_m: int) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim < 2:
        raise ValueError("logits must have a vocabulary dimension")
    if int(top_m) <= 0:
        raise ValueError("top_m must be positive")
    m = min(int(top_m), int(logits.shape[-1]))
    values, ids = torch.topk(logits, k=m, dim=-1)
    return ids, values


class _DrafterBlock(nn.Module):
    def __init__(self, config: SyncSpecDrafterConfig):
        super().__init__()
        d = config.hidden_size
        self.self_attn = nn.MultiheadAttention(d, config.heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, config.heads, batch_first=True)
        self.conv = nn.Conv1d(
            d, d, kernel_size=config.kernel_size, groups=config.groups,
            padding=config.kernel_size - 1,
        )
        self.dynamic_kernel = nn.Linear(d, config.groups * config.kernel_size)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.norm4 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn)
        # Causal convolution sees previous block slots only. Self-attention is
        # intentionally bidirectional within the draft block.
        conv = self.conv(x.transpose(1, 2)).transpose(1, 2)[..., : x.shape[1], :]
        # A second, context-conditioned grouped causal path supplies the
        # dynamic-convolution component used by the drafter design.  Padding
        # is explicit on the left, so slot j never sees a future slot.
        kernel = torch.softmax(
            self.dynamic_kernel(context.mean(dim=1)).reshape(
                x.shape[0], self.conv.groups, self.conv.kernel_size[0],
            ), dim=-1,
        )
        channels_per_group = x.shape[-1] // self.conv.groups
        history = F.pad(x.transpose(1, 2), (self.conv.kernel_size[0] - 1, 0))
        windows = history.unfold(-1, self.conv.kernel_size[0], 1)
        windows = windows.reshape(
            x.shape[0], self.conv.groups, channels_per_group, x.shape[1],
            self.conv.kernel_size[0],
        ).mean(dim=2)
        dynamic = (windows * kernel.unsqueeze(2)).sum(dim=-1)
        dynamic = dynamic.unsqueeze(2).expand(
            -1, -1, channels_per_group, -1,
        ).reshape(x.shape[0], x.shape[1], x.shape[-1])
        conv = conv + dynamic
        x = self.norm2(x + conv)
        cross, _ = self.cross_attn(x, context, context, need_weights=False)
        x = self.norm3(x + cross)
        return self.norm4(x + self.mlp(x))


class SyncSpecDrafter(nn.Module):
    def __init__(self, config: SyncSpecDrafterConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # Decoder-only target tokenizers normally have no dedicated [MASK]
        # token. Keep the sentinel ID in-vocabulary for cheap block creation,
        # but give masked slots their own trainable representation instead of
        # accidentally reusing the target embedding of a real token.
        self.mask_embedding = nn.Parameter(torch.empty(config.hidden_size))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)
        self.position = nn.Embedding(config.max_positions, config.hidden_size)
        self.layers = nn.ModuleList([_DrafterBlock(config) for _ in range(config.layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._tied_embedding = False
        self._requires_target_tie = False

    def tie_target_weights(self, embedding: nn.Embedding, lm_head: nn.Linear) -> None:
        if embedding.weight.shape != self.embedding.weight.shape:
            raise ValueError("target embedding shape does not match drafter config")
        if lm_head.weight.shape != self.lm_head.weight.shape:
            raise ValueError("target LM head shape does not match drafter config")
        # Target checkpoints used on B200 are commonly bfloat16. Cast the
        # trainable shallow blocks before binding the frozen target modules so
        # hidden states and the tied LM head have one device/dtype throughout.
        self.to(device=embedding.weight.device, dtype=embedding.weight.dtype)
        self.embedding = embedding
        self.lm_head = lm_head
        # SyncSpec training never updates the target model or shared lexical
        # space; only the shallow drafter blocks/position parameters learn.
        self.embedding.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)
        self._tied_embedding = True
        self._requires_target_tie = False

    def forward(
        self,
        masked_ids: torch.Tensor,
        target_anchor: torch.Tensor | None = None,
        recent_hidden: torch.Tensor | None = None,
        source_memory: torch.Tensor | None = None,
        position_offset: int | torch.Tensor = 0,
    ) -> DrafterOutput:
        if masked_ids.ndim != 2:
            raise ValueError("masked_ids must be [B,K_d]")
        if self._requires_target_tie:
            raise RuntimeError(
                "this compact checkpoint requires tie_target_weights before forward"
            )
        bsz, length = masked_ids.shape
        if length == 0:
            empty_h = torch.empty((bsz, 0, self.config.hidden_size), device=masked_ids.device)
            empty_l = torch.empty((bsz, 0, self.config.vocab_size), device=masked_ids.device)
            return DrafterOutput(empty_l, empty_h)
        offsets = torch.as_tensor(position_offset, dtype=torch.long, device=masked_ids.device)
        if (offsets < 0).any() or (offsets + length > self.config.max_positions).any():
            raise ValueError(
                "position_offset and block length must fit within max_positions"
            )
        if offsets.ndim == 0:
            positions = (
                torch.arange(length, device=masked_ids.device) + int(offsets.item())
            )
            position_embeddings = self.position(positions).unsqueeze(0)
        else:
            offsets = offsets.flatten()
            if offsets.numel() != bsz:
                raise ValueError("position_offset must be scalar or one value per batch row")
            positions = (
                offsets[:, None] + torch.arange(length, device=masked_ids.device)[None, :]
            )
            position_embeddings = self.position(positions)
        token_embeddings = self.embedding(masked_ids)
        if self.config.mask_token_id is not None:
            mask = masked_ids.eq(int(self.config.mask_token_id)).unsqueeze(-1)
            sentinel = self.mask_embedding.to(
                device=token_embeddings.device, dtype=token_embeddings.dtype,
            )
            token_embeddings = torch.where(mask, sentinel.reshape(1, 1, -1), token_embeddings)
        x = token_embeddings + position_embeddings
        d = self.config.hidden_size
        parts = []
        if target_anchor is not None:
            parts.append(target_anchor.reshape(bsz, 1, d))
        if recent_hidden is not None and recent_hidden.numel():
            parts.append(recent_hidden.reshape(bsz, -1, d))
        if source_memory is not None and source_memory.numel():
            parts.append(source_memory.reshape(bsz, -1, d))
        if not parts:
            parts.append(torch.zeros((bsz, 1, d), dtype=x.dtype, device=x.device))
        context = torch.cat([p.to(x.dtype) for p in parts], dim=1)
        for layer in self.layers:
            x = layer(x, context)
        return DrafterOutput(self.lm_head(x), x)

    def save_pretrained(self, path: Path | str, omit_tied_weights: bool = False) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n", encoding="utf-8"
        )
        state = self.state_dict()
        if omit_tied_weights:
            if not self._tied_embedding:
                raise ValueError("omit_tied_weights requires a target-tied drafter")
            state.pop("embedding.weight", None)
            state.pop("lm_head.weight", None)
            (output / "checkpoint_metadata.json").write_text(
                json.dumps({"schema_version": 1, "tied_target_weights": True}, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            metadata = output / "checkpoint_metadata.json"
            if metadata.exists():
                metadata.unlink()
        torch.save(state, output / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, path: Path | str, map_location: str | torch.device = "cpu") -> "SyncSpecDrafter":
        root = Path(path)
        config = SyncSpecDrafterConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
        model = cls(config)
        state = torch.load(root / "pytorch_model.bin", map_location=map_location, weights_only=True)
        metadata_path = root / "checkpoint_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        if metadata.get("tied_target_weights"):
            missing, unexpected = model.load_state_dict(state, strict=False)
            # `mask_embedding` was added after the first compact checkpoint
            # format. Allow an old checkpoint to initialize it normally while
            # still rejecting every other missing/unexpected parameter.
            allowed = {"embedding.weight", "lm_head.weight", "mask_embedding"}
            if set(missing) != allowed or unexpected:
                legacy_allowed = {"embedding.weight", "lm_head.weight"}
                if set(missing) != legacy_allowed or unexpected:
                    raise ValueError("invalid compact tied-target drafter checkpoint")
            model._requires_target_tie = True
        else:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if set(missing) not in (set(), {"mask_embedding"}) or unexpected:
                raise ValueError("invalid standalone drafter checkpoint")
        return model
