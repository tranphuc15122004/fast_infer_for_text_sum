"""Multi-resolution target memory cho MR-DFlash.

Module này chỉ phụ trách chuyển feature target thành hai memory view dùng
chung bởi training và inference. Không có cache global: inference giữ
``MRMemoryState`` và chỉ gọi ``append`` sau khi verifier accept token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _check_feature_tensor(features: torch.Tensor, name: str = "features") -> None:
    if features.ndim != 3:
        raise ValueError(f"{name} phải có dạng [batch, seq, width], got {tuple(features.shape)}")
    if features.shape[1] < 1:
        raise ValueError(f"{name} phải có ít nhất một token")


@dataclass
class MRMemoryState:
    """Snapshot của memory target tại một prefix đã được verifier chấp nhận."""

    hca: torch.Tensor
    hca_positions: torch.Tensor
    csa: torch.Tensor
    csa_positions: torch.Tensor
    local_hca: torch.Tensor
    local_csa: torch.Tensor
    local_positions: torch.Tensor
    pending_hca: torch.Tensor
    pending_csa: torch.Tensor
    pending_positions: torch.Tensor
    total_tokens: int


class TargetFeatureAdapter(nn.Module):
    """Tách concat hidden target thành hai không gian HCA và CSA."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        if input_dim < 1 or hidden_size < 1:
            raise ValueError("input_dim và hidden_size phải dương")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.hca = nn.Linear(input_dim, hidden_size, bias=False)
        self.csa = nn.Linear(input_dim, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Khởi tạo ổn định: hai view bắt đầu từ trung bình các layer feature."""
        with torch.no_grad():
            self.hca.weight.zero_()
            self.csa.weight.zero_()
            if self.input_dim % self.hidden_size == 0:
                num_layers = self.input_dim // self.hidden_size
                scale = 1.0 / float(num_layers)
                for offset in range(0, self.input_dim, self.hidden_size):
                    self.hca.weight[:, offset : offset + self.hidden_size].copy_(
                        torch.eye(self.hidden_size, device=self.hca.weight.device,
                                  dtype=self.hca.weight.dtype) * scale
                    )
                    self.csa.weight[:, offset : offset + self.hidden_size].copy_(
                        torch.eye(self.hidden_size, device=self.csa.weight.device,
                                  dtype=self.csa.weight.dtype) * scale
                    )
            else:
                nn.init.xavier_uniform_(self.hca.weight)
                nn.init.xavier_uniform_(self.csa.weight)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_feature_tensor(features)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature width={features.shape[-1]} không khớp input_dim={self.input_dim}"
            )
        return self.hca(features), self.csa(features)


class WeightedTokenPool(nn.Module):
    """Learned pooling trên từng nhóm token liên tiếp."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.score.weight.zero_()
            self.score.bias.zero_()
            self.value.weight.copy_(torch.eye(hidden_size))

    def forward(self, tokens: torch.Tensor, ratio: int) -> torch.Tensor:
        _check_feature_tensor(tokens, "tokens")
        if ratio < 1:
            raise ValueError(f"ratio phải >= 1, got {ratio}")
        batch, length, hidden = tokens.shape
        groups = (length + ratio - 1) // ratio
        padded_length = groups * ratio
        pad = padded_length - length
        if pad:
            tokens = F.pad(tokens, (0, 0, 0, pad))
        valid = torch.arange(padded_length, device=tokens.device).view(1, groups, ratio)
        valid = valid < length
        grouped = tokens.view(batch, groups, ratio, hidden)
        scores = self.score(grouped).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * self.value(grouped)).sum(dim=2)


def _pool_with_positions(
    pool: WeightedTokenPool,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool token groups và lấy position của token cuối mỗi group."""
    _check_feature_tensor(tokens, "tokens")
    if positions.shape != tokens.shape[:2]:
        raise ValueError(
            f"positions phải có shape {tuple(tokens.shape[:2])}, got {tuple(positions.shape)}"
        )
    length = tokens.shape[1]
    values = pool(tokens, ratio)
    groups = values.shape[1]
    last_indices = torch.minimum(
        torch.arange(1, groups + 1, device=tokens.device) * ratio - 1,
        tokens.new_tensor(length - 1, dtype=torch.long),
    )
    pooled_positions = positions[:, last_indices]
    return values, pooled_positions


class CSAIndexer(nn.Module):
    """Learned query/key selector cho CSA memory."""

    def __init__(self, hidden_size: int, indexer_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size phải dương")
        self.hidden_size = int(hidden_size)
        self.indexer_dim = int(indexer_dim or hidden_size)
        if self.indexer_dim < 1:
            raise ValueError("indexer_dim phải dương")
        self.q_proj = nn.Linear(hidden_size, self.indexer_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.indexer_dim, bias=False)
        self.scale = self.indexer_dim ** -0.5

    def select(
        self,
        query: torch.Tensor,
        csa_memory: torch.Tensor,
        top_k: int,
        allowed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or csa_memory.ndim != 3:
            raise ValueError("query và csa_memory phải có dạng [batch, seq, hidden]")
        if query.shape[0] != csa_memory.shape[0] or query.shape[-1] != self.hidden_size:
            raise ValueError("query/csa_memory không cùng batch hoặc hidden size")
        if csa_memory.shape[1] < 1:
            raise ValueError("csa_memory phải có ít nhất một slot")
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        scores = torch.matmul(self.q_proj(query), self.k_proj(csa_memory).transpose(-1, -2))
        scores = scores * self.scale
        if allowed_mask is not None:
            if allowed_mask.shape != scores.shape:
                raise ValueError(
                    f"allowed_mask phải có shape {tuple(scores.shape)}, got {tuple(allowed_mask.shape)}"
                )
            scores = scores.masked_fill(
                ~allowed_mask.to(dtype=torch.bool), torch.finfo(scores.dtype).min
            )
        k = min(int(top_k), csa_memory.shape[1])
        top_scores, top_indices = scores.topk(k=k, dim=-1)
        return top_indices, top_scores

    @staticmethod
    def gather(memory: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather selected memory: ``[B,Q,K,H]``."""
        batch, query_len, top_k = indices.shape
        expanded = memory.unsqueeze(1).expand(-1, query_len, -1, -1)
        return torch.gather(
            expanded,
            2,
            indices.unsqueeze(-1).expand(batch, query_len, top_k, memory.shape[-1]),
        )


class MRTargetMemory(nn.Module):
    """Xây và cập nhật HCA/CSA memory từ concat target features."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        *,
        hca_compression_ratio: int = 128,
        csa_compression_ratio: int = 4,
        local_window: int = 128,
        csa_top_k: int = 64,
        indexer_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("hca_compression_ratio", hca_compression_ratio),
            ("csa_compression_ratio", csa_compression_ratio),
            ("local_window", local_window),
            ("csa_top_k", csa_top_k),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} phải >= 1, got {value}")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.hca_compression_ratio = int(hca_compression_ratio)
        self.csa_compression_ratio = int(csa_compression_ratio)
        self.local_window = int(local_window)
        self.csa_top_k = int(csa_top_k)
        self.adapter = TargetFeatureAdapter(input_dim, hidden_size)
        self.hca_pool = WeightedTokenPool(hidden_size)
        self.csa_pool = WeightedTokenPool(hidden_size)
        self.indexer = CSAIndexer(hidden_size, indexer_dim=indexer_dim)

    def _positions(
        self,
        features: torch.Tensor,
        positions: Optional[torch.Tensor],
        start: int = 0,
    ) -> torch.Tensor:
        if positions is None:
            return torch.arange(
                start,
                start + features.shape[1],
                device=features.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(features.shape[0], -1)
        if positions.shape != features.shape[:2]:
            raise ValueError(
                f"positions phải có shape {tuple(features.shape[:2])}, got {tuple(positions.shape)}"
            )
        return positions.to(device=features.device, dtype=torch.long)

    def build(
        self,
        features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> MRMemoryState:
        _check_feature_tensor(features)
        pos = self._positions(features, positions)
        hca_tokens, csa_tokens = self.adapter(features)
        hca, hca_pos = _pool_with_positions(
            self.hca_pool, hca_tokens, pos, self.hca_compression_ratio
        )
        csa, csa_pos = _pool_with_positions(
            self.csa_pool, csa_tokens, pos, self.csa_compression_ratio
        )
        local_start = max(0, features.shape[1] - self.local_window)
        return MRMemoryState(
            hca=hca,
            hca_positions=hca_pos,
            csa=csa,
            csa_positions=csa_pos,
            local_hca=hca_tokens[:, local_start:],
            local_csa=csa_tokens[:, local_start:],
            local_positions=pos[:, local_start:],
            pending_hca=hca_tokens[:, 0:0],
            pending_csa=csa_tokens[:, 0:0],
            pending_positions=pos[:, 0:0],
            total_tokens=int(features.shape[1]),
        )

    def _append_stream(
        self,
        previous: torch.Tensor,
        new: torch.Tensor,
        previous_positions: torch.Tensor,
        new_positions: torch.Tensor,
        pool: WeightedTokenPool,
        ratio: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.cat([previous, new], dim=1)
        positions = torch.cat([previous_positions, new_positions], dim=1)
        complete = (tokens.shape[1] // ratio) * ratio
        if complete:
            compressed, compressed_positions = _pool_with_positions(
                pool,
                tokens[:, :complete],
                positions[:, :complete],
                ratio,
            )
        else:
            compressed = tokens[:, :0]
            compressed_positions = positions[:, :0]
        return (
            compressed,
            compressed_positions,
            tokens[:, complete:],
            positions[:, complete:],
        )

    def append(
        self,
        state: MRMemoryState,
        features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> MRMemoryState:
        """Append target features; caller phải chỉ truyền token đã accept."""
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features phải có dạng [B,S,{self.input_dim}], got {tuple(features.shape)}"
            )
        if features.shape[1] == 0:
            return state
        if features.shape[0] != state.local_hca.shape[0]:
            raise ValueError("batch của features không khớp memory state")
        pos = self._positions(features, positions, start=state.total_tokens)
        hca_new, csa_new = self.adapter(features)
        hca_add, hca_add_pos, hca_pending, hca_pending_pos = self._append_stream(
            state.pending_hca,
            hca_new,
            state.pending_positions,
            pos,
            self.hca_pool,
            self.hca_compression_ratio,
        )
        csa_add, csa_add_pos, csa_pending, csa_pending_pos = self._append_stream(
            state.pending_csa,
            csa_new,
            state.pending_positions,
            pos,
            self.csa_pool,
            self.csa_compression_ratio,
        )
        local_hca = torch.cat([state.local_hca, hca_new], dim=1)
        local_csa = torch.cat([state.local_csa, csa_new], dim=1)
        local_pos = torch.cat([state.local_positions, pos], dim=1)
        local_start = max(0, local_pos.shape[1] - self.local_window)
        return MRMemoryState(
            hca=torch.cat([state.hca, hca_add], dim=1),
            hca_positions=torch.cat([state.hca_positions, hca_add_pos], dim=1),
            csa=torch.cat([state.csa, csa_add], dim=1),
            csa_positions=torch.cat([state.csa_positions, csa_add_pos], dim=1),
            local_hca=local_hca[:, local_start:],
            local_csa=local_csa[:, local_start:],
            local_positions=local_pos[:, local_start:],
            pending_hca=hca_pending,
            pending_csa=csa_pending,
            pending_positions=hca_pending_pos,
            total_tokens=state.total_tokens + int(features.shape[1]),
        )


__all__ = [
    "CSAIndexer",
    "MRMemoryState",
    "MRTargetMemory",
    "TargetFeatureAdapter",
    "WeightedTokenPool",
]
