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

from .model import RMSNorm


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
    pending_hca_positions: torch.Tensor
    pending_csa: torch.Tensor
    pending_csa_positions: torch.Tensor
    total_tokens: int
    # Khi training flatten các anchor thành batch, global memory vẫn giữ batch
    # gốc; field này ánh xạ từng block query về sample tương ứng.
    block_batch_indices: Optional[torch.Tensor] = None

    def flatten_blocks(self, num_blocks: int) -> "MRMemoryState":
        """Đổi memory query-relative thành batch ``B * num_blocks``.

        Training MR-DFlash xử lý từng draft block như một phần tử batch để
        attention không phải dựng logits giữa các block độc lập. Global
        memory giữ batch gốc và dùng ``block_batch_indices``; local memory đã có layout
        ``[B, num_blocks, W, H]`` khi build với ``query_positions``.
        """
        if num_blocks < 1:
            raise ValueError("num_blocks phải >= 1")
        batch = self.hca.shape[0]

        def flatten_local(values: torch.Tensor, name: str) -> torch.Tensor:
            if values.ndim == 4:
                if values.shape[1] != num_blocks:
                    raise ValueError(
                        f"{name} có {values.shape[1]} blocks, cần {num_blocks}"
                    )
                return values.reshape(batch * num_blocks, *values.shape[2:])
            if values.ndim == 3:
                if name == "local_positions" and values.shape[1] == num_blocks:
                    return values.reshape(batch * num_blocks, values.shape[2])
                return values.repeat_interleave(num_blocks, dim=0)
            if values.ndim == 2:
                return values.repeat_interleave(num_blocks, dim=0)
            raise ValueError(f"{name} phải có dạng [B,W,H] hoặc [B,N,W,H]")

        return MRMemoryState(
            # Không repeat global hidden memory: joint attention/indexer dùng
            # block_batch_indices để contraction trực tiếp với batch [B,N].
            hca=self.hca,
            hca_positions=self.hca_positions,
            csa=self.csa,
            csa_positions=self.csa_positions,
            local_hca=flatten_local(self.local_hca, "local_hca"),
            local_csa=flatten_local(self.local_csa, "local_csa"),
            local_positions=flatten_local(self.local_positions, "local_positions"),
            pending_hca=self.pending_hca,
            pending_hca_positions=self.pending_hca_positions,
            pending_csa=self.pending_csa,
            pending_csa_positions=self.pending_csa_positions,
            total_tokens=self.total_tokens,
            block_batch_indices=torch.arange(
                batch, device=self.hca.device, dtype=torch.long
            ).repeat_interleave(num_blocks),
        )


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
        self.hca_norm = RMSNorm(hidden_size)
        self.csa_norm = RMSNorm(hidden_size)
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
        return self.hca_norm(self.hca(features)), self.csa_norm(self.csa(features))


class WeightedTokenPool(nn.Module):
    """Learned pooling trên các nhóm token đầy đủ, liên tiếp.

    Phần đuôi chưa đủ ``ratio`` token không thuộc pool. Caller giữ phần đó
    trong pending cache để ``build`` và ``append`` có cùng semantics.
    """

    def __init__(self, hidden_size: int, max_ratio: int = 128) -> None:
        super().__init__()
        if max_ratio < 1:
            raise ValueError("max_ratio phải >= 1")
        self.max_ratio = int(max_ratio)
        # One token weight per compressed channel (instead of one scalar per
        # token), plus a learnable within-group positional bias.
        self.score = nn.Linear(hidden_size, hidden_size)
        self.position_bias = nn.Parameter(torch.zeros(self.max_ratio, hidden_size))
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.score.weight.zero_()
            self.score.bias.zero_()
            self.value.weight.copy_(torch.eye(hidden_size))

    def forward(self, tokens: torch.Tensor, ratio: int) -> torch.Tensor:
        _check_feature_tensor(tokens, "tokens")
        if ratio < 1:
            raise ValueError(f"ratio phải >= 1, got {ratio}")
        if ratio > self.max_ratio:
            raise ValueError(f"ratio={ratio} vượt max_ratio={self.max_ratio}")
        batch, length, hidden = tokens.shape
        groups = length // ratio
        if groups < 1:
            raise ValueError(
                f"tokens phải có ít nhất một nhóm đầy đủ: length={length}, ratio={ratio}"
            )
        tokens = tokens[:, : groups * ratio]
        grouped = tokens.view(batch, groups, ratio, hidden)
        scores = self.score(grouped) + self.position_bias[:ratio].view(1, 1, ratio, hidden)
        weights = torch.softmax(scores, dim=2)
        return (weights * self.value(grouped)).sum(dim=2)


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
    """Lightning-inspired learned query/key selector cho CSA memory.

    Mỗi indexer head dùng interaction ``ReLU(q * k)`` và một trọng số head
    phụ thuộc query. Đây vẫn là bản torch thuần gọn hơn Lightning Indexer
    production, nhưng đã tách score path khỏi main attention projection.
    """

    def __init__(
        self,
        hidden_size: int,
        indexer_dim: Optional[int] = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size phải dương")
        self.hidden_size = int(hidden_size)
        self.indexer_dim = int(indexer_dim or hidden_size)
        if self.indexer_dim < 1:
            raise ValueError("indexer_dim phải dương")
        if num_heads < 1 or self.indexer_dim % int(num_heads):
            raise ValueError("indexer_dim phải chia hết cho indexer_num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = self.indexer_dim // self.num_heads
        self.q_proj = nn.Linear(hidden_size, self.indexer_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.indexer_dim, bias=False)
        self.weight_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        with torch.no_grad():
            self.weight_proj.weight.zero_()
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
        scores = self.score(query, csa_memory)
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

    def score(
        self,
        query: torch.Tensor,
        csa_memory: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Trả toàn bộ index scores để dense/indexer warm-up có gradient."""
        if query.ndim != 3 or csa_memory.ndim != 3:
            raise ValueError("query và csa_memory phải có dạng [batch, seq, hidden]")
        if query.shape[-1] != self.hidden_size:
            raise ValueError("query/csa_memory không cùng batch hoặc hidden size")
        if csa_memory.shape[1] < 1:
            raise ValueError("csa_memory phải có ít nhất một slot")
        batch, query_len, _ = query.shape
        memory_len = csa_memory.shape[1]
        q = self.q_proj(query).view(batch, query_len, self.num_heads, self.head_dim)
        memory_batch = csa_memory.shape[0]
        k = self.k_proj(csa_memory).view(
            memory_batch, memory_len, self.num_heads, self.head_dim
        )
        # Lightning-style head score: ReLU(dot(q, k)) per head. Applying
        # ReLU after the reduction is intentional; ``sum(ReLU(q*k))`` has a
        # different ranking and is not the agreed indexer formulation.
        if batch_indices is None:
            if batch != memory_batch:
                raise ValueError("query/csa_memory không cùng batch")
            similarity = torch.relu(
                (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1)
            )
            head_weights = 1.0 + self.weight_proj(query).unsqueeze(2)
            return (similarity * head_weights).sum(dim=-1) * self.scale

        batch_indices = batch_indices.to(device=query.device, dtype=torch.long)
        if batch_indices.shape != (batch,):
            raise ValueError("batch_indices phải có dạng [query_batch]")
        if batch == memory_batch:
            # Cho phép caller truyền identity mapping mà không đổi layout.
            if not torch.equal(
                batch_indices,
                torch.arange(batch, device=query.device, dtype=torch.long),
            ):
                raise ValueError("batch_indices không hợp lệ cho batch cùng kích thước")
            similarity = torch.relu(
                (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1)
            )
            head_weights = 1.0 + self.weight_proj(query).unsqueeze(2)
            return (similarity * head_weights).sum(dim=-1) * self.scale
        if batch % memory_batch or not torch.equal(
            batch_indices,
            torch.arange(memory_batch, device=query.device, dtype=torch.long)
            .repeat_interleave(batch // memory_batch),
        ):
            raise ValueError("batch_indices phải ánh xạ tuần tự các block về batch gốc")
        num_blocks = batch // memory_batch
        q = q.reshape(memory_batch, num_blocks, query_len, self.num_heads, self.head_dim)
        similarity = torch.relu(
            torch.einsum("bnqhd,bmhd->bnqhm", q, k)
        )
        head_weights = 1.0 + self.weight_proj(query).reshape(
            memory_batch, num_blocks, query_len, self.num_heads
        ).unsqueeze(-1)
        return (
            (similarity * head_weights)
            .sum(dim=-2)
            .reshape(batch, query_len, memory_len)
            * self.scale
        )

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
        indexer_num_heads: int = 1,
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
        self.hca_pool = WeightedTokenPool(hidden_size, self.hca_compression_ratio)
        self.csa_pool = WeightedTokenPool(hidden_size, self.csa_compression_ratio)
        self.indexer = CSAIndexer(
            hidden_size,
            indexer_dim=indexer_dim,
            num_heads=indexer_num_heads,
        )

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

    def _anchor_local_view(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather ``[B,Q,W,H]`` local views immediately before each anchor."""
        query_positions = query_positions.to(device=tokens.device, dtype=torch.long)
        if query_positions.ndim != 2 or query_positions.shape[0] != tokens.shape[0]:
            raise ValueError(
                "query_positions phải có dạng [batch, query] và cùng batch với features"
            )
        batch, query_len = query_positions.shape
        length = tokens.shape[1]
        offsets = torch.arange(
            self.local_window, device=tokens.device, dtype=torch.long
        )
        # Positions are chronological. Counting values strictly before an
        # anchor also handles non-zero prefix offsets without assuming that
        # positions are exactly arange(length).
        end = (positions.unsqueeze(1) < query_positions.unsqueeze(-1)).sum(dim=-1)
        indices = end.unsqueeze(-1) - self.local_window + offsets
        valid = (indices >= 0) & (indices < length)
        safe_indices = indices.clamp(min=0, max=max(length - 1, 0))
        expanded_tokens = tokens.unsqueeze(1).expand(-1, query_len, -1, -1)
        gathered = torch.gather(
            expanded_tokens,
            2,
            safe_indices.unsqueeze(-1).expand(-1, -1, -1, tokens.shape[-1]),
        )
        expanded_positions = positions.unsqueeze(1).expand(-1, query_len, -1)
        gathered_positions = torch.gather(expanded_positions, 2, safe_indices)
        gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0)
        invalid_position = torch.iinfo(torch.long).max
        gathered_positions = gathered_positions.masked_fill(~valid, invalid_position)
        return gathered.reshape(batch, query_len, self.local_window, -1), gathered_positions

    def build(
        self,
        features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        query_positions: Optional[torch.Tensor] = None,
    ) -> MRMemoryState:
        _check_feature_tensor(features)
        pos = self._positions(features, positions)
        hca_tokens, csa_tokens = self.adapter(features)
        hca_complete = (hca_tokens.shape[1] // self.hca_compression_ratio) * self.hca_compression_ratio
        csa_complete = (csa_tokens.shape[1] // self.csa_compression_ratio) * self.csa_compression_ratio
        if hca_complete:
            hca, hca_pos = _pool_with_positions(
                self.hca_pool,
                hca_tokens[:, :hca_complete],
                pos[:, :hca_complete],
                self.hca_compression_ratio,
            )
        else:
            hca = hca_tokens[:, :0]
            hca_pos = pos[:, :0]
        if csa_complete:
            csa, csa_pos = _pool_with_positions(
                self.csa_pool,
                csa_tokens[:, :csa_complete],
                pos[:, :csa_complete],
                self.csa_compression_ratio,
            )
        else:
            csa = csa_tokens[:, :0]
            csa_pos = pos[:, :0]
        if query_positions is None:
            local_start = max(0, features.shape[1] - self.local_window)
            local_hca = hca_tokens[:, local_start:]
            local_csa = csa_tokens[:, local_start:]
            local_pos = pos[:, local_start:]
        else:
            local_hca, local_pos = self._anchor_local_view(
                hca_tokens, pos, query_positions
            )
            local_csa, csa_local_pos = self._anchor_local_view(
                csa_tokens, pos, query_positions
            )
            # Both streams represent the same raw target positions. Keep one
            # position tensor in the state while retaining independent values.
            if not torch.equal(local_pos, csa_local_pos):
                raise RuntimeError("HCA/CSA local position views không nhất quán")
        return MRMemoryState(
            hca=hca,
            hca_positions=hca_pos,
            csa=csa,
            csa_positions=csa_pos,
            local_hca=local_hca,
            local_csa=local_csa,
            local_positions=local_pos,
            pending_hca=hca_tokens[:, hca_complete:],
            pending_hca_positions=pos[:, hca_complete:],
            pending_csa=csa_tokens[:, csa_complete:],
            pending_csa_positions=pos[:, csa_complete:],
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
        if state.local_hca.ndim != 3 or state.local_csa.ndim != 3:
            raise ValueError("append không hỗ trợ state có local view theo nhiều anchor")
        pos = self._positions(features, positions, start=state.total_tokens)
        hca_new, csa_new = self.adapter(features)
        hca_add, hca_add_pos, hca_pending, hca_pending_pos = self._append_stream(
            state.pending_hca,
            hca_new,
            state.pending_hca_positions,
            pos,
            self.hca_pool,
            self.hca_compression_ratio,
        )
        csa_add, csa_add_pos, csa_pending, csa_pending_pos = self._append_stream(
            state.pending_csa,
            csa_new,
            state.pending_csa_positions,
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
            pending_hca_positions=hca_pending_pos,
            pending_csa=csa_pending,
            pending_csa_positions=csa_pending_pos,
            total_tokens=state.total_tokens + int(features.shape[1]),
        )


__all__ = [
    "CSAIndexer",
    "MRMemoryState",
    "MRTargetMemory",
    "TargetFeatureAdapter",
    "WeightedTokenPool",
]
