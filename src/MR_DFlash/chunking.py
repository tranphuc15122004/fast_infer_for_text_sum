"""Chunked reduction dùng chung cho objective tốn bộ nhớ.

Port tự đóng gói từ ``specforge/core/chunking.py``. Khi ``chunk_size > 0`` và
gradient đang bật, mỗi lát được bọc qua activation checkpointing (non-reentrant)
để giới hạn bộ nhớ trung gian; tổng các số hạng cộng dồn theo chiều đã chọn.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch

ChunkTerms = tuple[torch.Tensor, ...]


def checkpointed_chunk_reduce(
    function: Callable[..., ChunkTerms],
    *aligned_tensors: Optional[torch.Tensor],
    chunk_size: int,
    dim: int = 0,
) -> ChunkTerms:
    """Cộng các số hạng additive trên các lát tensor thẳng hàng nhau.

    ``chunk_size=0`` đánh giá cả chiều một lần (không checkpointing).
    Các tensor ``None`` được giữ nguyên vị trí (tensor option đầu vào vẫn
    thẳng hàng với tensor bị cắt lát).
    """
    if chunk_size < 0:
        raise ValueError(f"chunk_size phải >= 0, got {chunk_size}")

    tensors = tuple(t for t in aligned_tensors if t is not None)
    if not tensors:
        raise ValueError("chunked reduction cần ít nhất một tensor")
    first = tensors[0]
    normalized_dim = dim if dim >= 0 else first.ndim + dim
    if normalized_dim < 0 or normalized_dim >= first.ndim:
        raise ValueError(f"dim {dim} không hợp lệ cho tensor {first.ndim}D")
    length = first.shape[normalized_dim]
    if length == 0:
        raise ValueError("chunked reduction nhận chiều rỗng")

    for tensor in tensors[1:]:
        t_dim = dim if dim >= 0 else tensor.ndim + dim
        if t_dim < 0 or t_dim >= tensor.ndim:
            raise ValueError(f"dim {dim} không hợp lệ cho tensor {tensor.ndim}D")
        if tensor.shape[t_dim] != length:
            raise ValueError(
                "đầu vào chunked reduction phải thẳng hàng: "
                f"kỳ vọng độ dài {length}, got {tensor.shape[t_dim]}"
            )

    effective_chunk_size = chunk_size or length
    totals: Optional[ChunkTerms] = None
    for start in range(0, length, effective_chunk_size):
        width = min(effective_chunk_size, length - start)
        chunk_args = tuple(
            (
                tensor.narrow(
                    dim if dim >= 0 else tensor.ndim + dim, start, width
                )
                if tensor is not None
                else None
            )
            for tensor in aligned_tensors
        )
        should_checkpoint = (
            chunk_size > 0
            and torch.is_grad_enabled()
            and any(t is not None and t.requires_grad for t in chunk_args)
        )
        if should_checkpoint:
            from torch.utils.checkpoint import checkpoint

            chunk_terms = checkpoint(function, *chunk_args, use_reentrant=False)
        else:
            chunk_terms = function(*chunk_args)

        if not isinstance(chunk_terms, tuple) or not all(
            isinstance(t, torch.Tensor) for t in chunk_terms
        ):
            raise TypeError("chunk function phải trả tuple các tensor")
        if totals is None:
            totals = chunk_terms
            continue
        if len(totals) != len(chunk_terms):
            raise ValueError("chunk function trả số lượng term khác nhau")
        totals = tuple(a + b for a, b in zip(totals, chunk_terms))

    assert totals is not None
    return totals


__all__ = ["checkpointed_chunk_reduce", "ChunkTerms"]
