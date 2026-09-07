"""Metrics chung cho training/evaluation DFlash-family."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def acceptance_proxy(
    predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Trả ratio counts cho accuracy theo offset và ``P(A >= j)``.

    Input có shape ``[..., block_size]``; offset 0 bị bỏ qua vì là token
    anchor. Hàm trả numerator/denominator, giúp caller aggregate đúng giữa
    micro-batch và DDP thay vì lấy trung bình các tỷ lệ riêng lẻ.
    """
    if predicted_ids.shape != target_ids.shape or predicted_ids.shape != valid_mask.shape:
        raise ValueError("predicted_ids/target_ids/valid_mask phải cùng shape")
    if predicted_ids.shape[-1] < 2:
        empty = predicted_ids.new_zeros((0,), dtype=torch.float32)
        return {"acc_pos": (empty, empty), "accept_ge": (empty, empty)}
    valid = valid_mask.to(dtype=torch.bool)[..., 1:]
    correct = predicted_ids.eq(target_ids)[..., 1:]
    acc_num = (correct & valid).sum(dim=tuple(range(correct.ndim - 1))).float()
    acc_den = valid.sum(dim=tuple(range(valid.ndim - 1))).float()
    prefix_valid = torch.cumprod(valid.float(), dim=-1)
    prefix_correct = torch.cumprod((correct & valid).float(), dim=-1) * prefix_valid
    accept_num = prefix_correct.sum(dim=tuple(range(correct.ndim - 1)))
    accept_den = prefix_valid.sum(dim=tuple(range(valid.ndim - 1)))
    return {"acc_pos": (acc_num, acc_den), "accept_ge": (accept_num, accept_den)}


__all__ = ["acceptance_proxy"]
