"""Horizon lịch train + LR schedule dùng chung (optimizer steps).

Port rút gọn từ ``specforge/training/schedule.py`` + ``lr_scheduler`` của
SpecForge. Nguyên tắc giữ nguyên: mọi mốc (global_step, LR, loss horizon,
Domino lambda decay) đều tính theo *completed optimizer updates* — không trộn
micro-batch với bước optimizer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR, Optimizer


def resolve_total_steps(
    *,
    total_steps: Optional[int],
    max_steps: Optional[int],
    num_samples: Optional[int],
    batch_size: int,
    accumulation_steps: int,
    num_epochs: int,
) -> int:
    """Xác định một horizon optimizer-step từ giới hạn tường minh hoặc dữ liệu hữu hạn."""
    if total_steps is not None:
        return int(total_steps)
    if max_steps is not None:
        return int(max_steps)
    if num_samples is None:
        raise ValueError(
            "dữ liệu streaming cần training.total_steps hoặc training.max_steps "
            "để optimizer và loss schedule chia sẻ một horizon"
        )
    micro_batches_per_epoch = int(num_samples) // int(batch_size)
    optimizer_steps = (micro_batches_per_epoch * int(num_epochs)) // int(
        accumulation_steps
    )
    if optimizer_steps < 1:
        raise ValueError(
            "dữ liệu không tạo ra optimizer step nào: "
            f"samples={num_samples}, batch_size={batch_size}, "
            f"accumulation_steps={accumulation_steps}, num_epochs={num_epochs}"
        )
    return optimizer_steps


def validate_fixed_accumulation_plan(
    *,
    num_samples: int,
    batch_size: int,
    accumulation_steps: int,
    num_epochs: int,
    max_steps: Optional[int],
) -> None:
    """Từ chối sớm một kế hoạch accumulation lẻ (trước khi dựng optimizer).

    Fixed-ref loader bỏ batch mẫu không đủ. Nếu số micro-batch tự nhiên không
    chia hết cho accumulation_steps thì công việc không thể commit bền vững;
    phát hiện từ đầu trừ khi ``max_steps`` chặn ở biên đầy đủ sớm hơn.
    """
    micro_batches = (int(num_samples) // int(batch_size)) * int(num_epochs)
    complete_steps, remainder = divmod(micro_batches, int(accumulation_steps))
    stops_before = max_steps is not None and int(max_steps) <= complete_steps
    if remainder and not stops_before:
        raise ValueError(
            "kế hoạch dữ liệu cố định kết thúc với gradient accumulation chưa "
            f"đủ: {micro_batches} micro-batch qua {num_epochs} epoch không chia "
            f"hết cho accumulation_steps={accumulation_steps} (remainder="
            f"{remainder}); điều chỉnh batch/accumulation/epochs hoặc đặt "
            f"max_steps <= {complete_steps}"
        )


def _warmup_cosine_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Hệ số LR: linear warmup rồi cosine decay về 0 tại total_steps."""
    if total_steps <= 0:
        raise ValueError(f"total_steps phải dương, got {total_steps}")
    if step < warmup_steps:
        if warmup_steps == 0:
            return 1.0
        return float(step) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def build_cosine_with_warmup(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Dựng scheduler cosine + linear warmup (tính trên optimizer step)."""

    def lr_lambda(step: int) -> float:
        return _warmup_cosine_factor(step, total_steps, warmup_steps)

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)


def current_lr(optimizer: Optimizer) -> float:
    """LR hiện tại của nhóm tham số đầu tiên (phục vụ logging)."""
    for group in optimizer.param_groups:
        return float(group["lr"])
    return 0.0


__all__ = [
    "resolve_total_steps",
    "validate_fixed_accumulation_plan",
    "build_cosine_with_warmup",
    "current_lr",
]
