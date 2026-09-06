"""Các helper distributed tối thiểu cho train MR-DFlash.

MR-DFlash không cần FSDP cho Qwen3-4B: target bị freeze và draft đủ nhỏ để
replicate trên mỗi B200. Vì vậy 2 GPU dùng DDP data-parallel, với một global
optimizer step cho mỗi micro-batch đồng bộ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

import torch
import torch.distributed as dist


DeviceLike = Union[str, torch.device]


@dataclass(frozen=True)
class DistributedContext:
    """Thông tin launch đã resolve cho một process."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} phải là integer") from exc


def _resolve_requested_device(requested: DeviceLike) -> torch.device:
    if isinstance(requested, torch.device):
        device = requested
    else:
        value = str(requested).strip().lower()
        if value == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda nhưng torch.cuda.is_available()=False; "
            "kiểm tra driver/CUDA runtime trước khi launch"
        )
    return device


def setup_distributed(requested_device: DeviceLike = "auto") -> DistributedContext:
    """Khởi tạo process group nếu đang chạy dưới ``torchrun``."""
    requested = _resolve_requested_device(requested_device)
    world_size = _env_int("WORLD_SIZE", 1)
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE phải >= 1, got {world_size}")

    if world_size == 1:
        if requested.type == "cuda":
            local_rank = requested.index if requested.index is not None else 0
            torch.cuda.set_device(local_rank)
            requested = torch.device("cuda", local_rank)
        return DistributedContext(0, 1, 0, requested)

    if not dist.is_available():
        raise RuntimeError("torch.distributed không khả dụng trong runtime này")
    rank = _env_int("RANK", -1)
    local_rank = _env_int("LOCAL_RANK", rank)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK không hợp lệ: rank={rank}, world_size={world_size}")
    if requested.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(rank, world_size, local_rank, device)


def current_context(device: torch.device) -> DistributedContext:
    """Lấy context hiện tại; không tự âm thầm init process group."""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = _env_int("LOCAL_RANK", rank)
        return DistributedContext(rank, world_size, local_rank, device)
    world_size = _env_int("WORLD_SIZE", 1)
    if world_size != 1:
        raise RuntimeError(
            "WORLD_SIZE>1 nhưng process group chưa init; hãy launch qua run_train.main"
        )
    return DistributedContext(0, 1, 0, device)


def rank_shard_indices(
    num_samples: int,
    *,
    batch_size: int,
    world_size: int,
    rank: int,
) -> list[int]:
    """Tạo shard disjoint, bỏ global tail để mọi rank có cùng số micro-batch."""
    if num_samples < 0 or batch_size < 1 or world_size < 1:
        raise ValueError("num_samples>=0, batch_size/world_size phải dương")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank không hợp lệ: {rank}/{world_size}")
    global_batch = batch_size * world_size
    usable = (num_samples // global_batch) * global_batch
    return list(range(usable))[rank:usable:world_size]


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """All-reduce một scalar/tensor và trả chính tensor đã cộng."""
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "DistributedContext",
    "setup_distributed",
    "current_context",
    "rank_shard_indices",
    "barrier",
    "all_reduce_sum",
    "destroy_process_group",
]
