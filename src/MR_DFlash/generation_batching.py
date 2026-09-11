"""Primitive nhỏ cho batching phase target regeneration.

Các hàm ở đây không phụ thuộc Transformers để có thể kiểm thử CPU. Worker
generation dùng chúng để left-pad prompt và chọn các sample có cùng budget;
nhờ vậy tăng batch thật mà không đổi giới hạn ``prompt + output``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


GENERATION_BATCH_PROFILE_SCHEMA_VERSION = "mr_dflash_generation_batch_profile_v1"


def left_pad_prompt_ids(
    prompt_ids: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad các prompt thành batch và trả cả attention mask."""
    if not prompt_ids:
        raise ValueError("prompt_ids không được rỗng")
    normalized = [torch.as_tensor(value, dtype=torch.long).flatten() for value in prompt_ids]
    max_length = max(int(value.numel()) for value in normalized)
    if max_length < 1:
        raise ValueError("prompt phải có ít nhất một token")
    device = normalized[0].device
    if any(value.device != device for value in normalized):
        raise ValueError("mọi prompt phải ở cùng device")
    input_ids = torch.full(
        (len(normalized), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, value in enumerate(normalized):
        length = int(value.numel())
        input_ids[row, -length:] = value
        attention_mask[row, -length:] = 1
    return input_ids, attention_mask


def select_generation_group(
    items: Sequence[Mapping[str, Any]],
    *,
    max_batch_size: int,
    schedule: "GenerationBatchSchedule | None" = None,
) -> list[Mapping[str, Any]]:
    """Chọn nhóm an toàn cùng generation budget.

    Các sample được xếp theo prompt ngắn đến dài. Với profile, khi thêm một
    sample thì số phần tử nhóm không được vượt batch an toàn của bucket chứa
    sample dài nhất. Cùng budget là điều kiện để một ``max_new_tokens`` chung
    không làm thay đổi output của sample nào.
    """
    if max_batch_size < 1:
        raise ValueError("max_batch_size phải >= 1")
    if not items:
        return []
    first_budget = int(items[0].get("generation_budget", 0))
    candidates = sorted(
        (
            item
            for item in items
            if int(item.get("generation_budget", -1)) == first_budget
        ),
        key=lambda item: int(item.get("prompt_tokens", 0)),
    )
    selected: list[Mapping[str, Any]] = []
    for item in candidates:
        if len(selected) >= int(max_batch_size):
            break
        allowed = int(max_batch_size)
        if schedule is not None:
            allowed = min(
                allowed,
                schedule.batch_size_for_length(int(item.get("prompt_tokens", 0))),
            )
        if len(selected) >= allowed:
            break
        selected.append(item)
    return selected


class GenerationBatchSchedule:
    """Lookup batch size theo độ dài prompt."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != GENERATION_BATCH_PROFILE_SCHEMA_VERSION:
            raise ValueError("generation batch profile schema không hợp lệ")
        self.payload = dict(payload)
        self.max_length = int(payload["max_length"])
        self.requested_max_new_tokens = int(payload["requested_max_new_tokens"])
        self.buckets = sorted(
            [dict(bucket) for bucket in payload.get("buckets", [])],
            key=lambda bucket: int(bucket["min_length"]),
        )
        if self.max_length < 1 or self.requested_max_new_tokens < 1 or not self.buckets:
            raise ValueError("generation batch profile thiếu metadata/buckets")
        previous_max = 0
        for bucket in self.buckets:
            minimum = int(bucket["min_length"])
            maximum = int(bucket["max_length"])
            selected = int(bucket["selected_batch_size"])
            if minimum != previous_max + 1 or maximum < minimum or selected < 1:
                raise ValueError("generation batch profile buckets không liên tục")
            previous_max = maximum
        if previous_max != self.max_length:
            raise ValueError("generation batch profile không phủ hết max_length")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GenerationBatchSchedule":
        return cls(payload)

    @classmethod
    def from_path(cls, path: str | Path) -> "GenerationBatchSchedule":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("generation batch profile phải là JSON object")
        return cls(payload)

    def batch_size_for_length(self, prompt_length: int) -> int:
        prompt_length = int(prompt_length)
        if prompt_length < 1 or prompt_length > self.max_length:
            raise ValueError(
                f"prompt length {prompt_length} ngoài [1, {self.max_length}]"
            )
        for bucket in self.buckets:
            if prompt_length <= int(bucket["max_length"]):
                return int(bucket["selected_batch_size"])
        raise AssertionError("unreachable")
