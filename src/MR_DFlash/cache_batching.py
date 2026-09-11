"""Lập lịch batch an toàn cho target-feature cache.

Module này chỉ chứa logic thuần Python để có thể kiểm thử không cần GPU. Một
profile được tạo bằng ``profile_cache_batches.py`` trên chính GPU sẽ cache.
Trong lúc cache, schedule là cố định theo bucket độ dài; cache không tự thử
batch khác sau khi đã bắt đầu ghi artifact. Điều này tránh tình trạng một
worker bị OOM giữa chừng nhưng phần cache đã ghi dở với các điều kiện không
đồng nhất.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CACHE_BATCH_PROFILE_SCHEMA_VERSION = "mr_dflash_cache_batch_profile_v1"


def default_bucket_boundaries(max_length: int, bucket_step: int = 8192) -> list[int]:
    """Trả về cận trên các bucket, bao phủ ``1..max_length``."""
    max_length = int(max_length)
    bucket_step = int(bucket_step)
    if max_length < 1:
        raise ValueError("max_length phải >= 1")
    if bucket_step < 1:
        raise ValueError("bucket_step phải >= 1")
    boundaries = list(range(bucket_step, max_length, bucket_step))
    if not boundaries or boundaries[-1] != max_length:
        boundaries.append(max_length)
    return boundaries


def make_bucket_specs(max_length: int, bucket_step: int = 8192) -> list[dict[str, int]]:
    """Tạo mô tả bucket liên tục, không chồng lấn."""
    result: list[dict[str, int]] = []
    lower = 1
    for upper in default_bucket_boundaries(max_length, bucket_step):
        result.append({"min_length": lower, "max_length": int(upper)})
        lower = int(upper) + 1
    return result


def candidate_batch_sizes(
    max_batch_size: int,
    requested: Sequence[int] | None = None,
) -> list[int]:
    """Chuẩn hóa candidates tăng dần và luôn giữ batch size 1."""
    maximum = int(max_batch_size)
    if maximum < 1:
        raise ValueError("max_batch_size phải >= 1")
    if requested is None:
        values: list[int] = []
        value = 1
        while value < maximum:
            values.append(value)
            value *= 2
        values.append(maximum)
    else:
        values = [int(value) for value in requested]
        if any(value < 1 for value in values):
            raise ValueError("candidate batch size phải >= 1")
        values.append(1)
    values = sorted({value for value in values if value <= maximum})
    if not values:
        raise ValueError("không có candidate batch size nào hợp lệ")
    return values


def _peak_bytes(result: Mapping[str, Any]) -> int | None:
    """Lấy reserved memory, fallback allocated cho profile cũ/fixture."""
    for key in ("peak_memory_reserved_bytes", "peak_memory_allocated_bytes"):
        value = result.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def choose_largest_safe_batch_size(
    results: Iterable[Mapping[str, Any]],
    *,
    vram_limit_bytes: int,
) -> int:
    """Chọn batch lớn nhất có status ``pass`` dưới ngưỡng VRAM.

    Batch 1 phải pass và nằm dưới ngưỡng. Nếu không, profile bị coi là không
    an toàn và cache không được phép bắt đầu.
    """
    limit = int(vram_limit_bytes)
    if limit < 1:
        raise ValueError("vram_limit_bytes phải >= 1")
    safe: list[int] = []
    batch_one_safe = False
    for result in results:
        try:
            batch = int(result["batch_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("profile result thiếu batch_size hợp lệ") from exc
        status = str(result.get("status", ""))
        peak = _peak_bytes(result)
        is_safe = status == "pass" and peak is not None and peak <= limit
        if batch == 1 and is_safe:
            batch_one_safe = True
        if is_safe:
            safe.append(batch)
    if not batch_one_safe:
        raise RuntimeError(
            "profile không chứng minh được batch size 1 an toàn dưới ngưỡng VRAM; "
            "dừng trước khi cache"
        )
    return max(safe)


@dataclass(frozen=True)
class CacheBatchSchedule:
    """Schedule batch cố định theo độ dài, được đọc từ profile JSON."""

    target_model_path: str
    feature_layer_ids: tuple[int, ...]
    max_length: int
    requested_torch_dtype: str
    attention_backend: str
    buckets: tuple[dict[str, int], ...]
    target_revision: str | None = None
    profile_path: str | None = None
    profile_sha256: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        profile_path: str | None = None,
    ) -> "CacheBatchSchedule":
        if payload.get("schema_version") != CACHE_BATCH_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "cache batch profile schema không tương thích: "
                f"{payload.get('schema_version')!r}"
            )
        try:
            target_model_path = str(payload["target_model_path"])
            layers = tuple(int(value) for value in payload["feature_layer_ids"])
            max_length = int(payload["max_length"])
            requested_dtype = str(payload["requested_torch_dtype"])
            attention_backend = str(payload["attention_backend"])
            raw_buckets = payload["buckets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("cache batch profile thiếu metadata bắt buộc") from exc
        if not layers:
            raise ValueError("cache batch profile phải có feature_layer_ids")
        if max_length < 1 or not isinstance(raw_buckets, list) or not raw_buckets:
            raise ValueError("cache batch profile có max_length/buckets không hợp lệ")
        buckets: list[dict[str, int]] = []
        expected_min = 1
        for raw in raw_buckets:
            if not isinstance(raw, Mapping):
                raise ValueError("mỗi bucket trong cache batch profile phải là object")
            try:
                lower = int(raw["min_length"])
                upper = int(raw["max_length"])
                selected = int(raw["selected_batch_size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("bucket profile thiếu min/max/selected_batch_size") from exc
            if lower != expected_min or upper < lower or upper > max_length or selected < 1:
                raise ValueError("bucket profile không liên tục hoặc batch không hợp lệ")
            buckets.append(
                {
                    "min_length": lower,
                    "max_length": upper,
                    "selected_batch_size": selected,
                }
            )
            expected_min = upper + 1
        if expected_min != max_length + 1:
            raise ValueError("bucket profile không bao phủ hết max_length")
        return cls(
            target_model_path=target_model_path,
            feature_layer_ids=layers,
            max_length=max_length,
            requested_torch_dtype=requested_dtype,
            attention_backend=attention_backend,
            buckets=tuple(buckets),
            target_revision=(
                str(payload["target_revision"])
                if payload.get("target_revision") is not None
                else None
            ),
            profile_path=profile_path,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "CacheBatchSchedule":
        profile = Path(path)
        if not profile.is_file():
            raise FileNotFoundError(f"không tìm thấy cache batch profile: {profile}")
        try:
            payload = json.loads(profile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"không đọc được cache batch profile: {profile}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"cache batch profile phải là JSON object: {profile}")
        schedule = cls.from_payload(payload, profile_path=str(profile))
        return replace(
            schedule,
            profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        )

    @property
    def max_batch_size(self) -> int:
        return max(int(bucket["selected_batch_size"]) for bucket in self.buckets)

    def batch_size_for_length(self, length: int) -> int:
        value = int(length)
        if value < 1 or value > self.max_length:
            raise ValueError(
                f"sample length {value} vượt max_length của profile {self.max_length}"
            )
        for bucket in self.buckets:
            if int(bucket["min_length"]) <= value <= int(bucket["max_length"]):
                return int(bucket["selected_batch_size"])
        raise ValueError(f"profile không có bucket cho sample length {value}")

    def validate(
        self,
        *,
        target_model_path: str,
        feature_layer_ids: Sequence[int],
        max_length: int,
        requested_torch_dtype: str,
        attention_backend: str,
        target_revision: str | None = None,
    ) -> None:
        """Fail fast nếu cache đang dùng profile của model/cấu hình khác."""
        mismatches: list[str] = []
        if self.target_model_path != str(target_model_path):
            mismatches.append("target_model_path")
        if self.feature_layer_ids != tuple(int(value) for value in feature_layer_ids):
            mismatches.append("feature_layer_ids")
        if self.max_length != int(max_length):
            mismatches.append("max_length")
        if self.requested_torch_dtype != str(requested_torch_dtype):
            mismatches.append("requested_torch_dtype")
        if self.attention_backend != str(attention_backend):
            mismatches.append("attention_backend")
        if self.target_revision != (str(target_revision) if target_revision is not None else None):
            mismatches.append("target_revision")
        if mismatches:
            raise RuntimeError(
                "cache batch profile không khớp cấu hình hiện tại ở: "
                + ", ".join(mismatches)
                + "; tạo profile mới"
            )


__all__ = [
    "CACHE_BATCH_PROFILE_SCHEMA_VERSION",
    "CacheBatchSchedule",
    "candidate_batch_sizes",
    "choose_largest_safe_batch_size",
    "default_bucket_boundaries",
    "make_bucket_specs",
]
