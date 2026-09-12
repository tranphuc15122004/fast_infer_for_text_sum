"""Profile và scheduler cache theo token budget.

Module này chỉ dùng thư viện chuẩn. Profile được tạo/đọc như một hợp đồng
deterministic giữa profiler và cache worker; nó không biết GPU, torch hay
SGLang để có thể kiểm thử trên CPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CACHE_THROUGHPUT_PROFILE_SCHEMA_VERSION = "mr_dflash_cache_throughput_profile_v1"
CACHE_THROUGHPUT_PROFILE_VERSION = 1


@dataclass(frozen=True)
class CacheThroughputBucket:
    """Một bucket độ dài và giới hạn batch/token tương ứng."""

    min_length: int
    max_length: int
    batch_size: int
    token_budget: int

    def to_payload(self) -> dict[str, int]:
        return {
            "min_length": int(self.min_length),
            "max_length": int(self.max_length),
            "batch_size": int(self.batch_size),
            "token_budget": int(self.token_budget),
        }


def _as_int(value: Any, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} phải là số nguyên") from exc


def _read_bucket_value(raw: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    raise KeyError(names[0])


@dataclass(frozen=True)
class CacheThroughputProfile:
    """Lịch batch cố định theo độ dài, có trần token cho batch đã pad."""

    buckets: tuple[CacheThroughputBucket, ...]
    schema_version: str = CACHE_THROUGHPUT_PROFILE_SCHEMA_VERSION
    version: int = CACHE_THROUGHPUT_PROFILE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_THROUGHPUT_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "schema_version profile cache không tương thích: "
                f"{self.schema_version!r}"
            )
        if int(self.version) != CACHE_THROUGHPUT_PROFILE_VERSION:
            raise ValueError(f"version profile cache không hỗ trợ: {self.version!r}")
        self._validate_buckets(self.buckets)

    @staticmethod
    def _validate_buckets(buckets: Sequence[CacheThroughputBucket]) -> None:
        if not buckets:
            raise ValueError("buckets profile cache không được rỗng")
        expected_min = 1
        previous_max = 0
        for bucket in buckets:
            values = (
                int(bucket.min_length),
                int(bucket.max_length),
                int(bucket.batch_size),
                int(bucket.token_budget),
            )
            if any(value < 0 for value in values):
                raise ValueError("bucket profile cache không được có giá trị âm")
            if bucket.min_length < 1 or bucket.max_length < 1:
                raise ValueError("độ dài bucket profile cache phải >= 1")
            if bucket.batch_size < 1:
                raise ValueError("batch_size bucket profile cache phải > 0")
            if bucket.token_budget < 1:
                raise ValueError("token_budget bucket profile cache phải > 0")
            if bucket.max_length <= previous_max:
                raise ValueError(
                    "max_length bucket profile cache phải tăng dần, "
                    "không được trùng hoặc chồng lấn"
                )
            if bucket.min_length != expected_min:
                raise ValueError(
                    "bucket profile cache phải liên tục, không có khoảng trống "
                    "hoặc chồng lấn"
                )
            if bucket.max_length < bucket.min_length:
                raise ValueError(
                    "bucket profile cache có max_length nhỏ hơn min_length"
                )
            previous_max = bucket.max_length
            expected_min = bucket.max_length + 1

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CacheThroughputProfile":
        """Parse và kiểm tra payload JSON của profile."""
        if not isinstance(payload, Mapping):
            raise ValueError("profile cache phải là JSON object")

        schema_version = payload.get("schema_version")
        if schema_version is not None and schema_version != CACHE_THROUGHPUT_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "schema_version profile cache không tương thích: "
                f"{schema_version!r}"
            )
        raw_version = payload.get("version")
        if raw_version is None:
            raw_version = payload.get("profile_version")
        if raw_version is None and schema_version is None:
            raise ValueError("profile cache thiếu version/schema_version")
        version = (
            CACHE_THROUGHPUT_PROFILE_VERSION
            if raw_version is None
            else _as_int(raw_version, field="version profile cache")
        )
        if version != CACHE_THROUGHPUT_PROFILE_VERSION:
            raise ValueError(f"version profile cache không hỗ trợ: {version!r}")

        raw_buckets = payload.get("buckets")
        if not isinstance(raw_buckets, list) or not raw_buckets:
            raise ValueError("buckets profile cache không được rỗng")

        buckets: list[CacheThroughputBucket] = []
        expected_min = 1
        previous_max = 0
        for raw in raw_buckets:
            if not isinstance(raw, Mapping):
                raise ValueError("mỗi bucket profile cache phải là JSON object")
            try:
                raw_min = raw.get("min_length", expected_min)
                min_length = _as_int(raw_min, field="min_length bucket")
                max_length = _as_int(
                    _read_bucket_value(raw, "max_length", "upper_bound"),
                    field="max_length bucket",
                )
                batch_size = _as_int(
                    _read_bucket_value(raw, "batch_size", "candidate_batch_size"),
                    field="batch_size bucket",
                )
                token_budget = _as_int(
                    _read_bucket_value(raw, "token_budget", "max_total_tokens"),
                    field="token_budget bucket",
                )
            except KeyError as exc:
                raise ValueError(
                    "bucket profile cache thiếu max_length/batch_size/token_budget"
                ) from exc

            values = (min_length, max_length, batch_size, token_budget)
            if any(value < 0 for value in values):
                raise ValueError("bucket profile cache không được có giá trị âm")
            if max_length <= previous_max:
                raise ValueError(
                    "max_length bucket profile cache phải tăng dần, "
                    "không được trùng hoặc chồng lấn"
                )
            if min_length != expected_min:
                raise ValueError(
                    "bucket profile cache phải liên tục, không có khoảng trống "
                    "hoặc chồng lấn"
                )
            if max_length < min_length:
                raise ValueError(
                    "bucket profile cache có max_length nhỏ hơn min_length"
                )
            bucket = CacheThroughputBucket(
                min_length=min_length,
                max_length=max_length,
                batch_size=batch_size,
                token_budget=token_budget,
            )
            buckets.append(bucket)
            previous_max = max_length
            expected_min = max_length + 1

        cls._validate_buckets(tuple(buckets))
        return cls(
            buckets=tuple(buckets),
            schema_version=CACHE_THROUGHPUT_PROFILE_SCHEMA_VERSION,
            version=version,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "CacheThroughputProfile":
        profile_path = Path(path)
        if not profile_path.is_file():
            raise FileNotFoundError(f"không tìm thấy throughput profile: {profile_path}")
        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"không đọc được throughput profile: {profile_path}") from exc
        return cls.from_payload(payload)

    @property
    def max_length(self) -> int:
        return int(self.buckets[-1].max_length)

    @property
    def max_batch_size(self) -> int:
        return max(int(bucket.batch_size) for bucket in self.buckets)

    @property
    def serialized_payload(self) -> str:
        """JSON canonical, không phụ thuộc thứ tự key của input payload."""
        return json.dumps(
            self.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def profile_sha256(self) -> str:
        return hashlib.sha256(self.serialized_payload.encode("utf-8")).hexdigest()

    @property
    def profile_hash(self) -> str:
        """Alias dễ dùng khi ghi provenance."""
        return self.profile_sha256

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": int(self.version),
            "buckets": [bucket.to_payload() for bucket in self.buckets],
        }

    def to_json(self) -> str:
        return self.serialized_payload

    def _bucket_for_length(self, length: int) -> CacheThroughputBucket:
        value = _as_int(length, field="sample length")
        if value < 1:
            raise ValueError("sample length phải >= 1")
        for bucket in self.buckets:
            if value <= bucket.max_length:
                return bucket
        raise ValueError(
            f"sample length {value} vượt bucket cuối {self.max_length}"
        )

    def batch_for_length(self, length: int) -> int:
        return int(self._bucket_for_length(length).batch_size)

    def token_budget_for_length(self, length: int) -> int:
        return int(self._bucket_for_length(length).token_budget)

    def cap_batch_size(
        self,
        *,
        padded_length: int,
        requested_batch_size: int | None = None,
    ) -> int:
        """Cap batch theo số token sau padding của batch thực tế."""
        padded = _as_int(padded_length, field="padded_length")
        if padded < 1:
            raise ValueError("padded_length phải >= 1")
        bucket = self._bucket_for_length(padded)
        requested = (
            int(bucket.batch_size)
            if requested_batch_size is None
            else _as_int(requested_batch_size, field="requested_batch_size")
        )
        if requested < 1:
            raise ValueError("requested_batch_size phải > 0")
        token_cap = int(bucket.token_budget) // padded
        if token_cap < 1:
            raise ValueError(
                f"token_budget {bucket.token_budget} không đủ cho padded_length {padded}"
            )
        return max(1, min(int(bucket.batch_size), requested, token_cap))

    def batch_for_lengths(
        self,
        lengths: Sequence[int],
        *,
        requested_batch_size: int | None = None,
    ) -> int:
        """Trả batch tối đa sau khi xét padded length của nhóm sample."""
        if not lengths:
            raise ValueError("lengths không được rỗng")
        padded_length = max(_as_int(length, field="sample length") for length in lengths)
        return self.cap_batch_size(
            padded_length=padded_length,
            requested_batch_size=requested_batch_size,
        )

    def validate(self, max_length: int, batch_limit: int) -> None:
        """Kiểm tra profile có dùng được trong worker hiện tại."""
        requested_max_length = _as_int(max_length, field="max_length")
        if requested_max_length < 1:
            raise ValueError("max_length phải >= 1")
        limit = _as_int(batch_limit, field="batch_limit")
        if limit < 1:
            raise ValueError("batch_limit phải > 0")
        self._validate_buckets(self.buckets)
        if self.max_length < requested_max_length:
            raise ValueError(
                f"profile không bao phủ max_length={requested_max_length}; "
                f"bucket cuối={self.max_length}"
            )
        if self.max_batch_size > limit:
            raise ValueError(
                f"batch_limit={limit} nhỏ hơn batch lớn nhất của profile "
                f"({self.max_batch_size})"
            )


__all__ = [
    "CACHE_THROUGHPUT_PROFILE_SCHEMA_VERSION",
    "CACHE_THROUGHPUT_PROFILE_VERSION",
    "CacheThroughputBucket",
    "CacheThroughputProfile",
]
