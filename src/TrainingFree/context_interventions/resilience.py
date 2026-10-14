"""Isolate failed compute units while preserving scientific and device integrity."""

from __future__ import annotations

import gc
import math
import sys
import traceback
from collections.abc import Callable
from typing import Any


class CampaignIntegrityError(RuntimeError):
    """Continuing could produce invalid comparisons or use an unusable GPU."""


def sanitize_failed_numbers(record: dict[str, Any]) -> dict[str, Any]:
    """Keep invalid compute values as explicit failures in strict JSON checkpoints."""
    paths = []

    def clean(value, path):
        if isinstance(value, float) and not math.isfinite(value):
            paths.append(path)
            return None
        if isinstance(value, dict):
            return {key: clean(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    result = clean(record, "record")
    if paths:
        result["status"] = "error"
        result["nonfinite_fields"] = paths
        result["reason"] = f"nonfinite compute values: {paths}"
    return result


_FATAL_MESSAGES = (
    "device-side assert", "illegal memory access", "unspecified launch failure",
    "misaligned address", "uncorrectable ecc", "cuda context is destroyed",
    "cuda driver shutting down", "cuda error: initialization error", "cuda driver error",
    "cublas_status_execution_failed", "device kernel image is invalid", "device has been lost",
    "no kernel image is available", "invalid device function",
    "replay target cache crop mismatch", "replay draft-bank logical length changed",
)


def is_fatal_compute_error(exc: Exception) -> bool:
    if isinstance(exc, (CampaignIntegrityError, OSError, ImportError)):
        return True
    return any(fragment in str(exc).lower() for fragment in _FATAL_MESSAGES)


def release_compute_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            raise CampaignIntegrityError(f"GPU cleanup failed: {exc}") from exc


def run_compute_unit(
    operation: Callable[[], Any],
    *,
    label: str,
    result_error: Callable[[Any], str | None] | None = None,
    attempts: int = 2,
) -> tuple[Any, dict[str, Any] | None]:
    """Retry computation only; checkpoint writes and provenance checks stay outside.

    A failed result or ordinary compute exception gets one fresh attempt. After
    both attempts the caller records the failed unit and continues. Fatal device,
    filesystem, dependency and integrity errors propagate immediately.
    """
    if attempts < 1:
        raise ValueError("compute attempts must be positive")
    value = None
    error = None
    for attempt in range(1, attempts + 1):
        value = None
        try:
            value = operation()
            reason = result_error(value) if result_error is not None else None
            if reason is None:
                return value, None
            if is_fatal_compute_error(RuntimeError(reason)):
                raise CampaignIntegrityError(f"{label}: {reason}")
            error = {"status": "error", "reason": reason, "attempts": attempt}
        except Exception as exc:
            if is_fatal_compute_error(exc):
                if isinstance(exc, CampaignIntegrityError):
                    raise
                raise CampaignIntegrityError(f"{label}: {type(exc).__name__}: {exc}") from exc
            error = {
                "status": "error", "reason": f"{type(exc).__name__}: {exc}",
                "exception_type": type(exc).__name__, "attempts": attempt,
                "traceback": "".join(traceback.format_exception(exc))[-6000:],
            }
            # Tracebacks otherwise keep the failed forward's CUDA tensors alive.
            exc.__traceback__ = None
        print(
            f"[compute recovery] {label} attempt={attempt}/{attempts} "
            f"reason={error['reason']}; {'retrying' if attempt < attempts else 'recording failed unit'}",
            file=sys.stderr, flush=True,
        )
        if attempt < attempts:
            value = None
        release_compute_memory()
    return value, error


class FailureStreak:
    """Treat several wholly failed neighboring units as a systemic failure."""

    def __init__(self, limit: int = 3) -> None:
        self.limit = limit
        self.consecutive = 0

    def observe(self, *, failed: bool, label: str) -> None:
        self.consecutive = self.consecutive + 1 if failed else 0
        if self.consecutive >= self.limit:
            raise CampaignIntegrityError(
                f"{self.consecutive} consecutive compute units failed at {label}; "
                "stopping systemic failure before more evidence is lost"
            )
