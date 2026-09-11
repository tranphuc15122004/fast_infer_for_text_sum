"""Heartbeat/progress artifact cho worker regenerate/cache.

Mỗi worker ghi một JSON nhỏ bằng ``os.replace`` thông qua ``_common.write_json``.
Artifact này được thiết kế để đọc trong lúc worker đang chạy: nếu process bị
kill giữa lúc ghi, người theo dõi chỉ thấy file cũ hoàn chỉnh thay vì JSON hỏng.
Không ghi hidden state hay output vào đây.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from _common import write_json


SCHEMA_VERSION = "mr_dflash_worker_progress_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: Any) -> str:
    """Format thời lượng ngắn gọn để log người đọc được."""
    if seconds is None:
        return "unknown"
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "unknown"
    if value < 60:
        return f"{value:.1f}s"
    minutes = int(value // 60)
    remainder = int(value % 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours = minutes // 60
    minutes %= 60
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days = hours // 24
    hours %= 24
    return f"{days}d{hours:02d}h"


def estimate_fields(
    *,
    total_samples: int,
    total_tokens: int,
    completed_samples: int,
    completed_tokens: int,
    run_tokens: int,
    active_elapsed_seconds: float,
) -> dict[str, Any]:
    """Tính ETA theo tổng token đã capture trong *lần chạy hiện tại*.

    ``completed_*`` bao gồm cả sample đã có từ ``--resume``; ``run_tokens``
    chỉ gồm token được xử lý trong process hiện tại để throughput không bị
    sai khi resume. ETA là thời gian còn lại của worker, không bao gồm phase
    loading model/preflight đã qua.
    """
    total_samples = max(0, int(total_samples))
    total_tokens = max(0, int(total_tokens))
    completed_samples = max(0, int(completed_samples))
    completed_tokens = max(0, int(completed_tokens))
    run_tokens = max(0, int(run_tokens))
    elapsed = max(0.0, float(active_elapsed_seconds))
    rate = run_tokens / elapsed if run_tokens > 0 and elapsed > 0 else None
    remaining_samples = max(0, total_samples - completed_samples)
    remaining_tokens = max(0, total_tokens - completed_tokens)
    eta = remaining_tokens / rate if rate and remaining_tokens else (0.0 if remaining_tokens == 0 else None)
    return {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "completed_samples": completed_samples,
        "completed_tokens": completed_tokens,
        "remaining_samples": remaining_samples,
        "remaining_tokens": remaining_tokens,
        "active_elapsed_seconds": round(elapsed, 3),
        "throughput_tokens_per_second": round(rate, 3) if rate is not None else None,
        "eta_seconds": round(eta, 3) if eta is not None else None,
        "eta_human": format_duration(eta),
        "estimated_total_seconds": round(elapsed + eta, 3) if eta is not None else None,
    }


def _env_gpu_id() -> Optional[int]:
    value = os.environ.get("MR_DFLASH_GPU_ID")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cuda_memory() -> dict[str, float]:
    """Đọc telemetry CUDA hiện tại mà không làm progress phụ thuộc vào CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        # Worker chỉ nhìn thấy đúng một GPU qua CUDA_VISIBLE_DEVICES nên device
        # mặc định là chính GPU vật lý được launcher gán.
        return {
            "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "cuda_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
            "cuda_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "cuda_max_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }
    except Exception:
        # Telemetry không được phép làm chết generation/cache.
        return {}


def write_progress(path: str | Path, *, phase: str, **fields: Any) -> dict[str, Any]:
    """Ghi heartbeat atomically và trả payload đã ghi."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pid": os.getpid(),
        "gpu_id": _env_gpu_id(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "phase": str(phase),
        "updated_at": _now(),
        "updated_at_unix": time.time(),
    }
    payload.update(_cuda_memory())
    payload.update(fields)
    write_json(path, payload)
    return payload


def read_progress(path: str | Path) -> dict[str, Any]:
    """Đọc heartbeat hiện tại; file chưa tồn tại được coi là not_started."""
    target = Path(path)
    if not target.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": "not_started",
            "updated_at_unix": None,
        }
    import json

    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": "progress_unreadable",
            "error": repr(exc),
            "updated_at_unix": None,
        }
    if not isinstance(value, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": "progress_invalid",
            "updated_at_unix": None,
        }
    return value


class ProgressReporter:
    """Reporter no-op khi không có path, heartbeat khi chạy parallel."""

    def __init__(self, path: str | Path | None, *, interval_tokens: int = 256) -> None:
        self.path = Path(path) if path else None
        self.interval_tokens = max(1, int(interval_tokens))
        self._last_token_write: Optional[int] = None
        self._context: dict[str, Any] = {}

    def set_context(self, **fields: Any) -> None:
        """Giữ các field bất biến qua mọi heartbeat của cùng worker.

        Mỗi lần ``update`` ghi một JSON snapshot mới. Vì vậy các metadata như
        tổng số sample phải được giữ ở reporter; nếu chỉ ghi ở heartbeat đầu,
        heartbeat sau sẽ làm mất chúng và parent sẽ hiển thị ``0/0``.
        """
        self._context.update(fields)

    def update(self, phase: str, **fields: Any) -> dict[str, Any] | None:
        if self.path is None:
            return None
        payload = dict(self._context)
        payload.update(fields)
        return write_progress(self.path, phase=phase, **payload)

    def maybe_tokens(self, generated_tokens: int, **fields: Any) -> dict[str, Any] | None:
        generated_tokens = int(generated_tokens)
        if (
            self._last_token_write is not None
            and generated_tokens - self._last_token_write < self.interval_tokens
        ):
            return None
        self._last_token_write = generated_tokens
        return self.update("generating", generated_tokens=generated_tokens, **fields)

    def reset_tokens(self) -> None:
        self._last_token_write = None


def install_exception_hook(reporter: ProgressReporter):
    """Cài hook tạm để lỗi chưa bắt được cũng xuất hiện trong heartbeat."""
    previous = sys.excepthook

    def hook(exc_type, exc_value, traceback):
        sys.excepthook = previous
        reporter.update(
            "failed",
            error_type=getattr(exc_type, "__name__", str(exc_type)),
            error=repr(exc_value),
        )
        previous(exc_type, exc_value, traceback)

    sys.excepthook = hook
    return previous
