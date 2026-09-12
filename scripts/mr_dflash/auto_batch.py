"""Small, backend-independent controller for adaptive inference batches.

The controller deliberately has no CUDA dependency.  Regeneration uses the
reported peak VRAM as its stopping signal, while the SpecForge cache backend
uses a preallocated static pool and therefore grows after each successful
batch without per-request telemetry.  Keeping this state in a pure module
makes both behaviours easy to test and keeps the retry policy identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class _BatchState:
    current: int
    target_reached: bool = False
    last_peak_vram_gb: Optional[float] = None
    oom_count: int = 0
    upper_bound: Optional[int] = None


class AdaptiveBatchController:
    """Grow a batch per length/budget bucket and back off after OOM.

    ``default`` in :meth:`batch_size` lets a caller provide a profiled starting
    point for a new bucket.  This is useful for cache workloads: the built-in
    SpecForge profile is a good warm start, then this controller can continue
    growing it to use the available VRAM.
    """

    def __init__(
        self,
        *,
        initial_batch_size: int,
        max_batch_size: int,
        growth_factor: float = 2.0,
        target_vram_gb: Optional[float] = None,
    ) -> None:
        if int(initial_batch_size) < 1:
            raise ValueError("initial_batch_size phải >= 1")
        if int(max_batch_size) < int(initial_batch_size):
            raise ValueError("max_batch_size phải >= initial_batch_size")
        if float(growth_factor) <= 1.0:
            raise ValueError("growth_factor phải > 1")
        if target_vram_gb is not None and float(target_vram_gb) <= 0.0:
            raise ValueError("target_vram_gb phải > 0")
        self.initial_batch_size = int(initial_batch_size)
        self.max_batch_size = int(max_batch_size)
        self.growth_factor = float(growth_factor)
        self.target_vram_gb = (
            None if target_vram_gb is None else float(target_vram_gb)
        )
        self._states: dict[str, _BatchState] = {}

    def _state(self, key: str, default: Optional[int] = None) -> _BatchState:
        name = str(key)
        state = self._states.get(name)
        if state is None:
            starting = self.initial_batch_size if default is None else int(default)
            starting = max(1, min(self.max_batch_size, starting))
            state = _BatchState(current=starting)
            self._states[name] = state
        return state

    def batch_size(self, key: str, default: Optional[int] = None) -> int:
        return int(self._state(key, default).current)

    def record_success(
        self,
        key: str,
        *,
        peak_vram_gb: Optional[float],
    ) -> int:
        state = self._state(key)
        if peak_vram_gb is not None:
            peak = float(peak_vram_gb)
            if peak < 0.0:
                raise ValueError("peak_vram_gb không được âm")
            state.last_peak_vram_gb = peak
            if self.target_vram_gb is not None and peak >= self.target_vram_gb:
                state.target_reached = True
        if state.target_reached or state.current >= self.max_batch_size:
            return int(state.current)
        if state.upper_bound is not None:
            # After an OOM, binary-search the interval between the last safe
            # batch and the failed candidate. This reaches the VRAM target
            # more closely than repeatedly doubling (which can freeze at a
            # batch far below 170GB after the first overshoot).
            if state.current >= state.upper_bound:
                return int(state.current)
            grown = (state.current + int(state.upper_bound) + 1) // 2
        else:
            grown = max(state.current + 1, math.ceil(state.current * self.growth_factor))
        state.current = min(self.max_batch_size, grown)
        return int(state.current)

    def record_oom(self, key: str, attempted_batch_size: Optional[int] = None) -> int:
        state = self._state(key)
        state.oom_count += 1
        attempted = state.current if attempted_batch_size is None else int(attempted_batch_size)
        if attempted < 1:
            raise ValueError("attempted_batch_size phải >= 1")
        state.target_reached = bool(
            self.target_vram_gb is not None
            and state.last_peak_vram_gb is not None
            and state.last_peak_vram_gb >= self.target_vram_gb
        )
        failed_upper = max(1, attempted - 1)
        state.upper_bound = (
            failed_upper
            if state.upper_bound is None
            else min(int(state.upper_bound), failed_upper)
        )
        if attempted <= state.current:
            state.current = max(1, (attempted + 1) // 2)
        return int(state.current)

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            key: {
                "batch_size": int(state.current),
                "target_reached": bool(state.target_reached),
                "last_peak_vram_gb": state.last_peak_vram_gb,
                "oom_count": int(state.oom_count),
                "upper_bound": state.upper_bound,
            }
            for key, state in sorted(self._states.items())
        }


def target_memory_fraction(
    target_vram_gb: Optional[float],
    total_vram_gb: float,
    *,
    requested_fraction: float,
) -> float:
    """Return a static-pool fraction that never exceeds the requested cap."""
    requested = float(requested_fraction)
    total = float(total_vram_gb)
    if not 0.0 < requested <= 1.0:
        raise ValueError("requested_fraction phải thuộc (0, 1]")
    if total <= 0.0:
        raise ValueError("total_vram_gb phải > 0")
    if target_vram_gb is None:
        return requested
    target = float(target_vram_gb)
    if target <= 0.0:
        raise ValueError("target_vram_gb phải > 0")
    return min(requested, target / total)


def cuda_memory_gb(device: object) -> tuple[float, float]:
    """Read total and currently reserved CUDA memory for ``device``.

    Imported lazily so tests and CPU-only preprocessing do not need CUDA.
    """
    import torch

    total = float(torch.cuda.get_device_properties(device).total_memory) / 2**30
    reserved = float(torch.cuda.memory_reserved(device)) / 2**30
    return total, reserved
