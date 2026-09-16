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
    last_success_batch_size: Optional[int] = None
    oom_count: int = 0
    upper_bound: Optional[int] = None
    last_effective_length: Optional[int] = None


class AdaptiveBatchController:
    """Grow a batch per length/budget bucket and back off after OOM.

    ``default`` in :meth:`batch_size` lets a caller provide a starting point
    for a new bucket. ``effective_length`` lets the caller look ahead at the
    next padded request; a longer request is scaled down from the last safe
    batch before the forward is issued. SGLang cache mode intentionally uses a
    small safe start, then applies a token-capacity guard and OOM backoff while
    growing.
    """

    def __init__(
        self,
        *,
        initial_batch_size: int,
        max_batch_size: int,
        growth_factor: float = 2.0,
        target_vram_gb: Optional[float] = None,
        hard_vram_gb: Optional[float] = None,
        safety_margin_gb: float = 0.0,
        length_safety_factor: float = 0.85,
    ) -> None:
        if int(initial_batch_size) < 1:
            raise ValueError("initial_batch_size phải >= 1")
        if int(max_batch_size) < int(initial_batch_size):
            raise ValueError("max_batch_size phải >= initial_batch_size")
        if float(growth_factor) <= 1.0:
            raise ValueError("growth_factor phải > 1")
        if target_vram_gb is not None and float(target_vram_gb) <= 0.0:
            raise ValueError("target_vram_gb phải > 0")
        if hard_vram_gb is not None and float(hard_vram_gb) <= 0.0:
            raise ValueError("hard_vram_gb phải > 0")
        if (
            target_vram_gb is not None
            and hard_vram_gb is not None
            and float(hard_vram_gb) < float(target_vram_gb)
        ):
            raise ValueError("hard_vram_gb phải >= target_vram_gb")
        if float(safety_margin_gb) < 0.0:
            raise ValueError("safety_margin_gb không được âm")
        if not 0.0 < float(length_safety_factor) <= 1.0:
            raise ValueError("length_safety_factor phải thuộc (0, 1]")
        if (
            target_vram_gb is not None
            and float(safety_margin_gb) >= float(target_vram_gb)
        ):
            raise ValueError("safety_margin_gb phải nhỏ hơn target_vram_gb")
        self.initial_batch_size = int(initial_batch_size)
        self.max_batch_size = int(max_batch_size)
        self.growth_factor = float(growth_factor)
        self.target_vram_gb = (
            None if target_vram_gb is None else float(target_vram_gb)
        )
        self.hard_vram_gb = (
            None if hard_vram_gb is None else float(hard_vram_gb)
        )
        self.safety_margin_gb = float(safety_margin_gb)
        self.length_safety_factor = float(length_safety_factor)
        self._states: dict[str, _BatchState] = {}
        self._last_success_batch_size: Optional[int] = None
        self._last_success_length: Optional[int] = None

    def _state(self, key: str, default: Optional[int] = None) -> _BatchState:
        name = str(key)
        state = self._states.get(name)
        if state is None:
            starting = self.initial_batch_size if default is None else int(default)
            starting = max(1, min(self.max_batch_size, starting))
            state = _BatchState(current=starting)
            self._states[name] = state
        return state

    def batch_size(
        self,
        key: str,
        default: Optional[int] = None,
        *,
        effective_length: Optional[int] = None,
    ) -> int:
        state = self._state(key, default)
        if effective_length is not None:
            length = int(effective_length)
            if length < 1:
                raise ValueError("effective_length phải >= 1")
            reference_batch = state.last_success_batch_size
            reference_length = state.last_effective_length
            if reference_batch is None or reference_length is None:
                reference_batch = self._last_success_batch_size
                reference_length = self._last_success_length
            if reference_batch is not None and reference_length is not None and length > reference_length:
                predicted = max(
                    1,
                    math.floor(
                        float(reference_batch)
                        * float(reference_length)
                        / float(length)
                        * self.length_safety_factor
                    ),
                )
                state.current = min(int(state.current), predicted)
        return int(state.current)

    def record_success(
        self,
        key: str,
        *,
        peak_vram_gb: Optional[float],
        max_next_batch_size: Optional[int] = None,
        effective_length: Optional[int] = None,
    ) -> int:
        state = self._state(key)
        if max_next_batch_size is not None and int(max_next_batch_size) < 1:
            raise ValueError("max_next_batch_size phải >= 1")
        ceiling = self.max_batch_size
        if max_next_batch_size is not None:
            ceiling = min(ceiling, int(max_next_batch_size))

        previous_peak = state.last_peak_vram_gb
        previous_batch = state.last_success_batch_size
        if effective_length is not None and int(effective_length) < 1:
            raise ValueError("effective_length phải >= 1")
        if peak_vram_gb is not None:
            peak = float(peak_vram_gb)
            if peak < 0.0:
                raise ValueError("peak_vram_gb không được âm")
            state.last_peak_vram_gb = peak
            state.last_success_batch_size = int(state.current)
            if self.target_vram_gb is not None and peak >= self._safe_vram_limit:
                state.target_reached = True
        if effective_length is not None:
            state.last_effective_length = int(effective_length)
        if effective_length is not None:
            self._last_success_batch_size = int(state.current)
            self._last_success_length = int(effective_length)
        if state.target_reached or state.current >= ceiling:
            return int(state.current)

        grown = None
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

        grown = min(ceiling, grown)

        # Predict the next request before launching it.  The first successful
        # observation establishes a baseline; the next one gives an online
        # per-sample slope.  Near the safety limit, switch from exponential
        # growth to the largest integer that is still predicted safe.  In
        # particular, if adding one sample is already unsafe, hold the current
        # batch and never issue that forward.
        if (
            self.target_vram_gb is not None
            and peak_vram_gb is not None
            and previous_peak is not None
            and previous_batch is not None
            and int(state.current) > int(previous_batch)
        ):
            slope = max(
                0.0,
                (float(peak_vram_gb) - float(previous_peak))
                / (int(state.current) - int(previous_batch)),
            )
            if slope > 0.0:
                safe_limit = self._safe_vram_limit
                one_more_prediction = float(peak_vram_gb) + slope
                if one_more_prediction >= safe_limit:
                    state.target_reached = True
                    return int(state.current)
                safe_delta = int(
                    math.floor((safe_limit - float(peak_vram_gb)) / slope)
                )
                grown = min(grown, int(state.current) + max(1, safe_delta))

        state.current = min(ceiling, grown)
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
            and state.last_peak_vram_gb >= self._safe_vram_limit
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
                "hard_vram_gb": self.hard_vram_gb,
                "last_success_batch_size": state.last_success_batch_size,
                "last_effective_length": state.last_effective_length,
                "length_safety_factor": self.length_safety_factor,
                "oom_count": int(state.oom_count),
                "upper_bound": state.upper_bound,
            }
            for key, state in sorted(self._states.items())
        }

    @property
    def _safe_vram_limit(self) -> float:
        if self.target_vram_gb is None:
            return float("inf")
        return float(self.target_vram_gb) - float(self.safety_margin_gb)


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
