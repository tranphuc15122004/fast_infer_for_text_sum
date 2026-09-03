"""Measured runtime profile schema consumed by the SyncSpec controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import statistics
import time
from typing import Callable, Any


@dataclass(frozen=True)
class ProfileKey:
    model: str
    checkpoint: str
    gpu: str
    precision: str
    context_bin: str
    batch_bin: str
    kd: int
    kv: int
    kernel: str = "pytorch"
    selector_checkpoint: str | None = None
    survival_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeProfiler:
    key: ProfileKey
    measurements_ms: dict[str, list[float]] = field(default_factory=dict)
    peak_memory_mb: float | None = None
    target_ar_tokens: int = 1
    source: str = "measured"

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise ValueError("profile source must be non-empty")

    def record(self, component: str, elapsed_ms: float) -> None:
        self.measurements_ms.setdefault(component, []).append(float(elapsed_ms))

    def measure(self, component: str, function: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        result = function()
        elapsed = (time.perf_counter() - start) * 1000.0
        self.record(component, elapsed)
        return result

    def record_peak_memory(self, memory_mb: float) -> None:
        value = float(memory_mb)
        self.peak_memory_mb = value if self.peak_memory_mb is None else max(self.peak_memory_mb, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": str(self.source),
            "key": self.key.to_dict(),
            "measurements_ms": {
                name: {
                    "count": len(values),
                    "mean": statistics.fmean(values) if values else 0.0,
                    "p50": statistics.median(values) if values else 0.0,
                    "p95": (
                        statistics.quantiles(values, n=20, method="inclusive")[-1]
                        if len(values) > 1 else (values[0] if values else 0.0)
                    ),
                    "samples": values,
                }
                for name, values in self.measurements_ms.items()
            },
            "peak_memory_mb": self.peak_memory_mb,
            "target_ar_tokens": int(self.target_ar_tokens),
        }

    def save(self, path: Path | str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def costs_from_profile(payload: dict[str, Any], component: str = "verify") -> dict[int, float]:
    """Extract `kv -> ms` from a profile directory/file without extrapolation."""
    result: dict[int, float] = {}
    records = payload if isinstance(payload, list) else [payload]
    for record in records:
        key = record.get("key", {})
        if component not in record.get("measurements_ms", {}):
            continue
        result[int(key["kv"])] = float(record["measurements_ms"][component]["mean"])
    return result


def round_costs_from_profile(payload: dict[str, Any]) -> dict[int, float]:
    """Extract measured speculative-round cost keyed by ``K_v``.

    The analytical controller's denominator includes the fixed draft,
    selector, survival and scheduler work in addition to target verification.
    Prefer the median of each available component; a legacy profile containing
    only ``verify`` remains valid, while a profile with only ``e2e`` is used as
    a conservative fallback.
    """
    result: dict[int, float] = {}
    records = payload if isinstance(payload, list) else [payload]
    components = ("draft", "selector", "survival", "verify", "scheduler")
    for record in records:
        key = record.get("key", {})
        measurements = record.get("measurements_ms", {})
        if "kv" not in key or not isinstance(measurements, dict):
            continue
        values = []
        for component in components:
            measurement = measurements.get(component)
            if not isinstance(measurement, dict):
                continue
            value = measurement.get("p50", measurement.get("mean"))
            if value is not None:
                values.append(float(value))
        if values:
            result[int(key["kv"])] = sum(values)
            continue
        measurement = measurements.get("e2e")
        if isinstance(measurement, dict):
            value = measurement.get("p50", measurement.get("mean"))
            if value is not None:
                result[int(key["kv"])] = float(value)
    return result


def ar_cost_from_profile(payload: dict[str, Any]) -> float | None:
    """Return measured target-AR latency per committed token.

    New profiles explicitly record ``target_ar_tokens``. For older profiles
    produced by this repository, ``target_ar`` measured a block of ``kv``
    tokens, so divide by that axis as a compatibility fallback.
    """
    values: list[float] = []
    records = payload if isinstance(payload, list) else [payload]
    for record in records:
        if not isinstance(record, dict):
            continue
        measurements = record.get("measurements_ms", {})
        measurement = measurements.get("target_ar") if isinstance(measurements, dict) else None
        if not isinstance(measurement, dict):
            continue
        raw = measurement.get("p50", measurement.get("mean"))
        if raw is None:
            continue
        key = record.get("key", {})
        tokens = record.get("target_ar_tokens")
        if tokens is None:
            tokens = key.get("kv", 1) if isinstance(key, dict) else 1
        try:
            cost = float(raw) / max(1, int(tokens))
        except (TypeError, ValueError):
            continue
        if cost > 0.0 and math.isfinite(cost):
            values.append(cost)
    return sum(values) / len(values) if values else None
