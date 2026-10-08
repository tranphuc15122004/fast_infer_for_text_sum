"""GPU-hour ledger shared by capture, labeling, and optimization phases."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def read_ledger(run_root: str | Path) -> dict[str, Any]:
    path = Path(run_root) / "resource_ledger.json"
    if not path.exists():
        return {"schema_version": "amr-v0", "total_gpu_seconds": 0.0, "phases": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "amr-v0":
        raise ValueError(f"unsupported resource ledger schema: {path}")
    return value


def remaining_gpu_seconds(
    run_root: str | Path,
    max_gpu_hours: float,
    *,
    device: torch.device,
    in_flight_seconds: float = 0.0,
) -> float:
    if device.type != "cuda":
        return float("inf")
    if max_gpu_hours < 0:
        raise ValueError("max_gpu_hours must be non-negative")
    limit = max_gpu_hours * 3600.0
    used = float(read_ledger(run_root).get("total_gpu_seconds", 0.0))
    return limit - used - max(0.0, in_flight_seconds)


def charge_phase(
    run_root: str | Path,
    phase: str,
    *,
    elapsed_seconds: float,
    device: torch.device,
    status: str = "complete",
) -> dict[str, Any]:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    ledger = read_ledger(root)
    phases = ledger.setdefault("phases", {})
    previous = phases.get(phase, {})
    gpu_seconds = elapsed_seconds if device.type == "cuda" else 0.0
    phases[phase] = {
        "gpu_seconds": float(previous.get("gpu_seconds", 0.0)) + gpu_seconds,
        "wall_seconds": float(previous.get("wall_seconds", 0.0)) + elapsed_seconds,
        "status": status,
        "device": str(device),
    }
    ledger["total_gpu_seconds"] = float(ledger.get("total_gpu_seconds", 0.0)) + gpu_seconds
    ledger["total_wall_seconds"] = float(ledger.get("total_wall_seconds", 0.0)) + elapsed_seconds
    path = root / "resource_ledger.json"
    fd, temporary_name = tempfile.mkstemp(prefix=".resource_ledger.", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(ledger, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return ledger
