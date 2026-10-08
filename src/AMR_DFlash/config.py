"""Offline-safe YAML and environment configuration for AMR-DFlash runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .core import MemoryConfig


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"AMR-DFlash config not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "amr-v0":
        raise ValueError("AMR-DFlash config must be a mapping with schema_version amr-v0")
    for key in ("model", "inference", "memory", "training", "data", "output"):
        if not isinstance(value.get(key), dict):
            raise ValueError(f"AMR-DFlash config is missing section {key!r}")
    model = value["model"]
    if int(model.get("draft_layers", 5)) != 5 or int(model.get("block_size", 16)) != 16:
        raise ValueError("AMR-DFlash V0 locks the pretrained five-layer, block-16 contract")
    if float(value["inference"].get("temperature", 0.0)) != 0.0:
        raise ValueError("AMR-DFlash V0 supports greedy verification only (temperature=0)")
    memory = value["memory"]
    MemoryConfig(
        raw_budget=int(memory.get("raw_budget", 4096)),
        num_slots=int(memory.get("num_slots", 128)),
        local_window=int(memory.get("local_window", 128)),
        slot_gate_init=float(memory.get("slot_gate_init", -2.0)),
        min_context_tokens=int(memory.get("min_context_tokens", 4096)),
        query_window=int(memory.get("query_window", 16)),
    )
    return value


def resolve_env_path(config: dict[str, Any], env_key: str, *, required: bool = True) -> Path | None:
    """Resolve a named master-env value; config files never embed model paths."""
    env_name = str(config["model"].get(env_key, ""))
    if not env_name:
        raise ValueError(f"model.{env_key} must name an environment variable")
    value = os.environ.get(env_name, "").strip()
    if not value:
        if required:
            raise ValueError(f"environment variable {env_name} is required")
        return None
    return Path(value).expanduser()


def memory_config(config: dict[str, Any]) -> MemoryConfig:
    raw = config["memory"]
    return MemoryConfig(
        raw_budget=int(raw.get("raw_budget", 4096)),
        num_slots=int(raw.get("num_slots", 128)),
        local_window=int(raw.get("local_window", 128)),
        slot_gate_init=float(raw.get("slot_gate_init", -2.0)),
        min_context_tokens=int(raw.get("min_context_tokens", 4096)),
        query_window=int(raw.get("query_window", 16)),
    )


def resolve_run_root(config: dict[str, Any], *, create: bool = False) -> Path:
    env_name = str(config["output"].get("run_root_env", "AMR_RUN_ROOT"))
    configured = os.environ.get(env_name, "").strip()
    run_id_env = str(config["output"].get("run_id_env", "AMR_RUN_ID"))
    run_id = os.environ.get(run_id_env, "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        from common.paths import ROOT

        root = ROOT / "outputs" / "amr_dflash" / (run_id or "pilot")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root
