"""Checkpoint metadata and strict asset-contract validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .artifacts import atomic_torch_save


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
}


def snapshot_fingerprint(path: str | Path) -> dict[str, Any]:
    """Hash local model weights and config/tokenizer files for run provenance."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model snapshot directory not found: {root}")
    selected: list[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        name = candidate.name
        if (
            name in TOKENIZER_NAMES
            or name in {"config.json", "generation_config.json"}
            or name.endswith((".safetensors.index.json", ".bin.index.json"))
            or name.endswith(".model")
            or name.endswith(".tiktoken")
            or (candidate.suffix in WEIGHT_SUFFIXES and "optimizer" not in name)
        ):
            selected.append(candidate)
    if not selected:
        raise ValueError(f"no local model/config/tokenizer files found in {root}")
    files = []
    combined = hashlib.sha256()
    for candidate in sorted(selected):
        file_hash = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                file_hash.update(chunk)
        relative = candidate.relative_to(root).as_posix()
        digest = file_hash.hexdigest()
        files.append({"path": relative, "size_bytes": candidate.stat().st_size, "sha256": digest})
        combined.update(relative.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return {
        "resolved_path": str(root),
        "files": files,
        "sha256": combined.hexdigest(),
    }


def contract_identity(value: Any) -> Any:
    """Absolute snapshot locations are provenance; file hashes define identity."""
    if isinstance(value, dict):
        is_snapshot = "sha256" in value and "files" in value
        return {key: contract_identity(item) for key, item in value.items()
                if not (is_snapshot and key == "resolved_path")}
    if isinstance(value, list):
        return [contract_identity(item) for item in value]
    return value


def feature_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_sha256": metadata["target_fingerprint"]["sha256"],
        "draft_sha256": metadata["draft_fingerprint"]["sha256"],
        "target_layer_ids": metadata["target_layer_ids"],
        "feature_hidden_offset": metadata["feature_hidden_offset"],
        "dtype": metadata["dtype"],
        "attention_backend": metadata["attention_backend"],
    }


def save_memory_checkpoint(
    path: str | Path,
    memory: torch.nn.Module,
    *,
    metadata: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
    step: int | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "format": "amr-v0",
        "memory_state_dict": {key: value.detach().cpu() for key, value in memory.state_dict().items()},
        "metadata": metadata,
        "step": step,
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    return atomic_torch_save(payload, path)


def load_memory_checkpoint(
    path: str | Path,
    memory: torch.nn.Module,
    *,
    expected_metadata: dict[str, Any],
    device: torch.device,
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "amr-v0":
        raise ValueError("unsupported AMR-DFlash checkpoint format")
    saved_metadata = payload.get("metadata") or {}
    for key, expected in expected_metadata.items():
        actual = saved_metadata.get(key)
        if contract_identity(actual) != contract_identity(expected):
            raise ValueError(
                f"AMR checkpoint fingerprint mismatch for {key}: expected {expected!r}, got {actual!r}"
            )
    missing, unexpected = memory.load_state_dict(
        payload.get("memory_state_dict", {}), strict=strict
    )
    if strict and (missing or unexpected):
        raise ValueError(f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    memory.to(device)
    return payload


def build_model_metadata(
    target_fingerprint: dict[str, Any],
    draft_fingerprint: dict[str, Any],
    *,
    target_layer_ids: list[int],
    block_size: int,
    memory_config: dict[str, Any],
    index_dim: int,
) -> dict[str, Any]:
    return {
        "schema_version": "amr-v0",
        "target_fingerprint": target_fingerprint,
        "draft_fingerprint": draft_fingerprint,
        "target_layer_ids": [int(value) for value in target_layer_ids],
        "block_size": int(block_size),
        "memory_config": memory_config,
        "index_dim": int(index_dim),
        "feature_hidden_offset": 1,
        "alignment_history": "captured_verifier_candidates",
    }
