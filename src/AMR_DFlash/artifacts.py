"""Atomic local artifacts and JSONL helpers for AMR-DFlash experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch


SCHEMA_VERSION = "amr-v0"


def read_jsonl(path: str | Path, *, drop_summary: bool = True) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            if drop_summary and row.get("record_type") == "summary":
                continue
            rows.append(row)
    return rows


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    summary: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            if summary is not None:
                final = {"record_type": "summary", **summary}
                handle.write(json.dumps(final, ensure_ascii=False, default=_json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def atomic_torch_save(value: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def load_torch(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location=map_location)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text[:120] or "sample"


def artifact_id(value: Any) -> str:
    """Keep filenames readable while hashing the entire, unsanitized identity."""
    return f"{safe_id(value)[:72]}-{sha256_text(str(value))}"


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def stable_split(document_id: str, *, train_fraction: float, validation_fraction: float) -> str:
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave a non-empty holdout")
    value = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "holdout"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")
