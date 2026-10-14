"""Atomic cell checkpoints and schema-consistent phase JSONL artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from TrainingFree.context_adaptive.schema import atomic_json


class CellStore:
    def __init__(self, phase_root: Path, *, signature: str, resume: bool) -> None:
        self.root = Path(phase_root)
        self.cells = self.root / "cells"
        self.cells.mkdir(parents=True, exist_ok=True)
        self._had_cells_at_start = any(self.cells.glob("*.json"))
        self.signature = str(signature)
        self.resume = bool(resume)

    @staticmethod
    def key(value: str) -> str:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

    def path(self, cell_id: str) -> Path:
        return self.cells / f"{self.key(cell_id)}.json"

    def load(self, cell_id: str) -> dict[str, Any] | None:
        path = self.path(cell_id)
        if not path.is_file():
            return None
        if not self.resume:
            raise FileExistsError(f"cell checkpoint exists; pass --resume: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("cell_id") != cell_id or row.get("run_signature") != self.signature:
            raise ValueError(f"resume signature/cell mismatch at {path}")
        return row

    def save(self, cell_id: str, row: Mapping[str, Any]) -> None:
        path = self.path(cell_id)
        payload = dict(row)
        payload["cell_id"] = cell_id
        payload["run_signature"] = self.signature
        if path.exists() and self.resume:
            existing = self.load(cell_id)
            if existing == payload:
                return
            if existing.get("status") != "error":
                raise ValueError(f"refusing to replace completed cell {cell_id}")
            # Error rows are failed attempts, not completed evidence. Replacing
            # them atomically lets a resumed phase retry transient cell errors.
            atomic_json(path, payload)
            return
        if path.exists():
            raise FileExistsError(f"cell checkpoint already exists: {path}")
        atomic_json(path, payload)

    def rows(self) -> list[dict[str, Any]]:
        paths = sorted(self.cells.glob("*.json"))
        if self._had_cells_at_start and not self.resume:
            raise FileExistsError(f"campaign cells already exist; pass --resume or choose a new run id: {self.cells}")
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        for path, row in zip(paths, rows):
            if row.get("run_signature") != self.signature or self.path(str(row.get("cell_id"))) != path:
                raise ValueError(f"resume signature mismatch at {path}")
        return sorted(rows, key=lambda row: str(row.get("cell_id", "")))

    def merge(
        self, output_path: Path, *, summary: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        # Use the repository's canonical writer, then atomically publish the
        # complete stream. A stopped phase retains per-cell checkpoints only.
        from common.io_util import JsonlWriter

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".campaign-", suffix=".jsonl", dir=output_path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
        rows = self.rows() if records is None else list(records)
        writer = JsonlWriter(temp_path)
        for row in rows:
            writer.add(row)
        summary_record = dict(summary)
        quality_rows = [row for row in rows if row.get("status") == "ok"]
        if any(any(row.get(key) is not None for key in ("rouge1", "rouge2", "rougeL")) for row in quality_rows):
            from common.rouge import aggregate_rouge

            try:
                summary_record.update(aggregate_rouge(quality_rows))
            except (TypeError, ValueError, ArithmeticError) as exc:
                summary_record["rouge_status"] = f"error:{type(exc).__name__}:{exc}"
        writer.finalize(summary_record)
        os.replace(temp_path, output_path)
