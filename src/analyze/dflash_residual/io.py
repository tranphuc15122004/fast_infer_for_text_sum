"""JSONL adapters and deterministic output helpers for residual experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import SCHEMA_VERSION, normalize_trace_row


def read_trace_jsonl(
    path: str | Path,
    *,
    strict: bool = False,
    dataset_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Read v1 trace rows; malformed rows become explicit errors by default."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("row must be a JSON object")
            if dataset_filter is not None:
                dataset = str(raw.get("task_regime", raw.get("dataset", ""))).lower()
                if dataset != str(dataset_filter).lower():
                    continue
            rows.append(normalize_trace_row(raw))
        except Exception as exc:
            if strict:
                raise ValueError(f"Invalid residual trace at {path}:{line_number}: {exc}") from exc
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "line_number": line_number,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return rows


def read_selection_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read minimal DFlash2 selection rows without requiring candidate fields."""

    path = Path(path)
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"Selection row at {path}:{line_number} is not an object")
        for field in ("run_id", "sample_id", "round_index", "draft_position"):
            if field not in raw:
                raise ValueError(f"Selection row at {path}:{line_number} missing {field}")
        if raw.get("dflash2_selected_token_id", raw.get("selected_token_id")) is None:
            raise ValueError(f"Selection row at {path}:{line_number} missing selected token")
        result.append(dict(raw))
    return result


def read_acceptance_records(path: str | Path, *, run_id: str) -> list[dict[str, Any]]:
    """Normalize benchmark outputs and trace rows to one record per block.

    DFlash's public benchmark writes ``acceptance_lengths`` including the
    target fallback token.  The analyzer subtracts exactly one fallback token
    and stores the result as ``accepted_draft_len``.
    """

    path = Path(path)
    result: list[dict[str, Any]] = []
    trace_rows = read_trace_jsonl(path)
    for index, row in enumerate(trace_rows):
        if row.get("status") != "ok":
            continue
        if "candidate_token_ids" in row:
            if not any(existing["sample_id"] == row["sample_id"] and existing["round_index"] == row["round_index"] for existing in result):
                result.append({
                    "run_id": run_id,
                    "sample_id": row["sample_id"],
                    "document_id": row["document_id"],
                    "dataset": row["dataset"],
                    "round_index": row["round_index"],
                    "accepted_draft_len": row.get("accepted_draft_len"),
                    "context_length": row["context_length"],
                })
            continue
        # This branch is reached only for files that use the legacy shape but
        # happened to pass through the v1 reader's error-safe path.
        acceptance = row.get("acceptance_lengths")
        if isinstance(acceptance, list):
            for round_index, committed in enumerate(acceptance):
                result.append({
                    "run_id": run_id,
                    "sample_id": str(row.get("id", index)),
                    "document_id": str(row.get("id", index)),
                    "dataset": str(row.get("dataset", "canonical")),
                    "round_index": round_index,
                    "accepted_draft_len": max(int(committed) - 1, 0),
                    "context_length": int(row.get("input_tokens", row.get("context_length", 0))),
                })
    if result:
        return result

    # Legacy rows cannot satisfy the candidate trace contract, so read them
    # separately without attempting to normalize them as token-level rows.
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        acceptance = raw.get("acceptance_lengths")
        if raw.get("accepted_draft_len") is not None:
            result.append({
                "run_id": run_id,
                "sample_id": str(raw.get("sample_id", raw.get("id", index))),
                "document_id": str(raw.get("document_id", raw.get("sample_id", raw.get("id", index)))),
                "dataset": str(raw.get("dataset", "canonical")),
                "round_index": int(raw.get("round_index", 0)),
                "accepted_draft_len": int(raw["accepted_draft_len"]),
                "context_length": int(raw.get("input_tokens", raw.get("context_length", 0))),
            })
            continue
        if not isinstance(acceptance, list):
            continue
        for round_index, committed in enumerate(acceptance):
            result.append({
                "run_id": run_id,
                "sample_id": str(raw.get("sample_id", raw.get("id", index))),
                "document_id": str(raw.get("sample_id", raw.get("id", index))),
                "dataset": str(raw.get("dataset", "canonical")),
                "round_index": round_index,
                "accepted_draft_len": max(int(committed) - 1, 0),
                "context_length": int(raw.get("input_tokens", raw.get("context_length", 0))),
            })
    return result


def _selection_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["run_id"]),
        str(row["sample_id"]),
        int(row["round_index"]),
        int(row["draft_position"]),
    )


def join_selection_trace(
    base_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join DFlash2 selected IDs to DFlash rows using an exact composite key."""

    lookup: dict[tuple[str, str, int, int], int] = {}
    for row in selection_rows:
        key = _selection_key(row)
        if key in lookup:
            raise ValueError(f"duplicate DFlash2 selection key: {key}")
        selected = row.get("dflash2_selected_token_id", row.get("selected_token_id"))
        if selected is None:
            raise ValueError(f"selection row missing selected token: {key}")
        lookup[key] = int(selected)
    result: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int, int]] = []
    for row in base_rows:
        key = _selection_key(row)
        if key not in lookup:
            missing.append(key)
            continue
        item = dict(row)
        item["dflash2_selected_token_id"] = lookup[key]
        result.append(item)
    if missing:
        raise ValueError(f"missing DFlash2 selection keys: {missing[:3]}")
    return result


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], child))
        return result
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value, ensure_ascii=False, sort_keys=True)}
    return {prefix: value}


def write_metrics_bundle(
    output_dir: str | Path,
    metrics: Mapping[str, Any],
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    report: str | None = None,
) -> dict[str, str]:
    """Write JSON, flat metrics CSV, optional tables and Markdown report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(dict(metrics), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flat = _flatten(metrics)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(flat))
        writer.writeheader()
        writer.writerow(flat)
    paths = {"metrics": str(metrics_path), "metrics_csv": str(output_dir / "metrics.csv")}
    for name, rows in (tables or {}).items():
        rows = list(rows)
        path = output_dir / f"{name}.csv"
        columns = sorted({str(key) for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns or ["status"])
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column) for column in writer.fieldnames})
        paths[name] = str(path)
    if report is not None:
        report_path = output_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        paths["report"] = str(report_path)
    return paths
