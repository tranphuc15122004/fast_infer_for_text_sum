"""Trace contract for the DFlash residual-headroom experiments.

The analyzer intentionally accepts JSON-safe mappings only.  Model-backed
collectors are responsible for producing this contract; the statistical code
must remain runnable on the CPU development host.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA_VERSION = "dflash_residual.trace.v1"
CONTEXT_BUCKETS = ("0-2k", "2-4k", "4-8k", "8-16k", "16-32k", ">32k")
REQUIRED_FIELDS = (
    "run_id",
    "sample_id",
    "document_id",
    "dataset",
    "context_length",
    "round_index",
    "draft_position",
    "max_depth",
    "target_token_id",
    "candidate_token_ids",
    "dflash_selected_token_id",
    "target_token_source",
)
VALID_TARGET_SOURCES = frozenset(("verifier_posterior", "canonical_continuation"))

_DATASET_TO_REGIME = {
    "canonical": "canonical",
    "gsm8k": "canonical",
    "math500": "canonical",
    "humaneval": "canonical",
    "mbpp": "canonical",
    "cnn_dailymail": "cnn_dm",
    "cnn_dm": "cnn_dm",
    "gov_report": "govreport",
    "govreport": "govreport",
    "multi_news": "multi_news",
    "multinews": "multi_news",
}


def context_bin(context_length: int) -> str:
    """Map a token count to a stable human-readable bucket."""

    value = int(context_length)
    if value < 0:
        raise ValueError("context_length must be non-negative")
    if value < 2048:
        return CONTEXT_BUCKETS[0]
    if value < 4096:
        return CONTEXT_BUCKETS[1]
    if value < 8192:
        return CONTEXT_BUCKETS[2]
    if value < 16384:
        return CONTEXT_BUCKETS[3]
    if value < 32768:
        return CONTEXT_BUCKETS[4]
    return CONTEXT_BUCKETS[5]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_trace_row(row: Mapping[str, Any]) -> list[str]:
    """Return deterministic validation messages for one trace row."""

    problems: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    problems.extend(f"missing field: {field}" for field in missing)
    if row.get("status", "ok") not in {"ok", "error"}:
        problems.append("status must be 'ok' or 'error'")
    if row.get("status", "ok") == "error":
        return problems
    for field in ("context_length", "round_index", "draft_position", "max_depth", "target_token_id"):
        if field in row and not _is_int(row[field]):
            problems.append(f"{field} must be an integer")
    if _is_int(row.get("context_length")) and row["context_length"] < 0:
        problems.append("context_length must be non-negative")
    if _is_int(row.get("round_index")) and row["round_index"] < 0:
        problems.append("round_index must be non-negative")
    if _is_int(row.get("draft_position")) and row["draft_position"] < 1:
        problems.append("draft_position must be 1-based")
    if _is_int(row.get("max_depth")) and row["max_depth"] < 1:
        problems.append("max_depth must be positive")
    candidates = row.get("candidate_token_ids")
    if not isinstance(candidates, list) or not candidates:
        problems.append("candidate_token_ids must be a non-empty list")
    elif any(not _is_int(token) for token in candidates):
        problems.append("candidate_token_ids must contain integers")
    if "accepted_draft_len" in row and row["accepted_draft_len"] is not None:
        if not _is_int(row["accepted_draft_len"]) or row["accepted_draft_len"] < 0:
            problems.append("accepted_draft_len must be a non-negative integer or null")
    source = row.get("target_token_source")
    if source not in VALID_TARGET_SOURCES:
        problems.append("target_token_source is unsupported")
    if "candidate_logits" in row and row["candidate_logits"] is not None:
        logits = row["candidate_logits"]
        if not isinstance(logits, list) or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in logits
        ):
            problems.append("candidate_logits must be a finite numeric list or null")
    return problems


def _regime(row: Mapping[str, Any]) -> str:
    explicit = row.get("task_regime")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    dataset = str(row.get("dataset", "other")).lower()
    return _DATASET_TO_REGIME.get(dataset, "other")


def task_regime_for_dataset(dataset: str) -> str:
    """Return the canonical workload regime name for a dataset label."""

    return _DATASET_TO_REGIME.get(str(dataset).lower(), str(dataset))


def normalize_trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one successful trace row.

    Aliases from early collector prototypes are accepted, but the returned
    mapping always uses the v1 field names.  This function never injects the
    target token into a candidate list.
    """

    source = dict(row)
    if source.get("status", "ok") == "error":
        return {
            **source,
            "schema_version": str(source.get("schema_version", SCHEMA_VERSION)),
            "status": "error",
        }
    aliases = {
        "candidate_ids": "candidate_token_ids",
        "target_token": "target_token_id",
        "dflash_token_id": "dflash_selected_token_id",
        "dflash2_token_id": "dflash2_selected_token_id",
    }
    for old, new in aliases.items():
        if new not in source and old in source:
            source[new] = source[old]
    source.setdefault("schema_version", SCHEMA_VERSION)
    source.setdefault("status", "ok")
    source["run_id"] = str(source.get("run_id", "run"))
    source["sample_id"] = str(source.get("sample_id"))
    source["document_id"] = str(source.get("document_id", source["sample_id"]))
    source["dataset"] = str(source.get("dataset", "other"))
    source["task_regime"] = _regime(source)
    for field in ("context_length", "round_index", "draft_position", "max_depth", "target_token_id"):
        if field in source:
            source[field] = int(source[field])
    source["candidate_token_ids"] = [int(token) for token in source["candidate_token_ids"]]
    source["dflash_selected_token_id"] = int(source["dflash_selected_token_id"])
    if source.get("dflash2_selected_token_id") is not None:
        source["dflash2_selected_token_id"] = int(source["dflash2_selected_token_id"])
    if source.get("accepted_draft_len") is not None:
        source["accepted_draft_len"] = int(source["accepted_draft_len"])
    source["context_bin"] = str(source.get("context_bin") or context_bin(source["context_length"]))
    problems = validate_trace_row(source)
    if problems:
        raise ValueError("Invalid DFlash residual trace row: " + "; ".join(problems))
    return source
