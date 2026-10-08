"""Validation helpers for Context-Adaptive DFlash artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


REQUEST_REQUIRED = (
    "method", "dataset", "model", "input_tokens", "retained_tokens", "output_tokens",
    "batch_size", "selector_latency_ms", "ttft_ms", "tpot_ms", "e2e_ms",
    "throughput_tok_s", "qps", "peak_memory_gb", "avg_accept_length",
    "acceptance_rate", "draft_latency_ms", "verification_latency_ms", "rejected_draft_ratio",
)
ROUND_REQUIRED = (
    "schema_version", "type", "status", "run_id", "sample_id", "variant", "round_index",
    "logical_length_before", "processed_output_before", "pending_anchor_position", "parent_query_index",
    "parent_entropy", "source_concentration", "history_acceptance", "requested_budget",
    "selected_context_tokens_by_layer", "physical_context_tokens_by_layer", "gamma_requested",
    "gamma_executed", "block_size", "length_mode", "selector_id", "refresh_required",
    "action_reason", "accepted_candidates", "processed_commits", "logical_length_after",
    "all_candidates_accepted", "prefix_right_censored", "eos_offset", "boundary_round",
    "trimmed_commits", "signal_ms", "controller_ms", "selection_ms", "bank_update_ms",
    "draft_ms", "verify_ms", "round_gpu_span_ms", "round_host_ms", "gather_bytes",
    "cost_observation_ready", "predicted_cost_per_commit", "fallback_reason", "wasted_work_ms",
)


def _check_finite(value: Any, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path} is non-finite")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            errors.extend(_check_finite(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            errors.extend(_check_finite(item, f"{path}[{index}]"))
    return errors


def validate_request(record: Mapping[str, Any]) -> list[str]:
    errors = [f"missing request field: {key}" for key in REQUEST_REQUIRED if key not in record]
    errors.extend(_check_finite(record))
    if record.get("type") not in {"sample", "error"}:
        errors.append("request type must be sample or error")
    if record.get("status") == "ok":
        for field in ("input_tokens", "output_tokens", "e2e_ms"):
            value = record.get(field)
            if value is None or float(value) < 0:
                errors.append(f"successful request requires nonnegative {field}")
        if record.get("batch_size") != 1:
            errors.append("V1 request batch_size must equal 1")
        if record.get("output_accounting_valid") is False:
            errors.append("generation output accounting invariant failed")
        if record.get("generation_temperature") == 0 and not record.get("output_ids_hash"):
            errors.append("greedy request is missing output_ids_hash")
    return errors


def validate_round(record: Mapping[str, Any]) -> list[str]:
    errors = [f"missing round field: {key}" for key in ROUND_REQUIRED if key not in record]
    errors.extend(_check_finite(record))
    if record.get("type") != "round":
        errors.append("round type must equal 'round'")
    try:
        gamma = int(record["gamma_executed"])
        accepted = int(record["accepted_candidates"])
        block_size = int(record["block_size"])
        if gamma < 0 or accepted < 0 or accepted > gamma:
            errors.append("round acceptance must satisfy 0 <= accepted_candidates <= gamma_executed")
        if block_size != gamma + 1:
            errors.append("block_size must equal gamma_executed + 1")
        if "logical_length_after" in record:
            expected = int(record["logical_length_before"]) + int(record["processed_commits"]) - int(record["trimmed_commits"])
            if int(record["logical_length_after"]) != expected:
                errors.append("logical length accounting does not match committed tokens")
    except (KeyError, TypeError, ValueError):
        pass
    return errors


def validate_calibration(payload: Mapping[str, Any], expected_signature: Mapping[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "cadflash.calibration.v1":
        errors.append("unsupported calibration schema_version")
    cutpoints = payload.get("entropy_cutpoints")
    if not isinstance(cutpoints, list) or len(cutpoints) != 3:
        errors.append("entropy_cutpoints must contain three values")
    else:
        try:
            values = [float(value) for value in cutpoints]
            if any(not math.isfinite(value) for value in values) or values != sorted(values):
                errors.append("entropy_cutpoints must be finite and nondecreasing")
        except (TypeError, ValueError):
            errors.append("entropy_cutpoints must be numeric")
    for key, row in payload.get("prefix_priors", {}).items():
        survival = row.get("survival", [])
        try:
            values = [float(value) for value in survival]
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
                errors.append(f"invalid survival probabilities at {key}")
            if values != sorted(values, reverse=True):
                errors.append(f"survival probabilities must be nonincreasing at {key}")
        except (TypeError, ValueError):
            errors.append(f"invalid survival probabilities at {key}")
    for key, row in payload.get("cost_priors", {}).items():
        try:
            cost = float(row["action_cost_ms"])
            if not math.isfinite(cost) or cost <= 0:
                errors.append(f"invalid action cost at {key}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"missing or invalid action cost at {key}")
    errors.extend(_check_finite(payload))
    if expected_signature is not None:
        actual = payload.get("signature", {})
        differences = [key for key, value in expected_signature.items() if actual.get(key) != value]
        if differences:
            errors.append("calibration signature mismatch: " + ", ".join(sorted(differences)))
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows
