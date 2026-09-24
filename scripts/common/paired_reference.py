"""Strict per-sample reference pairing for online benchmark speedups.

The reference pass is executed before an optimized baseline.  The baseline
then loads one reference row, validates that it describes the same workload,
and stores ESR/DSR alongside its raw timings.  Raw fields are intentionally
kept so a later audit can recompute every derived value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REFERENCE_TIMING_FIELDS = (
    "prefill_ms",
    "decode_ms",
    "e2e_ms",
)

REFERENCE_KEY_FIELDS = (
    "dataset",
    "sample_id",
    "prompt_hash",
    "run_config_hash",
)

SPEEDUP_SCOPE = "qwen3_4b_batch1_fixed_budget"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_hash(input_ids: Any) -> str:
    """Hash the exact token sequence used by a request.

    Accepts a Python sequence or a tensor-like object exposing ``detach`` and
    ``tolist``.  The helper deliberately hashes token IDs rather than decoded
    text, because whitespace/template differences must invalidate a pair.
    """

    value = input_ids
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("input_ids must be a sequence or tensor-like object")
    payload = [int(token) for token in value]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable hash for workload-defining configuration."""

    return hashlib.sha256(_canonical_json(dict(config)).encode("utf-8")).hexdigest()[:20]


def build_reference_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Build the strict join key shared by reference and method records."""

    missing = [field for field in REFERENCE_KEY_FIELDS if record.get(field) is None]
    if missing:
        raise ValueError(f"reference record is missing join fields: {', '.join(missing)}")
    return tuple(str(record[field]) for field in REFERENCE_KEY_FIELDS)  # type: ignore[return-value]


def write_reference_sidecar(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write sample reference rows and optional summary rows as JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def extract_reference_sidecar(source: Path, destination: Path) -> int:
    """Extract validated sample rows from a Vanilla output JSONL file."""

    rows: list[dict[str, Any]] = []
    with Path(source).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "summary" or record.get("scope") == "aggregate":
                continue
            build_reference_key(record)
            for field in REFERENCE_TIMING_FIELDS:
                if _positive_float(record.get(field)) is None:
                    raise ValueError(
                        f"reference row {source}:{line_number} has invalid {field}"
                    )
            if record.get("batch_size") != 1:
                raise ValueError(f"reference row {source}:{line_number} is not batch_size=1")
            rows.append(record)
    write_reference_sidecar(destination, rows)
    return len(rows)


def load_reference_sidecar(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Load only sample rows from a reference sidecar, rejecting duplicates."""

    path = Path(path)
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "summary" or record.get("scope") == "aggregate":
                continue
            key = build_reference_key(record)
            if key in result:
                raise ValueError(f"duplicate reference key at line {line_number}: {key}")
            result[key] = record
    return result


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _throughput(output_tokens: Any, decode_ms: Any) -> float | None:
    try:
        tokens = int(output_tokens)
        elapsed_ms = float(decode_ms)
    except (TypeError, ValueError):
        return None
    if tokens <= 0 or elapsed_ms <= 0:
        return None
    return tokens / (elapsed_ms / 1000.0)


def _pair_mismatch(
    record: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    expected_output_tokens: int | None,
) -> list[str]:
    mismatches: list[str] = []
    for field in ("dataset", "sample_id", "prompt_hash", "run_config_hash"):
        if str(record.get(field)) != str(reference.get(field)):
            mismatches.append(field)
    for field in (
        "target_model_revision",
        "tokenizer_revision",
        "tokenizer_name_or_path",
    ):
        reference_value = reference.get(field)
        if reference_value is not None and record.get(field) != reference_value:
            mismatches.append(field)
    if record.get("batch_size") != 1 or reference.get("batch_size") != 1:
        mismatches.append("batch_size")
    if expected_output_tokens is not None:
        expected = int(expected_output_tokens)
        if int(record.get("output_tokens") or 0) != expected:
            mismatches.append("output_tokens")
        if int(reference.get("output_tokens") or 0) != expected:
            mismatches.append("dense_output_tokens")
    return list(dict.fromkeys(mismatches))


def _visible_continuation(record: Mapping[str, Any], token_ids: list[int]) -> list[int]:
    count = record.get("quality_output_tokens")
    if isinstance(count, int) and 0 <= count <= len(token_ids):
        return token_ids[:count]
    eos_ids = record.get("eos_token_id")
    eos_set = (
        {int(value) for value in eos_ids}
        if isinstance(eos_ids, (list, tuple, set))
        else ({int(eos_ids)} if isinstance(eos_ids, int) else set())
    )
    eos_position = next(
        (index for index, token in enumerate(token_ids) if token in eos_set), None
    )
    return token_ids[:eos_position] if eos_position is not None else token_ids


def _token_parity_fields(
    record: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    method_ids = record.get("generated_token_ids")
    reference_ids = reference.get("generated_token_ids")
    fields = {
        "fixed_continuation_exact_match": None,
        "fixed_continuation_token_match_ratio": None,
        "fixed_continuation_first_divergence_index": None,
        "quality_prefix_exact_match": None,
        "quality_prefix_token_match_ratio": None,
        "output_parity_available": False,
    }
    if not isinstance(method_ids, list) or not isinstance(reference_ids, list):
        return fields
    method_tokens = [int(token) for token in method_ids]
    reference_tokens = [int(token) for token in reference_ids]

    def compare(left: list[int], right: list[int]) -> tuple[bool, float, int | None]:
        matches = sum(a == b for a, b in zip(left, right))
        denominator = max(len(left), len(right), 1)
        divergence = next(
            (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
            None,
        )
        if divergence is None and len(left) != len(right):
            divergence = min(len(left), len(right))
        return left == right, matches / denominator, divergence

    fixed_exact, fixed_ratio, fixed_divergence = compare(
        method_tokens, reference_tokens
    )
    quality_exact, quality_ratio, _ = compare(
        _visible_continuation(record, method_tokens),
        _visible_continuation(reference, reference_tokens),
    )
    fields.update(
        {
            "fixed_continuation_exact_match": fixed_exact,
            "fixed_continuation_token_match_ratio": fixed_ratio,
            "fixed_continuation_first_divergence_index": fixed_divergence,
            "quality_prefix_exact_match": quality_exact,
            "quality_prefix_token_match_ratio": quality_ratio,
            "output_parity_available": True,
        }
    )
    return fields


def attach_reference_timing(
    record: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    reference_baseline: str = "vanilla_hf",
    reference_run_id: str | None = None,
    expected_output_tokens: int | None = None,
    speedup_scope: str = SPEEDUP_SCOPE,
) -> dict[str, Any]:
    """Attach reference timing and online ESR/DSR to one method record.

    Invalid pairs retain the reference fields for diagnosis but receive
    ``speedup_valid=False`` and no derived speedup values.
    """

    result = dict(record)
    result["reference_baseline"] = reference_baseline
    result["reference_run_id"] = reference_run_id or reference.get("run_id")
    result["speedup_scope"] = speedup_scope
    result["dense_output_tokens"] = reference.get("output_tokens")
    result.update(_token_parity_fields(result, reference))
    for field in REFERENCE_TIMING_FIELDS:
        result[f"dense_{field}"] = reference.get(field)

    reasons = _pair_mismatch(
        result,
        reference,
        expected_output_tokens=expected_output_tokens,
    )
    method_e2e = _positive_float(result.get("e2e_ms"))
    method_decode = _positive_float(result.get("decode_ms"))
    dense_e2e = _positive_float(reference.get("e2e_ms"))
    dense_decode = _positive_float(reference.get("decode_ms"))
    if method_e2e is None or dense_e2e is None:
        reasons.append("e2e_ms")
    if method_decode is None or dense_decode is None:
        reasons.append("decode_ms")
    if result.get("status", "success") != "success":
        reasons.append("method_status")
    if reference.get("status", "success") != "success":
        reasons.append("reference_status")

    reasons = list(dict.fromkeys(reasons))
    result["speedup_valid"] = not reasons
    result["speedup_invalid_reason"] = ",".join(reasons) if reasons else None
    result["esr"] = dense_e2e / method_e2e if not reasons else None

    method_tok_s = _throughput(result.get("output_tokens"), method_decode)
    dense_tok_s = _throughput(reference.get("output_tokens"), dense_decode)
    result["method_decode_tok_s"] = method_tok_s
    result["dense_decode_tok_s"] = dense_tok_s
    result["dsr"] = method_tok_s / dense_tok_s if not reasons and method_tok_s and dense_tok_s else None
    result["prefill_speedup"] = (
        _positive_float(reference.get("prefill_ms"))
        / _positive_float(result.get("prefill_ms"))
        if not reasons
        and _positive_float(reference.get("prefill_ms"))
        and _positive_float(result.get("prefill_ms"))
        else None
    )
    return result
