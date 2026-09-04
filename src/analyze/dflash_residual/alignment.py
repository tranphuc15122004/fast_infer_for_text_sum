"""H0 official/custom acceptance alignment checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .metrics import survival


def _values(rows: Iterable[Mapping[str, Any]]) -> list[float]:
    return [float(row["accepted_draft_len"]) for row in rows if row.get("accepted_draft_len") is not None]


def _sample_keys(rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["sample_id"]), int(row["round_index"])) for row in rows}


def compare_acceptance(
    official_rows: Iterable[Mapping[str, Any]],
    custom_rows: Iterable[Mapping[str, Any]],
    *,
    min_blocks: int = 5,
    mat_tolerance: float = 0.15,
    official_metadata: Mapping[str, Any] | None = None,
    custom_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the H0 positive-acceptance and approximate-reproduction gate."""

    official = list(official_rows)
    custom = list(custom_rows)
    official_values = _values(official)
    custom_values = _values(custom)
    common = _sample_keys(official) & _sample_keys(custom)
    result: dict[str, Any] = {
        "status": "INCONCLUSIVE",
        "reason": None,
        "official_blocks": len(official_values),
        "custom_blocks": len(custom_values),
        "common_blocks": len(common),
        "mat_official": sum(official_values) / len(official_values) if official_values else None,
        "mat_custom": sum(custom_values) / len(custom_values) if custom_values else None,
        "survival_official": survival([int(value) for value in official_values]),
        "survival_custom": survival([int(value) for value in custom_values]),
        "mean_abs_delta": None,
        "protocol_check": {"status": "not_provided", "mismatches": {}},
    }
    if official_metadata is not None and custom_metadata is not None:
        fields = (
            "target_model",
            "draft_model",
            "tokenizer",
            "thinking_mode",
            "block_size",
            "native_block_size",
            "target_layer_ids",
        )
        mismatches = {
            field: {
                "official": official_metadata.get(field),
                "custom": custom_metadata.get(field),
            }
            for field in fields
            if official_metadata.get(field) is not None
            and custom_metadata.get(field) is not None
            and official_metadata.get(field) != custom_metadata.get(field)
        }
        result["protocol_check"] = {
            "status": "pass" if not mismatches else "fail",
            "mismatches": mismatches,
        }
        if mismatches:
            result["status"] = "FAIL"
            result["reason"] = "protocol_mismatch"
            return result
    if len(official_values) < min_blocks or len(custom_values) < min_blocks:
        result["reason"] = "insufficient_blocks"
        return result
    if not common:
        result["status"] = "FAIL"
        result["reason"] = "no_common_samples"
        return result
    official_by_key = {(str(row["sample_id"]), int(row["round_index"])): float(row["accepted_draft_len"]) for row in official if row.get("accepted_draft_len") is not None}
    custom_by_key = {(str(row["sample_id"]), int(row["round_index"])): float(row["accepted_draft_len"]) for row in custom if row.get("accepted_draft_len") is not None}
    deltas = [abs(official_by_key[key] - custom_by_key[key]) for key in sorted(common) if key in official_by_key and key in custom_by_key]
    result["mean_abs_delta"] = sum(deltas) / len(deltas) if deltas else None
    if result["mat_official"] <= 0 or result["mat_custom"] <= 0:
        result["status"] = "FAIL"
        result["reason"] = "non_positive_acceptance"
    elif result["mean_abs_delta"] is None or result["mean_abs_delta"] > mat_tolerance:
        result["status"] = "FAIL"
        result["reason"] = "runner_mismatch"
    else:
        result["status"] = "PASS"
        result["reason"] = "positive_and_aligned"
    return result
