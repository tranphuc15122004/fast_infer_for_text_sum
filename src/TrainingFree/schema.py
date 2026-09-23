"""Trace schema and finite-value validation for RECAP-KV V0."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping


TRACE_SCHEMA_VERSION = "recap.trace.v1"
LEASE_TRACE_SCHEMA_VERSION = "recap.lease.trace.v1"
HIERARCHY_TRACE_SCHEMA_VERSION = "recap.hierarchy.trace.v1"
TEMPORAL_TRACE_SCHEMA_VERSION = "recap.e43.temporal.trace.v1"


def _vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty vector")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def validate_trace_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached trace record.

    Error rows are allowed to carry only identifying metadata so a long Modal
    run can report per-sample failures without turning them into successes.
    """

    result = copy.deepcopy(dict(record))
    if result.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError("invalid trace schema_version")
    if result.get("status") != "ok":
        if not result.get("sample_id") or not result.get("error"):
            raise ValueError("error trace rows require sample_id and error")
        return result
    if not result.get("sample_id"):
        raise ValueError("trace requires sample_id")
    blocks = result.get("source_blocks")
    segments = result.get("segments")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("source_blocks must be a non-empty list")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments must be a non-empty list")

    dimensions: int | None = None
    for expected_index, block in enumerate(blocks):
        if not isinstance(block, Mapping) or int(block.get("index", -1)) != expected_index:
            raise ValueError("source block indices must be contiguous")
        embedding = _vector(block.get("embedding"), f"source_blocks[{expected_index}].embedding")
        prototypes = block.get("prototypes")
        if not isinstance(prototypes, list) or not prototypes:
            raise ValueError("each source block needs prototypes")
        prototype_vectors = [
            _vector(value, f"source_blocks[{expected_index}].prototypes")
            for value in prototypes
        ]
        dimensions = dimensions or len(embedding)
        if len(embedding) != dimensions or any(len(value) != dimensions for value in prototype_vectors):
            raise ValueError("source representations must have one dimension")

    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError("each segment must be an object")
        attention = segment.get("attention")
        if not isinstance(attention, list) or len(attention) != len(blocks):
            raise ValueError("segment attention width must match source blocks")
        attention_values = [float(value) for value in attention]
        if any(not math.isfinite(value) or value < 0.0 for value in attention_values):
            raise ValueError("attention must be finite and non-negative")
        if sum(attention_values) <= 0.0:
            raise ValueError("attention must have positive mass")
        query_vectors = segment.get("query_vectors")
        if not isinstance(query_vectors, list) or not query_vectors:
            raise ValueError(f"segment {index} requires query_vectors")
        for query in query_vectors:
            if len(_vector(query, f"segment {index}.query_vectors")) != dimensions:
                raise ValueError("query vector dimension mismatch")
        if len(_vector(segment.get("segment_vector"), f"segment {index}.segment_vector")) != dimensions:
            raise ValueError("segment vector dimension mismatch")
    return result


def validate_lease_trace_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate compact V2 query-drift lease events."""

    result = copy.deepcopy(dict(record))
    if result.get("schema_version") != LEASE_TRACE_SCHEMA_VERSION:
        raise ValueError("invalid lease trace schema_version")
    if result.get("status") != "ok":
        if not result.get("sample_id") or not result.get("error"):
            raise ValueError("error lease rows require sample_id and error")
        return result
    if not result.get("sample_id"):
        raise ValueError("lease trace requires sample_id")
    try:
        source_blocks = int(result.get("source_block_count", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("source_block_count must be positive") from exc
    if source_blocks <= 0:
        raise ValueError("source_block_count must be positive")
    steps = result.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("lease trace steps must be a non-empty list")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError("each lease step must be an object")
        for key in (
            "step",
            "anchor_step",
            "drift",
            "bound",
            "actual_cold_mass",
            "slack",
            "cold_fraction",
        ):
            value = step.get(key)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"lease step {index} field {key} must be finite") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"lease step {index} field {key} must be finite")
        if float(step["drift"]) < 0.0:
            raise ValueError("lease drift must be non-negative")
        for key in ("bound", "actual_cold_mass", "cold_fraction"):
            if not 0.0 <= float(step[key]) <= 1.0:
                raise ValueError(f"lease {key} must be in [0, 1]")
        if int(step["step"]) < int(step["anchor_step"]):
            raise ValueError("lease step must not precede anchor_step")
        for key in ("valid", "expired", "full_source_score"):
            if not isinstance(step.get(key), bool):
                raise ValueError(f"lease step {key} must be boolean")
    return result


def validate_hierarchy_trace_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate compact V3 hierarchy routing/audit events."""

    result = copy.deepcopy(dict(record))
    if result.get("schema_version") != HIERARCHY_TRACE_SCHEMA_VERSION:
        raise ValueError("invalid hierarchy trace schema_version")
    if result.get("status") != "ok":
        if not result.get("sample_id") or not result.get("error"):
            raise ValueError("error hierarchy rows require sample_id and error")
        return result
    if not result.get("sample_id"):
        raise ValueError("hierarchy trace requires sample_id")
    try:
        source_tokens = int(result.get("source_tokens", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("source_tokens must be positive") from exc
    if source_tokens <= 0:
        raise ValueError("source_tokens must be positive")
    steps = result.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("hierarchy trace steps must be a non-empty list")
    finite_fields = (
        "missed_attention_mass",
        "max_missed_attention_mass",
        "exact_expansion_fraction",
        "index_overhead",
        "routing_fraction",
        "routing_time_ms",
    )
    integer_fields = (
        "step",
        "upper_bound_violations",
        "active_qk_tokens",
        "full_qk_tokens",
        "routing_representatives",
    )
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError("each hierarchy step must be an object")
        for key in finite_fields:
            try:
                value = float(step[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"hierarchy step {index} field {key} must be finite") from exc
            if not math.isfinite(value):
                raise ValueError(f"hierarchy step {index} field {key} must be finite")
        for key in integer_fields:
            try:
                value = int(step[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"hierarchy step {index} field {key} must be integer") from exc
            if value < 0:
                raise ValueError(f"hierarchy step {index} field {key} must be non-negative")
        for key in ("missed_attention_mass", "max_missed_attention_mass", "exact_expansion_fraction"):
            if not 0.0 <= float(step[key]) <= 1.0:
                raise ValueError(f"hierarchy {key} must be in [0, 1]")
        if int(step["active_qk_tokens"]) > int(step["full_qk_tokens"]):
            raise ValueError("active_qk_tokens must not exceed full_qk_tokens")
        temporal = step.get("temporal")
        if temporal is not None:
            if not isinstance(temporal, Mapping):
                raise ValueError("temporal metrics must be an object")
            for group_name in ("gqa_oracle", "lag", "adaptive_lag", "recurrent"):
                group = temporal.get(group_name, {})
                if not isinstance(group, Mapping):
                    raise ValueError(f"temporal {group_name} must be an object")
                for config_name, metrics in group.items():
                    if not isinstance(config_name, str) or not isinstance(metrics, Mapping):
                        raise ValueError("temporal metric rows must be named objects")
                    for key in (
                        "mean_missed_mass",
                        "p99_missed_mass",
                        "gqa_missed_mass",
                        "gqa_p99_missed_mass",
                        "source_cost_ratio",
                        "working_source_cost_ratio",
                    ):
                        if key not in metrics:
                            continue
                        try:
                            metric = float(metrics[key])
                        except (TypeError, ValueError) as exc:
                            raise ValueError(f"temporal metric {key} must be finite") from exc
                        if not math.isfinite(metric):
                            raise ValueError(f"temporal metric {key} must be finite")
                    if "refresh" in metrics and not isinstance(metrics["refresh"], bool):
                        raise ValueError("temporal refresh must be boolean")
        head_rows = step.get("head_concentration")
        if head_rows is not None:
            if not isinstance(head_rows, list) or not head_rows:
                raise ValueError("head concentration must be a non-empty list")
            for row in head_rows:
                if not isinstance(row, Mapping):
                    raise ValueError("head concentration rows must be objects")
                try:
                    source_width = int(row["source_tokens"])
                    layer = int(row["layer"])
                    head = int(row["head"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("head concentration row is invalid") from exc
                if source_width <= 0 or layer < 0 or head < 0:
                    raise ValueError("head concentration row indices are invalid")
                try:
                    source_mass = float(row["source_mass"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("head concentration source mass is invalid") from exc
                if not math.isfinite(source_mass) or source_mass <= 0.0:
                    raise ValueError("head concentration source mass is invalid")
                for level in (90, 95, 99):
                    try:
                        count = int(row[f"k{level}"])
                        fraction = float(row[f"k{level}_fraction"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError("head concentration row is incomplete") from exc
                    if not 1 <= count <= source_width:
                        raise ValueError("head concentration k-value is invalid")
                    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                        raise ValueError("head concentration fraction is invalid")
    return result


def validate_temporal_trace_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate compact E43 temporal-support audit traces."""

    result = copy.deepcopy(dict(record))
    if result.get("schema_version") != TEMPORAL_TRACE_SCHEMA_VERSION:
        raise ValueError("invalid temporal trace schema_version")
    if result.get("status") != "ok":
        if not result.get("sample_id") or not result.get("error"):
            raise ValueError("error temporal rows require sample_id and error")
        return result
    if not result.get("sample_id"):
        raise ValueError("temporal trace requires sample_id")
    try:
        source_tokens = int(result.get("source_tokens", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("temporal source_tokens must be positive") from exc
    if source_tokens <= 0:
        raise ValueError("temporal source_tokens must be positive")
    steps = result.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("temporal trace steps must be a non-empty list")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError("each temporal step must be an object")
        try:
            if int(step["step"]) < 0:
                raise ValueError("step must be non-negative")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"temporal step {index} index is invalid") from exc
        temporal = step.get("temporal", step)
        if not isinstance(temporal, Mapping):
            raise ValueError("temporal metrics must be an object")
        for group_name in ("gqa_oracle", "lag", "adaptive_lag", "recurrent"):
            group = temporal.get(group_name, {})
            if not isinstance(group, Mapping):
                raise ValueError(f"temporal {group_name} must be an object")
            for config_name, metrics in group.items():
                if not isinstance(config_name, str) or not isinstance(metrics, Mapping):
                    raise ValueError("temporal metric rows must be named objects")
                for key, value in metrics.items():
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        continue
                    raise ValueError(f"temporal metric {key} must be finite")
    return result
