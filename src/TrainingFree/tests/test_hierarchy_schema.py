from __future__ import annotations

import pytest

from src.TrainingFree.schema import validate_hierarchy_trace_record


def test_hierarchy_schema_accepts_compact_trace() -> None:
    record = {
        "schema_version": "recap.hierarchy.trace.v1",
        "status": "ok",
        "sample_id": "x",
        "dataset": "gov_report",
        "source_tokens": 10,
        "steps": [
            {
                "step": 0,
                "missed_attention_mass": 0.0,
                "max_missed_attention_mass": 0.0,
                "exact_expansion_fraction": 0.2,
                "upper_bound_violations": 0,
                "index_overhead": 0.1,
                "routing_fraction": 0.1,
                "active_qk_tokens": 2,
                "full_qk_tokens": 10,
                "routing_representatives": 1,
                "routing_time_ms": 1.0,
            }
        ],
    }

    result = validate_hierarchy_trace_record(record)

    assert result["sample_id"] == "x"


def test_hierarchy_schema_rejects_non_finite_step_metric() -> None:
    record = {
        "schema_version": "recap.hierarchy.trace.v1",
        "status": "ok",
        "sample_id": "x",
        "dataset": "gov_report",
        "source_tokens": 10,
        "steps": [{"step": 0, "missed_attention_mass": float("nan")}],
    }

    with pytest.raises(ValueError, match="finite"):
        validate_hierarchy_trace_record(record)


def test_hierarchy_schema_validates_optional_head_concentration_rows() -> None:
    record = {
        "schema_version": "recap.hierarchy.trace.v1",
        "status": "ok",
        "sample_id": "x",
        "dataset": "gov_report",
        "source_tokens": 10,
        "steps": [
            {
                "step": 0,
                "missed_attention_mass": 0.0,
                "max_missed_attention_mass": 0.0,
                "exact_expansion_fraction": 0.2,
                "upper_bound_violations": 0,
                "index_overhead": 0.1,
                "routing_fraction": 0.1,
                "active_qk_tokens": 2,
                "full_qk_tokens": 10,
                "routing_representatives": 1,
                "routing_time_ms": 1.0,
                "head_concentration": [
                    {
                        "layer": 27,
                        "head": 0,
                        "source_tokens": 10,
                        "source_mass": 0.9,
                        "k90": 2,
                        "k90_fraction": 0.2,
                        "k95": 3,
                        "k95_fraction": 0.3,
                        "k99": 5,
                        "k99_fraction": 0.5,
                    }
                ],
            }
        ],
    }

    result = validate_hierarchy_trace_record(record)

    assert result["steps"][0]["head_concentration"][0]["k95"] == 3


def test_hierarchy_schema_rejects_invalid_head_concentration_row() -> None:
    record = {
        "schema_version": "recap.hierarchy.trace.v1",
        "status": "ok",
        "sample_id": "x",
        "dataset": "gov_report",
        "source_tokens": 10,
        "steps": [
            {
                "step": 0,
                "missed_attention_mass": 0.0,
                "max_missed_attention_mass": 0.0,
                "exact_expansion_fraction": 0.2,
                "upper_bound_violations": 0,
                "index_overhead": 0.1,
                "routing_fraction": 0.1,
                "active_qk_tokens": 2,
                "full_qk_tokens": 10,
                "routing_representatives": 1,
                "routing_time_ms": 1.0,
                "head_concentration": [{"layer": 27, "head": 0, "k95": 11}],
            }
        ],
    }

    with pytest.raises(ValueError, match="head concentration"):
        validate_hierarchy_trace_record(record)
