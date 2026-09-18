from __future__ import annotations

import pytest

from src.TrainingFree.schema import validate_lease_trace_record


def _record() -> dict:
    return {
        "schema_version": "recap.lease.trace.v1",
        "status": "ok",
        "sample_id": "sample-1",
        "dataset": "gov_report",
        "source_block_count": 2,
        "steps": [
            {
                "step": 0,
                "anchor_step": 0,
                "drift": 0.0,
                "bound": 0.08,
                "actual_cold_mass": 0.03,
                "slack": 0.05,
                "valid": True,
                "expired": False,
                "cold_fraction": 0.5,
                "full_source_score": True,
            }
        ],
    }


def test_validate_lease_trace_round_trip() -> None:
    result = validate_lease_trace_record(_record())

    assert result["schema_version"] == "recap.lease.trace.v1"
    assert result["steps"][0]["valid"] is True


def test_validate_lease_trace_rejects_non_finite_values() -> None:
    row = _record()
    row["steps"][0]["bound"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        validate_lease_trace_record(row)
