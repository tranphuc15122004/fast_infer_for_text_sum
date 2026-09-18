from __future__ import annotations

import pytest

from src.TrainingFree.schema import TRACE_SCHEMA_VERSION, validate_trace_record


def _record() -> dict:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "status": "ok",
        "sample_id": "s1",
        "dataset": "gov_report",
        "source_blocks": [
            {
                "index": 0,
                "start": 0,
                "end": 2,
                "embedding": [1.0, 0.0],
                "prototypes": [[1.0, 0.0]],
            },
            {
                "index": 1,
                "start": 2,
                "end": 4,
                "embedding": [0.0, 1.0],
                "prototypes": [[0.0, 1.0]],
            },
        ],
        "segments": [
            {
                "index": 0,
                "attention": [0.8, 0.2],
                "query_vectors": [[1.0, 0.0]],
                "segment_vector": [1.0, 0.0],
                "token_count": 2,
            },
            {
                "index": 1,
                "attention": [0.2, 0.8],
                "query_vectors": [[0.0, 1.0]],
                "segment_vector": [0.0, 1.0],
                "token_count": 2,
            },
        ],
    }


def test_valid_trace_is_normalized() -> None:
    record = validate_trace_record(_record())
    assert record["schema_version"] == TRACE_SCHEMA_VERSION
    assert len(record["source_blocks"]) == 2


def test_schema_rejects_attention_width_and_non_finite_values() -> None:
    record = _record()
    record["segments"][0]["attention"] = [1.0]
    with pytest.raises(ValueError, match="attention"):
        validate_trace_record(record)

    record = _record()
    record["segments"][0]["segment_vector"] = [float("nan"), 0.0]
    with pytest.raises(ValueError, match="finite"):
        validate_trace_record(record)


def test_schema_rejects_missing_future_features() -> None:
    record = _record()
    del record["segments"][0]["query_vectors"]
    with pytest.raises(ValueError, match="query_vectors"):
        validate_trace_record(record)
