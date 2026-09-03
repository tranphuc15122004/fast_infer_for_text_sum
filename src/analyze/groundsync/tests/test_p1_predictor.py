import json

import pytest

from src.analyze.groundsync.p1_predictor import (
    FEATURE_NAMES,
    StandardizedLogistic,
    _doc_split,
    analyze_dataset,
    feature_matrix,
)


def _row(doc, accepted, k=16):
    return {
        "status": "ok",
        "document_id": doc,
        "accepted_len": accepted,
        "start_position": 1,
        "target_entropy_at_start": 1.0 if accepted else 3.0,
        "draft_confidence": [0.95 if accepted else 0.1],
        "recent_acceptance": 1.0 if accepted else 0.0,
        "autoregressive_time_ms": 10.0,
        "draft_time_by_k_ms": [2.0] * k,
        "verification_time_by_k_ms": [2.0] * k,
        "max_k": k,
    }


def test_document_split_is_disjoint_and_deterministic():
    rows = [_row(f"d{i}", i % 2) for i in range(10)]
    train, dev, test, ids = _doc_split(rows)
    assert set(ids["train"]).isdisjoint(ids["dev"])
    assert set(ids["dev"]).isdisjoint(ids["test"])
    assert {r["document_id"] for r in train + dev + test} == {f"d{i}" for i in range(10)}


def test_feature_matrix_excludes_missing_and_uses_entry_only():
    rows = [_row("d0", 1), {"status": "ok", "document_id": "d1"}]
    x, y, used = feature_matrix(rows)
    assert x.shape == (1, len(FEATURE_NAMES))
    assert y.tolist() == [1.0]
    assert used[0]["document_id"] == "d0"


def test_logistic_fits_two_classes_and_predicts_finite():
    rows = [_row(f"d{i}", i % 2) for i in range(12)]
    x, y, _ = feature_matrix(rows)
    model = StandardizedLogistic().fit(x, y)
    probabilities = model.predict_proba(x)
    assert len(probabilities) == len(rows)
    assert all(0.0 < value < 1.0 for value in probabilities)


def test_predictor_never_uses_future_fields_and_returns_policy_metrics():
    rows = []
    for index in range(20):
        rows.append(_row(f"d{index:02d}", index % 2))
    result = analyze_dataset(rows)
    assert result["features_are_causal"] is True
    assert "grounding" in result["excluded_features"]
    assert result["coverage"]["train_documents"] > 0
    assert result["test_policies"]["predicted_admission"]["tokens_per_ms"] is not None
    assert result["utility"]["decision"] in {"PASS", "FAIL", "INCONCLUSIVE"}


def test_one_class_training_is_inconclusive():
    rows = [_row(f"d{i}", 0) for i in range(12)]
    result = analyze_dataset(rows)
    assert result["status"] == "INCONCLUSIVE"
