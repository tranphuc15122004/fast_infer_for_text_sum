from __future__ import annotations

from src.analyze.dflash_residual.prefix_probe import run_probe_suite, split_documents


def _rows(documents: int = 6) -> list[dict]:
    rows = []
    for document in range(documents):
        for position in range(1, 5):
            target = 100 + position
            candidates = [target, 200 + position, 300 + position, 400 + position]
            rows.append({
                "status": "ok",
                "run_id": "probe",
                "sample_id": str(document),
                "document_id": str(document),
                "task_regime": "toy",
                "dataset": "toy",
                "context_length": 32,
                "context_cap": 1024,
                "round_index": 0,
                "draft_position": position,
                "target_token_id": target,
                "candidate_token_ids": candidates,
                "candidate_logits": [4.0, 3.0, 2.0, 1.0],
                "target_candidate_logits": [4.0, 3.0, 2.0, 1.0],
                "accepted_draft_len": 0,
            })
    return rows


def test_split_documents_is_disjoint():
    train, test, metadata = split_documents(_rows(), test_fraction=0.33, seed=42)
    assert {row["document_id"] for row in train}.isdisjoint({row["document_id"] for row in test})
    assert metadata["train_documents"]
    assert metadata["test_documents"]


def test_probe_suite_reports_all_objectives():
    result = run_probe_suite(_rows(), epochs=2, learning_rate=1e-2, device="cpu")
    assert result["status"] == "ok"
    assert set(result["regimes"]["toy"]["objectives"]) == {
        "pointwise", "pairwise", "listwise", "prefix_utility"
    }
    assert result["regimes"]["toy"]["train_documents"] > 0
    assert result["regimes"]["toy"]["test_documents"] > 0
