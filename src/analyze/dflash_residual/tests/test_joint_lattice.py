from __future__ import annotations

from src.analyze.dflash_residual.joint_lattice import (
    analyze_decomposition,
    entropy_standardized_stats,
    lattice_stats,
    marginal_joint_decomposition,
)


def _row(doc: str, position: int, hit: bool, entropy: float = 1.0) -> dict:
    target = 10 + position
    candidates = [target if hit else 999, 20 + position, 30 + position, 40 + position]
    return {
        "status": "ok", "run_id": "r", "sample_id": doc, "document_id": doc,
        "task_regime": "toy", "dataset": "toy", "context_length": 8,
        "context_cap": 1024, "round_index": 0, "draft_position": position,
        "target_token_id": target, "candidate_token_ids": candidates,
        "candidate_logits": [4.0, 3.0, 2.0, 1.0],
        "accepted_draft_len": 0, "target_entropy": entropy,
    }


def test_lattice_stats_distinguishes_marginal_and_joint():
    rows = []
    for doc in ("a", "b"):
        rows.extend([_row(doc, 1, True), _row(doc, 2, doc == "a")])
    stats = lattice_stats(rows, max_position=2)
    assert stats["marginal_recall"] == {"1": 1.0, "2": 0.5}
    assert stats["joint_survival"] == {"1": 1.0, "2": 0.5}
    assert stats["mat_o16"] == 1.5


def test_decomposition_conserves_total_change():
    canonical = lattice_stats([_row("a", 1, True), _row("a", 2, True)], max_position=2)
    summary = lattice_stats([_row("b", 1, True), _row("b", 2, False)], max_position=2)
    result = marginal_joint_decomposition(canonical, summary)
    assert abs(result["total_degradation"] - result["marginal_component"] - result["joint_component"]) < 1e-9


def test_entropy_standardization_is_deterministic():
    current = [_row("s1", 1, True, 1.0), _row("s1", 2, False, 1.0)]
    reference = [_row("c1", 1, True, 1.0), _row("c1", 2, True, 1.0)]
    result = entropy_standardized_stats(current, reference, bins=2, max_position=2)
    assert result["status"] == "ok"
    assert result["mat_entropy_standardized"] <= result["mat_reference_entropy_standardized"]


def test_analyze_decomposition_groups_canonical_and_summary():
    rows = [{**_row("c", 1, True), "task_regime": "canonical"}, {**_row("c", 2, True), "task_regime": "canonical"}]
    rows += [{**_row("s", 1, True), "task_regime": "cnn_dm"}, {**_row("s", 2, False), "task_regime": "cnn_dm"}]
    result = analyze_decomposition(rows, max_position=2)
    assert result["status"] == "ok"
    assert "cnn_dm" in result["decomposition"]
