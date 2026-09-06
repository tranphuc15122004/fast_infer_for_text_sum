from __future__ import annotations

import pytest

from src.analyze.dflash_residual.prefix_alignment import (
    analyze_alignment,
    analyze_alignment_utility,
    row_alignment,
)


def _row(position: int, draft_logits: list[float], target_logits: list[float], *, target: int = 10) -> dict:
    candidates = [10, 11, 12, 13]
    return {
        "status": "ok",
        "run_id": "r",
        "sample_id": "s",
        "document_id": "d",
        "dataset": "cnn_dm",
        "task_regime": "cnn_dm",
        "context_length": 1024,
        "context_cap": 1024,
        "round_index": 0,
        "draft_position": position,
        "target_token_id": target,
        "candidate_token_ids": candidates,
        "candidate_logits": draft_logits,
        "target_candidate_logits": target_logits,
        "dflash_selected_token_id": candidates[0],
        "accepted_draft_len": 1,
    }


def test_row_alignment_detects_reverse_order_and_target_membership() -> None:
    aligned = row_alignment(_row(1, [4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]))
    reversed_row = row_alignment(_row(1, [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]))
    assert aligned is not None and reversed_row is not None
    assert aligned["kendall_tau"] == pytest.approx(1.0)
    assert aligned["pairwise_inversion_rate"] == pytest.approx(0.0)
    assert reversed_row["kendall_tau"] == pytest.approx(-1.0)
    assert reversed_row["pairwise_inversion_rate"] == pytest.approx(1.0)
    assert reversed_row["target_in_lattice"] is True


def test_e11_excludes_rows_without_target_logits() -> None:
    row = _row(1, [4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0])
    missing = dict(row)
    missing.pop("target_candidate_logits")
    result = analyze_alignment([row, missing])
    assert result["rows_with_target_logits"] == 1
    assert result["datasets"]["cnn_dm"]["documents"] == 1


def test_e12_reports_early_prefix_hazard_and_alignment_correlations() -> None:
    rows = [
        _row(1, [4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]),
        _row(2, [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]),
    ]
    result = analyze_alignment_utility(rows, max_prefix=2)
    dataset = result["datasets"]["cnn_dm"]
    assert dataset["blocks"] == 1
    assert "1" in dataset["dflash_first_rejection_hazard"]
    assert "2" in dataset["utility_loss_by_position"]
    assert dataset["alignment_vs_mat_d"]["n"] == 1
