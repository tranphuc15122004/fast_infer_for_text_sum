from __future__ import annotations

from src.analyze.dflash_residual.causal_screening_analysis import (
    _hit,
    _suffix_cmat,
    candidate_depth_sweep,
    reveal_oracle,
)
from src.analyze.dflash_residual.e22_rank_band import (
    BAND_SPECS,
    rank_band_repair_oracle,
    repaired_prefix_length,
)


def _row(position: int, target: int, rank: int, *, state: str = "s", reveal: int = 0) -> dict:
    return {
        "status": "ok",
        "run_id": "r",
        "sample_id": f"{state}-r{reveal}",
        "document_id": "d",
        "dataset": "multi_news",
        "task_regime": "multi_news",
        "context_length": 10,
        "round_index": 0,
        "draft_position": position,
        "max_depth": 3,
        "target_token_id": target,
        "candidate_token_ids": [target] if rank == 1 else [99, target],
        "candidate_logits": [2.0, 1.0][: 1 if rank == 1 else 2],
        "dflash_selected_token_id": target if rank == 1 else 99,
        "target_token_source": "verifier_posterior",
        "fixed_state_id": state,
        "state_mode": "on_policy",
        "reveal_count": reveal,
        "draft_target_rank": rank,
    }


def test_full_rank_controls_top_k_hit() -> None:
    row = _row(1, 7, 17)
    assert not _hit(row, 16)
    assert _hit(row, 32)


def test_suffix_cmat_counts_only_contiguous_suffix() -> None:
    block = [_row(1, 1, 1), _row(2, 2, 1), _row(3, 3, 17)]
    assert _suffix_cmat(block, 1, 16) == 2.0
    assert _suffix_cmat(block, 2, 16) == 1.0


def test_candidate_depth_sweep_reports_rank_gain(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    rows = [_row(1, 1, 1), _row(2, 2, 17), _row(3, 3, 33)]
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")
    result = candidate_depth_sweep({"multi_news": path}, ks=(16, 32, 64, 128), max_position=3)
    item = result["datasets"]["multi_news"]
    assert item["by_k"]["16"]["mat_o_k"] < item["by_k"]["32"]["mat_o_k"]
    assert item["by_k"]["32"]["mat_o_k"] < item["by_k"]["64"]["mat_o_k"]


def test_reveal_oracle_uses_same_state_baseline(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    rows = []
    for reveal in (0, 1):
        ranks = (17, 1, 17) if reveal == 0 else (1, 1, 1)
        rows.extend(_row(pos, pos, rank, reveal=reveal) for pos, rank in enumerate(ranks, 1))
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")
    result = reveal_oracle({"multi_news": path}, reveal_counts=(0, 1))
    item = result["datasets"]["multi_news"]["conditions"]["1"]
    assert item["conditional_cmat_o16"] > item["same_state_r0_baseline_cmat_o16"]
    assert item["relative_gain_vs_same_state_r0"] > 0


def test_repaired_prefix_only_changes_the_selected_rank_band() -> None:
    block = [
        _row(1, 1, 1),
        _row(2, 2, 17),
        _row(3, 3, 33),
    ]
    assert repaired_prefix_length(block, None) == 1
    assert repaired_prefix_length(block, BAND_SPECS["17-32"]) == 2
    assert repaired_prefix_length(block, BAND_SPECS["33-64"]) == 1
    assert repaired_prefix_length(block, BAND_SPECS["2-16"]) == 1


def test_rank_band_oracle_reports_non_additive_prefix_gain(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    rows = [
        _row(1, 1, 1),
        _row(2, 2, 17),
        _row(3, 3, 17),
    ]
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")
    result = rank_band_repair_oracle(
        {"multi_news": path},
        bootstrap_samples=25,
    )
    item = result["datasets"]["multi_news"]
    assert item["status"] == "ok"
    assert item["mat_d"] == 1.0
    assert item["bands"]["17-32"]["mat_repaired"] == 3.0
    assert item["bands"]["17-32"]["relative_gain_vs_mat_d"] == 2.0
    assert item["bands"]["33-64"]["mat_repaired"] == 1.0
    assert result["gate"]["decision"] == "FAIL"


def test_rank_band_oracle_rejects_missing_rank_rows(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    row = _row(1, 1, 1)
    row.pop("draft_target_rank")
    path.write_text(__import__("json").dumps(row) + "\n")
    result = rank_band_repair_oracle({"multi_news": path})
    item = result["datasets"]["multi_news"]
    assert item["status"] == "inconclusive"
    assert item["reason"] == "missing_full_vocabulary_rank"
