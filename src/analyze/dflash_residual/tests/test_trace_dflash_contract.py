from __future__ import annotations

import pytest
import torch

from src.analyze.dflash_residual.trace_dflash import (
    acceptance_length_from_tokens,
    build_position_rows,
    parse_context_caps,
    truncate_input_ids,
    build_parser,
)


def test_parse_context_caps_is_sorted_unique_and_rejects_non_positive() -> None:
    assert parse_context_caps("4096,1024,4096,2048") == [1024, 2048, 4096]
    with pytest.raises(ValueError, match="positive"):
        parse_context_caps("0,1024")


def test_truncate_input_ids_preserves_requested_side() -> None:
    ids = torch.arange(10).view(1, -1)
    assert truncate_input_ids(ids, 4, side="right").tolist() == [[0, 1, 2, 3]]
    assert truncate_input_ids(ids, 4, side="left").tolist() == [[6, 7, 8, 9]]
    with pytest.raises(ValueError, match="side"):
        truncate_input_ids(ids, 4, side="middle")


def test_acceptance_length_counts_only_consecutive_draft_tokens() -> None:
    proposed = torch.tensor([[4, 5, 6, 7]])
    posterior = torch.tensor([[4, 5, 99, 7, 100]])
    assert acceptance_length_from_tokens(proposed, posterior) == 2


def test_build_position_rows_records_rank_and_never_changes_candidate_ids() -> None:
    rows = build_position_rows(
        run_id="r1",
        sample_id="s1",
        document_id="d1",
        dataset="gov_report",
        context_length=4096,
        round_index=2,
        candidates=torch.tensor([[[10, 11, 12], [20, 21, 22]]]),
        candidate_logits=torch.tensor([[[3.0, 2.0, 1.0], [4.0, 3.0, 2.0]]]),
        target_tokens=torch.tensor([[11, 99]]),
        dflash_selected=torch.tensor([[10, 20]]),
        accepted_draft_len=1,
        block_size=3,
        native_block_size=3,
    )
    assert rows[0]["draft_position"] == 1
    assert rows[0]["target_rank"] == 2
    assert rows[1]["target_in_top16"] is False
    assert rows[0]["candidate_token_ids"] == [10, 11, 12]


def test_collector_parser_is_model_lazy_and_exposes_context_sweep() -> None:
    args = build_parser().parse_args([
        "--target-model", "target",
        "--draft-model", "draft",
        "--input", "data.jsonl",
        "--output", "out.jsonl",
        "--context-lengths", "1024,2048",
    ])
    assert args.context_lengths == "1024,2048"
