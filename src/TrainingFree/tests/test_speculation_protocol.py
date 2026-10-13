from __future__ import annotations

import pytest

from TrainingFree.speculation_experiments.protocol import (
    CampaignConfig,
    assign_group_folds,
    state_identity,
)


def test_campaign_defaults_use_supported_dflash_candidate_counts() -> None:
    config = CampaignConfig()

    assert config.gammas == (3, 7, 11, 15)
    assert config.draft_actions == ("gamma3", "gamma7", "gamma11", "gamma15")


@pytest.mark.parametrize("gammas", [(3, 7, 7), (0, 3), (3, 16), ()])
def test_campaign_rejects_invalid_or_duplicate_draft_shapes(gammas: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="gamma"):
        CampaignConfig(gammas=gammas)


def test_group_folds_are_deterministic_balanced_and_keep_source_rows_together() -> None:
    rows = [
        {"source_group_id": f"doc-{index}", "dataset": "gov_report", "round_index": round_index}
        for index in range(12)
        for round_index in range(3)
    ]

    folds = assign_group_folds(rows, n_splits=4, seed=42)

    assert folds == assign_group_folds(rows, n_splits=4, seed=42)
    assert set(folds.values()) == {0, 1, 2, 3}
    assert [sum(fold == index for fold in folds.values()) for index in range(4)] == [3, 3, 3, 3]
    assert len(folds) == 12


def test_duplicate_source_group_across_datasets_gets_one_fold() -> None:
    rows = [
        {"source_group_id": "shared-document", "dataset": "gov_report"},
        {"source_group_id": "shared-document", "dataset": "multi_news"},
        *[
            {"source_group_id": f"doc-{index}", "dataset": "gov_report"}
            for index in range(1, 8)
        ],
    ]

    folds = assign_group_folds(rows, n_splits=4, seed=19)

    assert folds["shared-document"] in range(4)
    assert len(folds) == 8


def test_group_folds_reject_missing_ids_and_too_few_documents() -> None:
    with pytest.raises(ValueError, match="source_group_id"):
        assign_group_folds([{"dataset": "gov_report"}], n_splits=2)
    with pytest.raises(ValueError, match="groups"):
        assign_group_folds(
            [{"source_group_id": "one", "dataset": "gov_report"}],
            n_splits=2,
        )


def test_state_identity_pairs_actions_at_the_same_prefix_without_gamma() -> None:
    identity = state_identity(
        dataset="gov_report",
        source_group_id="doc-1",
        prompt_token_ids=(11, 12, 13),
        output_prefix_ids=(21, 22),
        pending_anchor=23,
        anchor_position=5,
        round_index=2,
    )

    assert len(identity) == 64
    assert identity == state_identity(
        dataset="gov_report",
        source_group_id="doc-1",
        prompt_token_ids=(11, 12, 13),
        output_prefix_ids=(21, 22),
        pending_anchor=23,
        anchor_position=5,
        round_index=2,
    )
    assert identity != state_identity(
        dataset="gov_report",
        source_group_id="doc-1",
        prompt_token_ids=(11, 12, 13),
        output_prefix_ids=(21, 22, 24),
        pending_anchor=23,
        anchor_position=6,
        round_index=2,
    )


def test_state_identity_rejects_inconsistent_processed_prefix_position() -> None:
    with pytest.raises(ValueError, match="prompt plus processed output"):
        state_identity(
            dataset="gov_report",
            source_group_id="doc-1",
            prompt_token_ids=(11, 12, 13),
            output_prefix_ids=(21, 22),
            pending_anchor=23,
            anchor_position=4,
            round_index=2,
        )
