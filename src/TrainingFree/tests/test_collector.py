from __future__ import annotations

import torch

from src.TrainingFree.collector import (
    build_source_index,
    segment_token_indices,
)


def test_source_index_pools_blocks_and_multiple_prototypes() -> None:
    hidden = torch.tensor(
        [[[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 3.0]]]
    )

    blocks = build_source_index(
        hidden, source_start=0, source_end=4, block_size=2, num_prototypes=2
    )

    assert blocks[0]["embedding"] == [2.0, 0.0]
    assert blocks[0]["prototypes"] == [[1.0, 0.0], [3.0, 0.0]]
    assert blocks[1]["start"] == 2


def test_segment_boundaries_prefer_punctuation_and_have_fixed_fallback() -> None:
    tokens = ["A", " report", ".", " Next", " part", " final"]

    segments = segment_token_indices(tokens, max_tokens=3)

    assert segments == [(0, 3), (3, 6)]
