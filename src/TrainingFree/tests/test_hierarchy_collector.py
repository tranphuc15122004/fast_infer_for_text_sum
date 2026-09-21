from __future__ import annotations

import torch

from src.TrainingFree.hierarchy_collector import build_head_hierarchies


def test_build_head_hierarchies_slices_source_and_preserves_kv_heads() -> None:
    keys = torch.randn(1, 2, 9, 4)

    result = build_head_hierarchies(
        keys,
        source_start=1,
        source_end=8,
        region_size=4,
        block_size=2,
        reps_per_block=1,
        reps_per_region=1,
    )

    assert sorted(result) == [0, 1]
    assert all(item.source_tokens == 7 for item in result.values())
