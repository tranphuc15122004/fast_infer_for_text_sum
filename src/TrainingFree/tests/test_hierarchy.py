from __future__ import annotations

import pytest
import torch

from src.TrainingFree.hierarchy import (
    build_source_hierarchy,
    cluster_log_bounds,
    coverage_radius,
    select_representatives,
)


def test_farthest_point_representatives_are_deterministic_and_cover_points() -> None:
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]], dtype=torch.float32
    )

    first = select_representatives(points, 2)
    second = select_representatives(points, 2)

    assert first == second
    assert len(first) == 2
    assert coverage_radius(points, first) >= 0.0
    assert coverage_radius(points, list(range(len(points)))) == pytest.approx(0.0)


def test_cluster_upper_log_partition_dominates_exact_partition() -> None:
    keys = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [-1.0, 0.0]], dtype=torch.float32
    )
    hierarchy = build_source_hierarchy(
        keys, region_size=4, block_size=2, reps_per_block=1, reps_per_region=1
    )
    query = torch.tensor([1.0, 0.25], dtype=torch.float32)

    for block in hierarchy.blocks:
        upper, lower = cluster_log_bounds(query, keys, block)
        exact = torch.logsumexp(
            keys[block.start : block.end] @ query / (keys.shape[-1] ** 0.5), dim=0
        ).item()
        assert upper + 1e-6 >= exact
        assert lower <= exact + 1e-6


def test_hierarchy_tracks_regions_blocks_and_index_overhead() -> None:
    keys = torch.randn(10, 4)

    hierarchy = build_source_hierarchy(
        keys, region_size=6, block_size=4, reps_per_block=2, reps_per_region=2
    )

    assert len(hierarchy.regions) == 2
    assert len(hierarchy.blocks) == 3
    assert hierarchy.source_tokens == 10
    assert hierarchy.total_representatives > 0
    assert hierarchy.index_overhead > 0.0
    assert hierarchy.index_overhead < 1.0
