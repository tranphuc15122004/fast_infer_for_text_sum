from __future__ import annotations

import torch

from src.TrainingFree.hierarchy import build_source_hierarchy
from src.TrainingFree.hierarchy_routing import evaluate_routing_step, route_query


def test_routing_returns_active_blocks_and_audits_missed_attention() -> None:
    keys = torch.tensor(
        [
            [5.0, 0.0],
            [4.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.5],
            [-2.0, 0.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    hierarchy = build_source_hierarchy(
        keys, region_size=4, block_size=2, reps_per_block=1, reps_per_region=1
    )
    query = torch.tensor([1.0, 0.0], dtype=torch.float32)

    decision = route_query(hierarchy, query, mass_budget=0.20)
    metrics = evaluate_routing_step(
        hierarchy,
        query,
        decision,
        attention_block_masses=[0.60, 0.25, 0.10, 0.05],
    )

    assert decision.active_block_indices
    assert set(decision.active_block_indices).issubset(range(len(hierarchy.blocks)))
    assert 0.0 <= metrics["missed_attention_mass"] <= 1.0
    assert metrics["upper_bound_violations"] == 0
    assert 0.0 < metrics["exact_expansion_fraction"] <= 1.0


def test_routing_with_zero_budget_materializes_every_block() -> None:
    keys = torch.randn(9, 3)
    hierarchy = build_source_hierarchy(
        keys, region_size=5, block_size=3, reps_per_block=1, reps_per_region=1
    )
    decision = route_query(hierarchy, torch.ones(3), mass_budget=0.0)

    assert decision.active_block_indices == tuple(range(len(hierarchy.blocks)))
    assert decision.upper_missed_mass_bound == 0.0


def test_evaluation_rejects_wrong_attention_width() -> None:
    keys = torch.randn(4, 2)
    hierarchy = build_source_hierarchy(
        keys, region_size=4, block_size=2, reps_per_block=1, reps_per_region=1
    )
    decision = route_query(hierarchy, torch.ones(2), mass_budget=0.1)

    try:
        evaluate_routing_step(hierarchy, torch.ones(2), decision, [1.0])
    except ValueError as exc:
        assert "attention_block_masses" in str(exc)
    else:
        raise AssertionError("wrong attention width must be rejected")
