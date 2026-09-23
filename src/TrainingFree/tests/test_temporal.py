import torch

from src.TrainingFree.temporal import (
    TemporalAnalyzer,
    TemporalConfig,
    evaluate_supports,
    gqa_union_supports,
    select_adaptive_supports,
    select_top_supports,
)


def test_top_support_and_gqa_union_measure_current_source_mass() -> None:
    attention = torch.tensor(
        [
            [0.70, 0.20, 0.05, 0.05],
            [0.15, 0.05, 0.70, 0.10],
            [0.60, 0.10, 0.20, 0.10],
            [0.20, 0.10, 0.10, 0.60],
        ],
        dtype=torch.float32,
    )
    supports = select_top_supports(attention, fractions=(0.50,))
    assert supports[0.5][0] == {0, 1}
    unions = gqa_union_supports(supports[0.5], query_to_kv=(0, 0, 1, 1))
    assert unions == {0: {0, 1, 2}, 1: {0, 2, 3}}
    metrics = evaluate_supports(
        attention, supports[0.5], query_to_kv=(0, 0, 1, 1)
    )
    assert metrics["query_expansion_fraction"] == 0.5
    assert metrics["gqa_expansion_fraction"] == 0.75
    assert 0.0 < metrics["gqa_missed_mass"] < 0.1


def test_adaptive_support_uses_k95_count_and_alpha() -> None:
    attention = torch.tensor([[0.50, 0.30, 0.15, 0.05]], dtype=torch.float32)
    supports = select_adaptive_supports(attention, alpha=1.0, mass_level=0.95)
    assert supports[0] == {0, 1, 2}
    expanded = select_adaptive_supports(attention, alpha=1.5, mass_level=0.95)
    assert expanded[0] == {0, 1, 2, 3}


def test_recurrent_analyzer_uses_previous_refresh_support_only() -> None:
    config = TemporalConfig(
        lags=(1,),
        budgets=(0.5,),
        alphas=(1.0,),
        refresh_intervals=(2,),
        block_sizes=(2,),
    )
    analyzer = TemporalAnalyzer(
        query_to_kv=(0,),
        source_tokens=4,
        config=config,
    )
    first = analyzer.observe(torch.tensor([[0.90, 0.05, 0.03, 0.02]]))
    second = analyzer.observe(torch.tensor([[0.02, 0.03, 0.05, 0.90]]))
    assert first["recurrent"]["R2_B2_A1"]["refresh"] is True
    assert second["recurrent"]["R2_B2_A1"]["refresh"] is False
    assert second["recurrent"]["R2_B2_A1"]["mean_missed_mass"] > 0.8
    assert second["lag"]["L1_B0.5"]["mean_missed_mass"] > 0.8


def test_recurrent_analyzer_also_reports_fixed_budget_block_cache() -> None:
    config = TemporalConfig(
        lags=(1,),
        budgets=(0.5,),
        alphas=(1.0,),
        refresh_intervals=(2,),
        block_sizes=(2,),
    )
    analyzer = TemporalAnalyzer(
        query_to_kv=(0,),
        source_tokens=4,
        config=config,
    )

    first = analyzer.observe(torch.tensor([[0.90, 0.05, 0.03, 0.02]]))
    second = analyzer.observe(torch.tensor([[0.02, 0.03, 0.05, 0.90]]))

    assert first["recurrent"]["R2_C2_F0.5"]["refresh"] is True
    assert second["recurrent"]["R2_C2_F0.5"]["refresh"] is False
    assert second["recurrent"]["R2_C2_F0.5"]["source_cost_ratio"] == 0.5
