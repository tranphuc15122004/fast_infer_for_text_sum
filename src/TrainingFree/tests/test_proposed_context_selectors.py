from __future__ import annotations

import os
import pytest
import torch

from TrainingFree.context_adaptive.config import AdaptiveConfig
from TrainingFree.context_adaptive.signals import (
    target_multi_query_scores,
    target_duo_ensemble_scores,
)
from TrainingFree.context_adaptive.selection import rank_chunks, select_context
from TrainingFree.context_adaptive.types import PromptLayout, RoundState
from TrainingFree.context_interventions.attention import (
    multi_query_full_key_attention_scores,
    duo_ensemble_full_key_attention_scores,
    MultiLayerTargetQueryCapture,
)
from TrainingFree.context_interventions.masks import build_online_selection_policy


class MockDynamicCache:
    """Mock DynamicCache for unit testing attention score calculation."""

    def __init__(self, key_tensors_by_layer: dict[int, torch.Tensor]) -> None:
        self.layers = []
        max_layer = max(key_tensors_by_layer.keys()) if key_tensors_by_layer else -1
        for i in range(max_layer + 1):
            if i in key_tensors_by_layer:
                key = key_tensors_by_layer[i]
                layer_obj = type("CacheLayer", (), {"keys": key, "values": key})()
                self.layers.append(layer_obj)
            else:
                self.layers.append(type("CacheLayer", (), {"keys": None, "values": None})())


def _simple_layout() -> PromptLayout:
    # 10 prompt tokens: [global 0], source chunk 1 [1, 2, 3, 4], source chunk 2 [5, 6, 7, 8], [query 9]
    return PromptLayout(
        prompt_tokens=10,
        source_positions=(1, 2, 3, 4, 5, 6, 7, 8),
        global_positions=(0, 9),
        source_chunks=((1, 2, 3, 4), (5, 6, 7, 8)),
        prompt_hash="hash123",
        source_available=True,
    )


# =========================================================================
# 1. Config & Validation Tests
# =========================================================================

def test_adaptive_config_proposed_selectors_registered() -> None:
    cfg1 = AdaptiveConfig(selector="multi_query_max_pooling", query_pooling="max")
    assert cfg1.selector == "multi_query_max_pooling"
    assert cfg1.query_pooling == "max"

    cfg2 = AdaptiveConfig(
        selector="duo_ensemble",
        ensemble_alpha=0.6,
        duo_deep_layer=35,
        query_pooling="mean",
    )
    assert cfg2.selector == "duo_ensemble"
    assert cfg2.ensemble_alpha == 0.6
    assert cfg2.duo_deep_layer == 35
    assert cfg2.query_pooling == "mean"


def test_adaptive_config_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError, match="ensemble_alpha must be in"):
        AdaptiveConfig(selector="duo_ensemble", ensemble_alpha=1.5)

    with pytest.raises(ValueError, match="ensemble_alpha must be in"):
        AdaptiveConfig(selector="duo_ensemble", ensemble_alpha=-0.1)

    with pytest.raises(ValueError, match="query_pooling must be"):
        AdaptiveConfig(selector="multi_query_max_pooling", query_pooling="invalid_mode")


def test_adaptive_config_from_env() -> None:
    os.environ["CAD_SELECTOR"] = "duo_ensemble"
    os.environ["CAD_ENSEMBLE_ALPHA"] = "0.75"
    os.environ["CAD_DUO_DEEP_LAYER"] = "32"
    os.environ["CAD_QUERY_POOLING"] = "max"
    try:
        cfg = AdaptiveConfig.from_env()
        assert cfg.selector == "duo_ensemble"
        assert cfg.ensemble_alpha == 0.75
        assert cfg.duo_deep_layer == 32
        assert cfg.query_pooling == "max"
    finally:
        os.environ.pop("CAD_SELECTOR", None)
        os.environ.pop("CAD_ENSEMBLE_ALPHA", None)
        os.environ.pop("CAD_DUO_DEEP_LAYER", None)
        os.environ.pop("CAD_QUERY_POOLING", None)


# =========================================================================
# 2. Multi-Query Attention & Pooling Tests
# =========================================================================

def test_multi_query_full_key_attention_causality_and_pooling() -> None:
    # 2 query tokens, 1 head, head_dim=2
    # key has 4 positions: 0, 1, 2, 3
    # Q0 is at pos 1, Q1 is at pos 2
    # Q0 should only attend to keys [0, 1]; Q1 should attend to keys [0, 1, 2]
    # Key at pos 3 must receive 0 attention from both.
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # [1, 1, 2, 2]
    k = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [100.0, 100.0]]]])  # [1, 1, 4, 2]

    # Max pooling
    scores_max = multi_query_full_key_attention_scores(
        q,
        k,
        query_positions=[1, 2],
        head_groups=1,
        scaling=1.0,
        pooling_mode="max",
    )
    assert 3 not in scores_max  # Pos 3 is in future relative to queries 1 and 2
    assert 0 in scores_max and 1 in scores_max and 2 in scores_max

    # Mean pooling
    scores_mean = multi_query_full_key_attention_scores(
        q,
        k,
        query_positions=[1, 2],
        head_groups=1,
        scaling=1.0,
        pooling_mode="mean",
    )
    assert 3 not in scores_mean
    assert 0 in scores_mean and 1 in scores_mean and 2 in scores_mean


def test_target_multi_query_scores_chunk_aggregation() -> None:
    layout = _simple_layout()
    # 2 queries, 2 heads, head_dim=2
    q = torch.randn(1, 2, 2, 2)
    # 10 keys (positions 0..9)
    k = torch.randn(1, 2, 10, 2)
    cache = MockDynamicCache({18: k})

    scores, concentration = target_multi_query_scores(
        q,
        cache,
        layer_index=18,
        query_positions=[8, 9],
        source_chunks=layout.source_chunks,
        head_groups=1,
        scaling=1.0,
        pooling_mode="max",
    )

    # Scores must be computed for all source tokens
    for pos in layout.source_positions:
        assert pos in scores
        assert scores[pos] >= 0.0

    # Total score over source chunks should be positive and bounded by 1.0
    total_score = sum(scores[pos] for pos in layout.source_positions)
    assert 0.0 < total_score <= 1.0

    # Concentration should be a valid float between 0 and 1
    assert concentration is not None
    assert 0.0 <= concentration <= 1.0


# =========================================================================
# 3. Duo-Ensemble Attention Tests
# =========================================================================

def test_duo_ensemble_alpha_blending() -> None:
    # Set up key tensors for mid (18) and deep (35)
    # Key 18 strongly favors pos 1; Key 35 strongly favors pos 2
    k_mid = torch.tensor([[[[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]]]])
    k_deep = torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]]]])

    q_mid = torch.tensor([[[[1.0, 0.0]]]])
    q_deep = torch.tensor([[[[1.0, 0.0]]]])

    # alpha = 1.0 -> 100% mid layer
    scores_mid_only = duo_ensemble_full_key_attention_scores(
        q_mid,
        k_mid,
        q_deep,
        k_deep,
        query_positions=2,
        head_groups=1,
        scaling=1.0,
        alpha=1.0,
    )
    assert scores_mid_only[1] > scores_mid_only[2]

    # alpha = 0.0 -> 100% deep layer
    scores_deep_only = duo_ensemble_full_key_attention_scores(
        q_mid,
        k_mid,
        q_deep,
        k_deep,
        query_positions=2,
        head_groups=1,
        scaling=1.0,
        alpha=0.0,
    )
    assert scores_deep_only[2] > scores_deep_only[1]

    # alpha = 0.5 -> 50/50 blend
    scores_blend = duo_ensemble_full_key_attention_scores(
        q_mid,
        k_mid,
        q_deep,
        k_deep,
        query_positions=2,
        head_groups=1,
        scaling=1.0,
        alpha=0.5,
    )
    assert pytest.approx(scores_blend[1], rel=1e-2) == pytest.approx(scores_blend[2], rel=1e-2)


def test_target_duo_ensemble_scores_layer_validation() -> None:
    layout = _simple_layout()
    q = torch.randn(1, 2, 2, 2)
    k = torch.randn(1, 2, 10, 2)
    cache = MockDynamicCache({18: k, 35: k})

    # Missing layer 35 in query map should raise ValueError
    with pytest.raises(ValueError, match="Duo-Ensemble requires queries"):
        target_duo_ensemble_scores(
            {18: q},
            cache,
            layer_mid=18,
            layer_deep=35,
            query_positions=9,
            source_chunks=layout.source_chunks,
            head_groups=1,
            scaling=1.0,
        )

    # Valid invocation with both layers
    scores, concentration = target_duo_ensemble_scores(
        {18: q, 35: q},
        cache,
        layer_mid=18,
        layer_deep=35,
        query_positions=9,
        source_chunks=layout.source_chunks,
        head_groups=1,
        scaling=1.0,
        alpha=0.5,
    )
    assert len(scores) == len(layout.source_positions)
    assert concentration is not None


# =========================================================================
# 4. Selection & Ranking Integration Tests
# =========================================================================

def test_rank_chunks_proposed_selectors() -> None:
    layout = _simple_layout()
    # Chunk 0 has tokens [1, 2, 3, 4]; Chunk 1 has tokens [5, 6, 7, 8]
    # Set chunk 1 to have higher attention mass than chunk 0
    scores = {pos: (0.1 if pos in (1, 2, 3, 4) else 0.9) for pos in range(1, 9)}

    for selector_name in ("multi_query_max_pooling", "duo_ensemble"):
        rankings = rank_chunks(
            layout,
            selector=selector_name,
            scores_by_layer=[scores],
            num_draft_layers=2,
        )
        assert len(rankings) == 2
        # Chunk 1 (tokens 5..8) has higher mass, so chunk 1 should be ranked first
        assert rankings[0][0] == 1
        assert rankings[0][1] == 0


def test_select_context_proposed_selectors() -> None:
    layout = _simple_layout()
    # 2 draft layers
    rankings = ((1, 0), (1, 0))
    state = RoundState(
        round_index=1,
        logical_length=12,
        processed_output_tokens=2,
        remaining_output_tokens=10,
        parent_entropy=0.5,
        source_concentration=0.8,
        history_acceptance=0.7,
        ranking_age=0,
        refresh_required=False,
    )

    for selector_name in ("multi_query_max_pooling", "duo_ensemble"):
        selection = select_context(
            state=state,
            layout=layout,
            rankings=rankings,
            budget=8,
            processed_output_start=10,
            bank_length=12,
            source_anchors=1,
            recent_output=1,
            selector_id=selector_name,
        )
        assert selection is not None
        assert len(selection.positions_by_layer) == 2
        for positions in selection.positions_by_layer:
            assert len(positions) == 8
            # Must include global anchor (0)
            assert 0 in positions
            # Must include top-ranked chunk (chunk 1: 5, 6, 7, 8)
            assert {5, 6, 7, 8}.issubset(positions)


def test_build_online_selection_policy_proposed_selectors() -> None:
    layout = _simple_layout()
    scores = {pos: 0.1 for pos in range(12)}

    for selector_name in ("multi_query_max_pooling", "duo_ensemble"):
        policy = build_online_selection_policy(
            selector=selector_name,
            budget=5,
            source_anchors=1,
            recent_output=1,
        )
        selection = policy(
            state=None,
            layout=layout,
            bank_length=12,
            processed_output_start=10,
            requested_budget=5,
            round_index=0,
            refresh_required=False,
            target_parent_scores=scores,
            previous_dflash_scores={},
            num_draft_layers=2,
        )
        assert len(selection.positions_by_layer) == 2
        for positions in selection.positions_by_layer:
            assert len(positions) == 5
            assert 0 in positions


# =========================================================================
# 5. Generation Signal Dispatcher & MultiLayer Capture Lifecycle Tests
# =========================================================================

def test_capture_target_signals_dispatcher() -> None:
    from TrainingFree.context_adaptive.generation import _capture_target_signals

    layout = _simple_layout()
    # 2 heads, head_dim=4, 10 keys
    k_mid = torch.randn(1, 2, 10, 4)
    k_deep = torch.randn(1, 2, 10, 4)
    cache = MockDynamicCache({18: k_mid, 35: k_deep})
    attn_mock = type(
        "MockAttn",
        (),
        {"num_key_value_groups": 1, "scaling": 0.5, "sliding_window": None},
    )()
    # 3 queries (accepted=2)
    q = torch.randn(1, 2, 3, 4)

    # 1. Multi-query max pooling
    scores_multi, conc_multi = _capture_target_signals(
        q,
        cache,
        layout,
        attn_mock,
        selector="multi_query_max_pooling",
        target_layer_mid=18,
        target_layer_deep=35,
        accepted=2,
        logical_length=9,
        pooling_mode="max",
    )
    assert scores_multi is not None
    assert conc_multi is not None
    for pos in layout.source_positions:
        assert pos in scores_multi

    # 2. Duo ensemble
    scores_duo, conc_duo = _capture_target_signals(
        {18: q, 35: q},
        cache,
        layout,
        attn_mock,
        selector="duo_ensemble",
        target_layer_mid=18,
        target_layer_deep=35,
        accepted=2,
        logical_length=9,
        alpha=0.6,
        pooling_mode="max",
    )
    assert scores_duo is not None
    assert conc_duo is not None
    for pos in layout.source_positions:
        assert pos in scores_duo

    # 3. Backwards-compatible target_parent
    scores_parent, conc_parent = _capture_target_signals(
        q,
        cache,
        layout,
        attn_mock,
        selector="target_parent",
        target_layer_mid=18,
        target_layer_deep=35,
        accepted=2,
        logical_length=9,
    )
    assert scores_parent is not None
    assert conc_parent is not None


def test_multi_layer_target_query_capture_lifecycle() -> None:
    class MockAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.q_norm = torch.nn.Identity()
            self.head_dim = 4

    class MockLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = MockAttention()

    class MockTargetModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = [MockLayer() for _ in range(36)]
            self.model = type("Model", (), {"layers": torch.nn.ModuleList(layers)})()

    target = MockTargetModel()
    capture = MultiLayerTargetQueryCapture(target, [18, 35])
    assert 18 in capture.captures and 35 in capture.captures

    # Enable capture for all queries
    capture.enable(last_only=False)

    # Simulate forward pass on layers 18 and 35
    for layer_idx in (18, 35):
        attn = target.model.layers[layer_idx].self_attn
        dummy_input = torch.randn(1, 2, 8)
        pos_emb = (torch.ones(1, 2, 4), torch.zeros(1, 2, 4))
        # Trigger pre-hook with position_embeddings
        for hook in attn._forward_pre_hooks.values():
            hook(attn, (dummy_input,), {"position_embeddings": pos_emb})
        # Trigger q_proj forward
        _ = attn.q_proj(dummy_input)

    queries = capture.disable()
    assert 18 in queries
    assert 35 in queries
    # Shape should be [batch=1, heads=2, seq=2, head_dim=4]
    assert queries[18].shape == (1, 2, 2, 4)
    assert queries[35].shape == (1, 2, 2, 4)

    # Teardown
    capture.close()


def test_signal_dispatch_preserves_each_query_causal_position():
    from TrainingFree.context_adaptive.generation import _capture_target_signals

    layout = PromptLayout(1, (0,), (), ((0,),), "causal", True)
    query = torch.tensor([[[[1.], [-1.]]]])
    cache = MockDynamicCache({18: torch.tensor([[[[0.], [0.], [20.]]]])})
    attention = type("Attention", (), {"num_key_value_groups": 1, "scaling": 1., "sliding_window": None})()
    scores, _ = _capture_target_signals(
        query, cache, layout, attention, selector="multi_query_max_pooling",
        target_layer_mid=18, target_layer_deep=35, accepted=1, logical_length=1,
    )
    assert scores[0] == pytest.approx(.5, abs=1e-6)

