from __future__ import annotations

import pytest

from src.analyze.groundsync.target_kv_e1 import (
    MemoryBlockProbe,
    anchor_positions,
    pool_sequence_to_interface,
    pool_representation_dict,
    split_feature_rows_by_document,
    probe_metrics,
    wrong_document_indices,
    required_capture_layers,
)


def test_pool_sequence_to_interface_returns_fixed_memory_and_mask() -> None:
    torch = pytest.importorskip("torch")
    values = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    pooled, mask = pool_sequence_to_interface(
        values, max_memory_tokens=3, interface_dim=2
    )
    assert tuple(pooled.shape) == (3, 2)
    assert tuple(mask.shape) == (3,)
    assert mask.tolist() == [1.0, 1.0, 1.0]
    assert torch.isfinite(pooled).all()


def test_pool_sequence_pads_short_memory_and_tracks_mask() -> None:
    torch = pytest.importorskip("torch")
    pooled, mask = pool_sequence_to_interface(
        torch.ones(1, 4), max_memory_tokens=3, interface_dim=2
    )
    assert tuple(pooled.shape) == (3, 2)
    assert mask.tolist() == [1.0, 0.0, 0.0]
    assert pooled[1:].abs().sum().item() == 0.0


def test_representation_pooling_preserves_distinct_memory_masks() -> None:
    torch = pytest.importorskip("torch")
    features, masks = pool_representation_dict(
        {"hidden": torch.ones(1, 4), "kv": torch.ones(5, 4)},
        max_memory_tokens=3,
        interface_dim=2,
    )
    assert masks["hidden"].tolist() == [1.0, 0.0, 0.0]
    assert masks["kv"].tolist() == [1.0, 1.0, 1.0]
    assert tuple(features["hidden"].shape) == (3, 2)


def test_anchor_positions_are_deterministic_and_within_prefix() -> None:
    assert anchor_positions(100, count=4, minimum_prefix=8) == [25, 50, 75, 100]
    assert anchor_positions(5, count=4, minimum_prefix=8) == [5]
    with pytest.raises(ValueError, match="count"):
        anchor_positions(10, count=0, minimum_prefix=1)


def test_feature_split_is_document_disjoint() -> None:
    rows = [
        {"document_id": "a", "value": 1},
        {"document_id": "a", "value": 2},
        {"document_id": "b", "value": 3},
        {"document_id": "c", "value": 4},
        {"document_id": "d", "value": 5},
        {"document_id": "e", "value": 6},
    ]
    train, dev, test = split_feature_rows_by_document(rows)
    assert not ({r["document_id"] for r in train} & {r["document_id"] for r in dev})
    assert not ({r["document_id"] for r in train} & {r["document_id"] for r in test})
    assert not ({r["document_id"] for r in dev} & {r["document_id"] for r in test})
    assert len(train) + len(dev) + len(test) == len(rows)


def test_memory_probe_has_same_shape_and_parameter_count_for_all_representations() -> None:
    torch = pytest.importorskip("torch")
    model = MemoryBlockProbe(
        interface_dim=8, hidden_dim=16, horizon=3, vocab_size=11
    )
    memory = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4)
    logits = model(memory, mask)
    assert tuple(logits.shape) == (2, 3, 11)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_probe_metrics_reports_positionwise_exact_prefix_and_topk() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 3, 5)
    labels = torch.tensor([[1, 2, 3]])
    logits[0, 0, 1] = 10
    logits[0, 1, 2] = 10
    logits[0, 2, 4] = 10
    result = probe_metrics(logits, labels)
    assert result["acc1_by_position"][:2] == [1.0, 1.0]
    assert result["acc1_by_position"][2] == 0.0
    assert result["prefix_exact_by_position"] == [1.0, 1.0, 0.0]
    assert result["acc5_by_position"] == [1.0, 1.0, 1.0]


def test_wrong_document_indices_are_a_deterministic_cross_document_control() -> None:
    assert wrong_document_indices(["a", "a", "b", "b", "c"]) == [2, 3, 4, 4, 0]


def test_required_capture_layers_adds_final_layer_for_hidden_sequence() -> None:
    assert required_capture_layers([1, 9, 17], num_layers=20) == [1, 9, 17, 19]
