from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.TrainingFree.lease_collector import (
    block_log_geometry,
    compute_live_log_z,
    extract_cache_keys,
    map_query_heads_to_kv,
)


def test_gqa_query_to_kv_mapping_is_deterministic() -> None:
    assert map_query_heads_to_kv(4, 2) == [0, 0, 1, 1]


def test_gqa_mapping_rejects_non_divisible_head_counts() -> None:
    with pytest.raises(ValueError, match="divisible"):
        map_query_heads_to_kv(3, 2)


def test_block_geometry_returns_per_query_head_and_block_values() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    keys = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ]
    )

    log_z0, kappa = block_log_geometry(query, keys, source_start=0, source_end=2, block_size=1)

    assert len(log_z0) == 2
    assert len(log_z0[0]) == 2
    assert len(kappa) == 2
    assert all(value >= 0.0 for row in kappa for value in row)


def test_block_geometry_handles_partial_final_block() -> None:
    query = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]])

    log_z0, kappa = block_log_geometry(
        query, keys, source_start=0, source_end=3, block_size=2
    )

    assert len(log_z0[0]) == 2
    assert all(torch.isfinite(torch.tensor(row)).all() for row in log_z0)
    assert all(torch.isfinite(torch.tensor(row)).all() for row in kappa)


def test_live_log_z_excludes_source_span() -> None:
    query = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[[1.0, 0.0], [10.0, 0.0], [2.0, 0.0]]])

    live = compute_live_log_z(query, keys, source_start=1, source_end=2)

    expected = torch.logsumexp(torch.tensor([1.0, 2.0]) / 2.0**0.5, dim=0).item()
    assert live == pytest.approx([expected])


def test_extract_cache_keys_supports_dynamic_cache_like_layers() -> None:
    expected = torch.randn(1, 2, 5, 4)
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=expected)])

    result = extract_cache_keys(cache, 0)

    assert torch.equal(result, expected)
