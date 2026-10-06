"""Kiểm tra các denominator và span của phép đo attention."""

import importlib.util
from pathlib import Path

import numpy as np
import torch


def _module():
    path = Path(__file__).parents[1] / "scripts/probe_dflash_attention.py"
    spec = importlib.util.spec_from_file_location("attention_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_anchor_excluded_and_mass_partitioned():
    # Anchor attends prompt only; proposal attends prompt/generated/block.
    weights = torch.tensor([[[[1., 0., 0., 0., 0., 0.],
                               [.1, .2, .3, .1, .1, .2]]]])
    record, vector = _module().summarize_attention(weights, 3, 4, bin_size=2)
    np.testing.assert_allclose(vector, [.1, .2, .3], atol=1e-7)
    assert abs(record["prompt_mass"] - .6) < 1e-6
    assert abs(record["generated_mass"] - .1) < 1e-6
    assert abs(record["draft_block_mass"] - .3) < 1e-6
    np.testing.assert_allclose(record["prompt_bins_normalized"], [.5, .5])


def test_zero_prompt_mass_is_not_normalized_to_one():
    weights = torch.tensor([[[[0., 0., 1.], [0., 0., 1.]]]])
    record, _ = _module().summarize_attention(weights, 1, 1)
    assert record["prompt_mass"] == 0
    assert record["prompt_bins_normalized"] is None
    assert record["top_1k_prompt_coverage"] is None


def test_scattered_topk_and_contiguous_window_are_distinct():
    vector = torch.tensor([.45, .05, .05, .45])
    weights = vector.reshape(1, 1, 1, 4).repeat(1, 1, 2, 1)
    record, _ = _module().summarize_attention(weights, 4, 4, top_k=2)
    assert abs(record["top_1k_prompt_coverage"] - .9) < 1e-6
    assert abs(record["best_window_1k_prompt_coverage"] - .5) < 1e-6
