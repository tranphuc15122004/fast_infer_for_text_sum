from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def test_gather_paged_cache_preserves_logical_order_and_input():
    from Benchmark.vllm_research.context import gather_flash_kv
    cache = torch.arange(3 * 2 * 4 * 6).reshape(3, 2, 4, 6).float()
    before = cache.clone()
    key, value = gather_flash_kv(cache, torch.tensor([2, 0, 1]), torch.tensor([0, 3, 5]), 4, 3)
    assert torch.equal(key, torch.stack([cache[2, :, 0, :3], cache[2, :, 3, :3], cache[0, :, 1, :3]]))
    assert torch.equal(value, torch.stack([cache[2, :, 0, 3:], cache[2, :, 3, 3:], cache[0, :, 1, 3:]]))
    assert torch.equal(before, cache)


def test_attention_respects_noncausal_and_scores_include_block_denominator():
    from Benchmark.vllm_research.context import selected_attention
    q = torch.zeros(2, 2, 2)
    kctx = torch.zeros(3, 1, 2)
    vctx = torch.zeros_like(kctx)
    kb = torch.zeros(2, 1, 2)
    vb = torch.ones_like(kb)
    out, scores = selected_attention(q, kctx, vctx, kb, vb, torch.tensor([0, 2, 5]), torch.tensor([6, 7]), causal=False, scaling=1., collect_scores=True)
    assert torch.allclose(out, torch.full_like(q, .4))
    assert torch.allclose(scores, torch.full((3,), .2))
    causal, _ = selected_attention(q, kctx, vctx, kb, vb, torch.tensor([0, 2, 5]), torch.tensor([6, 7]), causal=True, scaling=1., collect_scores=False)
    assert torch.allclose(causal[0], torch.full((2, 2), .25))
    assert torch.allclose(causal[1], torch.full((2, 2), .4))


def test_selection_exact_budget_protected_new_keys_and_request_identity():
    from Benchmark.vllm_research.context import ContextState, select_positions
    state = ContextState(prompt_tokens=8, source_positions=(2, 3, 4), global_positions=(0, 7))
    state.rankings[0] = torch.tensor([0., 1., 8., 7., 6., 5., 4., 3.])
    state.ranking_length = 8
    result = select_positions(state, layer=0, bank_length=11, budget=7, selector='previous_dflash', source_anchors=1, recent_output=1)
    assert result.tolist() == [0, 2, 3, 7, 8, 9, 10]
    with pytest.raises(ValueError, match='protected'):
        select_positions(state, layer=0, bank_length=11, budget=2, selector='recency', source_anchors=1, recent_output=1)


def test_config_native_no_patch_and_research_worker_bootstrap(tmp_path):
    from Benchmark.vllm_research.compat import configure_worker_variant
    assert configure_worker_variant('native', {}, None)['env'] == {}
    config = configure_worker_variant('b3_fixed_k', {'k': 7}, str(tmp_path))
    assert config['engine_overrides']['max_num_seqs'] == 1
    assert config['engine_overrides']['async_scheduling'] is False
    env = config['env']
    assert 'fast_infer_research' in env['VLLM_PLUGINS'].split(',')
    payload = json.loads(Path(env['FAST_INFER_VLLM_RESEARCH_CONFIG']).read_text())
    assert payload['variant'] == 'b3_fixed_k'
    assert payload['parameters']['k'] == 7
    assert any((Path(p) / 'fast_infer_vllm_research-0.1.0.dist-info' / 'entry_points.txt').exists() for p in env['PYTHONPATH'].split(':'))


def test_invalid_policy_or_version_does_not_claim_supported():
    from Benchmark.vllm_research.compat import configure_worker_variant, inspect_capabilities
    assert inspect_capabilities(version='0.24.0')['a3']['available'] is False
    assert inspect_capabilities(version='0.30.0')['a3']['gpu_verified'] is False
    with pytest.raises(ValueError, match='parent'):
        configure_worker_variant('b3', {'model': {'variant': 'R-H'}}, None)
    with pytest.raises(ValueError, match='layout'):
        configure_worker_variant('a3', {'selector': 'recency', 'budget': 2048}, None)


def test_b3_keeps_draft_15_but_returns_only_verifier_prefix():
    from Benchmark.vllm_research.adapter import truncate_proposal
    proposal = torch.arange(15).reshape(1, 15)
    result = truncate_proposal(proposal, 7)
    assert proposal.shape[1] == 15
    assert result.tolist() == [list(range(7))]
    assert result.is_contiguous()
    with pytest.raises(ValueError):
        truncate_proposal(proposal.expand(2, -1), 7)
