"""Real tiny Qwen3/DFlash forwards validate replay state on CPU (eager only)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "externals/dflash")]
from dflash.model import DFlashDraftModel
from TrainingFree.context_adaptive.types import PromptLayout
from TrainingFree.context_interventions.replay import ReferencePrefix, rebuild_replay_state, evaluate_replay_action


@pytest.fixture
def replay_state():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        common = dict(vocab_size=32, hidden_size=32, intermediate_size=64,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=16384, eos_token_id=None, pad_token_id=0,
                      attention_dropout=0.0)
        target_config = Qwen3Config(num_hidden_layers=36, **common)
        target_config._attn_implementation = "eager"
        draft_config = Qwen3Config(num_hidden_layers=5, num_target_layers=36, block_size=16,
                                  dflash_config={"target_layer_ids": [1, 9, 17, 25, 33], "mask_token_id": 0},
                                  **common)
        draft_config._attn_implementation = "eager"
        runtime = {"target": Qwen3ForCausalLM(target_config).eval(), "draft": DFlashDraftModel(draft_config).eval()}
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        layout = PromptLayout(8, tuple(range(1, 7)), (0, 7), ((1, 2, 3), (4, 5, 6)), "tiny-prompt", True)
        prefix = ReferencePrefix(0, 0, 9, 32, "prefix", None)
        state = rebuild_replay_state(runtime, ids, [], prefix, layout, max_new_tokens=32)
        try:
            yield runtime, ids, layout, state
        finally:
            torch.set_num_threads(previous_threads)


def test_real_replay_keeps_prefix_cache_and_bank_immutable_between_actions(replay_state):
    runtime, ids, layout, state = replay_state
    original_keys = [layer.keys.clone() for layer in state.cache.layers]
    bank_keys = [storage.key[:, :, :state.bank.length].clone() for storage in state.bank.layer_storage]
    full = [tuple(range(state.bank.length))] * 5

    def evaluate(masks, budget, collect=False):
        return evaluate_replay_action(
            runtime, ids, state, layout, masks, requested_budget=budget,
            selector_id="cpu-check", remaining_output_tokens=32, collect_attention_scores=collect,
        )

    first = evaluate(full, "full", True)
    reduced = evaluate([(0, 1, 7)] * 5, 3)
    repeated = evaluate(full, "full", True)
    assert first["status"] == reduced["status"] == repeated["status"] == "ok"
    assert first["proposed_token_ids"] == repeated["proposed_token_ids"]
    assert first["accepted_candidates"] == repeated["accepted_candidates"]
    assert 0 <= first["accepted_candidates"] <= 15
    assert set(first["attention_scores_by_layer"]) == {0, 1, 2, 3, 4}
    assert state.bank.length == state.cache.get_seq_length() == 8
    for actual, expected in zip(state.cache.layers, original_keys):
        torch.testing.assert_close(actual.keys, expected, rtol=0, atol=0)
    for actual, expected in zip(state.bank.layer_storage, bank_keys):
        torch.testing.assert_close(actual.key[:, :, :8], expected, rtol=0, atol=0)
    assert set(state.target_scores) == {1, 9, 17, 18, 25, 33}
    assert all(sum(scores.values()) == pytest.approx(1.0) for scores in state.target_scores.values())
