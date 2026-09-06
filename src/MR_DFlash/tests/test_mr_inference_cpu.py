"""CPU inference contract cho MR-DFlash với target tiny local."""

from __future__ import annotations

import torch

from MR_DFlash.inference import MRDFlashInferenceEngine
from MR_DFlash.mr_model import MRDFlashDraftModel
from MR_DFlash.training import build_mr_draft_spec_from_target_config


def _target():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=False,
            use_qk_norm=False,
            attention_bias=False,
        )
    ).eval()


def test_prefill_draft_verify_updates_only_accepted_tokens_cpu() -> None:
    torch.manual_seed(7)
    target = _target()
    spec = build_mr_draft_spec_from_target_config(
        target.config,
        draft_num_hidden_layers=1,
        block_size=4,
        target_layer_ids=[1, 2],
        num_stages=2,
        hca_compression_ratio=4,
        csa_compression_ratio=2,
        local_window=4,
        csa_top_k=3,
        indexer_dim=8,
    )
    spec.mask_token_id = 127
    draft = MRDFlashDraftModel(spec).float()
    draft.init_from_target(target)
    engine = MRDFlashInferenceEngine(
        target,
        draft,
        mask_token_id=127,
        device=torch.device("cpu"),
    )
    prefix = torch.tensor([[3, 4, 5, 6, 7]])

    prefill = engine.prefill(prefix)
    assert prefill.memory.total_tokens == prefix.shape[1]
    assert torch.isfinite(prefill.target_logits).all()

    draft_output = engine.draft_block(prefix, prefill.memory)
    assert draft_output.proposed_ids.shape == (1, 3)
    assert torch.isfinite(draft_output.logits).all()

    wrong = (prefill.target_logits.argmax(dim=-1) + 1) % 128
    proposed = wrong.view(1, 1)
    verified = engine.verify(prefix, proposed, prefill.memory)
    assert verified.accepted_proposal_count == 0
    assert verified.accepted_ids.shape == (1, 1)
    assert not torch.equal(verified.accepted_ids, proposed)
    assert verified.memory.total_tokens == prefix.shape[1] + 1
    assert torch.isfinite(verified.target_logits).all()

    generated = engine.generate(prefix, max_new_tokens=2)
    assert generated.input_ids.shape[1] >= prefix.shape[1] + 2
    assert generated.input_ids.shape[0] == 1
    assert generated.accepted_proposal_tokens >= 0

