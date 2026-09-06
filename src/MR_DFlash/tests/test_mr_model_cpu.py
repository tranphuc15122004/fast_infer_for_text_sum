"""Contract test cho MR-DFlash draft model trên CPU."""

from __future__ import annotations

import torch

from MR_DFlash.memory import MRTargetMemory
from MR_DFlash.model import build_draft_spec
from MR_DFlash.mr_model import MRDFlashDraftModel, MRDraftSpec


def _tiny_target():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_qk_norm=False,
        attention_bias=False,
    )
    return Qwen3ForCausalLM(config).eval()


def test_initialized_mr_draft_forward_and_backward_cpu() -> None:
    torch.manual_seed(0)
    target = _tiny_target()
    dflash_spec = build_draft_spec(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        num_target_layers=4,
        draft_num_hidden_layers=1,
        block_size=4,
        target_layer_ids=[1, 2],
        head_dim=128,
        use_qk_norm=False,
        max_position_embeddings=128,
    )
    spec = MRDraftSpec.from_dflash(
        dflash_spec,
        num_stages=2,
        hca_compression_ratio=4,
        csa_compression_ratio=2,
        local_window=4,
        csa_top_k=3,
        indexer_dim=8,
    )
    draft = MRDFlashDraftModel(spec)
    copied = draft.init_from_target(target)
    assert copied

    features = torch.randn(2, 9, 64)
    memory = draft.build_memory(features)
    noise = torch.randn(2, 8, 32, requires_grad=True)
    draft_positions = torch.tensor([[1, 2, 3, 4, 10, 11, 12, 13]]).expand(2, -1)
    block_mask = torch.full((2, 1, 8, 8), torch.finfo(torch.float32).min)
    for start in (0, 4):
        block_mask[:, :, start : start + 4, start : start + 4] = torch.triu(
            torch.zeros((4, 4)), diagonal=0
        )
        block_mask[:, :, start : start + 4, start : start + 4].masked_fill_(
            torch.triu(torch.ones((4, 4), dtype=torch.bool), diagonal=1),
            torch.finfo(torch.float32).min,
        )

    output = draft(
        noise_embedding=noise,
        memory=memory,
        position_ids=draft_positions,
        attention_mask=block_mask,
    )
    assert output.shape == (2, 8, 32)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert noise.grad is not None
    assert torch.isfinite(noise.grad).all()
    assert any("block_attn" in key for key in copied)


def test_mr_context_does_not_read_target_tokens_after_anchor() -> None:
    torch.manual_seed(11)
    target = _tiny_target()
    dflash_spec = build_draft_spec(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        num_target_layers=4,
        draft_num_hidden_layers=1,
        block_size=4,
        target_layer_ids=[1, 2],
        head_dim=128,
        use_qk_norm=False,
        max_position_embeddings=128,
    )
    draft = MRDFlashDraftModel(
        MRDraftSpec.from_dflash(
            dflash_spec,
            num_stages=2,
            hca_compression_ratio=2,
            csa_compression_ratio=2,
            local_window=8,
            csa_top_k=4,
        )
    )
    features = torch.randn(1, 9, 64)
    changed = features.clone()
    changed[:, 7:] += 100.0
    memory = draft.build_memory(features)
    changed_memory = draft.build_memory(changed)
    noise = torch.randn(1, 8, 32)
    positions = torch.tensor([[3, 4, 5, 6, 10, 11, 12, 13]])
    mask = torch.full((1, 1, 8, 8), torch.finfo(torch.float32).min)
    allow = torch.zeros((4, 4))
    allow.masked_fill_(torch.triu(torch.ones((4, 4), dtype=torch.bool), diagonal=1), torch.finfo(torch.float32).min)
    mask[:, :, :4, :4] = allow
    mask[:, :, 4:, 4:] = allow

    first = draft(
        noise_embedding=noise,
        memory=memory,
        position_ids=positions,
        attention_mask=mask,
    )
    second = draft(
        noise_embedding=noise,
        memory=changed_memory,
        position_ids=positions,
        attention_mask=mask,
    )
    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-5, rtol=1e-5)
