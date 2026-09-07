"""CPU inference contract cho MR-DFlash với target tiny local."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from MR_DFlash.checkpoint import save_draft_weights
from MR_DFlash.config import ModelConfig, RunConfig
from MR_DFlash.inference import MRDFlashInferenceEngine, _load_checkpoint_model_config
from MR_DFlash.mr_model import MRDFlashDraftModel
from MR_DFlash.training import build_dflash_additive_mask, build_mr_draft_spec_from_target_config


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


def test_inference_block_mask_matches_default_training_mask_cpu() -> None:
    target = _target()
    spec = build_mr_draft_spec_from_target_config(
        target.config,
        draft_num_hidden_layers=1,
        block_size=4,
        target_layer_ids=[1, 2],
        num_stages=2,
    )
    draft = MRDFlashDraftModel(spec).float()
    engine = MRDFlashInferenceEngine(
        target,
        draft,
        mask_token_id=127,
        device=torch.device("cpu"),
    )
    train_mask = build_dflash_additive_mask(
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        S=0,
        block_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    inference_mask = engine._block_mask(1, 4, torch.float32)
    assert torch.equal(train_mask, inference_mask)

    sliding_spec = build_mr_draft_spec_from_target_config(
        target.config,
        draft_num_hidden_layers=1,
        block_size=4,
        target_layer_ids=[1, 2],
        layer_types=["sliding_attention"],
        sliding_window=2,
        num_stages=2,
    )
    sliding_engine = MRDFlashInferenceEngine(
        target,
        MRDFlashDraftModel(sliding_spec).float(),
        mask_token_id=127,
        device=torch.device("cpu"),
    )
    sliding_train_mask = build_dflash_additive_mask(
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        S=0,
        block_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sliding_window=2,
    )
    assert torch.equal(sliding_train_mask, sliding_engine._block_mask(1, 4, torch.float32))


def test_verify_commits_bonus_after_fully_accepted_block_cpu() -> None:
    torch.manual_seed(8)
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
    )
    draft = MRDFlashDraftModel(spec).float()
    engine = MRDFlashInferenceEngine(
        target,
        draft,
        mask_token_id=127,
        device=torch.device("cpu"),
    )
    prefix = torch.tensor([[3, 4, 5, 6]])
    prefill = engine.prefill(prefix)
    proposal = prefill.target_logits.argmax(dim=-1, keepdim=True)
    verified = engine.verify(prefix, proposal, prefill.memory)

    assert verified.accepted_proposal_count == 1
    assert verified.accepted_ids.shape == (1, 2)
    assert verified.memory.total_tokens == prefix.shape[1] + 2

    limited = engine.verify(
        prefix,
        proposal,
        prefill.memory,
        max_append_tokens=1,
    )
    assert limited.accepted_ids.shape == (1, 1)
    assert limited.memory.total_tokens == prefix.shape[1] + 1


def test_verify_stops_at_first_eos_before_memory_append_cpu() -> None:
    torch.manual_seed(9)
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
    )
    draft = MRDFlashDraftModel(spec).float()
    engine = MRDFlashInferenceEngine(
        target,
        draft,
        mask_token_id=127,
        device=torch.device("cpu"),
    )
    prefix = torch.tensor([[3, 4, 5, 6]])
    prefill = engine.prefill(prefix)
    eos_id = 9
    proposals = torch.tensor([[5, eos_id, 6]])

    def fake_target_forward(input_ids):
        length = input_ids.shape[1]
        vocab = 128
        logits = torch.full((1, length, vocab), -100.0)
        choices = [5, eos_id, 6]
        for offset, token in enumerate(choices):
            logits[:, prefix.shape[1] - 1 + offset, token] = 100.0
        logits[:, -1, 7] = 100.0
        hidden_states = [torch.zeros(1, length, 32) for _ in range(4)]
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    engine._target_forward = fake_target_forward
    verified = engine.verify(
        prefix,
        proposals,
        prefill.memory,
        eos_token_id=eos_id,
    )
    assert verified.accepted_proposal_count == 2
    assert verified.accepted_ids.tolist() == [[5, eos_id]]
    assert verified.memory.total_tokens == prefix.shape[1] + 2


def test_weights_checkpoint_reconstructs_mr_model_config_cpu(tmp_path) -> None:
    config_yaml = RunConfig(
        model=ModelConfig(
            architecture="mr_dflash",
            feature_layer_ids=[1, 9, 17, 25, 33],
            mr_num_stages=2,
            hca_compression_ratio=128,
            csa_compression_ratio=4,
            memory_local_window=128,
            csa_top_k=64,
        )
    ).dump_yaml()
    path = tmp_path / "draft.pt"
    save_draft_weights(str(path), {}, config_yaml=config_yaml)

    model_config = _load_checkpoint_model_config(str(path))
    assert model_config["feature_layer_ids"] == [1, 9, 17, 25, 33]
    assert model_config["hca_compression_ratio"] == 128
    assert model_config["csa_top_k"] == 64
