"""CPU train smoke cho MR-DFlash, giữ objective DFlash hiện tại."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from MR_DFlash.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from MR_DFlash.data import DFlashFeatureDataset, save_feature_file
from MR_DFlash.mr_model import MRDFlashDraftModel
from MR_DFlash.run_train import build_online_model
from MR_DFlash.trainer import Trainer
from MR_DFlash.training import (
    MRDFlashTrainStrategy,
    OnlineMRDFlashModel,
    build_mr_draft_spec_from_target_config,
)


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


def test_mr_train_smoke_and_checkpoint_reload_cpu() -> None:
    torch.manual_seed(42)
    target = _target()
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        features_dir = tmp / "features"
        for index in range(2):
            ids = torch.randint(5, 120, (20,))
            loss_mask = torch.zeros(20, dtype=torch.long)
            loss_mask[2:18] = 1
            save_feature_file(
                str(features_dir / f"sample_{index}.ckpt"),
                {
                    "input_ids": ids,
                    "loss_mask": loss_mask,
                    "hidden_states": torch.randn(20, 64),
                },
            )

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
        assert draft.init_from_target(target)
        model = OnlineMRDFlashModel(
            draft,
            target_lm_head=target.get_output_embeddings(),
            target_embed_tokens=target.get_input_embeddings(),
            mask_token_id=127,
            block_size=4,
            num_anchors=4,
            loss_decay_gamma=7.0,
            objective_chunk_blocks=0,
            loss_type="dflash",
            attention_backend="sdpa",
        )
        before = draft.memory.adapter.hca.weight.detach().clone()
        cfg = RunConfig(
            run_id="mr-smoke",
            output_dir=str(tmp / "out"),
            model=ModelConfig(
                target_model_path="tiny",
                architecture="mr_dflash",
                block_size=4,
                mask_token_id=127,
                target_layer_ids=[1, 2],
                torch_dtype="float32",
            ),
            data=DataConfig(hidden_states_path=str(features_dir), max_length=20),
            training=TrainingConfig(
                strategy="mr_dflash",
                num_epochs=1,
                max_steps=2,
                batch_size=1,
                accumulation_steps=1,
                learning_rate=1e-3,
                num_anchors=4,
                objective_chunk_blocks=0,
                save_interval=1,
                log_interval=1,
            ),
        )
        class _Tokenizer:
            def convert_tokens_to_ids(self, token):
                return 127 if token == "[MASK]" else -1

        configured = build_online_model(
            cfg,
            tokenizer=_Tokenizer(),
            target_config=target.config,
            embed_tokens=target.get_input_embeddings(),
            lm_head=target.get_output_embeddings(),
            device=torch.device("cpu"),
        )
        assert isinstance(configured, OnlineMRDFlashModel)
        assert configured.draft_model.spec.hca_compression_ratio == 128
        dataset = DFlashFeatureDataset(str(features_dir), max_len=20, run_id="smoke")
        trainer = Trainer(
            cfg,
            MRDFlashTrainStrategy(model),
            dataset,
            device=torch.device("cpu"),
        )
        summary = trainer.fit()

        assert summary["global_step"] == 2
        assert torch.isfinite(draft.memory.adapter.hca.weight).all()
        assert not torch.equal(before, draft.memory.adapter.hca.weight.detach())
        checkpoint = tmp / "out" / "checkpoint_final.pt"
        assert checkpoint.exists()

        from MR_DFlash.checkpoint import warm_start_draft_model

        restored = MRDFlashDraftModel(spec).float()
        loaded, unexpected = warm_start_draft_model(
            restored,
            str(checkpoint),
            key_prefix="draft_model.",
            strategy_name="mr_dflash",
        )
        assert loaded == []
        assert unexpected == []
        assert torch.equal(
            restored.memory.adapter.hca.weight,
            draft.memory.adapter.hca.weight,
        )
