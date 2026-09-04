"""Smoke test CPU end-to-end cho pipeline train DFlash (self-contained).

Dùng một Qwen3 siêu nhỏ dựng cục bộ bằng transformers (không download, không
GPU): tạo feature giả theo đúng contract offline, rồi chạy Trainer vài bước,
kiểm tra loss hữu hạn, checkpoint được ghi, accuracy hợp lệ.

Chạy: ``python tests/test_smoke_cpu.py`` (từ thư mục ``src`` để import
``MR_DFlash``) — hoặc pytest với PYTHONPATH=src.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/

from MR_DFlash.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from MR_DFlash.data import DFlashFeatureDataset, save_feature_file
from MR_DFlash.model import DFlashDraftModel
from MR_DFlash.training import (
    DFlashTrainStrategy,
    OnlineDFlashModel,
    build_draft_spec_from_target_config,
)
from MR_DFlash.trainer import Trainer


def make_tiny_qwen3(vocab: int = 512):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_qk_norm=False,
        attention_bias=False,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def make_synthetic_features(target, output_dir: Path, n_samples: int = 4,
                            seq_len: int = 48, layer_ids=(1, 2)):
    """Sinh feature offline giả: input_ids + loss_mask + hidden concat tại layer."""
    torch.manual_seed(0)
    vocab = target.config.vocab_size
    hidden_size = target.config.hidden_size
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):
        ids = torch.randint(5, vocab - 1, (seq_len,))
        # loss mask: đoạn assistant từ token 16..40 (có >=2 supervised liên tiếp)
        mask = torch.zeros(seq_len, dtype=torch.long)
        mask[16:40] = 1
        with torch.no_grad():
            outputs = target(
                ids.unsqueeze(0),
                output_hidden_states=True,
                use_cache=False,
            )
        hs = outputs.hidden_states
        feats = torch.cat([hs[l + 1] for l in layer_ids], dim=-1)[0]  # (seq, feat)
        save_feature_file(
            str(out_dir / f"sample_{i:04d}.ckpt"),
            {
                "input_ids": ids,
                "loss_mask": mask,
                "hidden_states": feats,
            },
        )
    return out_dir


def test_train_smoke():
    target = make_tiny_qwen3()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        features_dir = make_synthetic_features(target, tmp / "features")

        spec = build_draft_spec_from_target_config(
            target.config,
            draft_num_hidden_layers=1,
            block_size=4,
            target_layer_ids=[1, 2],
        )
        spec.mask_token_id = target.config.vocab_size - 1
        spec.num_anchors = 8
        draft = DFlashDraftModel(spec)
        copied = draft.init_from_target(target)
        assert len(copied) > 0, "init_from_target không copy được key nào"

        model = OnlineDFlashModel(
            draft,
            target_lm_head=target.get_output_embeddings(),
            target_embed_tokens=target.get_input_embeddings(),
            mask_token_id=spec.mask_token_id,
            block_size=4,
            num_anchors=8,
            loss_decay_gamma=7.0,
            objective_chunk_blocks=0,
            loss_type="dflash",
            attention_backend="sdpa",
        )

        cfg = RunConfig(
            run_id="smoke",
            output_dir=str(tmp / "out"),
            model=ModelConfig(target_model_path="tiny", block_size=4,
                              mask_token_id=spec.mask_token_id,
                              init_draft_from_target=True,
                              torch_dtype="float32"),
            data=DataConfig(hidden_states_path=str(features_dir), max_length=48),
            training=TrainingConfig(
                num_epochs=1,
                max_steps=4,
                batch_size=2,
                accumulation_steps=1,
                learning_rate=1e-3,
                warmup_ratio=0.1,
                max_grad_norm=1.0,
                num_anchors=8,
                loss_decay_gamma=7.0,
                objective_chunk_blocks=0,
                attention_backend="sdpa",
                save_interval=2,
                log_interval=1,
                seed=0,
            ),
        )

        dataset = DFlashFeatureDataset(str(features_dir), max_len=48, run_id="smoke")
        assert len(dataset) == 4

        strategy = DFlashTrainStrategy(model)
        trainer = Trainer(cfg, strategy, dataset, device=torch.device("cpu"))
        summary = trainer.fit()

        assert summary["global_step"] == 4
        out = Path(cfg.output_dir)
        assert (out / "checkpoint_final.pt").exists()
        assert (out / "checkpoint_step_2.pt").exists()
        assert (out / "draft_final.pt").exists()

        # loss hữu hạn, không NaN
        lines = (out / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 4
        for line in lines:
            import json

            m = json.loads(line)
            assert m["loss"] == m["loss"] and m["loss"] < 1e6
            assert 0.0 <= m["acc"] <= 1.0

        # --- run 2: accumulation_steps=2 + warmup (kiểm tra nhánh accumulation) ---
        cfg2 = RunConfig(
            run_id="smoke-accum",
            output_dir=str(tmp / "out2"),
            model=ModelConfig(target_model_path="tiny", block_size=4,
                              mask_token_id=spec.mask_token_id,
                              init_draft_from_target=True,
                              torch_dtype="float32"),
            data=DataConfig(hidden_states_path=str(features_dir), max_length=48),
            training=TrainingConfig(
                num_epochs=1,
                max_steps=2,
                batch_size=2,
                accumulation_steps=2,
                learning_rate=1e-3,
                warmup_ratio=0.5,
                max_grad_norm=1.0,
                num_anchors=8,
                loss_decay_gamma=7.0,
                objective_chunk_blocks=0,
                attention_backend="sdpa",
                save_interval=0,
                log_interval=1,
                seed=1,
            ),
        )
        # model mới để tránh nhiễu gradient cũ
        import copy

        draft2 = copy.deepcopy(draft)
        model2 = OnlineDFlashModel(
            draft2,
            target_lm_head=target.get_output_embeddings(),
            target_embed_tokens=target.get_input_embeddings(),
            mask_token_id=spec.mask_token_id,
            block_size=4,
            num_anchors=8,
            loss_decay_gamma=7.0,
            objective_chunk_blocks=0,
            loss_type="dflash",
            attention_backend="sdpa",
        )
        strategy2 = DFlashTrainStrategy(model2)
        dataset2 = DFlashFeatureDataset(str(features_dir), max_len=48, run_id="smoke2")
        trainer2 = Trainer(cfg2, strategy2, dataset2, device=torch.device("cpu"))
        summary2 = trainer2.fit()
        assert summary2["global_step"] == 2
        out2 = Path(cfg2.output_dir)
        assert (out2 / "checkpoint_final.pt").exists()
    print("SMOKE OK")


if __name__ == "__main__":
    test_train_smoke()
