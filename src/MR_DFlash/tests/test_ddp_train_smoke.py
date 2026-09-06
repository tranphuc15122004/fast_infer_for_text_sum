"""DDP launcher smoke; chạy explicit bằng torchrun world size 2."""

from __future__ import annotations

import os
import json
import math
from pathlib import Path

import pytest
import torch

from MR_DFlash.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from MR_DFlash.data import DFlashFeatureDataset
from MR_DFlash.distributed import barrier, destroy_process_group, setup_distributed
from MR_DFlash.run_train import build_online_model
from MR_DFlash.trainer import Trainer
from MR_DFlash.training import MRDFlashTrainStrategy


def _target():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            use_qk_norm=False,
            attention_bias=False,
        )
    ).eval()


def test_ddp_train_smoke_cpu(tmp_path: Path) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("chạy test này bằng torchrun --nproc_per_node=2")

    ctx = setup_distributed("cpu")
    root = Path(os.environ.get("MR_DFLASH_DDP_SMOKE_DIR", str(tmp_path)))
    feature_dir = root / "features"
    output_dir = root / "out"
    try:
        if ctx.is_main:
            feature_dir.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                torch.save(
                    {
                        "input_ids": torch.arange(8) + 5,
                        "loss_mask": torch.ones(8),
                        "hidden_states": torch.randn(8, 32),
                    },
                    feature_dir / f"sample_{index}.ckpt",
                )
        barrier()

        torch.manual_seed(17)
        target = _target()
        cfg = RunConfig(
            run_id="ddp-smoke",
            output_dir=str(output_dir),
            model=ModelConfig(
                target_model_path="tiny",
                architecture="mr_dflash",
                block_size=4,
                mask_token_id=63,
                feature_layer_ids=[1, 2],
                torch_dtype="float32",
            ),
            data=DataConfig(hidden_states_path=str(feature_dir), max_length=8),
            training=TrainingConfig(
                strategy="mr_dflash",
                num_epochs=1,
                max_steps=1,
                batch_size=1,
                num_anchors=2,
                objective_chunk_blocks=0,
                save_interval=1,
                log_interval=1,
                dp_world_size=2,
            ),
        )

        class _Tokenizer:
            def convert_tokens_to_ids(self, token: str) -> int:
                return 63 if token == "[MASK]" else -1

        model = build_online_model(
            cfg,
            tokenizer=_Tokenizer(),
            target_config=target.config,
            embed_tokens=target.get_input_embeddings(),
            lm_head=target.get_output_embeddings(),
            device=ctx.device,
        )
        dataset = DFlashFeatureDataset(
            str(feature_dir), max_len=8, expected_feature_width=32
        )
        summary = Trainer(
            cfg,
            MRDFlashTrainStrategy(model),
            dataset,
            device=ctx.device,
        ).fit()
        assert summary["global_step"] == 1
        assert summary["world_size"] == 2
        barrier()
        if ctx.is_main:
            assert (output_dir / "checkpoint_final.pt").exists()
            assert (output_dir / "draft_final.pt").exists()
            assert (output_dir / "metrics.jsonl").exists()
            metrics = json.loads((output_dir / "metrics.jsonl").read_text().splitlines()[-1])
            assert math.isfinite(float(metrics["loss"]))
            assert metrics["tokens_per_step"] == 16
    finally:
        destroy_process_group()
