"""Tests cho các seam cần thiết khi đưa MR-DFlash lên 1-2 GPU B200."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from MR_DFlash.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from MR_DFlash.data import (
    DFlashFeatureDataset,
    load_feature_manifest,
    save_feature_manifest,
)
from MR_DFlash.distributed import rank_shard_indices
from MR_DFlash.run_train import build_online_model
from MR_DFlash.trainer import Trainer
from MR_DFlash.training import MRDFlashTrainStrategy


def _tiny_target():
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


def test_feature_and_draft_init_layers_are_explicitly_separate() -> None:
    cfg = ModelConfig(
        feature_layer_ids=[1, 9, 17, 25, 33],
        draft_init_layer_ids=[18],
    )

    assert cfg.feature_layer_ids == [1, 9, 17, 25, 33]
    assert cfg.draft_init_layer_ids == [18]
    assert cfg.target_layer_ids is None

    with pytest.raises(ValueError, match="target_layer_ids.*feature_layer_ids"):
        ModelConfig(
            target_layer_ids=[18],
            feature_layer_ids=[1, 9, 17, 25, 33],
        )


def test_feature_manifest_roundtrip_and_dataset_width_validation(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    manifest = {
        "target_model_path": "Qwen/Qwen3-4B",
        "feature_layer_ids": [1, 9, 17, 25, 33],
        "hidden_size": 16,
        "feature_width": 80,
        "dtype": "bfloat16",
        "max_length": 32,
    }
    save_feature_manifest(str(feature_dir), manifest)
    loaded = load_feature_manifest(str(feature_dir))
    assert loaded["feature_width"] == 80
    assert json.loads((feature_dir / "manifest.json").read_text())["schema_version"]

    torch.save(
        {
            "input_ids": torch.arange(8),
            "loss_mask": torch.ones(8),
            "hidden_states": torch.randn(8, 80),
        },
        feature_dir / "sample.ckpt",
    )
    dataset = DFlashFeatureDataset(
        str(feature_dir), max_len=8, expected_feature_width=80
    )
    assert len(dataset) == 1
    assert dataset[0]["hidden_states"].shape[-1] == 80

    with pytest.raises(ValueError, match="feature_width"):
        DFlashFeatureDataset(
            str(feature_dir), max_len=8, expected_feature_width=81
        )


def test_rank_shard_indices_are_disjoint_and_drop_global_tail() -> None:
    rank0 = rank_shard_indices(10, batch_size=2, world_size=2, rank=0)
    rank1 = rank_shard_indices(10, batch_size=2, world_size=2, rank=1)

    assert rank0 == [0, 2, 4, 6]
    assert rank1 == [1, 3, 5, 7]
    assert set(rank0).isdisjoint(rank1)


def test_trainer_evaluates_feature_split_and_writes_metrics(tmp_path: Path) -> None:
    target = _tiny_target()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for index in range(2):
        torch.save(
            {
                "input_ids": torch.arange(8) + 5,
                "loss_mask": torch.ones(8),
                "hidden_states": torch.randn(8, 32),
            },
            feature_dir / f"sample_{index}.ckpt",
        )
    cfg = RunConfig(
        run_id="eval-smoke",
        output_dir=str(tmp_path / "out"),
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
            save_interval=0,
            log_interval=1,
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
        device=torch.device("cpu"),
    )
    dataset = DFlashFeatureDataset(
        str(feature_dir), max_len=8, expected_feature_width=32
    )
    summary = Trainer(
        cfg,
        MRDFlashTrainStrategy(model),
        dataset,
        device=torch.device("cpu"),
    ).fit(eval_dataset=dataset)

    assert torch.isfinite(torch.tensor(summary["eval"]["eval_loss"]))
    eval_path = tmp_path / "out" / "eval_metrics.json"
    assert json.loads(eval_path.read_text())["global_step"] == 1
