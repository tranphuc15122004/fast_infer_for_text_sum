"""Tests cho các seam cần thiết khi đưa MR-DFlash lên 1-2 GPU B200."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from MR_DFlash.config import DataConfig, ModelConfig, RunConfig, TrainingConfig, resolve_mr_stage_init_layer_ids
from MR_DFlash.data import (
    DFlashFeatureDataset,
    load_feature_manifest,
    save_feature_manifest,
)
from MR_DFlash.distributed import rank_shard_indices
from MR_DFlash.run_train import build_online_model, load_run_config
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


def test_mr_stage_init_layout_is_independent_from_draft_depth() -> None:
    cfg = ModelConfig(
        architecture="mr_dflash",
        draft_num_hidden_layers=1,
        mr_num_stages=2,
        draft_init_layer_ids=[18],
        mr_stage_init_layer_ids=[17, 18],
    )

    assert resolve_mr_stage_init_layer_ids(cfg, num_target_layers=36) == [17, 18]
    assert cfg.draft_num_hidden_layers == 1


def test_fair_qwen3_experiment_matrix_controls_features_and_depth() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "MR_DFlash" / "configs"
    configs = {
        name: load_run_config(str(config_dir / name))
        for name in (
            "qwen3_4b_dflash_1l.yaml",
            "qwen3_4b_dflash_2l.yaml",
            "qwen3_4b_mr_dflash.yaml",
        )
    }
    features = [1, 9, 17, 25, 33]

    assert {cfg.model.target_model_path for cfg in configs.values()} == {"Qwen/Qwen3-4B"}
    assert {tuple(cfg.model.feature_layer_ids or []) for cfg in configs.values()} == {
        tuple(features)
    }
    assert configs["qwen3_4b_dflash_1l.yaml"].model.architecture == "dflash"
    assert configs["qwen3_4b_dflash_1l.yaml"].model.draft_num_hidden_layers == 1
    assert configs["qwen3_4b_dflash_2l.yaml"].model.architecture == "dflash"
    assert configs["qwen3_4b_dflash_2l.yaml"].model.draft_num_hidden_layers == 2
    assert configs["qwen3_4b_mr_dflash.yaml"].model.architecture == "mr_dflash"
    assert configs["qwen3_4b_mr_dflash.yaml"].model.mr_num_stages == 2

    # Init policy and DFlash objective knobs are held constant in the pilot;
    # only architecture/depth differs across these three configs.
    assert configs["qwen3_4b_dflash_1l.yaml"].model.draft_init_layer_ids == [18]
    assert configs["qwen3_4b_dflash_2l.yaml"].model.draft_init_layer_ids == [17, 18]
    assert configs["qwen3_4b_mr_dflash.yaml"].model.mr_stage_init_layer_ids == [17, 18]
    assert {cfg.model.init_draft_from_target for cfg in configs.values()} == {True}
    for field in ("block_size", "torch_dtype"):
        assert len({getattr(cfg.model, field) for cfg in configs.values()}) == 1
    for field in (
        "num_anchors",
        "loss_decay_gamma",
        "objective_chunk_blocks",
        "learning_rate",
        "warmup_ratio",
        "max_length",
        "batch_size",
        "accumulation_steps",
        "attention_backend",
        "loss_type",
        "seed",
    ):
        values = []
        for cfg in configs.values():
            owner = cfg.training if field != "max_length" else cfg.data
            values.append(getattr(owner, field))
        assert len(set(values)) == 1, (field, values)


def test_qwen3_8b_dflash_configs_use_the_same_five_feature_layers() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "MR_DFlash" / "configs"
    baseline = load_run_config(str(config_dir / "qwen3_8b_dflash_offline.yaml"))
    control = load_run_config(str(config_dir / "qwen3_8b_dflash_2l.yaml"))
    mr = load_run_config(str(config_dir / "qwen3_8b_mr_dflash.yaml"))

    expected = [1, 9, 17, 25, 33]
    assert baseline.model.feature_layer_ids == expected
    assert control.model.feature_layer_ids == expected
    assert mr.model.feature_layer_ids == expected
    assert baseline.model.draft_num_hidden_layers == 1
    assert control.model.draft_num_hidden_layers == 2
    assert mr.model.mr_num_stages == 2


def test_primary_mr_training_configs_build_for_qwen3_and_llama31() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "MR_DFlash" / "configs"
    qwen_cfg = load_run_config(str(config_dir / "qwen3_4b_mr_dflash.yaml"))
    llama_cfg = load_run_config(str(config_dir / "llama3_1_8b_mr_dflash.yaml"))

    assert qwen_cfg.model.feature_layer_ids == [1, 9, 17, 25, 33]
    assert qwen_cfg.model.draft_intermediate_size == 9728
    assert qwen_cfg.model.mr_num_stages == 2
    assert llama_cfg.model.feature_layer_ids == [1, 8, 15, 22, 29]
    assert llama_cfg.model.draft_intermediate_size == 12288
    assert llama_cfg.model.mr_num_stages == 4

    from transformers import LlamaConfig, Qwen3Config

    from MR_DFlash.mr_model import MRDFlashDraftModel
    from MR_DFlash.training import build_mr_draft_spec_from_target_config

    qwen_target = Qwen3Config(
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
    )
    llama_target = LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        rms_norm_eps=1e-5,
    )

    with torch.device("meta"):
        qwen = MRDFlashDraftModel(
            build_mr_draft_spec_from_target_config(
                qwen_target,
                draft_num_hidden_layers=qwen_cfg.model.draft_num_hidden_layers,
                block_size=qwen_cfg.model.block_size,
                draft_intermediate_size=qwen_cfg.model.draft_intermediate_size,
                target_layer_ids=qwen_cfg.model.feature_layer_ids,
                mask_token_id=qwen_cfg.model.mask_token_id,
                num_stages=qwen_cfg.model.mr_num_stages,
                hca_compression_ratio=qwen_cfg.model.hca_compression_ratio,
                csa_compression_ratio=qwen_cfg.model.csa_compression_ratio,
                local_window=qwen_cfg.model.memory_local_window,
                csa_top_k=qwen_cfg.model.csa_top_k,
                indexer_dim=qwen_cfg.model.indexer_dim,
                indexer_num_heads=qwen_cfg.model.indexer_num_heads,
            )
        )
        llama = MRDFlashDraftModel(
            build_mr_draft_spec_from_target_config(
                llama_target,
                draft_num_hidden_layers=llama_cfg.model.draft_num_hidden_layers,
                block_size=llama_cfg.model.block_size,
                draft_intermediate_size=llama_cfg.model.draft_intermediate_size,
                target_layer_ids=llama_cfg.model.feature_layer_ids,
                mask_token_id=llama_cfg.model.mask_token_id,
                num_stages=llama_cfg.model.mr_num_stages,
                hca_compression_ratio=llama_cfg.model.hca_compression_ratio,
                csa_compression_ratio=llama_cfg.model.csa_compression_ratio,
                local_window=llama_cfg.model.memory_local_window,
                csa_top_k=llama_cfg.model.csa_top_k,
                indexer_dim=llama_cfg.model.indexer_dim,
                indexer_num_heads=llama_cfg.model.indexer_num_heads,
            )
        )

    assert qwen.spec.intermediate_size == 9728
    assert qwen.spec.context_feature_dim == 5 * 2560
    assert llama.spec.intermediate_size == 12288
    assert llama.spec.context_feature_dim == 5 * 4096


def test_llama31_mr_config_tracks_five_layer_dflash_parameter_budget() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "MR_DFlash" / "configs"
    dflash_cfg = load_run_config(str(config_dir / "llama3_1_8b_dflash_5l.yaml"))
    mr_cfg = load_run_config(str(config_dir / "llama3_1_8b_mr_dflash.yaml"))
    exact_cfg = load_run_config(
        str(config_dir / "llama3_1_8b_mr_dflash_exact_params.yaml")
    )

    assert dflash_cfg.model.target_model_path == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert mr_cfg.model.target_model_path == dflash_cfg.model.target_model_path
    assert dflash_cfg.model.feature_layer_ids == [1, 8, 15, 22, 29]
    assert mr_cfg.model.feature_layer_ids == dflash_cfg.model.feature_layer_ids
    assert mr_cfg.model.mr_num_stages == 4
    assert mr_cfg.model.indexer_dim == 4096
    assert dflash_cfg.model.draft_intermediate_size == 12288
    assert mr_cfg.model.draft_intermediate_size == 12288
    assert dflash_cfg.model.block_size == 16
    assert mr_cfg.model.block_size == 16
    assert dflash_cfg.model.mask_token_id == 128002
    assert mr_cfg.model.mask_token_id == 128002
    assert dflash_cfg.model.init_draft_from_target is False
    assert mr_cfg.model.init_draft_from_target is False
    assert dflash_cfg.model.draft_init_layer_ids == [1, 8, 15, 22, 29]
    assert mr_cfg.model.draft_init_layer_ids == [16]
    assert mr_cfg.model.mr_stage_init_layer_ids == [1, 10, 20, 29]
    assert exact_cfg.model.mr_num_stages == 4
    assert exact_cfg.model.indexer_dim == 5120
    assert exact_cfg.model.feature_layer_ids == dflash_cfg.model.feature_layer_ids
    assert exact_cfg.model.draft_intermediate_size == 12288

    from transformers import LlamaConfig

    target_config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        rms_norm_eps=1e-5,
    )
    from MR_DFlash.model import DFlashDraftModel
    from MR_DFlash.mr_model import MRDFlashDraftModel
    from MR_DFlash.training import (
        build_draft_spec_from_target_config,
        build_mr_draft_spec_from_target_config,
    )

    with torch.device("meta"):
        dflash = DFlashDraftModel(
            build_draft_spec_from_target_config(
                target_config,
                draft_num_hidden_layers=5,
                block_size=16,
                target_layer_ids=dflash_cfg.model.feature_layer_ids,
                draft_intermediate_size=dflash_cfg.model.draft_intermediate_size,
            )
        )
        mr = MRDFlashDraftModel(
            build_mr_draft_spec_from_target_config(
                target_config,
                draft_num_hidden_layers=1,
                block_size=16,
                target_layer_ids=mr_cfg.model.feature_layer_ids,
                draft_intermediate_size=mr_cfg.model.draft_intermediate_size,
                num_stages=4,
                hca_compression_ratio=128,
                csa_compression_ratio=4,
                local_window=128,
                csa_top_k=64,
                indexer_dim=mr_cfg.model.indexer_dim,
                indexer_num_heads=1,
            )
        )
        exact = MRDFlashDraftModel(
            build_mr_draft_spec_from_target_config(
                target_config,
                draft_num_hidden_layers=1,
                block_size=16,
                target_layer_ids=exact_cfg.model.feature_layer_ids,
                draft_intermediate_size=exact_cfg.model.draft_intermediate_size,
                num_stages=4,
                hca_compression_ratio=128,
                csa_compression_ratio=4,
                local_window=128,
                csa_top_k=64,
                indexer_dim=exact_cfg.model.indexer_dim,
                indexer_num_heads=1,
            )
        )

        dflash_count = sum(parameter.numel() for parameter in dflash.parameters())
        mr_count = sum(parameter.numel() for parameter in mr.parameters())
        exact_count = sum(parameter.numel() for parameter in exact.parameters())

    assert dflash_count == 1_048_626_432
    assert mr_count == 1_040_786_432
    assert exact_count == 1_049_175_040
    assert abs(mr_count - dflash_count) / dflash_count < 0.03
    assert abs(exact_count - dflash_count) / dflash_count < 0.001


def test_draft_intermediate_size_can_follow_released_dflash_config() -> None:
    from transformers import LlamaConfig

    from MR_DFlash.training import build_draft_spec_from_target_config

    target_config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
    )
    spec = build_draft_spec_from_target_config(
        target_config,
        draft_num_hidden_layers=5,
        block_size=16,
        target_layer_ids=[1, 8, 15, 22, 29],
        draft_intermediate_size=12288,
    )

    assert spec.intermediate_size == 12288
    assert target_config.intermediate_size == 14336


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
    with pytest.raises(ValueError, match="target_model_path"):
        DFlashFeatureDataset(
            str(feature_dir),
            max_len=8,
            expected_target_model_path="other-target",
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
