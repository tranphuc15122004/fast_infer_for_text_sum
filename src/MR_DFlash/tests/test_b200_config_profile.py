"""Regression tests cho profile train trên B200 100 GB."""

from pathlib import Path

from MR_DFlash.run_train import load_run_config


ROOT = Path(__file__).resolve().parents[2] / "MR_DFlash"
PILOT_DIR = ROOT / "configs" / "pilot_qwen3_4b"


def test_qwen_pilot_configs_use_b200_safe_fair_profile() -> None:
    names = (
        "dflash_2l_3k.yaml",
        "mr_dflash_2s_3k.yaml",
        "dflash_5l_3k.yaml",
        "dflash_2l_8k.yaml",
        "mr_dflash_2s_8k.yaml",
        "dflash_5l_8k.yaml",
    )
    configs = [load_run_config(str(PILOT_DIR / name)) for name in names]

    for cfg in configs:
        assert cfg.model.torch_dtype == "bfloat16"
        assert cfg.model.block_size == 16
        assert cfg.model.feature_layer_ids == [1, 9, 17, 25, 33]
        assert cfg.training.num_anchors == 512
        assert cfg.training.batch_size == 1
        assert cfg.training.accumulation_steps == 4
        assert cfg.training.objective_chunk_blocks == 64

    assert {cfg.data.max_length for cfg in configs[:3]} == {3072}
    assert {cfg.data.max_length for cfg in configs[3:]} == {8192}
    assert {cfg.training.batch_size * cfg.training.accumulation_steps for cfg in configs} == {4}


def test_llama_benchmark_configs_use_b200_safe_fair_profile() -> None:
    names = (
        "llama3_1_8b_dflash_5l.yaml",
        "llama3_1_8b_mr_dflash.yaml",
        "llama3_1_8b_mr_dflash_exact_params.yaml",
    )
    config_dir = ROOT / "configs"
    configs = [load_run_config(str(config_dir / name)) for name in names]

    for cfg in configs:
        assert cfg.training.batch_size == 1
        assert cfg.training.accumulation_steps == 4
        assert cfg.training.objective_chunk_blocks == 64
        assert cfg.training.batch_size * cfg.training.accumulation_steps == 4
