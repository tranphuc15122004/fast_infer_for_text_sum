from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "modal_trainingfree.py"


def load_runner():
    pytest = __import__("pytest")
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("modal_trainingfree", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_modal_command_uses_persistent_venv_and_recap_module() -> None:
    runner = load_runner()

    command = runner.build_runner_command(
        target_model="/opt/models/Qwen3-0.6B",
        inputs=["data/representative_100/govreport_representative.jsonl"],
        output_dir=Path("/mnt/fast-in/outputs/recap_kv_v0/smoke"),
        max_samples=1,
        max_new_tokens=32,
        smoke=True,
        python="/mnt/fast-in/venv/bin/python",
    )

    assert command[:3] == ["/mnt/fast-in/venv/bin/python", "-m", "src.TrainingFree.run"]
    assert "--smoke" in command
    assert command[command.index("--model") + 1] == "/opt/models/Qwen3-0.6B"
    assert "/workspace/storage-shared" not in " ".join(command)


def test_modal_environment_is_offline_for_existing_checkpoint() -> None:
    runner = load_runner()

    env = runner.build_runtime_env(
        target_model="/opt/models/Qwen3-0.6B",
        output_dir=Path("/mnt/fast-in/outputs/recap_kv_v0/smoke"),
        python=Path("/mnt/fast-in/venv/bin/python"),
    )

    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["VIRTUAL_ENV"] == "/mnt/fast-in/venv"
    assert env["FI_RECAP_TARGET_MODEL"] == "/opt/models/Qwen3-0.6B"


def test_modal_parser_has_smoke_and_pilot_defaults() -> None:
    runner = load_runner()

    parser = runner.build_parser()
    smoke = parser.parse_args(["--mode", "smoke"])
    pilot = parser.parse_args(["--mode", "pilot"])

    assert smoke.mode == "smoke"
    assert pilot.mode == "pilot"
    assert runner.resolve_options(smoke.mode, smoke.max_samples, smoke.max_new_tokens)[:2] == (1, 32)
    assert runner.resolve_options(pilot.mode, pilot.max_samples, pilot.max_new_tokens)[:2] == (20, 128)
    assert "Qwen3-0.6B" in smoke.target_model


def test_modal_lease_experiment_is_forwarded_to_trainingfree_runner() -> None:
    runner = load_runner()

    command = runner.build_runner_command(
        target_model="/opt/models/Qwen3-0.6B",
        inputs=["data/representative_100/govreport_representative.jsonl"],
        output_dir=Path("/mnt/fast-in/outputs/recap_kv_v2/smoke"),
        max_samples=1,
        max_new_tokens=32,
        smoke=True,
        python="/mnt/fast-in/venv/bin/python",
        experiment="lease",
    )

    assert "--experiment" in command
    assert command[command.index("--experiment") + 1] == "lease"


def test_modal_hierarchy_experiment_is_forwarded_to_trainingfree_runner() -> None:
    runner = load_runner()

    command = runner.build_runner_command(
        target_model="/opt/models/Qwen3-0.6B",
        inputs=["data/representative_100/govreport_representative.jsonl"],
        output_dir=Path("/mnt/fast-in/outputs/recap_kv_v3/smoke"),
        max_samples=1,
        max_new_tokens=32,
        smoke=True,
        python="/mnt/fast-in/venv/bin/python",
        experiment="hierarchy",
    )

    assert command[command.index("--experiment") + 1] == "hierarchy"


def test_modal_hierarchy_uses_a_separate_volume_output_root() -> None:
    runner = load_runner()

    assert runner.output_root_for_experiment("hierarchy").as_posix().endswith("recap_kv_v3")
    assert runner.output_root_for_experiment("lease").as_posix().endswith("recap_kv_v2")


def test_modal_hierarchy_config_is_forwarded() -> None:
    runner = load_runner()

    command = runner.build_runner_command(
        target_model="/opt/models/Qwen3-0.6B",
        inputs=["data/representative_100/govreport_representative.jsonl"],
        output_dir=Path("/mnt/fast-in/outputs/recap_kv_v3/smoke"),
        max_samples=1,
        max_new_tokens=32,
        smoke=True,
        python="/mnt/fast-in/venv/bin/python",
        experiment="hierarchy",
        region_size=512,
        block_size=64,
        reps_per_block=8,
        reps_per_region=8,
        hierarchy_mass_budget=0.01,
        hierarchy_layers="last",
    )

    assert command[command.index("--region-size") + 1] == "512"
    assert command[command.index("--block-size") + 1] == "64"
    assert command[command.index("--reps-per-block") + 1] == "8"
    assert command[command.index("--reps-per-region") + 1] == "8"


def test_modal_temporal_experiment_and_sweep_are_forwarded() -> None:
    runner = load_runner()

    command = runner.build_runner_command(
        target_model="/opt/models/Qwen3-0.6B",
        inputs=["data/longbench_100_14k/gov_report.jsonl"],
        output_dir=Path("/mnt/fast-in/outputs/recap_kv_e43/smoke"),
        max_samples=1,
        max_new_tokens=32,
        smoke=True,
        python="/mnt/fast-in/venv/bin/python",
        experiment="temporal",
        temporal_lags="1,2,4,8,16",
        temporal_budgets="0.1,0.2,0.3,0.4",
        temporal_alphas="1.0,1.25,1.5",
        temporal_refresh_intervals="2,4,8,16",
        temporal_block_sizes="16,32,64",
    )

    assert command[command.index("--experiment") + 1] == "temporal"
    assert command[command.index("--temporal-lags") + 1] == "1,2,4,8,16"
    assert command[command.index("--temporal-block-sizes") + 1] == "16,32,64"
    assert runner.output_root_for_experiment("temporal").as_posix().endswith("recap_kv_e43")
