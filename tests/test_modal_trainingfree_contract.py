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
