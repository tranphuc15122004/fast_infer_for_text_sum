from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_syncspec_is_registered_in_dispatcher_and_uses_shared_config() -> None:
    dispatcher = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/run_syncspec.sh").read_text(encoding="utf-8")
    config = (ROOT / "scripts/common/config.sh").read_text(encoding="utf-8")
    assert "syncspec)" in dispatcher
    assert "run_syncspec.sh" in dispatcher
    assert "fast_infer_load_config syncspec" in wrapper
    assert "fast_infer__load_syncspec" in config


def test_syncspec_launcher_exposes_adaptive_budget_profiles() -> None:
    wrapper = (ROOT / "scripts/run_syncspec.sh").read_text(encoding="utf-8")
    train_runner = (ROOT / "scripts/run_syncspec_b200_train_smoke.sh").read_text(encoding="utf-8")
    b200_runner = (ROOT / "scripts/run_syncspec_b200_smoke.sh").read_text(encoding="utf-8")
    assert "BUDGET_PROFILES" in wrapper
    assert "--budget-profiles" in wrapper
    assert "--budget-profiles" in train_runner
    assert "--budget-profiles" in b200_runner


def test_syncspec_train_smoke_uses_shared_config() -> None:
    runner = (ROOT / "scripts/run_syncspec_b200_train_smoke.sh").read_text(encoding="utf-8")
    config = (ROOT / "scripts/common/config.sh").read_text(encoding="utf-8")
    assert "fast_infer_load_config syncspec" in runner
    assert "SYNCSPEC_TRAIN" in runner
    assert "--train-batch-size" in runner
    assert "SYNCSPEC_TRAIN_SEED" in runner
    assert "fast_infer__load_syncspec" in config


def test_syncspec_master_example_defaults_to_binary_trajectory_cache() -> None:
    example = (ROOT / "docs/fast_infer_master.example.env").read_text(encoding="utf-8")
    assert (
        'SYNCSPEC_TRAIN_TRAJECTORY="${SYNCSPEC_TRAIN_TRAJECTORY:-'
        '$SYNCSPEC_TRAIN_ARTIFACT_DIR/trajectories.pt}"'
    ) in example


def test_syncspec_cuda_smoke_has_explicit_cuda_guard() -> None:
    runner = (ROOT / "scripts/run_syncspec_cuda_smoke.sh").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" in runner
    assert '"BLOCKED"' in runner
    assert "--backend synthetic" in runner
    assert "--device cuda" in runner


def test_syncspec_cuda_smoke_probes_fast_infer_venv_runtime(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env sh\n"
        "printf '%s\\n' \"$0\" >> \"$PROBE_MARKER\"\n"
        "case \"$*\" in *status*) printf '{\"status\": \"BLOCKED\"}\\n' ;; *) printf '0\\n' ;; esac\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    marker = tmp_path / "probe-marker"
    env = dict(os.environ)
    env.pop("FAST_INFER_PYTHON", None)
    env.pop("FI_PYTHON", None)
    env["FAST_INFER_VENV"] = str(tmp_path)
    env["PROBE_MARKER"] = str(marker)
    proc = subprocess.run(
        ["bash", "scripts/run_syncspec_cuda_smoke.sh", str(tmp_path / "missing.env")],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert proc.returncode == 2
    assert marker.read_text(encoding="utf-8").splitlines() == [
        str(fake_python), str(fake_python),
    ]
    assert '"status": "BLOCKED"' in proc.stdout


def test_syncspec_cpu_smoke_covers_stage0_train_profile_and_infer() -> None:
    runner = (ROOT / "scripts/run_syncspec_cpu_smoke.sh").read_text(encoding="utf-8")
    assert 'source "$ROOT/scripts/common/config.sh"' in runner
    assert "fast_infer_load_config syncspec" in runner
    assert 'source "$ROOT/scripts/common/runtime.sh"' in runner
    assert "build_syncspec_trajectories.py" in runner
    assert "--backend synthetic" in runner
    assert "train_syncspec.py" in runner
    assert "--stage joint" in runner
    assert "profile_syncspec.py" in runner
    assert "infer_syncspec.py" in runner
    assert "--check-exactness" in runner
    assert "pytorch_model.bin" in runner
    assert "selector.pt" in runner
    assert "survival.pt" in runner


def test_train_launcher_help_and_master_config_contract() -> None:
    launcher = ROOT / "scripts/train.sh"
    assert launcher.is_file()
    proc = subprocess.run(
        ["bash", str(launcher), "--help"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode == 0
    assert "--mode" in proc.stdout
    assert "smoke" in proc.stdout
    assert "full" in proc.stdout

    source = launcher.read_text(encoding="utf-8")
    for fragment in (
        "FAST_INFER_MASTER_CONFIG",
        "fast_infer_load_config syncspec",
        "check_syncspec_b200.py",
        "build_syncspec_trajectories.py",
        "train_syncspec.py",
        "profile_syncspec.py",
        "infer_syncspec.py",
        "--resume",
        "pytorch_model.bin",
        "selector.pt",
        "survival.pt",
        "SYNCSPEC_TRAIN_FULL_STEPS",
        "SYNCSPEC_TRAIN_FULL_MAX_SAMPLES",
        "SYNCSPEC_TRAIN_FULL_OUTPUT_DIR",
        "pipeline_status",
    ):
        assert fragment in source
