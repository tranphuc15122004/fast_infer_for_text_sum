import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "modal_longbench.py"


def load_modal_runner():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("modal_longbench", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_modal_master_env_uses_remote_cache_and_online_huggingface():
    runner = load_modal_runner()

    env = runner.build_modal_env(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        eagle_model="org/eagle",
        dflash_model="org/dflash",
        mode="representative",
        baselines="vanilla_hf eagle3",
        datasets="gov_report lcc",
        output_dir=Path("/mnt/fast-infer/outputs/longbench_100_14k"),
    )

    assert env["FI_OFFLINE"] == "0"
    assert env["HF_HOME"] == "/mnt/fast-infer/hf"
    assert env["LONG_BENCH_LOCAL_FILES_ONLY"] == "0"
    assert env["LONG_BENCH_MODEL"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert env["LONG_BENCH_EAGLE_MODEL"] == "org/eagle"
    assert env["LONG_BENCH_DFLASH_MODEL"] == "org/dflash"
    assert env["LONG_BENCH_MAGICDEC_MODEL_PTH"] == (
        "/mnt/fast-infer/checkpoints/magicdec/llama-3.1-8b/model.pth"
    )
    assert env["LONG_BENCH_MAGICDEC_MODEL_NAME"] == (
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    assert env["LONG_BENCH_OUTPUT_DIR"] == "/mnt/fast-infer/outputs/longbench_100_14k"
    assert "/workspace/storage-shared" not in "\n".join(
        f"{key}={value}" for key, value in env.items()
    )


def test_modal_master_env_allows_magicdec_checkpoint_override(tmp_path):
    runner = load_modal_runner()

    env = runner.build_modal_env(
        model="org/model",
        eagle_model="org/eagle",
        dflash_model="org/dflash",
        mode="representative",
        baselines="magicdec",
        datasets="lcc",
        output_dir=Path("/mnt/fast-infer/outputs/longbench_100_14k"),
        magicdec_model_pth=tmp_path / "model.pth",
        magicdec_model_name="org/tokenizer",
    )

    assert env["LONG_BENCH_MAGICDEC_MODEL_PTH"] == str(tmp_path / "model.pth")
    assert env["LONG_BENCH_MAGICDEC_MODEL_NAME"] == "org/tokenizer"


def test_modal_master_env_can_point_children_at_a_persistent_venv(tmp_path):
    runner = load_modal_runner()

    env = runner.build_modal_env(
        model="org/model",
        eagle_model="org/eagle",
        dflash_model="org/dflash",
        mode="smoke",
        baselines="vanilla_hf",
        datasets="gov_report",
        output_dir=Path("/mnt/fast-infer/outputs/longbench_100_14k"),
        python=tmp_path / "venv" / "bin" / "python",
    )

    assert env["FI_PYTHON"] == str(tmp_path / "venv" / "bin" / "python")


def test_modal_runtime_venv_is_created_with_system_site_packages(tmp_path):
    runner = load_modal_runner()

    python_path = runner.ensure_runtime_venv(tmp_path / "venv", sys.executable)

    probe = subprocess.run(
        [str(python_path), "-c", "import sys; print(sys.prefix != sys.base_prefix)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert python_path.is_file()
    assert probe.stdout.strip() == "True"


def test_modal_runner_command_forwards_benchmark_selection():
    runner = load_modal_runner()

    command = runner.build_runner_command(
        mode="representative",
        baselines="vanilla_hf eagle3",
        datasets="gov_report lcc",
        output_dir=Path("/mnt/fast-infer/outputs/longbench_100_14k"),
        run_id="modal-test-run",
        max_samples=20,
        max_new_tokens=2048,
        max_input_tokens=14000,
        timeout_seconds=3600,
        preflight_only=True,
    )

    assert command[:3] == [
        sys.executable,
        "/workspace/fast_infer_text_sum/scripts/run_longbench_200.py",
        "--mode",
    ]
    assert "representative" in command
    assert command[command.index("--baselines") + 1] == "vanilla_hf eagle3"
    assert command[command.index("--datasets") + 1] == "gov_report lcc"
    assert command[command.index("--max-samples") + 1] == "20"
    assert command[command.index("--max-input-tokens") + 1] == "14000"
    assert "--preflight-only" in command


def test_modal_source_mount_excludes_outputs_and_local_runtime_artifacts():
    runner = load_modal_runner()

    patterns = set(runner.SOURCE_IGNORE)

    assert "outputs" in patterns
    assert ".venv" in patterns
    assert "__pycache__" in patterns
    assert "checkpoints" in patterns


def test_modal_mounts_only_canonical_longbench_external_repositories():
    runner = load_modal_runner()

    assert {"EAGLE", "FAFO", "LongSpec", "MagicDec", "SSSD", "SpecExtend", "dflash"} == set(
        runner.CANONICAL_EXTERNAL_DIRS
    )
    assert "Sematic_selection" not in runner.CANONICAL_EXTERNAL_DIRS


def test_modal_optional_cuda_extensions_have_a_devel_image_path():
    runner = load_modal_runner()

    assert "cudnn-devel" in runner.DEFAULT_CUDA_IMAGE


def test_modal_function_env_propagates_secret_name_for_remote_import():
    runner = load_modal_runner()

    assert runner.build_function_env("team-huggingface") == {
        "PYTHONUNBUFFERED": "1",
        "MODAL_HF_SECRET": "team-huggingface",
    }
    assert runner.build_function_env("") == {"PYTHONUNBUFFERED": "1"}


def test_modal_cuda_extensions_are_installed_before_live_source_mounts(monkeypatch):
    runner = load_modal_runner()
    events = []

    class FakeImage:
        def __init__(self):
            self.events = events

        def apt_install(self, *packages):
            self.events.append(("apt", packages))
            return self

        def pip_install_from_requirements(self, requirements):
            self.events.append(("requirements", requirements))
            return self

        def pip_install(self, *packages, **kwargs):
            self.events.append(("pip", packages, kwargs))
            return self

        def env(self, variables):
            self.events.append(("env", variables))
            return self

        def add_local_dir(self, *args, **kwargs):
            self.events.append(("mount", args, kwargs))
            return self

    class FakeImageFactory:
        @staticmethod
        def from_registry(*args, **kwargs):
            return FakeImage()

        @staticmethod
        def debian_slim(*args, **kwargs):
            return FakeImage()

    monkeypatch.setattr(runner.modal, "Image", FakeImageFactory)
    monkeypatch.setenv("MODAL_INSTALL_FLASH_ATTN", "1")
    monkeypatch.setenv("MODAL_INSTALL_FLASHINFER", "1")

    runner._build_image()

    first_mount = next(index for index, event in enumerate(events) if event[0] == "mount")
    extension_installs = [
        index
        for index, event in enumerate(events)
        if event[0] == "pip"
        and any(package.startswith(("flash-attn", "flashinfer-")) for package in event[1])
    ]
    assert extension_installs
    assert all(index < first_mount for index in extension_installs)

    wheel_installs = [
        index
        for index, event in enumerate(events)
        if event[0] == "pip" and "wheel==0.45.1" in event[1]
    ]
    assert wheel_installs
    assert wheel_installs[0] < extension_installs[0]

    build_env = [event for event in events if event[0] == "env"]
    assert build_env
    assert build_env[0][1]["CC"] == "gcc"
    assert build_env[0][1]["CXX"] == "g++"

    ninja_installs = [
        index
        for index, event in enumerate(events)
        if event[0] == "pip" and "ninja==1.13.0" in event[1]
    ]
    assert ninja_installs
    assert ninja_installs[0] < extension_installs[0]
