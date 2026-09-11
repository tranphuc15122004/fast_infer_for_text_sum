#!/usr/bin/env python3
"""Run the canonical LongBench runner in a reproducible Modal container.

The server runner remains the source of truth for benchmark behavior.  This
file only supplies Modal-specific concerns: image construction, source/data
mounts, a persistent cache/output Volume, Hugging Face authentication, and a
container-local master environment that does not refer to the B200 server.

Typical commands:

    modal run scripts/modal_longbench.py --mode smoke
    modal run scripts/modal_longbench.py --mode representative \
        --baselines vanilla_hf --datasets "gov_report lcc" --max-samples 20

Set ``MODAL_GPU`` before ``modal run`` to choose the GPU (the default is
``A100-80GB``).  The default image is intentionally dependency-minimal;
optional kernels such as FlashAttention can be enabled at image-build time
with ``MODAL_INSTALL_FLASH_ATTN=1``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
VOLUME_MOUNT = Path("/mnt/fast-infer")
REMOTE_DATA_DIR = REMOTE_ROOT / "data" / "longbench_100_14k"
REMOTE_OUTPUT_DIR = VOLUME_MOUNT / "outputs" / "longbench_100_14k"
REMOTE_HF_HOME = VOLUME_MOUNT / "hf"
REMOTE_PYTHONPATH = os.pathsep.join(
    (
        str(REMOTE_ROOT / "scripts"),
        str(REMOTE_ROOT / "externals" / "dflash"),
        str(REMOTE_ROOT / "externals" / "LLMLingua"),
    )
)

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_EAGLE_MODEL = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
DEFAULT_DFLASH_MODEL = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
DEFAULT_BASELINES = "vanilla_hf"
DEFAULT_DATASETS = "gov_report lcc"
DEFAULT_VOLUME_NAME = "fast-infer-text-sum-cache"
DEFAULT_GPU = "A100-80GB"
DEFAULT_CUDA_IMAGE = "nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04"

# These paths can be mounted as live source files, but local artifacts should
# never become part of the image upload or overwrite the persistent Volume.
SOURCE_IGNORE = (
    ".git",
    ".venv",
    ".pytest_cache",
    ".jspace",
    ".serena",
    ".agents",
    ".codex",
    "outputs",
    "checkpoints",
    "__pycache__",
    "*.pyc",
)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes"}


def _build_image() -> modal.Image:
    """Build the portable core image and mount project code/data at runtime."""

    base_name = os.environ.get("MODAL_BASE_IMAGE", "").strip()
    if base_name:
        image = modal.Image.from_name(base_name)
    elif any(
        _truthy(name)
        for name in (
            "MODAL_INSTALL_FLASH_ATTN",
            "MODAL_INSTALL_VLLM",
            "MODAL_INSTALL_FLASHINFER",
        )
    ):
        # The optional CUDA extensions need nvcc during image build.  The
        # minimal Debian image remains the default for vanilla_hf preflight.
        image = modal.Image.from_registry(
            os.environ.get("MODAL_CUDA_IMAGE", DEFAULT_CUDA_IMAGE),
            add_python="3.12",
        )
    else:
        image = modal.Image.debian_slim(python_version="3.12")

    image = image.apt_install(
        "git", "build-essential", "libgl1", "libglib2.0-0"
    ).pip_install_from_requirements(str(ROOT / "requirements.modal.txt"))

    # These are deliberately opt-in because they compile or install large,
    # CUDA-version-specific extensions.  Preflight will report the exact
    # missing dependency if a selected baseline needs one.  Keep these image
    # build steps before add_local_dir(copy=False): Modal requires live source
    # mounts to be the final image operations.
    if any(
        _truthy(name)
        for name in (
            "MODAL_INSTALL_FLASH_ATTN",
            "MODAL_INSTALL_VLLM",
            "MODAL_INSTALL_FLASHINFER",
        )
    ):
        image = image.env({"CC": "gcc", "CXX": "g++"}).pip_install(
            "wheel==0.45.1", "ninja==1.13.0"
        )
    if _truthy("MODAL_INSTALL_FLASH_ATTN"):
        image = image.pip_install(
            "flash-attn==2.8.3.post1",
            extra_options="--no-build-isolation",
        )
    if _truthy("MODAL_INSTALL_VLLM"):
        image = image.pip_install("vllm==0.24.0")
    if _truthy("MODAL_INSTALL_FLASHINFER"):
        image = image.pip_install(
            "flashinfer-python==0.6.12",
            "flashinfer-cubin==0.6.12",
        )

    return (
        image.add_local_dir(
            ROOT / "scripts",
            str(REMOTE_ROOT / "scripts"),
            copy=False,
            ignore=SOURCE_IGNORE,
        )
        .add_local_dir(
            ROOT / "externals",
            str(REMOTE_ROOT / "externals"),
            copy=False,
            ignore=SOURCE_IGNORE,
        )
        .add_local_dir(
            ROOT / "data",
            str(REMOTE_ROOT / "data"),
            copy=False,
            ignore=SOURCE_IGNORE + ("raw", "normalized"),
        )
    )


def build_modal_env(
    *,
    model: str,
    eagle_model: str,
    dflash_model: str,
    mode: str,
    baselines: str,
    datasets: str,
    output_dir: Path,
    data_dir: Path = REMOTE_DATA_DIR,
    seed: int = 42,
    warmup_runs: int = 3,
    strict: bool = True,
    collect: bool = True,
    python: str | Path | None = None,
) -> dict[str, str]:
    """Return a server-independent master environment for the child runner."""

    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = REMOTE_ROOT / data_dir

    cache = str(REMOTE_HF_HOME)
    return {
        "FI_PYTHON": str(python or sys.executable),
        "FI_DEVICE": "cuda",
        "FI_GPU_IDS": "0",
        "FI_OFFLINE": "0",
        "HF_HOME": cache,
        "HF_HUB_CACHE": str(REMOTE_HF_HOME / "hub"),
        "TRANSFORMERS_CACHE": str(REMOTE_HF_HOME / "hub"),
        "TRITON_CACHE_DIR": str(VOLUME_MOUNT / "triton"),
        "TORCH_EXTENSIONS_DIR": str(VOLUME_MOUNT / "torch_extensions"),
        "FLASHINFER_WORKSPACE_BASE": str(VOLUME_MOUNT / "flashinfer"),
        "PYTHONPATH": REMOTE_PYTHONPATH,
        "MODEL_TARGET": model,
        "MODEL_EAGLE_DRAFT": eagle_model,
        "MODEL_DFLASH_DRAFT": dflash_model,
        "LONG_BENCH_DATA_DIR": str(data_dir),
        "LONG_BENCH_OUTPUT_DIR": str(output_dir),
        "LONG_BENCH_MODEL": model,
        "LONG_BENCH_EAGLE_MODEL": eagle_model,
        "LONG_BENCH_DFLASH_MODEL": dflash_model,
        "LONG_BENCH_DEVICE": "cuda",
        "LONG_BENCH_GPU_IDS": "0",
        "LONG_BENCH_BASELINES": baselines,
        "LONG_BENCH_DATASETS": datasets,
        "LONG_BENCH_MODE": mode,
        "LONG_BENCH_REFERENCE_BASELINE": "vanilla_fa",
        "LONG_BENCH_LOCAL_FILES_ONLY": "0",
        "LONG_BENCH_MAX_NEW_TOKENS": "2048",
        "LONG_BENCH_SMOKE_MAX_NEW_TOKENS": "8",
        "LONG_BENCH_SMOKE_MAX_INPUT_TOKENS": "4096",
        "LONG_BENCH_MAX_INPUT_TOKENS": "0",
        "LONG_BENCH_REPRESENTATIVE_TIMEOUT_SECONDS": "3600",
        "LONG_BENCH_FULL_TIMEOUT_SECONDS": "21600",
        "LONG_BENCH_TIMEOUT_SECONDS": "900",
        "LONG_BENCH_TEMPERATURE": "0",
        "LONG_BENCH_WARMUP_RUNS": str(warmup_runs),
        "LONG_BENCH_SEED": str(seed),
        "LONG_BENCH_MIN_FREE_GB": "32",
        "LONG_BENCH_STRICT": "1" if strict else "0",
        "LONG_BENCH_COLLECT": "1" if collect else "0",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "HF_DATASETS_OFFLINE": "0",
        "PYTHONUNBUFFERED": "1",
    }


def ensure_runtime_venv(venv_dir: Path, base_python: str) -> Path:
    """Create/reuse a persistent venv backed by the Modal image packages."""

    python_path = venv_dir / "bin" / "python"
    if not python_path.is_file():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [base_python, "-m", "venv", "--system-site-packages", str(venv_dir)],
            check=True,
        )
    return python_path


def write_master_config(path: Path, values: dict[str, str]) -> None:
    """Write a shell-safe, generated config consumed by child launchers."""

    lines = [
        "# Generated inside a Modal container; do not use as the server master.",
    ]
    lines.extend(f"{key}={shlex.quote(str(values[key]))}" for key in sorted(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_runner_command(
    *,
    mode: str,
    baselines: str,
    datasets: str,
    output_dir: Path,
    run_id: str,
    max_samples: int | None = None,
    max_new_tokens: int | None = None,
    max_input_tokens: int | None = None,
    timeout_seconds: int | None = None,
    preflight_only: bool = False,
    allow_unsupported: bool = False,
    python: str | None = None,
) -> list[str]:
    """Build the exact command executed in the remote container."""

    command = [
        python or sys.executable,
        str(REMOTE_ROOT / "scripts" / "run_longbench_200.py"),
        "--mode",
        mode,
        "--baselines",
        baselines,
        "--datasets",
        datasets,
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
    ]
    optional = (
        ("--max-samples", max_samples),
        ("--max-new-tokens", max_new_tokens),
        ("--max-input-tokens", max_input_tokens),
        ("--timeout-seconds", timeout_seconds),
    )
    for flag, value in optional:
        if value is not None:
            command.extend((flag, str(value)))
    if preflight_only:
        command.append("--preflight-only")
    if allow_unsupported:
        command.append("--allow-unsupported")
    return command


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"modal-{timestamp}-{uuid.uuid4().hex[:8]}"


def _remote_path(value: str, *, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else REMOTE_ROOT / path


def build_function_env(secret_name: str) -> dict[str, str]:
    """Keep Modal's local and remote dependency graphs identical.

    Modal imports this module once while assembling the local app and again
    while hydrating the remote container.  If ``MODAL_HF_SECRET`` exists only
    in the local shell, the conditional ``Secret.from_name`` changes the
    remote dependency count and the job fails before the function starts.
    Passing the secret *name* (never its value) through the function
    environment makes the second import resolve the same branch.
    """

    values = {"PYTHONUNBUFFERED": "1"}
    if secret_name:
        values["MODAL_HF_SECRET"] = secret_name
    return values


VOLUME_NAME = os.environ.get("MODAL_VOLUME", DEFAULT_VOLUME_NAME)
GPU_TYPE = os.environ.get("MODAL_GPU", DEFAULT_GPU)
CACHE_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
SECRET_NAME = os.environ.get("MODAL_HF_SECRET", "").strip()
SECRETS = [modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN"])] if SECRET_NAME else []
FUNCTION_ENV = build_function_env(SECRET_NAME)
IMAGE = _build_image()
app = modal.App("fast-infer-text-sum-longbench")


@app.function(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={str(VOLUME_MOUNT): CACHE_VOLUME},
    secrets=SECRETS,
    timeout=24 * 60 * 60,
    env=FUNCTION_ENV,
)
def run_benchmark(
    *,
    mode: str,
    baselines: str,
    datasets: str,
    model: str,
    eagle_model: str,
    dflash_model: str,
    data_dir: str = "data/longbench_100_14k",
    max_samples: int = -1,
    max_new_tokens: int = -1,
    max_input_tokens: int = -1,
    timeout_seconds: int = -1,
    seed: int = 42,
    warmup_runs: int = 3,
    run_id: str = "",
    preflight_only: bool = False,
    allow_unsupported: bool = False,
    strict: bool = True,
    collect: bool = True,
) -> dict[str, Any]:
    """Execute one serial benchmark matrix on one Modal GPU."""

    output_dir = REMOTE_OUTPUT_DIR
    resolved_data_dir = _remote_path(data_dir, default=REMOTE_DATA_DIR)
    selected_run_id = run_id or _new_run_id()
    venv_dir = VOLUME_MOUNT / "venv"
    runtime_python = ensure_runtime_venv(venv_dir, sys.executable)
    environment = build_modal_env(
        model=model,
        eagle_model=eagle_model,
        dflash_model=dflash_model,
        mode=mode,
        baselines=baselines,
        datasets=datasets,
        output_dir=output_dir,
        data_dir=resolved_data_dir,
        seed=seed,
        warmup_runs=warmup_runs,
        strict=strict,
        collect=collect,
        python=runtime_python,
    )
    environment["VIRTUAL_ENV"] = str(venv_dir)
    environment["PATH"] = os.pathsep.join(
        (str(venv_dir / "bin"), os.environ.get("PATH", ""))
    )
    environment["FAST_INFER_MASTER_CONFIG"] = "/tmp/modal-fast-infer-master.env"
    child_env = dict(os.environ)
    child_env.update(environment)
    write_master_config(Path(environment["FAST_INFER_MASTER_CONFIG"]), environment)

    command = build_runner_command(
        mode=mode,
        baselines=baselines,
        datasets=datasets,
        output_dir=output_dir,
        run_id=selected_run_id,
        max_samples=None if max_samples < 0 else max_samples,
        max_new_tokens=None if max_new_tokens < 0 else max_new_tokens,
        max_input_tokens=None if max_input_tokens < 0 else max_input_tokens,
        timeout_seconds=None if timeout_seconds < 0 else timeout_seconds,
        preflight_only=preflight_only,
        allow_unsupported=allow_unsupported,
        python=str(runtime_python),
    )
    print("[modal] GPU:", GPU_TYPE, flush=True)
    print("[modal] volume:", VOLUME_NAME, flush=True)
    print("[modal] command:", " ".join(shlex.quote(item) for item in command), flush=True)
    started = time.monotonic()
    process = subprocess.run(command, cwd=str(REMOTE_ROOT), env=child_env)

    run_dir = output_dir / selected_run_id
    manifest_path = run_dir / "run_manifest.json"
    try:
        CACHE_VOLUME.commit()
    except Exception as exc:
        print(f"[modal] volume commit warning: {exc}", file=sys.stderr, flush=True)

    result: dict[str, Any] = {
        "status": "success" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "run_id": selected_run_id,
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "volume": VOLUME_NAME,
        "python": str(runtime_python),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result["cell_count"] = manifest.get("cell_count")
            result["failure_count"] = manifest.get("failure_count")
        except (OSError, json.JSONDecodeError):
            result["manifest_readable"] = False
    print("[modal] result:", json.dumps(result, ensure_ascii=False), flush=True)
    return result


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    baselines: str = DEFAULT_BASELINES,
    datasets: str = DEFAULT_DATASETS,
    model: str = DEFAULT_MODEL,
    eagle_model: str = DEFAULT_EAGLE_MODEL,
    dflash_model: str = DEFAULT_DFLASH_MODEL,
    data_dir: str = "data/longbench_100_14k",
    max_samples: int = -1,
    max_new_tokens: int = -1,
    max_input_tokens: int = -1,
    timeout_seconds: int = -1,
    seed: int = 42,
    warmup_runs: int = 3,
    run_id: str = "",
    preflight_only: bool = False,
    allow_unsupported: bool = False,
    strict: bool = True,
    collect: bool = True,
) -> None:
    """Parse ``modal run`` flags and submit one remote benchmark job."""

    result = run_benchmark.remote(
        mode=mode,
        baselines=baselines,
        datasets=datasets,
        model=model,
        eagle_model=eagle_model,
        dflash_model=dflash_model,
        data_dir=data_dir,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        timeout_seconds=timeout_seconds,
        seed=seed,
        warmup_runs=warmup_runs,
        run_id=run_id,
        preflight_only=preflight_only,
        allow_unsupported=allow_unsupported,
        strict=strict,
        collect=collect,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("returncode", 1) != 0:
        raise SystemExit(int(result["returncode"]))
