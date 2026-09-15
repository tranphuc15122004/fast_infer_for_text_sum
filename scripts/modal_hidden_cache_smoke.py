"""Run the MR-DFlash hidden-cache smoke on a Modal B200.

This is intentionally a small, reproducible remote debug harness.  It uses a
Qwen3-0.6B fixture (same Qwen3 architecture as the B200 target), installs the
cu130 stack, runs the full Phase 1 chain, and persists logs/artifacts to a
Modal Volume.  The production pipeline is not changed by this file.

Run from the repository root:

    modal run scripts/modal_hidden_cache_smoke.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/mr_dflash")
REMOTE_MODEL = Path("/opt/models/Qwen3-0.6B")
REMOTE_INPUT = REMOTE_ROOT / "data" / "modal_phase1_prompts.jsonl"
REMOTE_OUTPUT = Path("/outputs/mr_dflash_phase1_sglang_path_guard")

artifacts = modal.Volume.from_name(
    "mr-dflash-hidden-cache-debug",
    create_if_missing=True,
)


def _build_image() -> modal.Image:
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.1-devel-ubuntu24.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install(
            "libnuma1",
            "libnuma-dev",
        )
        .pip_install(
            "torch==2.11.0+cu130",
            extra_options="--index-url https://download.pytorch.org/whl/cu130",
        )
        .pip_install(
            "torchvision==0.26.0+cu130",
            extra_options="--index-url https://download.pytorch.org/whl/cu130",
        )
        .pip_install(
            "transformers==5.12.1",
            "accelerate==1.14.0",
            "datasets==5.0.0",
            "numpy==2.2.6",
            "safetensors==0.8.0",
            "sentencepiece==0.2.1",
            "tqdm==4.68.3",
            "PyYAML==6.0.3",
            "pydantic==2.13.4",
            "psutil==7.2.2",
            "requests==2.34.2",
            "openai-harmony==0.0.8",
            "packaging==26.2",
            "einops==0.8.2",
            "ipython==9.15.0",
            "orjson",
            "pillow==12.2.0",
            "partial-json-parser==0.2.1.1.post7",
            "pyzmq",
            "uvloop",
            "uvicorn",
            "fastapi",
            "aiohttp",
            "gguf==0.19.0",
            "msgspec==0.21.1",
            "setproctitle",
            "watchdog",
            "ninja==1.13.0",
            "triton==3.6.0",
            "cuda-python==13.3.1",
            "cuda-tile==1.3.0",
            "cuda-bindings==13.3.1",
            "cuda-core==1.0.1",
            "cuda-pathfinder==1.5.5",
            "apache-tvm-ffi==0.1.9",
            "compressed-tensors==0.17.0",
            "nvidia-ml-py==13.595.45",
            "nvidia-cuda-tileiras==13.2.78",
            "pybase64==1.4.3",
            "openai==2.44.0",
        )
        .pip_install(
            "flashinfer-python==0.6.12",
            "flashinfer-cubin==0.6.12",
            extra_options=(
                "--index-url https://flashinfer.ai/whl "
                "--extra-index-url https://pypi.org/simple --no-deps"
            ),
        )
        .pip_install(
            "flashinfer-jit-cache==0.6.12+cu130",
            extra_options=(
                "--index-url https://flashinfer.ai/whl/cu130 "
                "--extra-index-url https://pypi.org/simple --no-deps"
            ),
        )
        .pip_install(
            "sglang-kernel==0.4.2",
            "yunchang==0.6.4",
            "sglang==0.5.14",
            extra_options="--no-deps",
        )
        .add_local_dir(str(ROOT / "scripts"), remote_path=str(REMOTE_ROOT / "scripts"))
        .add_local_dir(str(ROOT / "src"), remote_path=str(REMOTE_ROOT / "src"))
        .add_local_dir(
            str(ROOT / "externals" / "SSSD" / "python"),
            remote_path=str(REMOTE_ROOT / "externals" / "SSSD" / "python"),
        )
        .add_local_dir(
            str(ROOT / "externals" / "SpecForge"),
            remote_path=str(REMOTE_ROOT / "externals" / "SpecForge"),
        )
        .add_local_file(
            str(ROOT / "data" / "debug" / "modal_phase1_prompts.jsonl"),
            remote_path=str(REMOTE_INPUT),
        )
        .add_local_dir(str(Path("/home/tuantb/models/Qwen3-0.6B")), remote_path=str(REMOTE_MODEL))
    )
    return image


image = _build_image()
app = modal.App("mr-dflash-hidden-cache-b200-debug")


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": ":".join(
                [
                    str(REMOTE_ROOT / "src"),
                    str(REMOTE_ROOT / "scripts"),
                    # Simulate the server's shared SSSD environment. The
                    # SpecForge adapter must remove this shadowing source
                    # before importing the pinned SGLang wheel.
                    str(REMOTE_ROOT / "externals" / "SSSD" / "python"),
                    str(REMOTE_ROOT / "externals" / "SpecForge"),
                ]
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "FAST_INFER_CACHE_ROOT": "/outputs/runtime-cache",
            "FLASHINFER_WORKSPACE_BASE": "/outputs/runtime-cache/flashinfer",
            "TRITON_CACHE_DIR": "/outputs/runtime-cache/triton",
            "TORCH_EXTENSIONS_DIR": "/outputs/runtime-cache/torch_extensions",
            "FI_OFFLINE": "1",
        }
    )
    return env


def _print_runtime_probe(env: dict[str, str]) -> None:
    probe = r'''
import importlib
import json
import subprocess
import sys
import torch

sys.path.insert(0, "/opt/mr_dflash/scripts/mr_dflash")
from specforge_capture import _ensure_specforge_importable

_ensure_specforge_importable()

modules = [
    "transformers", "flashinfer", "flashinfer_cubin", "flashinfer_jit_cache",
    "sgl_kernel", "yunchang", "sglang", "sglang.srt.managers.schedule_batch",
    "specforge",
]
result = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    "modules": {},
}
for name in modules:
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
            "file": getattr(module, "__file__", "unknown"),
        }
    except Exception as exc:
        result["modules"][name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
print(json.dumps(result, default=str, sort_keys=True))
failed = {
    name: value["error"]
    for name, value in result["modules"].items()
    if not value["ok"]
}
if failed:
    raise RuntimeError(f"Modal runtime probe failed: {failed}")
'''
    subprocess.run([sys.executable, "-c", probe], env=env, check=True)
    subprocess.run(["nvidia-smi", "-L"], env=env, check=True)


@app.function(
    image=image,
    gpu="B200",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={"/outputs": artifacts},
)
def run_hidden_cache_smoke() -> dict[str, object]:
    env = _runtime_env()
    output = REMOTE_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    try:
        _print_runtime_probe(env)
        command = [
            sys.executable,
            str(REMOTE_ROOT / "scripts" / "mr_dflash" / "run_phase1_smoke.py"),
            "--input",
            str(REMOTE_INPUT),
            "--output-root",
            str(output),
            "--target-model-path",
            str(REMOTE_MODEL),
            "--num-samples",
            "4",
            "--seed",
            "7",
            "--max-length",
            "1024",
            "--max-new-tokens",
            "16",
            "--target-layer-ids",
            "1",
            "9",
            "17",
            "25",
            "--device",
            "cuda",
            "--torch-dtype",
            "bfloat16",
            "--local-files-only",
            "--resume",
            "--cache-backend",
            "specforge_sglang",
            "--cache-attention-backend",
            "sdpa",
            "--cache-auto-batch",
            "--cache-auto-batch-start-size",
            "1",
            "--cache-auto-batch-safety-fraction",
            "0.95",
            "--cache-auto-batch-target-vram-gb",
            "170",
            "--cache-auto-batch-max-size",
            "4",
            "--cache-io-threads",
            "1",
            "--cache-io-queue-size",
            "2",
        ]
        subprocess.run(command, cwd=str(REMOTE_ROOT), env=env, check=True)
        summary_path = output / "pipeline_summary.json"
        summary = json.loads(summary_path.read_text())
        return {"status": summary.get("status"), "summary": summary}
    finally:
        artifacts.commit()


@app.local_entrypoint()
def main() -> None:
    result = run_hidden_cache_smoke.remote()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
