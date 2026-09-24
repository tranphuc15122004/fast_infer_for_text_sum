#!/usr/bin/env python3
"""Run the FidelityKV E44 teacher-forced scan on the existing Modal GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
VOLUME_MOUNT = Path("/mnt/fast-in")
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "fast-infer-text-sum-cache")
GPU_TYPE = os.environ.get("MODAL_GPU", "A10G")
REMOTE_MODEL = Path("/opt/models/Qwen3-0.6B")
LOCAL_MODEL = Path(os.environ.get("MODAL_RECAP_TARGET_LOCAL", "/home/tuantb/models/Qwen3-0.6B"))
REFERENCE_TRACE = (
    VOLUME_MOUNT
    / "outputs/recap_kv_e43/e43-temporal-pilot-qwen06b-a10g-20260923/temporal_trace.jsonl"
)
DEFAULT_INPUTS = (
    "data/longbench_100_14k/gov_report.jsonl",
    "data/longbench_100_14k/multi_news.jsonl",
)


def _build_image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
        .pip_install_from_requirements(str(ROOT / "requirements.modal.txt"))
        .add_local_dir(ROOT / "src", REMOTE_ROOT / "src", copy=False)
        .add_local_dir(ROOT / "scripts", REMOTE_ROOT / "scripts", copy=False)
        .add_local_dir(ROOT / "data", REMOTE_ROOT / "data", copy=False)
    )
    if LOCAL_MODEL.is_dir():
        image = image.add_local_dir(LOCAL_MODEL, REMOTE_MODEL)
    return image


CACHE_VOLUME = modal.Volume.from_name(VOLUME_NAME)
IMAGE = _build_image()
APP = modal.App("fast-in-text-sum-fidelity-kv")


def ensure_runtime_venv(venv_dir: Path, base_python: str) -> Path:
    python_path = venv_dir / "bin" / "python"
    if not python_path.is_file():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [base_python, "-m", "venv", "--system-site-packages", str(venv_dir)],
            check=True,
        )
    return python_path


def _runtime_environment(python: Path, output_dir: Path) -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REMOTE_ROOT),
        "VIRTUAL_ENV": str(python.parent.parent),
        "PATH": os.pathsep.join((str(python.parent), os.environ.get("PATH", ""))),
        "HF_HOME": str(VOLUME_MOUNT / "hf"),
        "HF_HUB_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
        "TRANSFORMERS_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "FI_FIDELITY_OUTPUT_DIR": str(output_dir),
    }


def _build_command(
    *,
    python: Path,
    output_dir: Path,
    inputs: list[str],
    model: str,
    configs: str,
    max_samples: int,
    max_new_tokens: int,
    seed: int,
    run_id: str,
    sample_ids: str = "",
) -> list[str]:
    command = [
        str(python),
        "-m",
        "src.TrainingFree.fidelity_run",
        "--model",
        model,
        "--reference-trace",
        str(REFERENCE_TRACE),
        "--output",
        str(output_dir),
        "--configs",
        configs,
        "--max-samples",
        str(max_samples),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--run-id",
        run_id,
    ]
    if sample_ids:
        command.extend(("--sample-ids", sample_ids))
    for input_path in inputs:
        command.extend(("--input", input_path))
    return command


def _new_run_id(mode: str) -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"e44-{mode}-qwen06b-a10g-{timestamp}-{uuid.uuid4().hex[:6]}"


@APP.function(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={str(VOLUME_MOUNT): CACHE_VOLUME},
    timeout=6 * 60 * 60,
)
def run_fidelity(
    *,
    mode: str,
    model: str,
    inputs: list[str],
    configs: str,
    max_samples: int,
    max_new_tokens: int,
    seed: int,
    run_id: str = "",
    sample_ids: str = "",
) -> dict[str, object]:
    if mode not in {"smoke", "pilot", "full", "joint"}:
        raise ValueError("mode must be smoke, pilot, full, or joint")
    selected_run_id = run_id or _new_run_id(mode)
    output_dir = VOLUME_MOUNT / "outputs/fidelity_kv" / selected_run_id
    runtime_python = ensure_runtime_venv(VOLUME_MOUNT / "venv", sys.executable)
    command = _build_command(
        python=runtime_python,
        output_dir=output_dir,
        inputs=inputs,
        model=model,
        configs=configs,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        seed=seed,
        run_id=selected_run_id,
        sample_ids=sample_ids,
    )
    environment = dict(os.environ)
    environment.update(_runtime_environment(runtime_python, output_dir))
    print(f"[modal] gpu={GPU_TYPE} volume={VOLUME_NAME} run_id={selected_run_id}", flush=True)
    print("[modal] command:", " ".join(shlex.quote(part) for part in command), flush=True)
    process = subprocess.run(command, cwd=str(REMOTE_ROOT), env=environment)
    try:
        CACHE_VOLUME.commit()
    except Exception as exc:
        print(f"[modal] volume commit warning: {exc}", file=sys.stderr, flush=True)
    manifest_path = output_dir / "manifest.json"
    result: dict[str, object] = {
        "status": "success" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "run_id": selected_run_id,
        "output_dir": str(output_dir),
        "volume": VOLUME_NAME,
        "gpu": GPU_TYPE,
        "python": str(runtime_python),
    }
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result["manifest"] = manifest
        except (OSError, json.JSONDecodeError):
            result["manifest_readable"] = False
    print("[modal] result:", json.dumps(result, ensure_ascii=False), flush=True)
    return result


@APP.local_entrypoint()
def main(
    mode: str = "smoke",
    model: str = str(REMOTE_MODEL),
    inputs: str = "",
    configs: str = "K16V16,K16V8,K16V4,K8V16,K4V16",
    max_samples: int | None = None,
    max_new_tokens: int | None = None,
    seed: int = 42,
    sample_ids: str = "",
    run_id: str = "",
) -> None:
    if mode not in {"smoke", "pilot", "full", "joint"}:
        raise SystemExit("mode must be smoke, pilot, full, or joint")
    if max_samples is None:
        max_samples = 1 if mode == "smoke" else (2 if mode == "pilot" else 20)
    if max_new_tokens is None:
        max_new_tokens = 4 if mode == "smoke" else 32
    selected_inputs = [item.strip() for item in inputs.split(",") if item.strip()]
    if not selected_inputs:
        selected_inputs = list(DEFAULT_INPUTS)
    if mode == "smoke":
        selected_inputs = selected_inputs[:1]
    result = run_fidelity.remote(
        mode=mode,
        model=model,
        inputs=selected_inputs,
        configs=configs,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        seed=seed,
        run_id=run_id,
        sample_ids=sample_ids,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("returncode", 1) != 0:
        raise SystemExit(int(result["returncode"]))


if __name__ == "__main__":
    main()
