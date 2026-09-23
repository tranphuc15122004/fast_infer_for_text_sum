#!/usr/bin/env python3
"""Kiểm chứng auto-pause của launcher Phase 1 trên GPU Modal B200.

Runner này dùng chính ``run_b200_vllm_phase1.sh`` trong container Modal:
  1. tạo prepared train/val/test nhỏ trên Volume;
  2. chạy lần đầu với timer 1 phút;
  3. xác nhận ``stop_parallel`` được tạo và vLLM được dừng;
  4. chạy lại cùng output root với timer tắt để kiểm tra ``--resume``.

Đây là smoke test runtime, không phải benchmark throughput/quality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # local contract tests do not need Modal SDK
    modal = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
VOLUME_MOUNT = Path("/mnt/mr-dflash")
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_GPU = os.environ.get("MODAL_GPU", "B200")
DEFAULT_VOLUME = "fast-infer-text-sum-mr-dflash-vllm"
SECRET_NAME = os.environ.get("MODAL_HF_SECRET", "").strip()


def build_launcher_command(
    *,
    launcher: Path,
    run_root: Path,
    prepared_root: Path,
    target_model: Path,
    pause_minutes: int,
) -> list[str]:
    """Build the exact production launcher command used by the remote test."""

    return [
        "bash",
        str(launcher),
        "2",
        "--run-root",
        str(run_root),
        "--prepared-root",
        str(prepared_root),
        "--target-model",
        str(target_model),
        "--auto-pause-minutes",
        str(int(pause_minutes)),
    ]


def _build_image() -> Any:
    if modal is None:
        return None
    return (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.1-devel-ubuntu24.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("curl", "libnuma1", "libnuma-dev")
        .pip_install(
            "torch==2.11.0+cu130",
            extra_options="--index-url https://download.pytorch.org/whl/cu130",
        )
        .pip_install_from_requirements(str(ROOT / "requirements.modal.txt"))
        .pip_install("vllm==0.24.0")
        .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=False)
        .add_local_dir(
            ROOT / "scripts" / "mr_dflash",
            str(REMOTE_ROOT / "scripts" / "mr_dflash"),
            copy=False,
        )
    )


def _ensure_model(model_id: str, destination: Path) -> Path:
    if destination.is_dir() and (destination / "config.json").is_file():
        return destination
    from huggingface_hub import snapshot_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(destination))
    return destination


def _write_prepared_dataset(root: Path, *, rows_per_split: int, prompt_repeats: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    text = (
        "This synthetic scientific article describes a reproducible method, "
        "experiments, limitations, and conclusions. Summarize it faithfully.\n"
    ) * int(prompt_repeats)
    for split in ("train", "val", "test"):
        path = root / "normalized" / f"{split}_prompts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for index in range(int(rows_per_split)):
                row = {
                    "id": f"modal_timer_{split}_{index}",
                    "source": "modal_synthetic",
                    "conversations": [{"role": "user", "content": text}],
                    "metadata": {"split": split, "synthetic": True},
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_launcher(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-80:])
    return {
        "returncode": result.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path),
        "tail": tail,
    }


def _gpu_info() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


VOLUME_NAME = os.environ.get("MODAL_MR_DFLASH_TIMER_VOLUME", DEFAULT_VOLUME)
GPU_TYPE = os.environ.get("MODAL_GPU", DEFAULT_GPU)


class _NoopApp:
    def function(self, **_kwargs):
        return lambda fn: fn

    def local_entrypoint(self):
        return lambda fn: fn


if modal is not None:
    IMAGE = _build_image()
    VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    SECRETS = [modal.Secret.from_name(SECRET_NAME)] if SECRET_NAME else []
    app = modal.App("mr-dflash-phase1-timer-smoke")
else:
    IMAGE = None
    VOLUME = None
    SECRETS = []
    app = _NoopApp()


@app.function(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={str(VOLUME_MOUNT): VOLUME},
    secrets=SECRETS,
    timeout=6 * 60 * 60,
)
def run_smoke(
    *,
    target_model: str = DEFAULT_MODEL,
    rows_per_split: int = 256,
    prompt_repeats: int = 12,
    pause_minutes: int = 1,
    run_id: str = "",
) -> dict[str, Any]:
    if rows_per_split < 1 or prompt_repeats < 1 or pause_minutes < 1:
        raise ValueError("rows_per_split/prompt_repeats/pause_minutes phải >= 1")
    run_id = run_id or f"modal-phase1-timer-{uuid.uuid4().hex[:10]}"
    run_root = VOLUME_MOUNT / "runs" / run_id
    prepared_root = VOLUME_MOUNT / "prepared" / run_id
    model_path = VOLUME_MOUNT / "models" / "Qwen3-4B"
    run_root.mkdir(parents=True, exist_ok=False)
    _write_prepared_dataset(
        prepared_root,
        rows_per_split=rows_per_split,
        prompt_repeats=prompt_repeats,
    )
    model_path = _ensure_model(target_model, model_path)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": os.pathsep.join(
                (str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts" / "mr_dflash"))
            ),
            "HF_HOME": str(VOLUME_MOUNT / "hf"),
            "HF_HUB_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
            "TRANSFORMERS_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
            "PHASE1_CONTEXT_LENGTH": "1024",
            "PHASE1_MAX_NEW_TOKENS": "256",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
            "VLLM_BATCH_SIZE": "1",
            "VLLM_BATCH_TOKENS": "1024",
            "VLLM_BATCH_START_SIZE": "1",
            "VLLM_BATCH_START_TOKENS": "1024",
            "CACHE_BATCH_SIZE_3K": "1",
            "CACHE_BATCH_SIZE_LONG_CONTEXT": "1",
            "CACHE_IO_THREADS": "1",
            "CACHE_IO_QUEUE_SIZE": "2",
            "VLLM_PORT": "38147",
            "PYTHONUNBUFFERED": "1",
        }
    )
    launcher = REMOTE_ROOT / "scripts" / "mr_dflash" / "run_b200_vllm_phase1.sh"
    first_command = build_launcher_command(
        launcher=launcher,
        run_root=run_root,
        prepared_root=prepared_root,
        target_model=model_path,
        pause_minutes=pause_minutes,
    )
    first = _run_launcher(
        first_command,
        cwd=REMOTE_ROOT,
        env=env,
        log_path=run_root / "logs" / "timer_first.log",
    )
    stop_file = run_root / "stop_parallel"
    paused = stop_file.is_file()
    if not paused:
        raise RuntimeError(
            "timer không tạo stop_parallel; tăng rows_per_split hoặc prompt_repeats "
            f"và xem log {first['log']}\n{first['tail']}"
        )

    resume_command = build_launcher_command(
        launcher=launcher,
        run_root=run_root,
        prepared_root=prepared_root,
        target_model=model_path,
        pause_minutes=0,
    )
    resume = _run_launcher(
        resume_command,
        cwd=REMOTE_ROOT,
        env=env,
        log_path=run_root / "logs" / "timer_resume.log",
    )
    report = {
        "status": "success" if resume["returncode"] == 0 else "failed",
        "gpu": _gpu_info(),
        "run_id": run_id,
        "target_model": str(model_path),
        "rows_per_split": rows_per_split,
        "prompt_repeats": prompt_repeats,
        "pause_minutes": pause_minutes,
        "first_run": {**first, "stop_file_created": paused},
        "resume_run": resume,
        "run_root": str(run_root),
        "prepared_root": str(prepared_root),
    }
    (run_root / "timer_smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    VOLUME.commit()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if report["status"] != "success":
        raise RuntimeError(f"resume run failed: {resume['tail']}")
    return report


@app.local_entrypoint()
def main(
    target_model: str = DEFAULT_MODEL,
    rows_per_split: int = 256,
    prompt_repeats: int = 12,
    pause_minutes: int = 1,
    run_id: str = "",
) -> None:
    result = run_smoke.remote(
        target_model=target_model,
        rows_per_split=rows_per_split,
        prompt_repeats=prompt_repeats,
        pause_minutes=pause_minutes,
        run_id=run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("status") != "success":
        raise SystemExit(1)
