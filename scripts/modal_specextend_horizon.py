#!/usr/bin/env python3
"""Run the SpecExtend Horizon-CMR fidelity gate on a real Modal GPU.

This runner is intentionally separate from the repository-wide LongBench
runner.  SpecExtend's classic implementation was released against an older
Transformers/Torch stack, so the image is pinned close to its upstream
requirements.  The first jobs are reproduction gates only; no horizon
intervention is enabled by this file.

Examples::

    MODAL_GPU=A100-40GB modal run scripts/modal_specextend_horizon.py \
      --stage h0-1k --samples 10 --max-gen-tokens 128

    MODAL_GPU=A100-40GB modal run scripts/modal_specextend_horizon.py \
      --stage h0-4k --samples 5 --max-gen-tokens 256

    MODAL_GPU=A100-40GB modal run scripts/modal_specextend_horizon.py \
      --stage d0-prefix-4k --samples 10 --max-gen-tokens 256 \
      --trace-prefix

    MODAL_GPU=A100-40GB modal run scripts/modal_specextend_horizon.py \
      --stage h1-4k --policy recent32 --samples 10 --max-gen-tokens 256

The public Vicuna checkpoints are downloaded into a persistent Modal Volume.
The job returns a manifest, while raw JSONL/log/trace/report artifacts remain
in that Volume and can be copied with ``modal volume get``.
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
REMOTE_OUTPUT_ROOT = VOLUME_MOUNT / "outputs" / "specextend_horizon_modal"
REMOTE_D0_D3_ROOT = VOLUME_MOUNT / "outputs" / "specextend_d0_d3"
REMOTE_HF_HOME = VOLUME_MOUNT / "hf"

TARGET_REPO = "lmsys/vicuna-7b-v1.5-16k"
DRAFT_REPO = "double7/vicuna-68m"
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "fast-infer-text-sum-cache")
GPU_TYPE = os.environ.get("MODAL_GPU", "A100-40GB")


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes"}


def _build_image() -> modal.Image:
    # SpecExtend's published requirements use torch 2.4/cu121 and
    # transformers 4.41.  Keeping this stack in a CUDA registry image avoids
    # silently installing CPU Torch from PyPI inside Modal.
    image = modal.Image.from_registry(
        os.environ.get(
            "MODELEXTEND_IMAGE",
            "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        )
    )
    image = image.apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
    image = image.pip_install(
        "accelerate==0.21.0",
        "huggingface_hub>=0.23,<1.0",
        "safetensors>=0.4.3",
        "sentencepiece==0.1.99",
        "termcolor==2.4.0",
        "tqdm==4.67.1",
        "transformers==4.41.0",
    )
    # FlashAttention is optional for the fidelity gate.  Omitting it keeps the
    # image reproducible and lets the official eager fallback run on A100.
    image = image.add_local_dir(
        ROOT / "scripts",
        str(REMOTE_ROOT / "scripts"),
        copy=False,
        ignore=("__pycache__", "*.pyc", ".pytest_cache"),
    )
    image = image.add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=False,
        ignore=("__pycache__", "*.pyc", ".pytest_cache"),
    )
    return image.add_local_dir(
        ROOT / "externals" / "SpecExtend",
        str(REMOTE_ROOT / "externals" / "SpecExtend"),
        copy=False,
        ignore=("__pycache__", "*.pyc", ".git"),
    )


VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
IMAGE = _build_image()
app = modal.App("fast-infer-text-sum-specextend-horizon")


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"modal-{stamp}-{uuid.uuid4().hex[:8]}"


def _download_models() -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    cache_dir = str(REMOTE_HF_HOME / "hub")
    base = snapshot_download(TARGET_REPO, cache_dir=cache_dir)
    draft = snapshot_download(DRAFT_REPO, cache_dir=cache_dir)
    return base, draft


def _stage_config(stage: str) -> dict[str, int | str]:
    values: dict[str, dict[str, int | str]] = {
        "h0-1k": {"level": "1k", "input": 1024, "samples": 10, "gen": 128},
        "h0-4k": {"level": "4k", "input": 4096, "samples": 5, "gen": 256},
        "h1-4k": {"level": "4k", "input": 4096, "samples": 10, "gen": 256},
        "d0-prefix-4k": {"level": "4k", "input": 4096, "samples": 10, "gen": 256},
        "d2-4k": {"level": "4k", "input": 4096, "samples": 5, "gen": 256},
        "d3-1k": {"level": "1k", "input": 1024, "samples": 10, "gen": 128},
        "d3-2k": {"level": "2k", "input": 2048, "samples": 10, "gen": 256},
        "d3-4k": {"level": "4k", "input": 4096, "samples": 10, "gen": 256},
    }
    if stage not in values:
        raise ValueError(f"Unsupported stage: {stage}; choose {sorted(values)}")
    return values[stage]


@app.function(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={str(VOLUME_MOUNT): VOLUME},
    timeout=24 * 60 * 60,
    env={
        "HF_HOME": str(REMOTE_HF_HOME),
        "HF_HUB_CACHE": str(REMOTE_HF_HOME / "hub"),
        "TRANSFORMERS_CACHE": str(REMOTE_HF_HOME / "hub"),
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "PYTHONUNBUFFERED": "1",
    },
)
def run_stage(
    *,
    stage: str = "h0-1k",
    samples: int = -1,
    max_gen_tokens: int = -1,
    warmup_runs: int = 0,
    policy: str = "cmr32",
    trace_prefix: bool = False,
    run_id: str = "",
) -> dict[str, Any]:
    config = _stage_config(stage)
    if samples > 0:
        config["samples"] = samples
    if max_gen_tokens > 0:
        config["gen"] = max_gen_tokens
    selected_run_id = run_id or _run_id()
    run_root = REMOTE_OUTPUT_ROOT / selected_run_id
    run_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "status": "started",
        "stage": stage,
        "run_id": selected_run_id,
        "gpu": GPU_TYPE,
        "target_repo": TARGET_REPO,
        "draft_repo": DRAFT_REPO,
        "python": sys.executable,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "retrieval_policy": policy,
    }
    manifest_path = run_root / "modal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    try:
        base_model, draft_model = _download_models()
        manifest["target_snapshot"] = base_model
        manifest["draft_snapshot"] = draft_model
        command = [
            "bash",
            str(REMOTE_ROOT / "scripts" / "run_specextend_horizon.sh"),
            str(config["level"]),
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "HORIZON_OUTPUT_DIR": str(run_root),
                "HORIZON_PYTHON": sys.executable,
                "HORIZON_SAMPLES": str(config["samples"]),
                "HORIZON_INPUT_LIMIT": str(config["input"]),
                "HORIZON_MAX_NEW_TOKENS": str(config["gen"]),
                "HORIZON_WARMUP_RUNS": str(warmup_runs),
                # Synchronize CUDA around stage boundaries so H0.5 timing
                # decomposition is measured rather than host-side launch
                # time. This adds a small diagnostic overhead intentionally.
                "SPECEXTEND_TIMING": "1",
                "SPECEXTEND_RETRIEVAL_POLICY": policy,
                "SPECEXTEND_TRACE_PREFIX": "1" if trace_prefix else "0",
                "SPECEXTEND_BASE_MODEL": base_model,
                "SPECEXTEND_DRAFT_MODEL": draft_model,
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            }
        )
        log_path = run_root / "modal_job.log"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(
                "$ "
                + " ".join(shlex.quote(item) for item in command)
                + "\n"
            )
            process = subprocess.run(
                command,
                cwd=str(REMOTE_ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        manifest.update(
            {
                "status": "success" if process.returncode == 0 else "failed",
                "returncode": process.returncode,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "log": str(log_path),
                "result": str(run_root / f"spec_extend_{config['level']}.jsonl"),
                "report": str(run_root / "reports" / f"{config['level']}_report.md"),
                "trace": str(run_root / "traces" / f"spec_extend_{config['level']}.jsonl"),
            }
        )
    except Exception as exc:  # preserve a diagnosable artifact on Modal errors
        manifest.update(
            {
                "status": "blocked",
                "returncode": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    VOLUME.commit()
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


@app.function(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={str(VOLUME_MOUNT): VOLUME},
    timeout=24 * 60 * 60,
    env={
        "HF_HOME": str(REMOTE_HF_HOME),
        "HF_HUB_CACHE": str(REMOTE_HF_HOME / "hub"),
        "TRANSFORMERS_CACHE": str(REMOTE_HF_HOME / "hub"),
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "PYTHONUNBUFFERED": "1",
    },
)
def run_prefix_replay(
    *,
    state_bank: str,
    output: str,
    max_docs: int = 10,
    cycles_per_doc: int = 3,
    policies: str = "cmr32,recent32,shuffled32,full",
    run_id: str = "",
) -> dict[str, Any]:
    """Run matched-prefix replay from a trace state bank on the same GPU image."""
    selected_run_id = run_id or _run_id()
    # A previous Modal function may have committed a newer Volume snapshot
    # after this app image was initialized.  Reload the named Volume before
    # handing the state bank to the child process.
    VOLUME.reload()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_model, draft_model = _download_models()
    command = [
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "replay_specextend_prefix.py"),
        "--state-bank", state_bank,
        "--output", str(output_path),
        "--target-model", base_model,
        "--draft-model", draft_model,
        "--max-docs", str(max_docs),
        "--cycles-per-doc", str(cycles_per_doc),
        "--policies", policies,
    ]
    manifest_path = output_path.with_suffix(".manifest.json")
    log_path = output_path.with_suffix(".log")
    manifest: dict[str, Any] = {
        "status": "started",
        "run_id": selected_run_id,
        "gpu": GPU_TYPE,
        "state_bank": state_bank,
        "output": str(output_path),
        "policies": policies,
        "max_docs": max_docs,
        "cycles_per_doc": cycles_per_doc,
        "target_repo": TARGET_REPO,
        "draft_repo": DRAFT_REPO,
        "python": sys.executable,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    process = None
    try:
        process = subprocess.run(
            command,
            cwd=str(REMOTE_ROOT),
            env={**os.environ, "SPECEXTEND_TIMING": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_path.write_text(
            "$ " + " ".join(shlex.quote(item) for item in command) + "\n"
            + (process.stdout or ""),
            encoding="utf-8",
        )
        manifest.update({
            "status": "success" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "log": str(log_path),
            "stdout_tail": (process.stdout or "")[-4000:],
        })
        print((process.stdout or "")[-4000:], flush=True)
    except Exception as exc:
        manifest.update({
            "status": "blocked",
            "returncode": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        print(f"[replay runner exception] {type(exc).__name__}: {exc}", flush=True)
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        VOLUME.commit()
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


@app.function(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={str(VOLUME_MOUNT): VOLUME},
    timeout=24 * 60 * 60,
    env={
        "HF_HOME": str(REMOTE_HF_HOME),
        "HF_HUB_CACHE": str(REMOTE_HF_HOME / "hub"),
        "TRANSFORMERS_CACHE": str(REMOTE_HF_HOME / "hub"),
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "PYTHONUNBUFFERED": "1",
    },
)
def run_state_and_replay(
    *,
    state_run_id: str,
    replay_run_id: str,
    samples: int = 10,
    cycles_per_doc: int = 3,
    policies: str = "cmr32,recent32,shuffled32,full",
    intervention: bool = False,
) -> dict[str, Any]:
    """Create a fresh prefix bank and replay it in one Modal container.

    Keeping both subprocesses in one container avoids relying on a newly
    started container observing a Volume commit made by an earlier function
    call.  The state bank never leaves the configured Volume.
    """
    base_model, draft_model = _download_models()
    state_root = REMOTE_OUTPUT_ROOT / state_run_id
    state_root.mkdir(parents=True, exist_ok=True)
    config = {"level": "4k", "input": 4096, "samples": samples, "gen": 256}
    state_result = state_root / "spec_extend_4k.jsonl"
    state_trace = state_root / "traces" / "spec_extend_4k.jsonl"
    state_preflight = state_root / "preflight_4k.json"
    state_log = state_root / "modal_job.log"
    state_report = state_root / "reports" / "4k_report.md"
    command = [
        "bash", str(REMOTE_ROOT / "scripts" / "run_specextend_horizon.sh"), "4k"
    ]
    env = dict(os.environ)
    env.update({
        "HORIZON_OUTPUT_DIR": str(state_root),
        "HORIZON_PYTHON": sys.executable,
        "HORIZON_SAMPLES": str(samples),
        "HORIZON_INPUT_LIMIT": "4096",
        "HORIZON_MAX_NEW_TOKENS": "256",
        "HORIZON_WARMUP_RUNS": "1",
        "SPECEXTEND_TIMING": "1",
        "SPECEXTEND_RETRIEVAL_POLICY": "cmr32",
        "SPECEXTEND_TRACE_PREFIX": "1",
        "SPECEXTEND_BASE_MODEL": base_model,
        "SPECEXTEND_DRAFT_MODEL": draft_model,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
    state_started = time.monotonic()
    with state_log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(item) for item in command) + "\n")
        state_process = subprocess.run(
            command, cwd=str(REMOTE_ROOT), env=env,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    state_status = "success" if state_process.returncode == 0 else "failed"
    replay_root = REMOTE_D0_D3_ROOT / replay_run_id
    replay_root.mkdir(parents=True, exist_ok=True)
    replay_output = replay_root / "replay.jsonl"
    replay_log = replay_root / "replay.log"
    replay_manifest = replay_root / "replay.manifest.json"
    manifest: dict[str, Any] = {
        "status": "started",
        "gpu": GPU_TYPE,
        "state_run_id": state_run_id,
        "replay_run_id": replay_run_id,
        "state_config": config,
        "state_status": state_status,
        "state_result": str(state_result),
        "state_trace": str(state_trace),
        "replay_output": str(replay_output),
        "policies": policies,
        "cycles_per_doc": cycles_per_doc,
        "intervention": intervention,
        "target_repo": TARGET_REPO,
        "draft_repo": DRAFT_REPO,
        "python": sys.executable,
        "state_elapsed_seconds": round(time.monotonic() - state_started, 3),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if state_process.returncode != 0 or not state_trace.exists():
        manifest.update({
            "status": "failed",
            "returncode": state_process.returncode,
            "reason": "state-bank generation failed or trace missing",
        })
        manifest_path = replay_root / "combined.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        VOLUME.commit()
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
        return manifest

    replay_script = (
        "intervene_specextend_cycle.py"
        if intervention else "replay_specextend_prefix.py"
    )
    replay_command = [
        sys.executable, str(REMOTE_ROOT / "scripts" / replay_script),
        "--state-bank", str(state_trace),
        "--output", str(replay_output),
        "--target-model", base_model,
        "--draft-model", draft_model,
        "--max-docs", str(samples),
        "--cycles-per-doc", str(cycles_per_doc),
        "--policies", policies,
    ]
    if intervention:
        replay_command += [
            "--input-file",
            str(REMOTE_ROOT / "externals" / "SpecExtend" / "specextend" / "data" / "govreport" / "govreport_4K.jsonl"),
            "--max-input-tokens", "4096",
        ]
    replay_started = time.monotonic()
    with replay_log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(item) for item in replay_command) + "\n")
        replay_process = subprocess.run(
            replay_command, cwd=str(REMOTE_ROOT), env=env,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    manifest.update({
        "status": "success" if replay_process.returncode == 0 else "failed",
        "returncode": replay_process.returncode,
        "replay_elapsed_seconds": round(time.monotonic() - replay_started, 3),
        "state_report": str(state_report),
        "state_preflight": str(state_preflight),
        "state_log": str(state_log),
        "replay_log": str(replay_log),
        "replay_manifest": str(replay_manifest),
    })
    manifest_path = replay_root / "combined.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    replay_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    VOLUME.commit()
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


@app.local_entrypoint()
def main(
    stage: str = "h0-1k",
    samples: int = -1,
    max_gen_tokens: int = -1,
    warmup_runs: int = 0,
    policy: str = "cmr32",
    trace_prefix: bool = False,
    run_id: str = "",
    replay_state_bank: str = "",
    replay_output: str = "",
    replay_max_docs: int = 10,
    replay_cycles_per_doc: int = 3,
    replay_policies: str = "cmr32,recent32,shuffled32,full",
    state_and_replay: bool = False,
    cycle_intervention: bool = False,
    state_run_id: str = "d0-prefix-bank-combined-4k-10",
    replay_run_id: str = "d1-matched-prefix-combined-4k-10",
) -> None:
    if state_and_replay:
        result = run_state_and_replay.remote(
            state_run_id=state_run_id,
            replay_run_id=replay_run_id,
            samples=replay_max_docs,
            cycles_per_doc=replay_cycles_per_doc,
            policies=replay_policies,
            intervention=cycle_intervention,
        )
    elif replay_state_bank:
        if not replay_output:
            raise SystemExit("--replay-output is required with --replay-state-bank")
        result = run_prefix_replay.remote(
            state_bank=replay_state_bank,
            output=replay_output,
            max_docs=replay_max_docs,
            cycles_per_doc=replay_cycles_per_doc,
            policies=replay_policies,
            run_id=run_id,
        )
    else:
        result = run_stage.remote(
            stage=stage,
            samples=samples,
            max_gen_tokens=max_gen_tokens,
            warmup_runs=warmup_runs,
            policy=policy,
            trace_prefix=trace_prefix,
            run_id=run_id,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") != "success":
        raise SystemExit(1)
