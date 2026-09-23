#!/usr/bin/env python3
"""Run RECAP-KV V0 on Modal with a persistent Python 3.12 venv.

The default target is the existing local Qwen3-0.6B checkpoint.  This runner
does not download a new model: pass a checkpoint path already present in the
Modal Volume with ``--target-model`` when using another target.
"""

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
REMOTE_OUTPUT_ROOT_V2 = VOLUME_MOUNT / "outputs" / "recap_kv_v2"
REMOTE_OUTPUT_ROOT_V3 = VOLUME_MOUNT / "outputs" / "recap_kv_v3"
REMOTE_OUTPUT_ROOT_E43 = VOLUME_MOUNT / "outputs" / "recap_kv_e43"
REMOTE_TARGET_MODEL = Path("/opt/models/Qwen3-0.6B")
LOCAL_TARGET_MODEL = Path(
    os.environ.get("MODAL_RECAP_TARGET_LOCAL", "/home/tuantb/models/Qwen3-0.6B")
)
DEFAULT_VOLUME = "fast-infer-text-sum-cache"
DEFAULT_GPU = "A100-80GB"
DEFAULT_SMOKE_INPUTS = ("data/representative_100/govreport_representative.jsonl",)
DEFAULT_PILOT_INPUTS = (
    "data/longbench_100_14k/gov_report.jsonl",
    "data/longbench_100_14k/multi_news.jsonl",
)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes"}


def _build_image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
        .pip_install_from_requirements(str(ROOT / "requirements.modal.txt"))
        .add_local_dir(ROOT / "src", REMOTE_ROOT / "src", copy=False)
        .add_local_dir(ROOT / "scripts", REMOTE_ROOT / "scripts", copy=False)
        .add_local_dir(ROOT / "data", REMOTE_ROOT / "data", copy=False)
    )
    if LOCAL_TARGET_MODEL.is_dir():
        image = image.add_local_dir(LOCAL_TARGET_MODEL, REMOTE_TARGET_MODEL)
    return image


VOLUME_NAME = os.environ.get("MODAL_VOLUME", DEFAULT_VOLUME)
GPU_TYPE = os.environ.get("MODAL_GPU", DEFAULT_GPU)
CACHE_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
IMAGE = _build_image()
app = modal.App("fast-in-text-sum-recap-kv-v0")


def ensure_runtime_venv(venv_dir: Path, base_python: str) -> Path:
    """Create or reuse the Volume-backed venv using image site packages."""

    python_path = venv_dir / "bin" / "python"
    if not python_path.is_file():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [base_python, "-m", "venv", "--system-site-packages", str(venv_dir)],
            check=True,
        )
    return python_path


def build_runtime_env(*, target_model: str, output_dir: Path, python: Path) -> dict[str, str]:
    """Build an offline environment for local/Volume checkpoints."""

    venv_dir = str(python.parent.parent)
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REMOTE_ROOT),
        "VIRTUAL_ENV": venv_dir,
        "PATH": os.pathsep.join((str(python.parent), os.environ.get("PATH", ""))),
        "HF_HOME": str(VOLUME_MOUNT / "hf"),
        "HF_HUB_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
        "TRANSFORMERS_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "FI_RECAP_TARGET_MODEL": str(target_model),
        "FI_RECAP_OUTPUT_DIR": str(output_dir),
    }


def build_runner_command(
    *,
    target_model: str,
    inputs: list[str],
    output_dir: Path,
    max_samples: int,
    max_new_tokens: int,
    smoke: bool,
    python: str,
    experiment: str = "recap",
    region_size: int = 1024,
    block_size: int = 128,
    reps_per_block: int = 4,
    reps_per_region: int = 4,
    hierarchy_mass_budget: float = 0.01,
    hierarchy_layers: str = "last",
    temporal_lags: str = "1,2,4,8,16",
    temporal_budgets: str = "0.1,0.2,0.3,0.4",
    temporal_alphas: str = "1.0,1.25,1.5",
    temporal_refresh_intervals: str = "2,4,8,16",
    temporal_block_sizes: str = "16,32,64",
    temporal_mass_level: float = 0.95,
) -> list[str]:
    command = [
        python,
        "-m",
        "src.TrainingFree.run",
        "--experiment",
        experiment,
        "--model",
        target_model,
        "--output",
        str(output_dir),
        "--max-samples",
        str(max_samples),
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--region-size",
        str(region_size),
        "--block-size",
        str(block_size),
        "--reps-per-block",
        str(reps_per_block),
        "--reps-per-region",
        str(reps_per_region),
        "--hierarchy-mass-budget",
        str(hierarchy_mass_budget),
        "--hierarchy-layers",
        hierarchy_layers,
        "--temporal-lags",
        temporal_lags,
        "--temporal-budgets",
        temporal_budgets,
        "--temporal-alphas",
        temporal_alphas,
        "--temporal-refresh-intervals",
        temporal_refresh_intervals,
        "--temporal-block-sizes",
        temporal_block_sizes,
        "--temporal-mass-level",
        str(temporal_mass_level),
    ]
    for input_path in inputs:
        command.extend(("--input", input_path))
    if smoke:
        command.append("--smoke")
    return command


def resolve_options(mode: str, max_samples: int | None, max_new_tokens: int | None) -> tuple[int, int, bool]:
    if mode not in {"smoke", "pilot"}:
        raise ValueError("mode must be smoke or pilot")
    smoke = mode == "smoke"
    return (
        int(max_samples if max_samples is not None else (1 if smoke else 20)),
        int(max_new_tokens if max_new_tokens is not None else (32 if smoke else 128)),
        smoke,
    )


def _new_run_id(mode: str) -> str:
    return f"{mode}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def output_root_for_experiment(experiment: str) -> Path:
    if experiment == "temporal":
        return REMOTE_OUTPUT_ROOT_E43
    if experiment == "hierarchy":
        return REMOTE_OUTPUT_ROOT_V3
    if experiment in {"recap", "lease"}:
        return REMOTE_OUTPUT_ROOT_V2
    raise ValueError("unsupported experiment")


@app.function(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={str(VOLUME_MOUNT): CACHE_VOLUME},
    timeout=6 * 60 * 60,
)
def run_recap(
    *,
    mode: str,
    experiment: str,
    target_model: str,
    inputs: list[str],
    max_samples: int | None = None,
    max_new_tokens: int | None = None,
    run_id: str = "",
    region_size: int = 1024,
    block_size: int = 128,
    reps_per_block: int = 4,
    reps_per_region: int = 4,
    hierarchy_mass_budget: float = 0.01,
    hierarchy_layers: str = "last",
    temporal_lags: str = "1,2,4,8,16",
    temporal_budgets: str = "0.1,0.2,0.3,0.4",
    temporal_alphas: str = "1.0,1.25,1.5",
    temporal_refresh_intervals: str = "2,4,8,16",
    temporal_block_sizes: str = "16,32,64",
    temporal_mass_level: float = 0.95,
) -> dict[str, object]:
    sample_limit, token_limit, smoke = resolve_options(mode, max_samples, max_new_tokens)
    selected_run_id = run_id or _new_run_id(mode)
    output_dir = output_root_for_experiment(experiment) / selected_run_id
    runtime_python = ensure_runtime_venv(VOLUME_MOUNT / "venv", sys.executable)
    command = build_runner_command(
        target_model=target_model,
        inputs=inputs,
        output_dir=output_dir,
        max_samples=sample_limit,
        max_new_tokens=token_limit,
        smoke=smoke,
        python=str(runtime_python),
        experiment=experiment,
        region_size=region_size,
        block_size=block_size,
        reps_per_block=reps_per_block,
        reps_per_region=reps_per_region,
        hierarchy_mass_budget=hierarchy_mass_budget,
        hierarchy_layers=hierarchy_layers,
        temporal_lags=temporal_lags,
        temporal_budgets=temporal_budgets,
        temporal_alphas=temporal_alphas,
        temporal_refresh_intervals=temporal_refresh_intervals,
        temporal_block_sizes=temporal_block_sizes,
        temporal_mass_level=temporal_mass_level,
    )
    environment = dict(os.environ)
    environment.update(
        build_runtime_env(
            target_model=target_model,
            output_dir=output_dir,
            python=runtime_python,
        )
    )
    print("[modal] gpu:", GPU_TYPE, flush=True)
    print("[modal] volume:", VOLUME_NAME, flush=True)
    print("[modal] command:", " ".join(shlex.quote(item) for item in command), flush=True)
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
        "manifest": str(manifest_path),
        "volume": VOLUME_NAME,
        "python": str(runtime_python),
    }
    if manifest_path.is_file():
        try:
            result["manifest_data"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result["manifest_readable"] = False
    print("[modal] result:", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--experiment", choices=("recap", "lease", "hierarchy", "temporal"), default="lease")
    parser.add_argument("--target-model", default=str(REMOTE_TARGET_MODEL))
    parser.add_argument("--inputs", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--region-size", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--reps-per-block", type=int, default=4)
    parser.add_argument("--reps-per-region", type=int, default=4)
    parser.add_argument("--hierarchy-mass-budget", type=float, default=0.01)
    parser.add_argument("--hierarchy-layers", default="last")
    parser.add_argument("--temporal-lags", default="1,2,4,8,16")
    parser.add_argument("--temporal-budgets", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--temporal-alphas", default="1.0,1.25,1.5")
    parser.add_argument("--temporal-refresh-intervals", default="2,4,8,16")
    parser.add_argument("--temporal-block-sizes", default="16,32,64")
    parser.add_argument("--temporal-mass-level", type=float, default=0.95)
    return parser


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    experiment: str = "lease",
    target_model: str = str(REMOTE_TARGET_MODEL),
    inputs: str = "",
    max_samples: int | None = None,
    max_new_tokens: int | None = None,
    run_id: str = "",
    region_size: int = 1024,
    block_size: int = 128,
    reps_per_block: int = 4,
    reps_per_region: int = 4,
    hierarchy_mass_budget: float = 0.01,
    hierarchy_layers: str = "last",
    temporal_lags: str = "1,2,4,8,16",
    temporal_budgets: str = "0.1,0.2,0.3,0.4",
    temporal_alphas: str = "1.0,1.25,1.5",
    temporal_refresh_intervals: str = "2,4,8,16",
    temporal_block_sizes: str = "16,32,64",
    temporal_mass_level: float = 0.95,
) -> None:
    _, _, smoke = resolve_options(mode, max_samples, max_new_tokens)
    selected_inputs = tuple(
        item.strip() for item in inputs.split(",") if item.strip()
    ) or (DEFAULT_SMOKE_INPUTS if smoke else DEFAULT_PILOT_INPUTS)
    result = run_recap.remote(
        mode=mode,
        experiment=experiment,
        target_model=target_model,
        inputs=list(selected_inputs),
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        run_id=run_id,
        region_size=region_size,
        block_size=block_size,
        reps_per_block=reps_per_block,
        reps_per_region=reps_per_region,
        hierarchy_mass_budget=hierarchy_mass_budget,
        hierarchy_layers=hierarchy_layers,
        temporal_lags=temporal_lags,
        temporal_budgets=temporal_budgets,
        temporal_alphas=temporal_alphas,
        temporal_refresh_intervals=temporal_refresh_intervals,
        temporal_block_sizes=temporal_block_sizes,
        temporal_mass_level=temporal_mass_level,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("returncode", 1) != 0:
        raise SystemExit(int(result["returncode"]))


if __name__ == "__main__":
    # ``modal run`` owns the normal entrypoint.  This guard helps static tools
    # discover the module without accidentally starting a remote job.
    main()
