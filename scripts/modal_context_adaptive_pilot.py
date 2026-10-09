#!/usr/bin/env python3
"""Run a bounded Context-Adaptive DFlash pilot on Modal L40S.

This entrypoint reuses the existing Qwen3-4B Modal volume, verifies its content
hashes, then runs a short smoke, calibration, and paired dev pilot. See the
experiment record under outputs/modal_cadflash_pilot_20261009/ after download.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
ASSET_VOLUME_NAME = "cadflash-context-adaptive-pilot-20261009"
TARGET_VOLUME_NAME = "fast-infer-text-sum-mr-dflash-timer"
RUN_ID = "cadflash_modal_pilot_20261009"
# `modal volume put <directory> /` preserves the source directory's basename.
ASSET_ROOT = Path("/assets/assets")
# The reusable Modal volume stores the checkpoint below `models/Qwen3-4B`.
# Since the whole volume is mounted at `/models`, that path is doubled here.
TARGET_ROOT = Path("/models/models/Qwen3-4B")
REPO_ROOT = Path("/workspace/fast_infer_text_sum")
RUN_ROOT = ASSET_ROOT / "runs" / RUN_ID

app = modal.App("context-adaptive-dflash-pilot")
asset_volume = modal.Volume.from_name(ASSET_VOLUME_NAME)
target_volume = modal.Volume.from_name(TARGET_VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.12.1",
        "accelerate==1.15.0",
        "huggingface_hub==1.33.0",
        "safetensors==0.8.0",
        "numpy==2.3.5",
    )
    .add_local_dir(ROOT / "src/TrainingFree", remote_path=str(REPO_ROOT / "src/TrainingFree"), copy=True)
    .add_local_dir(ROOT / "scripts", remote_path=str(REPO_ROOT / "scripts"), copy=True)
    .add_local_dir(ROOT / "externals/dflash", remote_path=str(REPO_ROOT / "externals/dflash"), copy=True)
    .env(
        {
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'scripts'}:{REPO_ROOT / 'externals/dflash'}",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, object]:
    files = []
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_hash = _sha256(path)
        files.append({"path": relative, "bytes": size, "sha256": file_hash})
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
        total_bytes += size
    return {
        "file_count": len(files),
        "bytes": total_bytes,
        "content_sha256": digest.hexdigest(),
        "files": files,
    }


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    cpu=4,
    memory=8192,
    timeout=900,
    volumes={"/assets": asset_volume, "/models": target_volume},
)
def verify_target_volume() -> dict[str, object]:
    expected_path = ASSET_ROOT / "expected_target_signature.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = _tree_manifest(TARGET_ROOT)
    actual_files = {entry["path"]: entry for entry in actual["files"]}
    mismatches = [
        entry for entry in expected["files"]
        if actual_files.get(entry["path"]) != entry
    ]
    matches = not mismatches
    result = {
        "matches_local_snapshot": matches,
        "mismatched_or_missing_files": mismatches,
        "extra_files_ignored": sorted(set(actual_files) - {entry["path"] for entry in expected["files"]}),
        "actual": actual,
        "expected": expected,
    }
    output_path = ASSET_ROOT / "target_volume_verification.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    asset_volume.commit()
    if not matches:
        raise RuntimeError(
            f"Existing Modal Qwen3-4B differs from pinned local snapshot; see {output_path}"
        )
    print(f"Verified existing Modal target snapshot: {actual['content_sha256']}", flush=True)
    return {"matches_local_snapshot": True, "content_sha256": actual["content_sha256"]}


def _run_cli(arguments: list[str], env: dict[str, str]) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/infer_context_adaptive_dflash.py"),
        *arguments,
    ]
    print("[modal-pilot] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    asset_volume.commit()


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=3600,
    volumes={"/assets": asset_volume, "/models": target_volume},
)
def run_pilot() -> str:
    import torch

    os.chdir(REPO_ROOT)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["MODEL_TARGET"] = str(TARGET_ROOT)
    env["MODEL_DFLASH_DRAFT"] = str(ASSET_ROOT / "models/Qwen3-4B-DFlash-b16")
    env["PYTHONPATH"] = ":".join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts"), str(REPO_ROOT / "externals/dflash")]
    )
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CAD_SELECTOR"] = "target_parent"
    env["CAD_TARGET_LAYER"] = "middle"

    hardware = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version.split()[0],
    }
    try:
        import flash_attn  # noqa: F401

        hardware["flash_attn"] = getattr(flash_attn, "__version__", "installed")
    except Exception:
        hardware["flash_attn"] = None
    (RUN_ROOT / "evidence").mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "evidence/modal_hardware.json").write_text(
        json.dumps(hardware, indent=2) + "\n", encoding="utf-8"
    )
    asset_volume.commit()

    common = [
        "--target-model", str(TARGET_ROOT),
        "--draft-model", env["MODEL_DFLASH_DRAFT"],
        "--data-dir", str(ASSET_ROOT / "inputs/primary"),
        "--split-manifest", str(ASSET_ROOT / "split_manifest.json"),
        "--run-id", RUN_ID,
        "--output-root", str(RUN_ROOT),
        "--temperature", "0",
        "--max-input-tokens", "0",
        "--max-new-tokens", "128",
        "--fixed-budget", "full",
        "--fixed-gamma", "15",
        "--gamma-reference", "15",
        "--budgets", "1024,full",
        "--gammas", "3,7,15",
        "--seed", "42",
    ]

    _run_cli(
        ["--phase", "preflight", *common, "--model-check"],
        env,
    )

    smoke_common = [
        "--data-dir", str(ASSET_ROOT / "inputs/smoke_dev"),
        "--output-root", str(ASSET_ROOT / "runs" / f"{RUN_ID}_smoke"),
        "--run-id", f"{RUN_ID}_smoke",
        "--target-model", str(TARGET_ROOT),
        "--draft-model", env["MODEL_DFLASH_DRAFT"],
        "--temperature", "0",
        "--max-new-tokens", "32",
        "--fixed-budget", "full",
        "--fixed-gamma", "15",
        "--gamma-reference", "15",
        "--seed", "42",
        "--max-samples", "6",
        "--repetitions", "1",
        "--warmup-runs", "1",
    ]
    for variant in ("ar", "dflash_full_fixed"):
        _run_cli(["--phase", "smoke", "--variant", variant, *smoke_common], env)

    calibration_args = [
        "--phase", "calibrate",
        "--max-samples", "6",
        "--calibration-checkpoints", "0,64",
        "--repetitions", "1",
        "--warmup-runs", "1",
        "--max-new-tokens", "128",
        *common,
    ]
    _run_cli(calibration_args, env)

    dev_args = [
        "--phase", "dev",
        "--max-samples", "12",
        "--repetitions", "1",
        "--warmup-runs", "1",
        "--max-new-tokens", "128",
        "--calibration-file", str(RUN_ROOT / "calibration/calibration.json"),
        *common,
    ]
    for variant in ("ar", "dflash_full_fixed", "joint"):
        _run_cli([*dev_args, "--variant", variant], env)

    _run_cli(
        [
            "--phase", "report",
            "--output-root", str(RUN_ROOT),
            "--report-phase", "dev",
            "--report-statistics-mode", "online",
            "--bootstrap-samples", "1000",
        ],
        env,
    )
    asset_volume.commit()
    return str(RUN_ROOT)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={"/assets": asset_volume},
)
def archive_pilot_outputs() -> str:
    archive_path = ASSET_ROOT / "context_adaptive_dflash_pilot_results.tar.gz"
    smoke_root = ASSET_ROOT / "runs" / f"{RUN_ID}_smoke"
    paths = [RUN_ROOT, smoke_root]
    for name in ("expected_target_signature.json", "target_volume_verification.json"):
        candidate = ASSET_ROOT / name
        if candidate.is_file():
            paths.append(candidate)
    with tarfile.open(archive_path, "w:gz") as bundle:
        for path in paths:
            if path.exists():
                bundle.add(path, arcname=path.relative_to(ASSET_ROOT).as_posix())
    asset_volume.commit()
    return str(archive_path)


@app.local_entrypoint()
def main() -> None:
    verification = verify_target_volume.remote()
    if not verification.get("matches_local_snapshot"):
        raise RuntimeError("Modal target checkpoint verification failed")
    run_pilot.remote()
