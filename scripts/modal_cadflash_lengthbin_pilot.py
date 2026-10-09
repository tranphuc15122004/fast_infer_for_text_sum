#!/usr/bin/env python3
"""Run Context-Adaptive DFlash and ablations on length-stratified data via Modal.

This is a bounded, paired exploratory benchmark. Data selection and split
manifests are staged locally under outputs/cadflash_lengthbins_modal_20261009;
the target model is read from the existing verified Modal volume.
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
STAGE_ROOT = ROOT / "outputs/cadflash_lengthbins_modal_20261009/assets"
ASSET_VOLUME_NAME = "cadflash-lengthbins-20261009"
TARGET_VOLUME_NAME = "fast-infer-text-sum-mr-dflash-timer"
RUN_ID = "cadflash_lengthbins_modal_20261009_001"
ASSET_ROOT = Path("/assets/assets")
TARGET_ROOT = Path("/models/models/Qwen3-4B")
REPO_ROOT = Path("/workspace/fast_infer_text_sum")
RUN_ROOT = ASSET_ROOT / "runs" / RUN_ID
DATA_FILE = ASSET_ROOT / "inputs/lengthbin_pilot.jsonl"
SPLIT_FILE = ASSET_ROOT / "split_manifest.json"

VARIANTS = (
    "ar",
    "dflash_full_fixed",
    "a_only",
    "b_only_history",
    "b_only_entropy",
    "independent_ab",
    "joint_no_entropy",
    "joint_no_source_relevance",
    "joint",
)

app = modal.App("cadflash-lengthbin-pilot-20261009")
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
    if mismatches:
        raise RuntimeError(f"Target checkpoint volume hash mismatch: {mismatches[:3]}")
    result = {
        "matches_local_snapshot": True,
        "actual": actual,
        "expected_content_sha256": expected.get("content_sha256"),
    }
    out = ASSET_ROOT / "target_volume_verification.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    asset_volume.commit()
    print(f"Verified target volume SHA-256: {actual['content_sha256']}", flush=True)
    return result


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={"/assets": asset_volume, "/models": target_volume},
)
def run_experiment() -> str:
    import torch

    os.chdir(REPO_ROOT)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MODEL_TARGET": str(TARGET_ROOT),
            "MODEL_DFLASH_DRAFT": str(ASSET_ROOT / "models/Qwen3-4B-DFlash-b16"),
            "PYTHONPATH": ":".join(
                [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts"), str(REPO_ROOT / "externals/dflash")]
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CAD_SELECTOR": "target_parent",
            "CAD_TARGET_LAYER": "middle",
            "CAD_BUDGETS": "1024,full",
            "CAD_GAMMAS": "3,7,15",
            "CAD_REFRESH_PERIOD": "4",
            "CAD_SIGNAL_UPDATE_PERIOD": "1",
            "CAD_SOURCE_CHUNK_SIZE": "128",
            "CAD_SOURCE_ANCHORS": "64",
            "CAD_RECENT_OUTPUT": "256",
            "CAD_PRIOR_STRENGTH": "16",
            "CAD_MIN_STATE_SUPPORT": "8",
            "CAD_STATISTICS_DECAY": "0.95",
            "CAD_HISTORY_ALPHA": "0.1",
            "CAD_LATENCY_EMA_ALPHA": "0.1",
            "CAD_CONTROLLER_MARGIN": "0.02",
            "CAD_ENTROPY_SIGNAL_TEMPERATURE": "1",
            "CAD_TIMING_MODE": "wall",
            "CAD_COST_UPDATE_MODE": "frozen_cost",
            "CAD_STATISTICS_UPDATE_MODE": "online",
            "CAD_LENGTH_MODE": "draft_shape",
        }
    )
    hardware = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "run_id": RUN_ID,
        "data_sha256": _sha256(DATA_FILE),
        "split_sha256": _sha256(SPLIT_FILE),
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
        "--data-file", str(DATA_FILE),
        "--split-manifest", str(SPLIT_FILE),
        "--run-id", RUN_ID,
        "--output-root", str(RUN_ROOT),
        "--temperature", "0",
        "--max-input-tokens", "0",
        "--max-new-tokens", "128",
        "--fixed-output-tokens", "128",
        "--fixed-budget", "full",
        "--fixed-gamma", "15",
        "--gamma-reference", "15",
        "--budgets", "1024,full",
        "--gammas", "3,7,15",
        "--seed", "42",
    ]
    cli = str(REPO_ROOT / "scripts/infer_context_adaptive_dflash.py")

    def run_cli(args: list[str]) -> None:
        command = [sys.executable, cli, *args]
        print("[cadflash-lengthbin] " + " ".join(command), flush=True)
        try:
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        except BaseException:
            asset_volume.commit()
            raise
        asset_volume.commit()

    try:
        run_cli(["--phase", "preflight", *common, "--model-check"])
        run_cli(
            [
                "--phase", "calibrate", "--variant", "dflash_full_fixed", *common,
                "--calibration-checkpoints", "0,64",
                "--repetitions", "3", "--warmup-runs", "1",
            ]
        )
        calibration = str(RUN_ROOT / "calibration/calibration.json")
        for variant in VARIANTS:
            print(f"[cadflash-lengthbin] START dev variant={variant}", flush=True)
            args = [
                "--phase", "dev", "--variant", variant, *common,
                "--calibration-file", calibration,
                "--repetitions", "3", "--warmup-runs", "1",
            ]
            run_cli(args)
        run_cli(
            [
                "--phase", "report", "--output-root", str(RUN_ROOT),
                "--report-phase", "dev", "--report-statistics-mode", "online",
                "--bootstrap-samples", "1000",
            ]
        )
    finally:
        asset_volume.commit()

    archive_path = ASSET_ROOT / "cadflash_lengthbin_results.tar.gz"
    with tarfile.open(archive_path, "w:gz") as bundle:
        bundle.add(RUN_ROOT, arcname=f"runs/{RUN_ID}")
        bundle.add(ASSET_ROOT / "sample_manifest.json", arcname="sample_manifest.json")
        bundle.add(ASSET_ROOT / "preregistration.md", arcname="preregistration.md")
        bundle.add(SPLIT_FILE, arcname="split_manifest.json")
    asset_volume.commit()
    return str(archive_path)


@app.local_entrypoint()
def main() -> None:
    verification = verify_target_volume.remote()
    if not verification.get("matches_local_snapshot"):
        raise RuntimeError("Modal target model volume did not match its pinned local signature")
    archive = run_experiment.remote()
    print(f"Experiment finished; result archive: {archive}")
