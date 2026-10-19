#!/usr/bin/env python3
"""Run DFlash Speculation Phase B on Modal using profile phuclk30.

Usage:
    MODAL_PROFILE=phuclk30 .venv/bin/modal run --profile phuclk30 scripts/modal_dflash_speculation_b.py \
        --run-id phase_b_gpu_pilot_20261019 --mode run --cohort-phase pilot
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/workspace/fast_infer_text_sum")
VOLUME = modal.Volume.from_name("fast-infer-text-sum-cache")

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.12.1",
        "accelerate==1.14.0",
        "numpy==2.2.6",
        "scipy==1.15.2",
        "einops==0.8.2",
        "huggingface_hub==1.21.0",
        "rouge_score==0.1.2",
    )
    .add_local_dir(ROOT / "scripts/common", str(REMOTE / "scripts/common"), ignore=["__pycache__", "*.pyc"])
    .add_local_dir(ROOT / "src/TrainingFree", str(REMOTE / "src/TrainingFree"), ignore=["__pycache__", "*.pyc"])
    .add_local_dir(ROOT / "externals/dflash", str(REMOTE / "externals/dflash"), ignore=["__pycache__", "*.pyc", ".git"])
    .add_local_dir(ROOT / "data/longbench_200", "/bootstrap/longbench_200")
    .add_local_dir(ROOT / "data/dflash_context_A_audit", "/bootstrap/dflash_context_A_audit")
    .add_local_file(ROOT / "scripts/infer_dflash.py", str(REMOTE / "scripts/infer_dflash.py"))
    .add_local_file(ROOT / "scripts/run_dflash_speculation_B.py", str(REMOTE / "scripts/run_dflash_speculation_B.py"))
)

app = modal.App("dflash-speculation-b")


@app.function(
    image=IMAGE,
    gpu="L40S",
    cpu=8,
    memory=49152,
    timeout=7200,
    volumes={"/mnt/fast-infer": VOLUME},
)
def run_remote(
    run_id: str,
    mode: str = "run",
    cohort_phase: str = "pilot",
    attention_backend: str = "sdpa",
    opportunity_threshold_eta: float = 0.0,
    repetitions: int = 4,
    selection_repetitions: int = 2,
    max_new_tokens: int = 1024,
    validation_tokens: int = 128,
    e2e_repetitions: int = 3,
    e2e_warmups: int = 1,
    allow_gated_b3: bool = True,
    b3_variants: str = "",
    run_b3: bool = True,
) -> bytes:
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/mnt/fast-infer/hf"
    os.environ["CAD_ATTENTION_BACKEND"] = attention_backend
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    try:
        target_dir = snapshot_download("Qwen/Qwen3-4B", cache_dir="/mnt/fast-infer/hf/hub", local_files_only=True)
    except Exception:
        target_dir = snapshot_download("Qwen/Qwen3-4B", cache_dir="/mnt/fast-infer/hf/hub", local_files_only=False)
    try:
        draft_dir = snapshot_download("z-lab/Qwen3-4B-DFlash-b16", cache_dir="/mnt/fast-infer/hf/hub", local_files_only=True)
    except Exception:
        draft_dir = snapshot_download("z-lab/Qwen3-4B-DFlash-b16", cache_dir="/mnt/fast-infer/hf/hub", local_files_only=False)

    out_root = Path("/mnt/fast-infer/outputs/dflash_speculation_B")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PYTHONPATH=f"{REMOTE}:{REMOTE}/src:{REMOTE}/scripts:{REMOTE}/externals/dflash",
    )

    def build_cmd(target_mode: str) -> list[str]:
        base = [
            sys.executable,
            "-u",
            str(REMOTE / "scripts/run_dflash_speculation_B.py"),
            "--mode",
            target_mode,
            "--cohort-phase",
            cohort_phase,
            "--target-model",
            str(target_dir),
            "--draft-model",
            str(draft_dir),
            "--data-dir",
            "/bootstrap/longbench_200",
            "--exposure-manifest",
            "/bootstrap/dflash_context_A_audit/combined_exposure_manifest.json",
            "--output-root",
            str(out_root),
            "--run-id",
            run_id,
            "--attention-backend",
            attention_backend,
            "--opportunity-threshold-eta",
            str(opportunity_threshold_eta),
            "--repetitions",
            str(repetitions),
            "--selection-repetitions",
            str(selection_repetitions),
            "--max-new-tokens",
            str(max_new_tokens),
            "--validation-tokens",
            str(validation_tokens),
            "--e2e-repetitions",
            str(e2e_repetitions),
            "--e2e-warmups",
            str(e2e_warmups),
            "--allow-partial-cohort",
            "--resume",
        ]
        if allow_gated_b3:
            base.append("--allow-gated-b3")
        if b3_variants:
            base.extend(["--b3-variants", b3_variants])
        return base

    cmd = build_cmd(mode)
    print(f"Executing: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REMOTE, env=env, check=True)
    VOLUME.commit()

    if run_b3 and mode == "run" and cohort_phase in ("pilot", "dev", "confirm"):
        b3_cmd = build_cmd("b3-dev")
        print(f"Executing B3-dev: {' '.join(b3_cmd)}", flush=True)
        try:
            subprocess.run(b3_cmd, cwd=REMOTE, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[B3-dev] gate closed or exited with {exc.returncode}; B1/B2 artifacts preserved", flush=True)
        VOLUME.commit()

    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz", compresslevel=1) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".log", ".txt"):
                archive.add(path, arcname=str(path.relative_to(run_dir)))
    return data.getvalue()


@app.local_entrypoint()
def main(
    run_id: str | None = None,
    mode: str = "run",
    cohort_phase: str = "pilot",
    attention_backend: str = "sdpa",
    opportunity_threshold_eta: float = 0.005,
    repetitions: int = 4,
    selection_repetitions: int = 2,
    max_new_tokens: int = 1024,
    validation_tokens: int = 128,
    e2e_repetitions: int = 3,
    e2e_warmups: int = 1,
    allow_gated_b3: bool = True,
    b3_variants: str = "",
    run_b3: bool = True,
):
    if os.environ.get("MODAL_PROFILE") != "phuclk30":
        raise ValueError("MODAL_PROFILE=phuclk30 bắt buộc; sử dụng modal run --profile phuclk30")

    if run_id is None:
        run_id = f"b_pilot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    print(f"=== Starting Phase B Modal GPU Job ===")
    print(f"Run ID: {run_id}")
    print(f"Mode: {mode}")
    print(f"Cohort Phase: {cohort_phase}")
    print(f"Backend: {attention_backend}")
    print(f"Opportunity Threshold Eta: {opportunity_threshold_eta}")
    print(f"Repetitions: {repetitions} (selection: {selection_repetitions})")
    print(f"E2E Repetitions: {e2e_repetitions} (warmups: {e2e_warmups})")
    print(f"Allow Gated B3: {allow_gated_b3}, B3 Variants: {b3_variants or 'default'}")

    payload = run_remote.remote(
        run_id=run_id,
        mode=mode,
        cohort_phase=cohort_phase,
        attention_backend=attention_backend,
        opportunity_threshold_eta=opportunity_threshold_eta,
        repetitions=repetitions,
        selection_repetitions=selection_repetitions,
        max_new_tokens=max_new_tokens,
        validation_tokens=validation_tokens,
        e2e_repetitions=e2e_repetitions,
        e2e_warmups=e2e_warmups,
        allow_gated_b3=allow_gated_b3,
        b3_variants=b3_variants,
        run_b3=run_b3,
    )

    out = ROOT / "outputs/dflash_speculation_B" / run_id
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(out, filter="data")

    print(f"\n=== Phase B GPU Run Complete ===")
    print(f"Local Artifacts: {out}")
    if (out / "reports/decision.json").exists():
        decision = json.loads((out / "reports/decision.json").read_text(encoding="utf-8"))
        print("\nDecision summary:")
        print(json.dumps(decision, indent=2))
    if (out / "b3/dev_summary.json").exists():
        b3_sum = json.loads((out / "b3/dev_summary.json").read_text(encoding="utf-8"))
        print("\nB3 E2E Dev Summary:")
        print(json.dumps(b3_sum, indent=2))

