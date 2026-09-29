#!/usr/bin/env python3
"""Run the paired coverage-prefill pilot on a cached Modal GPU/model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
VOLUME_ROOT = Path("/mnt/fast-infer")
volume = modal.Volume.from_name("fast-infer-text-sum-cache")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.modal.txt"))
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=False,
                   ignore=("__pycache__", "*.pyc"))
    .add_local_dir(ROOT / "externals" / "Sematic_selection",
                   str(REMOTE_ROOT / "externals" / "Sematic_selection"), copy=False,
                   ignore=("__pycache__", "*.pyc"))
    .add_local_dir(ROOT / "data" / "longbench_100_14k",
                   str(REMOTE_ROOT / "data" / "longbench_100_14k"), copy=False)
)
app = modal.App("fast-infer-coverage-prefill-pilot")


@app.function(image=image, gpu="A100-80GB", timeout=7200,
              volumes={str(VOLUME_ROOT): volume})
def run_pilot(run_id: str, datasets: str, max_samples: int,
              max_new_tokens: int, budget: int) -> dict:
    output_dir = VOLUME_ROOT / "outputs" / "coverage_prefill" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output_file = output_dir / "paired.jsonl"
    env = dict(os.environ)
    env.update({
        "HF_HOME": str(VOLUME_ROOT / "hf"),
        "HF_HUB_CACHE": str(VOLUME_ROOT / "hf" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    command = [
        sys.executable, str(REMOTE_ROOT / "scripts" / "probe_coverage_prefill.py"),
        "--datasets", datasets, "--max-samples", str(max_samples),
        "--max-new-tokens", str(max_new_tokens), "--budget", str(budget),
        "--output", str(output_file),
    ]
    subprocess.run(command, env=env, check=True)
    volume.commit()
    summary = json.loads(output_file.with_suffix(".summary.json").read_text())
    return {"run_id": run_id, "output_dir": str(output_dir), "summary": summary}


@app.local_entrypoint()
def main(run_id: str = "coverage-prefill-pilot-20260928",
         datasets: str = "gov_report,qmsum", max_samples: int = 5,
         max_new_tokens: int = 128, budget: int = 6144) -> None:
    print(json.dumps(run_pilot.remote(run_id, datasets, max_samples,
                                      max_new_tokens, budget), indent=2))
