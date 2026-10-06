#!/usr/bin/env python3
"""Chạy đối chiếu attention DFlash/target trên Modal, dùng snapshot đã cache."""

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
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
MOUNT = Path("/mnt/fast-infer")
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "fast-infer-text-sum-cache")
GPU = os.environ.get("MODAL_GPU", "L40S")
VOLUME = modal.Volume.from_name(VOLUME_NAME)
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.modal.txt"))
    .add_local_dir(ROOT / "scripts/common", str(REMOTE_ROOT / "scripts/common"), copy=False,
                   ignore=("__pycache__", "*.pyc"))
    .add_local_file(ROOT / "scripts/probe_dflash_attention.py",
                    str(REMOTE_ROOT / "scripts/probe_dflash_attention.py"), copy=False)
    .add_local_file(ROOT / "scripts/probe_dflash_target_comparison.py",
                    str(REMOTE_ROOT / "scripts/probe_dflash_target_comparison.py"), copy=False)
    .add_local_dir(ROOT / "externals/dflash", str(REMOTE_ROOT / "externals/dflash"),
                   copy=False, ignore=("__pycache__", "*.pyc", "cache"))
)
app = modal.App("fast-infer-dflash-target-attention-comparison")


@app.function(image=IMAGE, gpu=GPU, volumes={str(MOUNT): VOLUME},
              timeout=3600, cpu=8, memory=49152)
def run_comparison(run_id: str, source_run_id: str, caps: str,
                   samples: int, max_rounds: int,
                   capture_parent_attention: bool = False):
    import torch

    source = MOUNT / "outputs/dflash_attention_probe" / source_run_id
    output = MOUNT / "outputs/dflash_attention_probe" / run_id
    command = [sys.executable, "-u", str(REMOTE_ROOT / "scripts/probe_dflash_target_comparison.py"),
               "--source-run", str(source), "--output", str(output),
               "--caps", caps, "--samples", str(samples),
               "--max-rounds", str(max_rounds)]
    if capture_parent_attention:
        command.append("--capture-parent-attention")
    environment = dict(os.environ)
    environment.update({
        "HF_HOME": str(MOUNT / "hf"), "HF_HUB_CACHE": str(MOUNT / "hf/hub"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    print(f"[modal] gpu={torch.cuda.get_device_name()} source={source_run_id} "
          f"run={run_id} caps={caps} samples={samples}", flush=True)
    try:
        subprocess.run(command, env=environment, cwd=REMOTE_ROOT, check=True)
    finally:
        VOLUME.commit()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz", compresslevel=1) as bundle:
        for path in sorted(output.iterdir()):
            bundle.add(path, arcname=path.name)
    return {"run_id": run_id, "remote_output": str(output), "volume": VOLUME_NAME,
            "summary": json.loads((output / "summary.json").read_text()),
            "archive": archive.getvalue()}


@app.local_entrypoint()
def main(run_id: str = "", source_run_id: str = "dflash-attention-fullmass-govreport10-l40s-20261006",
         caps: str = "3072,5120,8192,16384", samples: int = 10,
         max_rounds: int = 512, capture_parent_attention: bool = False):
    run_id = run_id or "dflash-target-attention-compare-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if Path(run_id).name != run_id or Path(source_run_id).name != source_run_id:
        raise ValueError("run-id và source-run-id phải là tên directory đơn")
    result = run_comparison.remote(run_id, source_run_id, caps, samples, max_rounds,
                                   capture_parent_attention)
    output = ROOT / "outputs/dflash_attention_probe" / run_id
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(result.pop("archive")), mode="r:gz") as bundle:
        bundle.extractall(output, filter="data")
    (output / "modal_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({key: value for key, value in result.items() if key != "summary"},
                     indent=2, ensure_ascii=False))
    print(f"[local] runs={result['summary']['run_count']} rounds={result['summary']['draft_rounds']}")
    print(f"[local] artifacts: {output}")


if __name__ == "__main__":
    main()
