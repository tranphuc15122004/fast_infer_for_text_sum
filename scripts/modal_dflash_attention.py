#!/usr/bin/env python3
"""Chạy probe DFlash trên Modal và tải artifact về outputs/ local."""

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
GPU = os.environ.get("MODAL_GPU", "A100-80GB")
VOLUME = modal.Volume.from_name(VOLUME_NAME)
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.modal.txt"))
    .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=False,
                   ignore=("__pycache__", "*.pyc"))
    .add_local_dir(ROOT / "externals/dflash", str(REMOTE_ROOT / "externals/dflash"),
                   copy=False, ignore=("__pycache__", "*.pyc", "cache"))
    .add_local_file(ROOT / "data/longbench_200/gov_report.jsonl",
                    str(REMOTE_ROOT / "data/longbench_200/gov_report.jsonl"), copy=False)
)
app = modal.App("fast-infer-dflash-attention-probe")


@app.function(image=IMAGE, gpu=GPU, volumes={str(MOUNT): VOLUME},
              timeout=3600, cpu=4, memory=32768)
def run_probe(run_id: str, caps: str, max_rounds: int, max_new_tokens: int,
              samples: int, sample_seed: int):
    import torch
    output = MOUNT / "outputs/dflash_attention_probe" / run_id
    environment = dict(os.environ)
    environment.update({"HF_HOME": str(MOUNT / "hf"),
        "HF_HUB_CACHE": str(MOUNT / "hf/hub"), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0"})
    command = [sys.executable, str(REMOTE_ROOT / "scripts/probe_dflash_attention.py"),
               "--output", str(output), "--caps", caps,
               "--max-rounds", str(max_rounds), "--max-new-tokens", str(max_new_tokens),
               "--samples", str(samples), "--sample-seed", str(sample_seed)]
    print(f"[modal] gpu={torch.cuda.get_device_name()} run={run_id} caps={caps}", flush=True)
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
def main(run_id: str = "", caps: str = "3072,5120,8192,16384",
         max_rounds: int = 512, max_new_tokens: int = 512,
         samples: int = 10, sample_seed: int = 42):
    run_id = run_id or "dflash-attention-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if Path(run_id).name != run_id:
        raise ValueError("run-id phải là tên directory đơn")
    result = run_probe.remote(run_id, caps, max_rounds, max_new_tokens, samples, sample_seed)
    output = ROOT / "outputs/dflash_attention_probe" / run_id
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(result.pop("archive")), mode="r:gz") as bundle:
        bundle.extractall(output, filter="data")
    (output / "modal_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({key: value for key, value in result.items() if key != "summary"},
                     indent=2, ensure_ascii=False))
    print(f"[local] runs={result['summary']['run_count']} drafts={result['summary']['draft_rounds']}")
    print(f"[local] artifacts: {output}")


if __name__ == "__main__":
    main()
