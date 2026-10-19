#!/usr/bin/env python3
"""Đánh giá riêng Phase A trên snapshot có metadata, trong profile phuclk30."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
MOUNT = Path("/mnt/fast-infer")
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "fast-infer-text-sum-cache")
VOLUME = modal.Volume.from_name(VOLUME_NAME)

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.2.6")
    .add_local_dir(ROOT / "src/TrainingFree", str(REMOTE_ROOT / "src/TrainingFree"), ignore=["__pycache__", "*.pyc"])
    .add_local_dir(ROOT / "scripts/common", str(REMOTE_ROOT / "scripts/common"), ignore=["__pycache__", "*.pyc"])
    .add_local_file(
        ROOT / "scripts/run_candidate_benchmarks.py",
        str(REMOTE_ROOT / "scripts/run_candidate_benchmarks.py"),
        copy=False,
    )
)

app = modal.App("fast-infer-candidate-benchmarks")


@app.function(
    image=IMAGE,
    volumes={str(MOUNT): VOLUME},
    timeout=1800,
    cpu=8,
    memory=16384,
)
def run_benchmark_on_modal(
    run_id: str = "dflash-target-attention-compare-all36layers-allrounds-govreport10",
    all_rounds: bool = False,
):
    source_dir = MOUNT / "outputs/dflash_attention_probe" / run_id
    target_dir = source_dir / "vector_snapshots"
    out_dir = MOUNT / "outputs/phase_a_evaluations" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[modal] Bắt đầu chạy Candidate Benchmarks trên Volume: {source_dir}...", flush=True)
    cmd = [
        sys.executable,
        str(REMOTE_ROOT / "scripts/run_candidate_benchmarks.py"),
        "--target-dir",
        str(target_dir),
        "--output-dir",
        str(out_dir),
    ]
    if all_rounds:
        cmd.append("--all-rounds")

    subprocess.run(cmd, check=True)
    VOLUME.commit()

    print(f"[modal] Đang đóng gói kết quả bài báo tại {out_dir} vào in-memory bundle...", flush=True)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz", compresslevel=6) as bundle:
        for file_path in out_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(out_dir)
                bundle.add(file_path, arcname=str(rel_path))

    size_mb = archive.tell() / (1024 * 1024)
    print(f"[modal] Đóng gói hoàn tất! Kích thước kết quả bài báo: {size_mb:.2f} MB", flush=True)
    return {"status": "success", "size_mb": size_mb, "archive": archive.getvalue()}


@app.local_entrypoint()
def main(
    run_id: str = "dflash-target-attention-compare-all36layers-allrounds-govreport10",
    all_rounds: bool = False,
):
    if os.environ.get("MODAL_PROFILE") != "phuclk30":
        raise ValueError("Cần MODAL_PROFILE=phuclk30 và modal run --profile phuclk30")
    if Path(run_id).name != run_id:
        raise ValueError("run_id phải là tên directory đơn")
    print(f"[local] Đang gửi job chạy benchmark cho 7 Candidates lên Modal CPU (run_id={run_id})...")
    result = run_benchmark_on_modal.remote(run_id=run_id, all_rounds=all_rounds)
    print(f"[local] Job thành công! Kích thước bundle nhận về: {result['size_mb']:.2f} MB.")

    local_out_dir = ROOT / "outputs/phase_a_evaluations" / run_id
    local_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[local] Đang giải nén kết quả vào {local_out_dir}...")
    with tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz") as bundle:
        bundle.extractall(local_out_dir, filter="data")

    print(f"[local] Hoàn tất giải nén toàn bộ 7 Candidate và Ma trận Experiment A tại: {local_out_dir}")


if __name__ == "__main__":
    main()
