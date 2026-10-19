#!/usr/bin/env python3
"""Chạy thực nghiệm đánh giá Layer Sweep (L0, L12, L14, L16, L18, L20, L27, L35) trên 960 snapshots Modal.

Snapshot source: /mnt/fast-infer/outputs/phase_a_paper/phase-a-paper-20261019/coverage/vector_snapshots
Gồm 960 snapshots của 10 documents GovReport tại các context 3K, 5K, 8K, 16K.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/workspace/fast_infer_text_sum")
VOLUME = modal.Volume.from_name("fast-infer-text-sum-cache")

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.2.6", "matplotlib", "scipy", "pandas")
    .add_local_dir(ROOT / "src/TrainingFree", str(REMOTE / "src/TrainingFree"), ignore=["__pycache__", "*.pyc"])
    .add_local_file(ROOT / "scripts/evaluate_layer_sweep_matrix.py", str(REMOTE / "scripts/evaluate_layer_sweep_matrix.py"))
)

app = modal.App("evaluate-layer-sweep")


@app.function(image=IMAGE, cpu=8, memory=32768, timeout=1800, volumes={"/mnt/fast-infer": VOLUME})
def run_evaluation_remote():
    import sys
    sys.path.insert(0, str(REMOTE))
    sys.path.insert(0, str(REMOTE / "src"))
    sys.path.insert(0, str(REMOTE / "scripts"))

    from evaluate_layer_sweep_matrix import evaluate_snapshots

    snapshot_dir = Path("/mnt/fast-infer/outputs/phase_a_paper/phase-a-paper-20261019/coverage/vector_snapshots")
    output_dir = Path("/tmp/layer_sweep_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[modal] Bắt đầu đánh giá Layer Sweep trên {snapshot_dir}...", flush=True)
    report = evaluate_snapshots(snapshot_dir, output_dir)
    print(f"[modal] Đánh giá hoàn tất! Đóng gói kết quả...", flush=True)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in output_dir.rglob("*"):
            if p.is_file():
                tar.add(p, arcname=p.relative_to(output_dir).as_posix())
    return buf.getvalue()


@app.local_entrypoint()
def main(output_dir: str = "outputs/_PAPER_EXP/dflash_layer_sweep_benchmark_20261019"):
    if os.environ.get("MODAL_PROFILE") != "phuclk30":
        os.environ["MODAL_PROFILE"] = "phuclk30"

    print("[local] Gửi job đánh giá Layer Sweep lên Modal phuclk30...")
    tar_bytes = run_evaluation_remote.remote()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    print(f"[local] Nhận {len(tar_bytes)} bytes từ Modal. Giải nén vào {out_p}...")

    buf = io.BytesIO(tar_bytes)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(path=out_p)

    print(f"[local] Giải nén hoàn tất! Kiểm tra các file tại {out_p}:")
    for f in out_p.glob("*"):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
