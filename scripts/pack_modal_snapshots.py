#!/usr/bin/env python3
"""Gói các snapshots trên Modal Volume thành 1 file tar duy nhất để tải siêu tốc về local."""

from __future__ import annotations

from pathlib import Path
import subprocess
import modal

VOLUME_NAME = "fast-infer-text-sum-cache"
VOLUME = modal.Volume.from_name(VOLUME_NAME)
IMAGE = modal.Image.debian_slim(python_version="3.12")
app = modal.App("pack-modal-snapshots")


@app.function(image=IMAGE, volumes={"/mnt/fast-infer": VOLUME}, timeout=600, cpu=4)
def pack_snapshots(run_id: str):
    base_dir = Path("/mnt/fast-infer/outputs/dflash_attention_probe") / run_id
    tar_path = base_dir / "snapshots.tar"
    print(f"[modal] Bắt đầu đóng gói snapshots tại {base_dir} vào {tar_path}...", flush=True)
    
    cmd = [
        "tar", "-cf", str(tar_path),
        "-C", str(base_dir),
        "parent_transition_snapshots",
        "vector_snapshots",
    ]
    subprocess.run(cmd, check=True)
    VOLUME.commit()
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"[modal] Hoàn tất đóng gói! Kích thước: {size_mb:.2f} MB", flush=True)
    return {"status": "success", "tar_file": str(tar_path), "size_mb": size_mb}


@app.local_entrypoint()
def main(run_id: str = "dflash-target-attention-compare-all36layers-allrounds-govreport10"):
    result = pack_snapshots.remote(run_id)
    print(f"[local] Kết quả: {result}")


if __name__ == "__main__":
    main()
