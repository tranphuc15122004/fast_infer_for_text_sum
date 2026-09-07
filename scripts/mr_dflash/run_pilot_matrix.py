"""Orchestrate pilot runs tuần tự, không tự ý chọn/chiếm GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from _common import REPO_ROOT


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the three MR-DFlash pilot configs")
    parser.add_argument("--config-dir", default=str(REPO_ROOT / "src/MR_DFlash/configs/pilot_qwen3_4b"))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true", help="xác nhận chạy subprocess train")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args(argv)
    names = ["dflash_2l_8k.yaml", "mr_dflash_2s_8k.yaml", "dflash_5l_8k.yaml"]
    if args.only:
        names = args.only
    commands = []
    for name in names:
        path = Path(args.config_dir) / name
        if not path.exists():
            raise FileNotFoundError(path)
        command = [sys.executable, "-m", "MR_DFlash.run_train", "--config", str(path), "--device", args.device]
        if args.max_steps is not None:
            command.extend(["--max-steps", str(args.max_steps)])
        if args.resume_from and len(names) == 1:
            command.extend(["--resume-from", args.resume_from])
        commands.append(command)
    for command in commands:
        print("[run_pilot_matrix]", " ".join(command))
    if not args.run:
        print("[run_pilot_matrix] dry-run; thêm --run để xác nhận chạy.")
        return
    environment = dict(os.environ)
    environment.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    # CUDA_VISIBLE_DEVICES is intentionally inherited, never inferred here.
    for command in commands:
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
