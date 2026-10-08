#!/usr/bin/env python3
"""B200-only AMR-DFlash asset/runtime preflight."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                "bash",
                str(ROOT / "scripts" / "run.sh"),
                "amr_dflash",
                "preflight",
                "--require-b200",
                *sys.argv[1:],
            ],
            cwd=ROOT,
        )
    )
