#!/usr/bin/env python3
"""Direct CLI entry point for the AMR-DFlash benchmark adapter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amr_dflash.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["infer", *sys.argv[1:]]))
