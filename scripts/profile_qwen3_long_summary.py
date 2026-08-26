#!/usr/bin/env python3
"""Backward-compatible entry point for the canonical full-infer profiler."""

from pathlib import Path
import sys


PROFILE_ROOT = Path(__file__).resolve().parents[1] / "src" / "analyze" / "full_infer"
sys.path.insert(0, str(PROFILE_ROOT))

from profile_qwen3_long_summary import main  # noqa: E402


if __name__ == "__main__":
    main()
