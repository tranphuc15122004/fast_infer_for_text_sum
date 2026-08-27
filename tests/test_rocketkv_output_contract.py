"""Regression test for the RocketKV smoke output metadata."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def test_rocketkv_smoke_records_numeric_budget_k(tmp_path: Path) -> None:
    """The output's ``k`` field is the selected-token budget, not a tensor dump."""
    output = tmp_path / "rocketkv.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": f"{ROOT / 'scripts'}:{ROOT / 'externals' / 'RocketKV' / 'gpt-fast'}",
        }
    )
    result = subprocess.run(
        [
            str(PYTHON),
            str(ROOT / "scripts" / "infer_rocketkv.py"),
            "--token-budget",
            "16",
            "--seq-len",
            "64",
            "--max-new-tokens",
            "1",
            "--num-runs",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(record["k"], int)
