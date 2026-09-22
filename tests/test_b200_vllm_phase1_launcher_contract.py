from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/mr_dflash/run_b200_vllm_phase1.sh"


def test_sourcing_launcher_returns_without_killing_parent_shell() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'. "{LAUNCHER}" 1; rc=$?; echo "shell-alive rc=$rc"',
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "shell-alive rc=2" in result.stdout
    assert "không source launcher này" in result.stderr
