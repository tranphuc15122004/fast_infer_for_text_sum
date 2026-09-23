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


def test_phase1_launcher_consumes_prepared_data_without_building_it() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '"$PROJECT_ROOT/scripts/mr_dflash/prepare_server_data.py"' not in text
    assert '"$PROJECT_ROOT/scripts/mr_dflash/build_pilot_dataset.py"' not in text
    assert 'PREPARED_ROOT="${PREPARED_ROOT:-' in text
    assert "--from-stage regenerate_full_train" in text
    assert "normalized/${split}_prompts.jsonl" in text


def test_phase1_launcher_has_resumable_graceful_auto_pause() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PHASE1_AUTO_PAUSE_MINUTES="${PHASE1_AUTO_PAUSE_MINUTES:-0}"' in text
    assert "--auto-pause-minutes" in text
    assert "start_pause_timer" in text
    assert "touch \"$STOP_FILE\"" in text
    assert "--worker-stop-file \"$STOP_FILE\"" in text
    assert 'exit "$PHASE1_RC"' in text
