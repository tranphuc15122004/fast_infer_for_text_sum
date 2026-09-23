from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_launcher_timer_smoke_command_uses_production_launcher() -> None:
    from modal_b200_phase1_timer_smoke import build_launcher_command

    command = build_launcher_command(
        launcher=Path("/workspace/fast_infer_text_sum/scripts/mr_dflash/run_b200_vllm_phase1.sh"),
        run_root=Path("/mnt/mr-dflash/runs/timer-test"),
        prepared_root=Path("/mnt/mr-dflash/prepared/timer-test"),
        target_model=Path("/mnt/mr-dflash/models/Qwen3-4B"),
        pause_minutes=1,
    )

    assert command[:2] == ["bash", "/workspace/fast_infer_text_sum/scripts/mr_dflash/run_b200_vllm_phase1.sh"]
    assert "--auto-pause-minutes" in command
    assert command[command.index("--auto-pause-minutes") + 1] == "1"
    assert command[command.index("--run-root") + 1] == str(Path("/mnt/mr-dflash/runs/timer-test"))
    assert command[command.index("--prepared-root") + 1] == str(Path("/mnt/mr-dflash/prepared/timer-test"))
