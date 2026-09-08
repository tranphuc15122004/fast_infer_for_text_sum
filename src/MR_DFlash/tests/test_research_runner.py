from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.mr_dflash.run_e23_e24 import (
    OBJECTIVE_VARIANTS,
    build_command,
    build_eval_command,
)


def test_e23_e24_matrix_has_three_frontier_and_two_hard_negative_conditions() -> None:
    assert set(OBJECTIVE_VARIANTS) == {
        "e23_fixed_decay",
        "e23_dpace",
        "e23_spec_auf",
        "e24_dpace_hn_all",
        "e24_dpace_hn_shallow",
    }


def test_runner_command_changes_only_objective_knobs() -> None:
    command = build_command(
        python="python3",
        config="base.yaml",
        output_dir="out/e23_dpace",
        variant="e23_dpace",
        max_steps=3,
    )
    assert command[:4] == ["python3", "-m", "MR_DFlash.run_train", "--config"]
    assert "--loss-type" in command
    assert command[command.index("--loss-type") + 1] == "dpace"
    assert command[command.index("--max-steps") + 1] == "3"
    assert "--hard-negative-k" not in command


def test_runner_eval_command_enforces_exactness() -> None:
    command = build_eval_command(
        python="python3",
        config="base.yaml",
        checkpoint="out/checkpoint_final.pt",
        input_path="data/test.jsonl",
        output_path="out/generation_eval.jsonl",
        device="cuda",
        max_new_tokens=32,
        max_samples=4,
        local_files_only=True,
    )
    assert "--exactness-check" in command
    assert command[command.index("--max-samples") + 1] == "4"
    assert "--local-files-only" in command
