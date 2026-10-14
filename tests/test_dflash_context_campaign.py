from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_dflash_context_experiments as campaign


def test_replay_uses_full_dflash_ids_when_ar_trajectory_diverges() -> None:
    trace = {
        "ar_ids": [10, 99, 30],
        "adapter_ids": [10, 20, 30],
        "native_ids": [10, 20, 30],
        "parity": {
            "status": "pass",
            "native_equals_adapter": True,
            "adapter_equals_ar": False,
        },
    }

    assert campaign._replay_reference_ids(trace) == [10, 20, 30]


def test_replay_rejects_trace_without_native_adapter_parity() -> None:
    trace = {
        "ar_ids": [10, 20, 30],
        "adapter_ids": [10, 20, 30],
        "native_ids": [10, 21, 30],
        "parity": {"status": "fail", "native_equals_adapter": False},
    }

    with pytest.raises(ValueError, match="native full DFlash and adapter IDs differ"):
        campaign._replay_reference_ids(trace)


def test_b200_campaign_launcher_uses_shared_config_and_python_runtime() -> None:
    launcher = (ROOT / "scripts" / "run_dflash_context_A_b200.sh").read_text(encoding="utf-8")

    assert 'source "$ROOT/scripts/common/config.sh"' in launcher
    assert "fast_infer_load_config dflash" in launcher
    assert 'source "$ROOT/scripts/common/runtime.sh" || exit 1' in launcher
    assert '"$FAST_INFER_PYTHON" - "$run_root"' in launcher
    assert "python3" not in launcher
