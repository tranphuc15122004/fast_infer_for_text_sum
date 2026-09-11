"""Contract tests cho runner smoke MR-DFlash trên Modal."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_modal_smoke_plan_uses_full_context_and_tracker_paths() -> None:
    from modal_mr_dflash_preprocess import build_smoke_plan

    plan = build_smoke_plan(
        remote_root=Path("/workspace/fast_infer_text_sum"),
        run_root=Path("/mnt/mr-dflash/runs/test"),
        target_model="Qwen/Qwen3-4B",
        max_length=32768,
        max_new_tokens=64,
    )

    assert plan["regenerate"][plan["regenerate"].index("--preserve-full-input")] == "--preserve-full-input"
    assert plan["regenerate"][plan["regenerate"].index("--output-batch-size") + 1] == "1"
    assert plan["cache"][plan["cache"].index("--batch-size") + 1] == "1"
    layer_flag = plan["cache"].index("--target-layer-ids")
    assert plan["cache"][layer_flag + 1 : layer_flag + 6] == ["1", "9", "17", "25", "33"]
    assert str(plan["regenerate_status"]).endswith(".parallel_regenerate_train/status.json")
    assert str(plan["cache_status"]).endswith(".parallel_cache_train/status.json")


def test_modal_smoke_report_requires_tracking_and_cache_audit() -> None:
    from modal_mr_dflash_preprocess import validate_smoke_report

    report = validate_smoke_report(
        {
            "regenerate": {"status": "success", "tracker": True},
            "cache": {"status": "success", "tracker": True},
            "audit": {"valid": True},
        }
    )
    assert report["status"] == "success"
