"""Contract tests for the Modal vLLM + Phase 1 smoke runner."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_vllm_server_command_forces_flashattention_and_writes_metrics() -> None:
    from modal_mr_dflash_vllm_smoke import build_vllm_server_command

    command = build_vllm_server_command(
        model="Qwen/Qwen3-4B",
        port=38147,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=16384,
    )

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-4B"]
    assert command[command.index("--port") + 1] == "38147"
    assert command[command.index("--attention-config") + 1] == '{"backend":"FLASH_ATTN"}'
    assert "--kv-cache-metrics" in command


def test_vllm_phase1_plan_uses_vllm_only_for_regeneration() -> None:
    from modal_mr_dflash_vllm_smoke import build_vllm_phase1_plan

    plan = build_vllm_phase1_plan(
        remote_root=Path("/workspace/fast_infer_text_sum"),
        run_root=Path("/mnt/mr-dflash/runs/test"),
        target_model="Qwen/Qwen3-4B",
        server_address="http://127.0.0.1:38147/v1",
        max_length=2048,
        max_new_tokens=32,
        request_concurrency=8,
        max_batched_tokens=16384,
    )

    regenerate = plan["regenerate"]
    assert "--regenerate-backend" in regenerate
    assert regenerate[regenerate.index("--regenerate-backend") + 1] == "vllm"
    assert regenerate[regenerate.index("--vllm-server-addresses") + 1] == "http://127.0.0.1:38147/v1"
    assert "--device" not in regenerate

    assert "--attention-backend" in plan["cache"]
    assert plan["cache"][plan["cache"].index("--attention-backend") + 1] == "sdpa"
    assert "--regenerate-backend" not in plan["cache"]


def test_vllm_smoke_report_requires_all_phase1_artifacts() -> None:
    from modal_mr_dflash_vllm_smoke import validate_vllm_smoke_report

    report = validate_vllm_smoke_report(
        {
            "server": {"status": "ready"},
            "regenerate": {"status": "success"},
            "validate": {"status": "success"},
            "tokenize": {"status": "success"},
            "cache": {"status": "success"},
            "audit": {"valid": True},
        }
    )
    assert report["status"] == "success"
