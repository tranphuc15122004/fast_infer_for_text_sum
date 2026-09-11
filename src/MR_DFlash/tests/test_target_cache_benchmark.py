"""Unit tests cho helper không cần tải target model của benchmark cache."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_benchmark_removes_only_final_assistant_from_regenerated_row() -> None:
    from target_cache_benchmark import generation_messages

    messages = [
        {"role": "user", "content": "Document"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Summarize it"},
        {"role": "assistant", "content": "Target answer"},
    ]

    assert generation_messages(messages) == messages[:-1]
    assert generation_messages(messages[:-1]) == messages[:-1]


def test_benchmark_token_report_detects_exact_and_non_exact_outputs() -> None:
    from target_cache_benchmark import compare_token_ids

    exact = compare_token_ids([1, 2, 3], [1, 2, 3])
    mismatch = compare_token_ids([1, 2, 3], [1, 9, 3, 4])

    assert exact == {"exact": True, "reference_length": 3, "candidate_length": 3, "first_mismatch": None}
    assert mismatch == {
        "exact": False,
        "reference_length": 3,
        "candidate_length": 4,
        "first_mismatch": 1,
    }


def test_benchmark_defaults_to_both_attention_backends() -> None:
    from target_cache_benchmark import resolve_attention_implementations

    assert resolve_attention_implementations(None, None) == ["sdpa", "flash_attention_2"]
    assert resolve_attention_implementations("sdpa", None) == ["sdpa"]
    assert resolve_attention_implementations(None, ["eager", "sdpa"]) == ["eager", "sdpa"]


def test_benchmark_rejects_conflicting_backend_options() -> None:
    import pytest

    from target_cache_benchmark import resolve_attention_implementations

    with pytest.raises(ValueError, match="không được dùng đồng thời"):
        resolve_attention_implementations("sdpa", ["flash_attention_2"])


def test_benchmark_builds_cross_backend_speed_and_output_report() -> None:
    from target_cache_benchmark import compare_backend_results

    reports = {
        "sdpa": {
            "status": "success",
            "hf_generate": {"generated_ids": [1, 2], "runs": [{"seconds": 4.0}]},
            "current_causal_lm_full_capture": {"runs": [{"seconds": 6.0}]},
            "backbone_only_full_capture": {"runs": [{"seconds": 3.0}]},
            "fused_generate_capture": {"runs": [{"seconds": 5.0}]},
            "summary": {"fused_generate_capture_s": 5.0},
        },
        "flash_attention_2": {
            "status": "success",
            "hf_generate": {"generated_ids": [1, 2], "runs": [{"seconds": 2.0}]},
            "current_causal_lm_full_capture": {"runs": [{"seconds": 3.0}]},
            "backbone_only_full_capture": {"runs": [{"seconds": 1.5}]},
            "fused_generate_capture": {"runs": [{"seconds": 2.5}]},
            "summary": {"fused_generate_capture_s": 2.5},
        },
    }

    comparison = compare_backend_results(reports)

    assert comparison["reference_backend"] == "sdpa"
    assert comparison["flash_attention_2_vs_sdpa"]["hf_generate_speedup"] == 2.0
    assert comparison["flash_attention_2_vs_sdpa"]["fused_generate_capture_speedup"] == 2.0
    assert comparison["cross_backend_output"]["flash_attention_2"]["exact"] is True
