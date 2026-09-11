from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mr_dflash"))


def test_cache_progress_bar_uses_valid_sample_total_and_resume_offset() -> None:
    from cache_target_features import _cache_progress_kwargs

    kwargs = _cache_progress_kwargs(
        {"valid_samples": 100_000, "existing_samples": 12_345}
    )

    assert kwargs["total"] == 100_000
    assert kwargs["initial"] == 12_345
    assert kwargs["unit"] == "sample"
    assert kwargs["desc"] == "Cache target features"


def test_parallel_progress_uses_sample_counter_for_cache_workers() -> None:
    from watch_parallel_stage import _progress_bar_state

    state = _progress_bar_state(
        {
            "phase": "capturing",
            "total_samples": 100_000,
            "completed_samples": 12_345,
            "sample_id": "sample-12345",
        }
    )

    assert state == {
        "mode": "sample",
        "total": 100_000,
        "n": 12_345,
        "unit": "sample",
    }


def test_parallel_progress_keeps_token_counter_for_generation_workers() -> None:
    from watch_parallel_stage import _progress_bar_state

    state = _progress_bar_state(
        {
            "phase": "generating",
            "generation_budget": 256,
            "generated_tokens": 64,
        }
    )

    assert state == {
        "mode": "token",
        "total": 256,
        "n": 64,
        "unit": "token",
    }
