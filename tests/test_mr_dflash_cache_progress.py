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


def test_parallel_watcher_uses_tqdm_by_default(monkeypatch) -> None:
    import watch_parallel_stage

    calls = []
    monkeypatch.setattr(
        watch_parallel_stage,
        "_watch_tqdm",
        lambda path, *, interval, once: calls.append((path, interval, once)) or 17,
    )
    monkeypatch.setattr(
        watch_parallel_stage,
        "_watch_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("text watcher used")),
    )

    result = watch_parallel_stage.main(["--status", "/tmp/status.json", "--once"])

    assert result == 17
    assert calls == [(Path("/tmp/status.json"), 5.0, True)]


def test_parallel_watcher_prefers_global_input_total_over_partial_aggregate() -> None:
    from watch_parallel_stage import _fixed_total

    assert _fixed_total(aggregate_total=25, input_total=100) == 100
    assert _fixed_total(aggregate_total=100, input_total=0) == 100


def test_parallel_watcher_reads_cache_total_from_tokenized_manifest(tmp_path: Path) -> None:
    import json

    from watch_parallel_stage import _input_sample_count

    tokenized_path = tmp_path / "tokenized"
    tokenized_path.mkdir()
    (tokenized_path / "manifest.json").write_text(
        json.dumps({"num_samples": 5}),
        encoding="utf-8",
    )
    status_path = tmp_path / "status.json"
    (tmp_path / "parallel_plan.json").write_text(
        json.dumps(
            {
                "mode": "cache",
                "input": str(tmp_path / "regenerated.jsonl"),
                "tokenized_path": str(tokenized_path),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "regenerated.jsonl").write_text(
        '{"id":"s0"}\n{"id":"s1"}\n',
        encoding="utf-8",
    )

    assert _input_sample_count(status_path) == 5
