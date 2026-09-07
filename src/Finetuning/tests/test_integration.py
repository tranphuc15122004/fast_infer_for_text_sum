from __future__ import annotations

import json
import math

import pytest

try:
    from Finetuning.config import RunConfig, load_run_config
    from Finetuning.run_train import evaluate_checkpoint, run_training
except ModuleNotFoundError as exc:  # Red phase: assembly is not ported yet.
    _IMPORT_ERROR = exc


def _require_api() -> None:
    if "_IMPORT_ERROR" in globals():
        pytest.fail(f"Finetuning integration API is not implemented: {_IMPORT_ERROR}")


def test_synthetic_end_to_end_training(tmp_path) -> None:
    _require_api()
    config = RunConfig.from_synthetic(
        output_dir=tmp_path / "run",
        max_steps=4,
        batch_size=1,
        eval_interval=1,
        attention_backend="eager",
    )
    final_checkpoint = run_training(config)
    assert final_checkpoint.is_dir()
    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    steps = [row for row in records if row.get("type") == "step"]
    assert len(steps) == 4
    losses = [row["loss"] for row in steps]
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]
    restored = evaluate_checkpoint(final_checkpoint, config)
    assert math.isfinite(restored["loss"])


def test_cli_rejects_missing_feature_source(tmp_path) -> None:
    _require_api()
    path = tmp_path / "invalid.yaml"
    path.write_text("model: {}\ndata: {}\ntraining: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_run_config(path)
