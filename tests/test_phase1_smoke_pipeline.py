from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mr_dflash"))


def _rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": f"sample-{index}",
            "source": "sharegpt" if index % 2 == 0 else "arxiv",
            "conversations": [{"role": "user", "content": f"prompt {index}"}],
        }
        for index in range(count)
    ]


def test_select_subset_is_deterministic_and_preserves_unique_rows() -> None:
    from run_phase1_smoke import select_subset

    rows = _rows(100)
    first = select_subset(rows, limit=20, seed=42)
    second = select_subset(rows, limit=20, seed=42)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 20
    assert len({row["id"] for row in first}) == 20
    assert rows[0]["id"] == "sample-0"


def test_build_stage_plan_runs_complete_phase1_chain(tmp_path: Path) -> None:
    from run_phase1_smoke import SmokeOptions, build_stage_plan

    options = SmokeOptions(
        input_path=tmp_path / "pilot_prompts.jsonl",
        output_root=tmp_path / "phase1_smoke",
        target_model_path="/models/Qwen3-4B",
        num_samples=20,
    )
    plan = build_stage_plan(options)

    assert [stage.name for stage in plan] == [
        "regenerate_smoke_train",
        "validate_smoke_train",
        "tokenize_smoke_train",
        "cache_smoke_train",
    ]
    assert all(stage.command[0] == sys.executable for stage in plan)
    assert "regenerate_pilot.py" in plan[0].command[1]
    assert "validate_pilot_dataset.py" in plan[1].command[1]
    assert "tokenize_dataset.py" in plan[2].command[1]
    assert "cache_target_features.py" in plan[3].command[1]
    assert "--require-generated" in plan[1].command
    assert options.output_root / "target_features_smoke" / "train" / "manifest.json" in plan[3].artifacts


def test_build_stage_plan_keeps_cache_provenance_paths_aligned(tmp_path: Path) -> None:
    from run_phase1_smoke import SmokeOptions, build_stage_plan

    options = SmokeOptions(
        input_path=tmp_path / "pilot_prompts.jsonl",
        output_root=tmp_path / "phase1_smoke",
        target_model_path="/models/Qwen3-4B",
        max_length=3072,
        target_layer_ids=(1, 9, 17, 25, 33),
    )
    plan = build_stage_plan(options)

    regenerated = str(options.output_root / "regenerated_smoke" / "train.jsonl")
    tokenized = str(options.output_root / "tokenized_smoke" / "train")
    cache = str(options.output_root / "target_features_smoke" / "train")
    assert regenerated in plan[0].command
    assert regenerated in plan[1].command
    assert regenerated in plan[2].command
    assert tokenized in plan[3].command
    assert cache in plan[3].command


def test_materialize_subset_writes_audit_manifest_without_touching_source(tmp_path: Path) -> None:
    import json

    from run_phase1_smoke import SmokeOptions, materialize_subset

    source = tmp_path / "pilot_prompts.jsonl"
    source_text = "\n".join(json.dumps(row) for row in _rows(8)) + "\n"
    source.write_text(source_text, encoding="utf-8")
    options = SmokeOptions(
        input_path=source,
        output_root=tmp_path / "phase1_smoke",
        target_model_path="/models/Qwen3-4B",
        num_samples=4,
    )

    manifest = materialize_subset(options)

    subset = options.output_root / "normalized" / "smoke_prompts.jsonl"
    assert subset.is_file()
    assert len(subset.read_text(encoding="utf-8").splitlines()) == 4
    assert manifest["num_selected"] == 4
    assert sum(manifest["source_counts"].values()) == 4
    assert set(manifest["source_counts"]) == {"arxiv", "sharegpt"}
    assert source.read_text(encoding="utf-8") == source_text
