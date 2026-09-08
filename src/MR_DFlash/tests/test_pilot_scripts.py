"""Smoke tests không cần model thật cho các bước prepare/split/regenerate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_prepare_split_regenerate_validate_smoke(tmp_path: Path) -> None:
    from build_pilot_dataset import main as build_main
    from prepare_arxiv import main as arxiv_main
    from prepare_sharegpt import main as sharegpt_main
    from regenerate_pilot import main as regenerate_main
    from validate_pilot_dataset import main as validate_main

    share_raw = tmp_path / "share.jsonl"
    arxiv_raw = tmp_path / "arxiv.jsonl"
    _write_jsonl(
        share_raw,
        [
            {"id": "1", "conversations": [{"from": "human", "value": "Question one"}, {"from": "gpt", "value": "old"}]},
            {"id": "2", "conversations": [{"from": "human", "value": "Question two"}]},
        ],
    )
    _write_jsonl(arxiv_raw, [{"id": "p1", "text": "A paper document."}, {"id": "p2", "article": "Another document."}, {"id": "p3", "body": "Third document."}])
    normalized = tmp_path / "normalized"
    share_path = normalized / "sharegpt_prompts.jsonl"
    arxiv_path = normalized / "arxiv_prompts.jsonl"
    sharegpt_main(["--input", str(share_raw), "--output", str(share_path)])
    arxiv_main(["--input", str(arxiv_raw), "--output", str(arxiv_path)])
    build_main([
        "--sharegpt", str(share_path),
        "--arxiv", str(arxiv_path),
        "--output-root", str(tmp_path),
        "--sharegpt-count", "2",
        "--arxiv-count", "3",
        "--allow-short",
    ])
    responses = [{"id": row["id"], "assistant": f"Generated response for {row['id']}"} for row in map(json.loads, (tmp_path / "normalized" / "train_prompts.jsonl").read_text().splitlines())]
    response_path = tmp_path / "responses.jsonl"
    _write_jsonl(response_path, responses)
    regenerated = tmp_path / "regenerated" / "train.jsonl"
    regenerate_main([
        "--input", str(tmp_path / "normalized" / "train_prompts.jsonl"),
        "--output", str(regenerated),
        "--responses-jsonl", str(response_path),
        "--skipped-report", str(tmp_path / "regenerated" / "train.skipped.jsonl"),
    ])
    validate_main(["--input", str(regenerated), "--require-generated"])
    rows = list(regenerated.open("r", encoding="utf-8"))
    assert rows
    assert (tmp_path / "regenerated" / "train.skipped.jsonl").exists()


def test_regeneration_skips_invalid_sample_and_records_reason(tmp_path: Path) -> None:
    from regenerate_pilot import main as regenerate_main

    source = tmp_path / "prompts.jsonl"
    _write_jsonl(
        source,
        [
            {
                "id": "valid",
                "source": "sharegpt",
                "conversations": [{"role": "user", "content": "Question"}],
            },
            {
                "id": "invalid",
                "source": "sharegpt",
                "conversations": [],
            },
        ],
    )
    responses = tmp_path / "responses.jsonl"
    _write_jsonl(responses, [{"id": "valid", "assistant": "Answer"}])
    output = tmp_path / "regenerated.jsonl"
    skipped = tmp_path / "regenerated.skipped.jsonl"
    regenerate_main(
        [
            "--input", str(source),
            "--output", str(output),
            "--responses-jsonl", str(responses),
            "--sample-error-policy", "skip",
            "--skipped-report", str(skipped),
        ]
    )
    assert [json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()] == ["valid"]
    skipped_rows = [json.loads(line) for line in skipped.read_text(encoding="utf-8").splitlines()]
    assert skipped_rows == [
        {
            "id": "invalid",
            "kind": "invalid",
            "error": "sample không có conversations",
            "prompt_tokens": None,
            "max_length": 8192,
            "requested_max_new_tokens": 768,
        }
    ]


def test_regeneration_repairs_truncated_last_jsonl_line(tmp_path: Path) -> None:
    from regenerate_pilot import _load_status_ids

    path = tmp_path / "status.jsonl"
    path.write_text(
        '{"id": "complete"}\n{"id": "interrupted"',
        encoding="utf-8",
    )
    assert _load_status_ids(path) == {"complete"}
    assert path.read_text(encoding="utf-8") == '{"id": "complete"}\n'


def test_validate_pilot_dataset_writes_machine_readable_report(tmp_path: Path) -> None:
    from validate_pilot_dataset import main as validate_main

    source = tmp_path / "regenerated.jsonl"
    _write_jsonl(
        source,
        [
            {
                "id": "sharegpt_1",
                "source": "sharegpt",
                "conversations": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                "metadata": {"generation_model": "/models/Qwen3-4B"},
            }
        ],
    )
    report = tmp_path / "reports" / "validation.json"
    validate_main(
        [
            "--input",
            str(source),
            "--expected-target-model",
            "/models/Qwen3-4B",
            "--require-generated",
            "--report",
            str(report),
        ]
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["valid"] == 1
    assert payload["source_counts"] == {"sharegpt": 1}


def test_prepare_sharegpt_accepts_json_array(tmp_path: Path) -> None:
    import json

    from prepare_sharegpt import main as sharegpt_main

    source = tmp_path / "ShareGPT_V3.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "conversation-1",
                    "conversations": [
                        {"from": "human", "value": "First question"},
                        {"from": "gpt", "value": "Old answer"},
                        {"from": "human", "value": "Final question"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "sharegpt_prompts.jsonl"
    sharegpt_main(["--input", str(source), "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["id"] == "sharegpt_conversation-1"
    assert rows[0]["conversations"][-1] == {
        "role": "user",
        "content": "Final question",
    }
    assert rows[0]["metadata"]["original_turn_count"] == 3


def test_prepare_sharegpt_skips_records_without_user_turn(tmp_path: Path) -> None:
    from prepare_sharegpt import main as sharegpt_main

    source = tmp_path / "ShareGPT_V3.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "system-only",
                    "conversations": [
                        {"from": "system", "value": "Instruction only"},
                    ],
                },
                {
                    "id": "valid",
                    "conversations": [
                        {"from": "human", "value": "A real question"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "sharegpt_prompts.jsonl"
    sharegpt_main(["--input", str(source), "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["sharegpt_valid"]


def test_prepare_arxiv_joins_paragraphs_and_preserves_reference(tmp_path: Path) -> None:
    import json

    from prepare_arxiv import main as arxiv_main

    source = tmp_path / "train.label.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "paper-1",
                "text": ["Paragraph one.", "Paragraph two."],
                "summary": ["Summary one.", "Summary two."],
                "label": [1, 3],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "arxiv_prompts.jsonl"
    arxiv_main(["--input", str(source), "--output", str(output)])
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["conversations"][0]["content"].endswith(
        "Paragraph one.\n\nParagraph two."
    )
    assert row["metadata"]["reference_summary"] == "Summary one.\n\nSummary two."
    assert row["metadata"]["label"] == [1, 3]


def test_prepare_server_sources_builds_current_pilot_layout(tmp_path: Path) -> None:
    import json

    from prepare_server_data import main as prepare_server_main

    share_source = tmp_path / "sharegpt.json"
    share_source.write_text(
        json.dumps(
            [
                {
                    "id": "s1",
                    "conversations": [
                        {"from": "human", "value": "Share question"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    arxiv_source = tmp_path / "arxiv.jsonl"
    arxiv_source.write_text(
        json.dumps({"id": "a1", "text": ["Document"], "summary": ["Reference"]})
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "pilot"
    prepare_server_main(
        [
            "--sharegpt-source",
            str(share_source),
            "--arxiv-source",
            str(arxiv_source),
            "--output-root",
            str(output_root),
            "--sharegpt-count",
            "1",
            "--arxiv-count",
            "1",
            "--allow-short",
        ]
    )
    assert (output_root / "normalized" / "train_prompts.jsonl").exists()
    manifest = json.loads(
        (output_root / "manifests" / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sharegpt_input"] == str(share_source)
    assert manifest["arxiv_input"] == str(arxiv_source)
    assert manifest["sharegpt_count"] == 1
    assert manifest["arxiv_count"] == 1


def test_analyze_pilot_data_reports_small_sample(tmp_path: Path) -> None:
    import json

    from analyze_pilot_data import main as analyze_main

    source = tmp_path / "prompts.jsonl"
    rows = [
        {
            "id": "sharegpt_s1",
            "source": "sharegpt",
            "conversations": [{"role": "user", "content": "Question"}],
            "metadata": {"source_length": 8},
        },
        {
            "id": "arxiv_a1",
            "source": "arxiv",
            "conversations": [{"role": "user", "content": "Document"}],
            "metadata": {"source_length": 8, "source_token_length": 2, "reference_summary": "Summary"},
        },
    ]
    _write_jsonl(source, rows)
    report_path = tmp_path / "analysis.json"
    analyze_main(["--input", str(source), "--output", str(report_path), "--limit", "2"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rows"] == 2
    assert report["source_counts"] == {"arxiv": 1, "sharegpt": 1}
    assert report["duplicate_ids"] == []
    assert report["reference_rows"] == 1


def test_server_data_defaults_balance_sharegpt_and_arxiv() -> None:
    from build_pilot_dataset import DEFAULT_ARXIV_COUNT, DEFAULT_SHAREGPT_COUNT
    from prepare_server_data import DEFAULT_ARXIV_COUNT as WRAPPER_ARXIV_COUNT
    from prepare_server_data import DEFAULT_OUTPUT_ROOT
    from prepare_server_data import DEFAULT_SHAREGPT_COUNT as WRAPPER_SHAREGPT_COUNT

    assert DEFAULT_SHAREGPT_COUNT == 50000
    assert DEFAULT_ARXIV_COUNT == 50000
    assert WRAPPER_SHAREGPT_COUNT == 50000
    assert WRAPPER_ARXIV_COUNT == 50000
    assert DEFAULT_OUTPUT_ROOT == (
        "/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot"
    )


def test_preprocess_pipeline_plan_contains_debuggable_stages(tmp_path: Path) -> None:
    from run_preprocess_pipeline import PipelineOptions, build_stage_plan

    options = PipelineOptions(
        repo_root=tmp_path,
        data_root=tmp_path / "pilot",
        target_model_path="/models/Qwen3-4B",
        sharegpt_count=4,
        arxiv_count=6,
        max_lengths=(3072, 8192),
        cache_splits=("train", "val"),
    )
    plan = build_stage_plan(options)
    names = [stage.name for stage in plan]
    assert names[0] == "prepare"
    assert "analyze" in names
    assert "regenerate_3k_train" in names
    assert "validate_8k_val" in names
    assert "tokenize_3k_train" in names
    assert "cache_8k_val" in names
    assert all(stage.command for stage in plan)
    assert all(str(tmp_path / "pilot") in " ".join(stage.command) for stage in plan)


def test_preprocess_pipeline_full_context_preserves_generation_input(tmp_path: Path) -> None:
    from run_preprocess_pipeline import PipelineOptions, build_stage_plan

    options = PipelineOptions(
        repo_root=tmp_path,
        data_root=tmp_path / "pilot",
        target_model_path="/models/Qwen3-4B",
        full_context=True,
        full_context_length=32768,
        max_new_tokens=2048,
        overflow_policy="skip",
        sample_error_policy="skip",
    )
    plan = build_stage_plan(options)
    names = [stage.name for stage in plan]
    assert "regenerate_full_train" in names
    assert "regenerate_3k_train" not in names
    stage = next(stage for stage in plan if stage.name == "regenerate_full_train")
    assert "--preserve-full-input" in stage.command
    assert "--overflow-policy" in stage.command
    assert stage.command[stage.command.index("--overflow-policy") + 1] == "skip"
    assert "--sample-error-policy" in stage.command
    assert stage.command[stage.command.index("--sample-error-policy") + 1] == "skip"
    assert tmp_path / "pilot" / "regenerated_full" / "train.skipped.jsonl" in stage.artifacts
    assert "32768" in stage.command
    assert str(tmp_path / "pilot" / "regenerated_full" / "train.jsonl") in stage.command


def test_generation_budget_clips_only_response_not_full_prompt() -> None:
    from regenerate_pilot import resolve_generation_budget

    assert resolve_generation_budget(30_000, 32_768, 2_048) == (2_048, False)
    assert resolve_generation_budget(32_000, 32_768, 2_048) == (768, True)
    with pytest.raises(ValueError, match="prompt đã chiếm"):
        resolve_generation_budget(32_768, 32_768, 2_048)


def test_preprocess_pipeline_plan_options_are_json_serializable(tmp_path: Path) -> None:
    import json

    from run_preprocess_pipeline import PipelineOptions, _options_payload

    options = PipelineOptions(
        repo_root=tmp_path,
        data_root=tmp_path / "pilot",
        target_model_path="/models/Qwen3-4B",
    )
    # pipeline_plan.json được ghi bằng _common.write_json, nên payload phải
    # serialize được ngay cả khi options dùng Path nội bộ.
    json.dumps(_options_payload(options))


def test_preprocess_pipeline_failure_state_is_resumable(tmp_path: Path) -> None:
    from run_preprocess_pipeline import (
        PipelineOptions,
        Stage,
        run_stage,
    )

    options = PipelineOptions(
        repo_root=tmp_path,
        data_root=tmp_path / "pilot",
        target_model_path="/models/Qwen3-4B",
        max_lengths=(3072,),
    )
    stage = Stage(
        name="unit_failure",
        command=["/bin/sh", "-c", "exit 7"],
        artifacts=[],
    )
    with pytest.raises(RuntimeError, match="unit_failure"):
        run_stage(stage, options, config_hash="test-hash")
    state = tmp_path / "pilot" / "pipeline_state" / "unit_failure.failed.json"
    assert state.exists()
    assert '"return_code": 7' in state.read_text(encoding="utf-8")


def test_preprocess_pipeline_success_marker_allows_resume(tmp_path: Path) -> None:
    from run_preprocess_pipeline import PipelineOptions, Stage, run_stage

    data_root = tmp_path / "pilot"
    artifact = data_root / "artifact.txt"
    options = PipelineOptions(
        repo_root=tmp_path,
        data_root=data_root,
        target_model_path="/models/Qwen3-4B",
    )
    stage = Stage(
        name="unit_success",
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(artifact)!r}).write_text('ok')",
        ],
        artifacts=[artifact],
    )
    first = run_stage(stage, options, config_hash="test-hash")
    second = run_stage(stage, options, config_hash="test-hash")
    assert first["status"] == "success"
    assert second["status"] == "skipped"
    assert artifact.read_text(encoding="utf-8") == "ok"
    assert (data_root / "pipeline_logs" / "unit_success.log").exists()


def test_preprocess_pipeline_lock_prevents_concurrent_writers(tmp_path: Path) -> None:
    from run_preprocess_pipeline import _PipelineLock

    first = _PipelineLock(tmp_path / "pilot", "hash-a")
    second = _PipelineLock(tmp_path / "pilot", "hash-b")
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="đang được một pipeline khác"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_parallel_regeneration_merge_preserves_input_order(tmp_path: Path) -> None:
    from parallel_stage import merge_regenerated_outputs

    source = tmp_path / "prompts.jsonl"
    _write_jsonl(
        source,
        [{"id": f"s{i}", "source": "sharegpt"} for i in range(4)],
    )
    worker_root = tmp_path / "workers"
    _write_jsonl(
        worker_root / "rank_00" / "output.jsonl",
        [{"id": "s0", "value": 0}, {"id": "s2", "value": 2}],
    )
    _write_jsonl(
        worker_root / "rank_01" / "output.jsonl",
        [{"id": "s1", "value": 1}, {"id": "s3", "value": 3}],
    )
    _write_jsonl(worker_root / "rank_00" / "skipped.jsonl", [])
    _write_jsonl(worker_root / "rank_01" / "skipped.jsonl", [])
    manifest = merge_regenerated_outputs(
        input_path=source,
        worker_roots=[worker_root / "rank_00", worker_root / "rank_01"],
        output_path=tmp_path / "merged.jsonl",
        skipped_path=tmp_path / "merged.skipped.jsonl",
        manifest_path=tmp_path / "merged.manifest.json",
        gpu_ids=[1, 2],
    )
    assert [row["id"] for row in map(json.loads, (tmp_path / "merged.jsonl").read_text().splitlines())] == [
        "s0", "s1", "s2", "s3"
    ]
    assert manifest["stats"]["written"] == 4
    assert manifest["stats"]["missing"] == 0


def test_parallel_pipeline_plan_targets_explicit_gpu_ids(tmp_path: Path) -> None:
    from run_preprocess_pipeline import PipelineOptions, build_stage_plan

    options = PipelineOptions(
        repo_root=tmp_path,
        data_root=tmp_path / "pilot",
        target_model_path="/models/Qwen3-4B",
        full_context=True,
        full_context_length=32768,
        max_new_tokens=2048,
        overflow_policy="skip",
        sample_error_policy="skip",
        parallel_gpu_ids=(1, 2, 3),
    )
    stage = next(stage for stage in build_stage_plan(options) if stage.name == "regenerate_full_train")
    assert stage.command[1].endswith("parallel_stage.py")
    assert stage.command[stage.command.index("--gpu-ids") + 1 : stage.command.index("--input")] == ["1", "2", "3"]
    cache_stage = next(stage for stage in build_stage_plan(options) if stage.name == "cache_full_train")
    assert cache_stage.command[1].endswith("parallel_stage.py")


def test_parallel_cache_merge_validates_all_samples(tmp_path: Path) -> None:
    from parallel_stage import merge_feature_caches
    from MR_DFlash.offline_features import write_feature_shard, write_sharded_feature_manifest

    worker_root = tmp_path / "workers" / "rank_00"
    worker_root.mkdir(parents=True)
    write_feature_shard(worker_root / "shard_00000.pt", [
        {"id": "s0", "input_ids": [1, 2], "loss_mask": [1, 1], "hidden_states": [[0.0, 1.0], [1.0, 2.0]]},
    ])
    write_sharded_feature_manifest(
        worker_root,
        target_model_path="tiny",
        feature_layer_ids=[1],
        hidden_size=2,
        feature_width=2,
        max_length=8,
        requested_torch_dtype="float32",
        shards=[{"path": "shard_00000.pt", "count": 1, "ids": ["s0"], "lengths": [2]}],
        sample_ids=["s0"],
    )
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, [{"id": "s0"}])
    manifest = merge_feature_caches(
        input_path=input_path,
        worker_roots=[worker_root],
        output_path=tmp_path / "merged_cache",
        manifest_path=tmp_path / "cache.manifest.json",
        gpu_ids=[1],
    )
    assert manifest["num_samples"] == 1
    assert (tmp_path / "merged_cache" / "manifest.json").exists()
