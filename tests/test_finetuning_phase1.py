from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts" / "run_finetuning_b200.py"


def _launcher_module():
    spec = importlib.util.spec_from_file_location("run_finetuning_b200", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase1_plan_validates_teacher_before_caching_and_excludes_training(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    args = launcher._parser().parse_args(
        [
            "--config", str(tmp_path / "config.yaml"),
            "--train-input", str(tmp_path / "train.jsonl"),
            "--eval-input", str(tmp_path / "eval.jsonl"),
            "--output-root", str(tmp_path / "run"),
            "--nproc-per-node", "2",
            "--phase1-only",
        ]
    )
    config = {
        "model": {
            "target_model_path": str(tmp_path / "qwen3"),
            "torch_dtype": "bfloat16",
            "num_draft_layers": 5,
        },
        "data": {
            "max_length": 2048,
            "max_source_tokens": 1536,
            "max_summary_tokens": 384,
            "chat_template": "qwen3",
            "prompt_template": "Tóm tắt: {document}",
        },
        "training": {"adaptive_batch_size": False},
    }
    paths = launcher.resolve_paths(args, "phase1-test")

    commands = launcher.build_commands(
        config, paths, args, python_bin=sys.executable, nproc_per_node=2
    )

    names = [name for name, _command, _artifact in commands]
    assert names == [
        "generate_train",
        "generate_eval",
        "validate_teacher_train",
        "validate_teacher_eval",
        "cache_train",
        "cache_eval",
    ]
    assert "--nproc_per_node" in commands[0][1]
    assert "Finetuning.validate_targets" in commands[2][1]
    assert "Finetuning.capture_features" in commands[4][1]


@pytest.mark.parametrize("unsafe_flag", ["--skip-validation", "--validate-warn-only"])
def test_phase1_refuses_options_that_bypass_the_quality_gate(
    tmp_path: Path, unsafe_flag: str
) -> None:
    launcher = _launcher_module()
    args = launcher._parser().parse_args(
        [
            "--config", str(tmp_path / "config.yaml"),
            "--train-input", str(tmp_path / "train.jsonl"),
            "--eval-input", str(tmp_path / "eval.jsonl"),
            "--output-root", str(tmp_path / "run"),
            unsafe_flag,
        ]
    )

    with pytest.raises(launcher.LauncherError, match="quality gate cannot be bypassed"):
        launcher.build_commands({}, None, args, python_bin=sys.executable, nproc_per_node=2)


def test_phase1_refuses_partial_stage_selection_that_skips_the_quality_gate(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    args = launcher._parser().parse_args(
        [
            "--config", str(tmp_path / "config.yaml"),
            "--train-input", str(tmp_path / "train.jsonl"),
            "--eval-input", str(tmp_path / "eval.jsonl"),
            "--output-root", str(tmp_path / "run"),
            "--stages", "cache_train",
        ]
    )

    with pytest.raises(launcher.LauncherError, match="partial stage selection is disabled"):
        launcher.build_commands({}, None, args, python_bin=sys.executable, nproc_per_node=2)


def test_anomaly_filter_writes_a_separate_teacher_file_used_by_cache(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    args = launcher._parser().parse_args(
        [
            "--config", str(tmp_path / "config.yaml"),
            "--train-input", str(tmp_path / "train.jsonl"),
            "--eval-input", str(tmp_path / "eval.jsonl"),
            "--output-root", str(tmp_path / "run"),
            "--nproc-per-node", "2",
            "--phase1-only",
            "--filter-anomalies",
        ]
    )
    config = {
        "model": {"target_model_path": str(tmp_path / "qwen3"), "num_draft_layers": 5},
        "data": {
            "max_length": 2048,
            "max_source_tokens": 1536,
            "max_summary_tokens": 384,
            "prompt_template": "Tóm tắt: {document}",
        },
    }
    paths = launcher.resolve_paths(args, "phase1-test")

    commands = launcher.build_commands(
        config, paths, args, python_bin=sys.executable, nproc_per_node=2
    )
    by_name = {name: command for name, command, _artifact in commands}

    assert "--clean-output" in by_name["validate_teacher_train"]
    clean_path = by_name["validate_teacher_train"][
        by_name["validate_teacher_train"].index("--clean-output") + 1
    ]
    cache_input = by_name["cache_train"][by_name["cache_train"].index("--input") + 1]
    assert cache_input == clean_path
    assert cache_input != str(paths.teacher_train)


def test_resume_rejects_a_teacher_report_that_failed_the_gate(tmp_path: Path) -> None:
    launcher = _launcher_module()
    report = tmp_path / "teacher_report.json"
    report.write_text('{"passed_gate": false}\n', encoding="utf-8")

    with pytest.raises(launcher.LauncherError, match="failed the quality gate"):
        launcher._valid_report(report)


def test_resume_refuses_changed_source_data_under_the_same_run_root(tmp_path: Path) -> None:
    launcher = _launcher_module()
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    train.write_text('{"id":"1","document":"a","summary":"b"}\n', encoding="utf-8")
    evaluation.write_text('{"id":"2","document":"c","summary":"d"}\n', encoding="utf-8")
    args = launcher._parser().parse_args(
        [
            "--config", str(tmp_path / "config.yaml"),
            "--train-input", str(train),
            "--eval-input", str(evaluation),
            "--target-model-path", str(tmp_path / "model"),
            "--output-root", str(tmp_path / "run"),
        ]
    )
    config = {"model": {"target_model_path": str(tmp_path / "model")}}
    paths = launcher.resolve_paths(args, "same-run")
    launcher._write_run_manifest(paths, config=config, args=args, nproc_per_node=2)

    train.write_text('{"id":"1","document":"changed","summary":"b"}\n', encoding="utf-8")

    with pytest.raises(launcher.LauncherError, match="use a new --output-root"):
        launcher._write_run_manifest(paths, config=config, args=args, nproc_per_node=2)
