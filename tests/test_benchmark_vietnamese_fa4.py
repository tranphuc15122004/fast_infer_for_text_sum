"""Tests for LongBench + Vietnamese dataset support in native FA4 benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from Benchmark.common.benchmark_data import (
    LONGBENCH_DATASETS,
    VIETNAMESE_DATASETS,
    DATASETS,
    render_prompt,
    load_prompt_templates,
    VIETNAMESE_PROMPT_TEMPLATE,
    read_jsonl,
)
from Benchmark.native_fa4_benchmark import (
    resolve_dataset_file,
    detect_default_datasets,
    _validate_longbench_data,
)
from Benchmark.fa4_server import build_parser, runner_kwargs


ROOT = Path(__file__).resolve().parents[1]
LONGBENCH_DIR = ROOT / "data" / "longbench_100_14k"
EVAL_100_DIR = ROOT / "data" / "eval_100"


def test_dataset_constants():
    assert "gov_report" in LONGBENCH_DATASETS
    assert "vietnews" in VIETNAMESE_DATASETS
    assert "wikilingua" in VIETNAMESE_DATASETS
    assert "vims" in VIETNAMESE_DATASETS
    assert "vlsp" in VIETNAMESE_DATASETS
    for name in LONGBENCH_DATASETS:
        assert name in DATASETS
    for name in VIETNAMESE_DATASETS:
        assert name in DATASETS


def test_render_prompt_vietnamese():
    row = {
        "dataset": "vietnews",
        "document": "Nội dung bài báo tiếng Việt.",
        "reference": "Bản tóm tắt mẫu.",
    }
    prompt = render_prompt(row)
    assert prompt == VIETNAMESE_PROMPT_TEMPLATE.format(document="Nội dung bài báo tiếng Việt.")
    assert "Hãy tóm tắt văn bản sau bằng tiếng Việt" in prompt
    assert "Nội dung bài báo tiếng Việt." in prompt


def test_render_prompt_longbench():
    row = {
        "dataset": "gov_report",
        "context": "Report content here.",
        "input": "",
    }
    prompt = render_prompt(row)
    assert "Report content here." in prompt
    assert "one-page summary" in prompt


def test_load_prompt_templates_contains_all():
    templates = load_prompt_templates()
    for name in LONGBENCH_DATASETS:
        assert name in templates
    for name in VIETNAMESE_DATASETS:
        assert name in templates
        assert templates[name] == VIETNAMESE_PROMPT_TEMPLATE


def test_resolve_dataset_file_longbench():
    if LONGBENCH_DIR.is_dir():
        path = resolve_dataset_file(LONGBENCH_DIR, "gov_report")
        assert path.is_file()
        assert path.name == "gov_report.jsonl"


def test_resolve_dataset_file_vietnamese():
    if EVAL_100_DIR.is_dir():
        for name in VIETNAMESE_DATASETS:
            path = resolve_dataset_file(EVAL_100_DIR, name)
            assert path.is_file()
            assert path.name in (f"{name}_100.jsonl", f"{name}.jsonl")


def test_detect_default_datasets():
    if LONGBENCH_DIR.is_dir():
        detected_lb = detect_default_datasets(LONGBENCH_DIR)
        assert set(detected_lb) == set(LONGBENCH_DATASETS)
    if EVAL_100_DIR.is_dir():
        detected_vn = detect_default_datasets(EVAL_100_DIR)
        assert set(detected_vn) == set(VIETNAMESE_DATASETS)


def test_validate_data_vietnamese_eval_100():
    if EVAL_100_DIR.is_dir():
        result = _validate_longbench_data(VIETNAMESE_DATASETS, data_root=EVAL_100_DIR)
        assert result["schema_version"] == "vietnamese-eval_100-v1"
        assert set(result["datasets"].keys()) == set(VIETNAMESE_DATASETS)
        for name in VIETNAMESE_DATASETS:
            entry = result["datasets"][name]
            assert entry["sample_count"] == 100
            assert len(entry["sha256"]) == 64


def test_validate_data_longbench():
    if LONGBENCH_DIR.is_dir() and (LONGBENCH_DIR / "manifest.json").is_file():
        result = _validate_longbench_data(LONGBENCH_DATASETS, data_root=LONGBENCH_DIR)
        assert result["schema_version"] == "longbench-canonical-v1"
        assert set(result["datasets"].keys()) == set(LONGBENCH_DATASETS)


def test_fa4_server_cli_supports_vietnamese_data_and_datasets():
    parser = build_parser()
    args = parser.parse_args([
        "--data-dir", str(EVAL_100_DIR),
        "--datasets", "vietnews,wikilingua",
        "--mode", "smoke",
    ])
    kwargs = runner_kwargs(args)
    assert kwargs["data_dir"] == str(EVAL_100_DIR)
    assert kwargs["datasets"] == "vietnews,wikilingua"
    assert kwargs["mode"] == "smoke"
