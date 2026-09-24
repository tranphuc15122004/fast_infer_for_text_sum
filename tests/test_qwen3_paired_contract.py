from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.qwen3_paired import (  # noqa: E402
    DEFAULT_INPUT_CAP,
    DEFAULT_SPEED_OUTPUT_TOKENS,
    build_run_config,
    input_distribution,
    prepare_input_ids,
    profile_defaults,
)
from run_qwen3_4b_paired import (  # noqa: E402
    snapshot_dataset,
    validate_paired_records,
)
from common.vanilla_inference import build_parser as build_vanilla_parser  # noqa: E402
from common.vanilla_inference import _timed_generate  # noqa: E402


def test_speed_profiles_use_exact_fixed_output_budgets():
    assert profile_defaults("smoke")["max_new_tokens"] == 32
    assert profile_defaults("full")["max_new_tokens"] == DEFAULT_SPEED_OUTPUT_TOKENS
    assert profile_defaults("full")["max_samples"] == 100


def test_prepare_input_ids_reports_deterministic_head_tail_truncation():
    input_ids = torch.arange(DEFAULT_INPUT_CAP + 10).reshape(1, -1)

    result = prepare_input_ids(input_ids, DEFAULT_INPUT_CAP)

    assert result.input_tokens == DEFAULT_INPUT_CAP
    assert result.original_input_tokens == DEFAULT_INPUT_CAP + 10
    assert result.input_truncated is True
    assert result.input_ids.shape == (1, DEFAULT_INPUT_CAP)
    assert torch.equal(result.input_ids[:, :4], input_ids[:, :4])
    assert torch.equal(result.input_ids[:, -4:], input_ids[:, -4:])


def test_build_run_config_excludes_non_workload_metadata():
    config = build_run_config(
        dataset="gov_report",
        target_model="Qwen3-4B",
        input_cap=14000,
        output_tokens=1024,
        seed=42,
        dtype="bfloat16",
        attention_backend="eager",
        extra={"run_id": "should-not-affect-pair"},
    )

    assert config["dataset"] == "gov_report"
    assert config["output_tokens"] == 1024
    assert "run_id" not in config


def test_input_distribution_reports_per_dataset_and_global_percentiles(tmp_path):
    path = tmp_path / "gov_report.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": i, "input_tokens": value})
            for i, value in enumerate((100, 200, 300))
        )
        + "\n",
        encoding="utf-8",
    )

    distribution = input_distribution(tmp_path)

    assert distribution["datasets"]["gov_report"]["count"] == 3
    assert distribution["datasets"]["gov_report"]["max"] == 300
    assert distribution["global"]["p50"] == pytest.approx(200.0)


def test_paired_record_validator_requires_raw_fixed_budget_and_online_speedups():
    row = {
        "status": "success",
        "batch_size": 1,
        "sample_id": "s1",
        "prompt_hash": "prompt",
        "run_config_hash": "config",
        "output_tokens": 4,
        "speed_output_tokens": 4,
        "dense_output_tokens": 4,
        "measurement_scope": "full_e2e",
        "prefill_ms": 10.0,
        "decode_ms": 5.0,
        "e2e_ms": 20.0,
        "throughput_tok_s": 200.0,
        "decode_throughput_tok_s": 800.0,
        "dense_prefill_ms": 8.0,
        "dense_decode_ms": 4.0,
        "dense_e2e_ms": 24.0,
        "method_decode_tok_s": 800.0,
        "dense_decode_tok_s": 1000.0,
        "generated_token_ids": [1, 2, 3, 4],
        "fixed_budget_reached": True,
        "quality_text": "answer",
        "full_output_text": "answer",
        "quality_output_tokens": 4,
        "acceptance_lengths": [3, 2],
        "avg_accept_length": 2.5,
        "tau_scope": "fixed_budget",
        "reference_baseline": "vanilla_hf",
        "reference_run_id": "run-1",
        "speedup_scope": "qwen3_4b_batch1_fixed_budget",
        "output_parity_available": True,
        "fixed_continuation_exact_match": True,
        "quality_prefix_exact_match": True,
        "speedup_valid": True,
        "esr": 1.2,
        "dsr": 0.8,
    }

    assert validate_paired_records([row], baseline="dflash", expected_samples=1, output_tokens=4) == []
    errors = validate_paired_records(
        [{**row, "speedup_valid": False, "output_tokens": 3}],
        baseline="dflash",
        expected_samples=1,
        output_tokens=4,
    )
    assert any("output_tokens" in error for error in errors)
    assert any("speedup_valid" in error for error in errors)


def test_paired_record_validator_rejects_missing_main_timing_phases():
    row = {
        "status": "success",
        "batch_size": 1,
        "sample_id": "s1",
        "output_tokens": 2,
        "speed_output_tokens": 2,
        "generated_token_ids": [1, 2],
        "fixed_budget_reached": True,
        "quality_text": "answer",
        "full_output_text": "answer",
        "quality_output_tokens": 2,
        "output_parity_available": True,
        "fixed_continuation_exact_match": True,
        "quality_prefix_exact_match": True,
        "prompt_hash": "prompt",
        "run_config_hash": "config",
        "reference_baseline": "vanilla_hf",
        "reference_run_id": "run-1",
        "speedup_scope": "qwen3_4b_batch1_fixed_budget",
        "speedup_valid": True,
        "esr": 1.0,
        "dsr": 1.0,
    }

    errors = validate_paired_records(
        [row], baseline="vanilla_fa", expected_samples=1, output_tokens=2
    )

    assert any("measurement_scope" in error for error in errors)
    assert any("prefill_ms" in error for error in errors)
    assert any("dense_prefill_ms" in error for error in errors)


def test_dataset_snapshot_freezes_exact_selected_source_rows(tmp_path):
    source = tmp_path / "gov_report.jsonl"
    snapshot = tmp_path / "inputs" / "gov_report.source.jsonl"
    raw_lines = [
        '{"id":"a","prompt":"first"}',
        '{"id":"b","prompt":"second"}',
        '{"id":"c","prompt":"not selected"}',
    ]
    source.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    count = snapshot_dataset(source, snapshot, max_samples=2)

    assert count == 2
    assert snapshot.read_text(encoding="utf-8").splitlines() == raw_lines[:2]


def test_vanilla_parser_exposes_fixed_output_and_pairing_arguments():
    args = build_vanilla_parser("eager", "test").parse_args(
        [
            "--model",
            "Qwen3-4B",
            "--data-file",
            "data.jsonl",
            "--output",
            "out.jsonl",
            "--fixed-output-tokens",
            "1024",
            "--reference-file",
            "ref.jsonl",
            "--run-config-hash",
            "cfg",
        ]
    )

    assert args.fixed_output_tokens == 1024
    assert args.reference_file == "ref.jsonl"
    assert args.run_config_hash == "cfg"


def test_vanilla_fixed_output_ignores_eos_until_the_budget():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, input_ids, **_kwargs):
            logits = torch.zeros((1, input_ids.shape[1], 3))
            logits[:, :, 0] = 10.0  # tokenizer EOS on every step
            return SimpleNamespace(logits=logits, past_key_values=None)

    tokenizer = SimpleNamespace(eos_token_id=0)
    args = SimpleNamespace(
        max_new_tokens=4,
        fixed_output_tokens=4,
        temperature=0.0,
        dtype="float32",
        attention_backend="eager",
    )
    output_ids, _ = _timed_generate(
        FakeModel(), torch.tensor([[1, 2]]), tokenizer, args, torch.device("cpu")
    )

    assert output_ids.shape[1] == 2 + 4
