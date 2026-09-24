from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.paired_reference import (  # noqa: E402
    attach_reference_timing,
    build_reference_key,
    extract_reference_sidecar,
    load_reference_sidecar,
    prompt_hash,
    write_reference_sidecar,
)
from common.benchmark_runtime import build_sample_record  # noqa: E402


def _reference_record(**overrides):
    record = {
        "dataset": "gov_report",
        "sample_id": "gov-0",
        "prompt_hash": prompt_hash([1, 2, 3]),
        "run_config_hash": "cfg-qwen3",
        "status": "success",
        "input_tokens": 3,
        "output_tokens": 1024,
        "prefill_ms": 10.0,
        "decode_ms": 100.0,
        "e2e_ms": 115.0,
        "batch_size": 1,
        "model": "/models/Qwen3-4B",
        "max_new_tokens": 1024,
    }
    record.update(overrides)
    return record


def _method_record(**overrides):
    record = {
        "method": "dflash",
        "dataset": "gov_report",
        "sample_id": "gov-0",
        "prompt_hash": prompt_hash([1, 2, 3]),
        "run_config_hash": "cfg-qwen3",
        "status": "success",
        "input_tokens": 3,
        "output_tokens": 1024,
        "prefill_ms": 5.0,
        "decode_ms": 50.0,
        "e2e_ms": 60.0,
        "batch_size": 1,
        "model": "/models/Qwen3-4B",
        "max_new_tokens": 1024,
    }
    record.update(overrides)
    return record


def test_reference_key_is_stable_for_the_same_sample_contract():
    key = build_reference_key(_reference_record())

    assert key == build_reference_key(_reference_record())
    assert key == (
        "gov_report",
        "gov-0",
        prompt_hash([1, 2, 3]),
        "cfg-qwen3",
    )


def test_prompt_hash_is_stable_for_token_ids_and_changes_when_input_changes():
    assert prompt_hash([1, 2, 3]) == prompt_hash([1, 2, 3])
    assert prompt_hash([1, 2, 3]) != prompt_hash([1, 2, 4])


def test_sidecar_round_trip_ignores_summary_records(tmp_path):
    sidecar = tmp_path / "vanilla_hf.jsonl"
    write_reference_sidecar(
        sidecar,
        [_reference_record(), {"type": "summary", "num_samples": 1}],
    )

    loaded = load_reference_sidecar(sidecar)

    assert list(loaded) == [build_reference_key(_reference_record())]
    assert loaded[build_reference_key(_reference_record())]["decode_ms"] == 100.0


def test_extract_reference_sidecar_validates_sample_rows(tmp_path):
    source = tmp_path / "vanilla.jsonl"
    destination = tmp_path / "sidecar.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in (_reference_record(), {"type": "summary", "num_samples": 1})
        )
        + "\n",
        encoding="utf-8",
    )

    assert extract_reference_sidecar(source, destination) == 1
    assert len(load_reference_sidecar(destination)) == 1


def test_attach_reference_computes_online_esr_and_token_normalized_dsr():
    record = attach_reference_timing(
        _method_record(),
        _reference_record(),
        reference_run_id="qwen3-ref-001",
        expected_output_tokens=1024,
    )

    assert record["speedup_valid"] is True
    assert record["dense_e2e_ms"] == 115.0
    assert record["dense_decode_ms"] == 100.0
    assert record["esr"] == pytest.approx(115.0 / 60.0)
    assert record["dsr"] == pytest.approx(2.0)
    assert record["reference_run_id"] == "qwen3-ref-001"
    assert record["speedup_scope"] == "qwen3_4b_batch1_fixed_budget"


def test_online_speedup_pair_does_not_require_matching_generated_text():
    reference = _reference_record(text="reference continuation")
    method = _method_record(text="different speculative continuation")

    record = attach_reference_timing(
        method,
        reference,
        expected_output_tokens=1024,
    )

    assert record["speedup_valid"] is True
    assert record["esr"] == pytest.approx(115.0 / 60.0)
    assert record["dsr"] == pytest.approx(2.0)


def test_online_pair_records_output_parity_separately_from_speedup_validity():
    reference = _reference_record(
        generated_token_ids=[10, 11, 99, 12],
        first_eos_index=2,
        quality_output_tokens=2,
    )
    method = _method_record(
        generated_token_ids=[10, 12, 99, 13],
        first_eos_index=2,
        quality_output_tokens=2,
    )

    record = attach_reference_timing(
        method,
        reference,
        expected_output_tokens=1024,
    )

    assert record["speedup_valid"] is True
    assert record["fixed_continuation_exact_match"] is False
    assert record["fixed_continuation_first_divergence_index"] == 1
    assert record["quality_prefix_exact_match"] is False
    assert record["quality_prefix_token_match_ratio"] == pytest.approx(0.5)


def test_attach_reference_rejects_prompt_or_fixed_budget_mismatch():
    record = attach_reference_timing(
        _method_record(prompt_hash=prompt_hash([9, 9, 9]), output_tokens=512),
        _reference_record(),
        expected_output_tokens=1024,
    )

    assert record["speedup_valid"] is False
    assert record["esr"] is None
    assert record["dsr"] is None
    assert "prompt_hash" in record["speedup_invalid_reason"]
    assert "output_tokens" in record["speedup_invalid_reason"]


def test_attach_reference_rejects_checkpoint_revision_mismatch():
    record = attach_reference_timing(
        _method_record(target_model_revision="target-rev"),
        _reference_record(target_model_revision="different-target-rev"),
        expected_output_tokens=1024,
    )

    assert record["speedup_valid"] is False
    assert record["esr"] is None
    assert "target_model_revision" in record["speedup_invalid_reason"]


def test_attach_reference_rejects_non_batch_one_pair():
    record = attach_reference_timing(
        _method_record(batch_size=2),
        _reference_record(),
        expected_output_tokens=1024,
    )

    assert record["speedup_valid"] is False
    assert "batch_size" in record["speedup_invalid_reason"]


def test_shared_record_builder_preserves_paired_input_contract_fields():
    record = build_sample_record(
        method="vanilla_hf",
        dataset="gov_report",
        sample_id="gov-0",
        model="Qwen3-4B",
        input_tokens=3,
        output_tokens=1024,
        timing={"prefill_ms": 10.0, "decode_ms": 100.0, "e2e_ms": 115.0},
        config={
            "batch_size": 1,
            "prompt_hash": prompt_hash([1, 2, 3]),
            "run_config_hash": "cfg-qwen3",
            "original_input_tokens": 3,
            "input_truncated": False,
            "speed_output_tokens": 1024,
        },
    )

    assert record["prompt_hash"] == prompt_hash([1, 2, 3])
    assert record["run_config_hash"] == "cfg-qwen3"
    assert record["original_input_tokens"] == 3
    assert record["input_truncated"] is False
    assert record["speed_output_tokens"] == 1024
