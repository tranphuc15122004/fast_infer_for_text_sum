from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_fafo_overall_gen_is_not_public_output_tokens():
    from infer_fafo import _parse_log

    parsed = _parse_log(
        "FAFO LOG - OVERALL GEN: 202752 STEPS: 99 AVG COMPRESS RATIO: 2048\n"
    )

    assert parsed["lookahead_tokens"] == 202752
    assert parsed["output_tokens"] is None


def test_fafo_sidecar_records_public_tokens_and_quality_fields():
    from infer_fafo import build_fafo_sample_records

    records = build_fafo_sample_records(
        [
            {
                "id": "s1",
                "prompt": "Question",
                "reference": "42",
                "raw": {"task_type": "qa"},
            }
        ],
        [{
            "sample_index": 0,
            "input_tokens": 11,
            "output_tokens": 4,
            "e2e_ms": 20.0,
            "decode_ms": 20.0,
            "text": "42",
        }],
        method="fafo_stream-llm",
        dataset="toy",
        model="model",
        max_new_tokens=8,
    )

    assert len(records) == 1
    assert records[0]["status"] == "success"
    assert records[0]["output_tokens"] == 4
    assert records[0]["input_tokens"] == 11
    assert records[0]["text"] == "42"
    assert records[0]["decode_ms"] is None
    assert records[0]["tpot_ms"] is None
    assert records[0]["decode_throughput_tok_s"] is None
    assert records[0]["rouge1"] is not None


def test_fafo_warmup_is_injected_for_multi_sample_runs():
    from infer_fafo import prepare_fafo_records

    records = [{"id": "s1"}, {"id": "s2"}]

    prepared = prepare_fafo_records(records, smoke=False)

    assert len(prepared) == 3
    assert prepared[0]["id"] == "__fafo_warmup__s1"
    assert [row["id"] for row in prepared[1:]] == ["s1", "s2"]


def test_fafo_sidecar_coverage_requires_every_real_sample():
    from infer_fafo import validate_fafo_sidecar

    runtime_records = [
        {"id": "__fafo_warmup__s1"},
        {"id": "s1"},
        {"id": "s2"},
    ]

    complete, reason = validate_fafo_sidecar(
        runtime_records,
        [{"sample_index": 0}, {"sample_index": 1}, {"sample_index": 2}],
    )
    assert complete is True
    assert reason is None

    incomplete, reason = validate_fafo_sidecar(
        runtime_records,
        [{"sample_index": 0}, {"sample_index": 1}],
    )
    assert incomplete is False
    assert "s2" in str(reason)


def test_specextend_stats_are_normalized_to_full_timing_and_text():
    from infer_specextend import build_specextend_sample_record

    record = build_specextend_sample_record(
        source={"id": "s1", "reference": "answer", "task_type": "qa"},
        stats={
            "input_tokens": 100,
            "output_tokens": 8,
            "text": "answer",
            "prefill_ms": 12.0,
            "decode_ms": 20.0,
            "e2e_ms": 32.0,
            "peak_memory_gb": 3.0,
        },
        method="specextend_eagle",
        dataset="toy",
        model="model",
        draft_model="draft",
        script="run_eagle.py",
        returncode=0,
    )

    assert record["measurement_scope"] == "full_e2e"
    assert record["prefill_ms"] == 12.0
    assert record["ttft_ms"] == 12.0
    assert record["decode_ms"] == 20.0
    assert record["e2e_ms"] == 32.0
    assert record["text"] == "answer"
    assert record["rouge1"] is not None
    assert record["acceptance_lengths"] is None


def test_dflash_acceptance_metrics_are_derived_from_block_acceptance_lengths():
    from infer_dflash import summarize_acceptance

    metrics = summarize_acceptance([3, 4], block_size=4)

    assert metrics["avg_accept_length"] == 3.5
    assert metrics["acceptance_rate"] == 0.8333
    assert metrics["rejected_draft_ratio"] == 0.1667


def test_eagle_timing_fields_are_canonical():
    from eagle3_infer_qwen3 import build_eagle_timing_fields

    fields = build_eagle_timing_fields(
        input_tokens=100,
        output_tokens=10,
        prefill_ms=25.0,
        decode_ms=50.0,
        e2e_ms=75.0,
    )

    assert fields == {
        "input_tokens": 100,
        "output_tokens": 10,
        "prefill_ms": 25.0,
        "ttft_ms": 25.0,
        "decode_ms": 50.0,
        "e2e_ms": 75.0,
        "tpot_ms": 5.0,
        "throughput_tok_s": 133.333,
        "decode_throughput_tok_s": 200.0,
        "measurement_scope": "full_e2e",
    }


def test_flash_attention_does_not_use_static_cache_by_default():
    from common.vanilla_inference import _should_use_static_cache

    assert _should_use_static_cache("flash_attention_2") is False
    assert _should_use_static_cache("eager") is True


def test_degenerate_output_guard_is_conservative():
    from common.quality_guard import is_degenerate_output

    assert is_degenerate_output("the of the of " * 40)
    assert not is_degenerate_output(
        "This is a normal answer with varied words and a concise conclusion."
    )


def test_fafo_sidecar_is_not_merged_as_aggregate_only():
    from scripts.run_longbench_200 import AGGREGATE_ONLY_BASELINES

    assert "fafo" not in AGGREGATE_ONLY_BASELINES
    assert "sssd" in AGGREGATE_ONLY_BASELINES


def test_fafo_preflight_is_not_aggregate_only():
    import common.longbench_adapter as adapter

    result = adapter.preflight_baseline(
        "fafo",
        config={"model": "org/model"},
        cuda_available=True,
    )

    assert result["status"] == "ready"
