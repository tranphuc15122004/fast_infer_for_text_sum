from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_metric_audit_reports_missing_full_e2e_fields_and_quality():
    from common.metric_audit import audit_record

    audit = audit_record(
        {
            "sample_id": "s1",
            "status": "success",
            "measurement_scope": "full_e2e",
            "input_tokens": 100,
            "output_tokens": 10,
            "decode_ms": 20.0,
            "e2e_ms": None,
            "text": None,
            "reference_output": "reference",
            "max_new_tokens": 32,
        }
    )

    assert audit["timing"]["status"] == "partial"
    assert "e2e_ms" in audit["timing"]["missing"]
    assert audit["quality"]["status"] == "missing_text"
    assert "missing_e2e_ms" in audit["issues"]
    assert "missing_text" in audit["issues"]
    assert audit["tokens"]["budget_valid"] is True


def test_metric_audit_summary_counts_fields_scopes_and_issues():
    from common.metric_audit import summarize_audits

    audits = [
        {
            "sample_id": "s1",
            "status": "success",
            "measurement_scope": "decode_only",
            "timing": {"present": ["decode_ms"], "missing": []},
            "quality": {"status": "not_available"},
            "issues": [],
        },
        {
            "sample_id": "s2",
            "status": "failed",
            "measurement_scope": "full_e2e",
            "timing": {"present": [], "missing": ["e2e_ms"]},
            "quality": {"status": "missing_text"},
            "issues": ["failed_record", "missing_e2e_ms"],
        },
    ]

    summary = summarize_audits(audits)

    assert summary["num_records"] == 2
    assert summary["status_counts"] == {"success": 1, "failed": 1}
    assert summary["scope_counts"] == {"decode_only": 1, "full_e2e": 1}
    assert summary["issue_counts"]["missing_e2e_ms"] == 1
    assert summary["quality_status_counts"]["missing_text"] == 1


def test_metric_audit_file_is_machine_readable(tmp_path):
    from common.metric_audit import audit_output_file

    output = tmp_path / "toy.jsonl"
    output.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "status": "success",
                "measurement_scope": "e2e_only",
                "input_tokens": 5,
                "output_tokens": 2,
                "e2e_ms": 10.0,
                "throughput_tok_s": 200.0,
                "text": "answer",
                "reference_output": "answer",
                "rouge1": 1.0,
            }
        )
        + "\n"
        + json.dumps({"type": "summary", "status": "success"})
        + "\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "toy.metrics.json"

    summary = audit_output_file(
        output,
        baseline="toy",
        dataset="toy",
        audit_path=audit_path,
        expected_output_tokens=2,
        expected_samples=1,
    )

    assert summary["num_records"] == 1
    assert summary["metric_contract"]["status"] == "metric_incomplete"
    assert summary["metric_contract"]["baseline"] == "toy"
    assert audit_path.is_file()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["baseline"] == "toy"
    assert payload["summary"]["quality_records"] == 1
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["metric_audit"]["quality"]["status"] == "complete"
    assert rows[1]["metric_audit_summary"]["num_records"] == 1


def test_metric_contract_marks_missing_required_scope_metrics_as_incomplete():
    from common.metric_audit import validate_cell_metric_contract

    result = validate_cell_metric_contract(
        [
            {
                "sample_id": "s1",
                "status": "success",
                "measurement_scope": "full_e2e",
                "input_tokens": 100,
                "output_tokens": 8,
                "e2e_ms": 20.0,
                "throughput_tok_s": 400.0,
                "text": "answer",
                "reference_output": "answer",
                "rouge1": 1.0,
            }
        ],
        baseline="eagle3",
        expected_samples=1,
        expected_output_tokens=8,
    )

    assert result["status"] == "metric_incomplete"
    assert result["expected_scope"] == "full_e2e"
    assert result["missing_field_counts"]["prefill_ms"] == 1
    assert result["missing_field_counts"]["decode_ms"] == 1


def test_metric_contract_allows_fafo_e2e_only_without_decode_metrics():
    from common.metric_audit import validate_cell_metric_contract

    result = validate_cell_metric_contract(
        [
            {
                "sample_id": "s1",
                "status": "success",
                "measurement_scope": "e2e_only",
                "input_tokens": 100,
                "retained_tokens": 100,
                "output_tokens": 8,
                "batch_size": 1,
                "model_load_ms": 100.0,
                "e2e_ms": 20.0,
                "peak_memory_gb": 2.0,
                "device": "cuda:0",
                "throughput_tok_s": 400.0,
                "text": "answer",
                "reference_output": "answer",
                "rouge1": 1.0,
                "rouge1_p": 1.0,
                "rouge1_r": 1.0,
                "rouge1_f": 1.0,
                "rouge2_p": 1.0,
                "rouge2_r": 1.0,
                "rouge2_f": 1.0,
                "rougeL_p": 1.0,
                "rougeL_r": 1.0,
                "rougeL_f": 1.0,
                "rougeLsum_p": 1.0,
                "rougeLsum_r": 1.0,
                "rougeLsum_f": 1.0,
                "bleu1": 1.0,
                "bleu2": 1.0,
                "bleu3": 1.0,
                "bleu4": 1.0,
                "length_ratio": 1.0,
            }
        ],
        baseline="fafo",
        expected_samples=1,
        expected_output_tokens=8,
    )

    assert result["status"] == "complete"
    assert result["expected_scope"] == "e2e_only"
    assert result["decode_metrics_available"] is False


def test_metric_contract_rejects_missing_quality_when_reference_is_present():
    from common.metric_audit import validate_cell_metric_contract

    result = validate_cell_metric_contract(
        [
            {
                "sample_id": "s1",
                "status": "success",
                "measurement_scope": "full_e2e",
                "input_tokens": 100,
                "output_tokens": 8,
                "prefill_ms": 5.0,
                "ttft_ms": 5.0,
                "decode_ms": 15.0,
                "e2e_ms": 20.0,
                "throughput_tok_s": 400.0,
                "text": "answer",
                "reference_output": "reference",
            }
        ],
        baseline="vanilla_hf",
        expected_samples=1,
        expected_output_tokens=8,
    )

    assert result["status"] == "metric_incomplete"
    assert result["issue_counts"]["missing_quality_metric"] == 1


def test_metric_contract_requires_raw_speculative_metrics_but_not_speedups():
    from common.metric_audit import validate_cell_metric_contract

    record = {
        "sample_id": "s1",
        "status": "success",
        "measurement_scope": "full_e2e",
        "input_tokens": 100,
        "retained_tokens": 100,
        "output_tokens": 8,
        "batch_size": 1,
        "model_load_ms": 100.0,
        "prefill_ms": 5.0,
        "ttft_ms": 5.0,
        "decode_ms": 15.0,
        "e2e_ms": 20.0,
        "peak_memory_gb": 4.0,
        "device": "cuda:0",
        "acceptance_lengths": [3, 4],
        "draft_latency_ms": 2.0,
        "verification_latency_ms": 8.0,
        "draft_tokens_proposed": 6,
        "draft_tokens_accepted": 5,
        "text": "answer",
        "reference_output": "answer",
        "rouge1_p": 1.0,
        "rouge1_r": 1.0,
        "rouge1_f": 1.0,
        "rouge2_p": 1.0,
        "rouge2_r": 1.0,
        "rouge2_f": 1.0,
        "rougeL_p": 1.0,
        "rougeL_r": 1.0,
        "rougeL_f": 1.0,
        "rougeLsum_p": 1.0,
        "rougeLsum_r": 1.0,
        "rougeLsum_f": 1.0,
        "bleu1": 1.0,
        "bleu2": 1.0,
        "bleu3": 1.0,
        "bleu4": 1.0,
        "length_ratio": 1.0,
        # No speedup/ESR/DSR fields: they are collector-derived.
    }

    result = validate_cell_metric_contract(
        [record],
        baseline="dflash",
        expected_samples=1,
        expected_output_tokens=8,
    )

    assert result["status"] == "complete"
    assert result["missing_direct_field_counts"] == {}
    assert "draft_latency_ms" in result["required_direct_metrics"]
    assert result["valid_speedup_pairs"] == 0


def test_magicdec_self_spec_requires_acceptance_trace_fields():
    from common.metric_audit import required_direct_metrics

    required = required_direct_metrics("magicdec")

    assert "acceptance_lengths" in required
    assert "draft_latency_ms" in required
    assert "verification_latency_ms" in required
