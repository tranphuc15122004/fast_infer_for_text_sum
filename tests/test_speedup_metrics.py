from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import metrics  # noqa: E402
import collect_metrics  # noqa: E402
from infer_llmlingua import _fits_model_context  # noqa: E402


def test_aggregate_speedup_uses_ratio_of_mean_timings():
    records = [
        {
            "e2e_ms": 80.0,
            "dense_e2e_ms": 160.0,
            "decode_ms": 30.0,
            "dense_decode_ms": 60.0,
            "prefill_ms": 50.0,
            "dense_prefill_ms": 100.0,
            "ttft_ms": 55.0,
            "dense_ttft_ms": 110.0,
        },
        {
            "e2e_ms": 120.0,
            "dense_e2e_ms": 180.0,
            "decode_ms": 50.0,
            "dense_decode_ms": 75.0,
            "prefill_ms": 70.0,
            "dense_prefill_ms": 105.0,
            "ttft_ms": 75.0,
            "dense_ttft_ms": 115.0,
        },
    ]

    result = metrics.aggregate_speedup(records)

    assert result["esr"] == pytest.approx(340.0 / 200.0)
    assert result["dsr"] == pytest.approx(135.0 / 80.0)
    assert result["prefill_speedup"] == pytest.approx(1.7083)
    assert result["ttft_speedup"] == pytest.approx(1.7308)


def test_speedup_pair_coverage_reports_valid_ratio_by_metric():
    coverage = metrics.speedup_pair_coverage(
        [
            {
                "e2e_ms": 80.0,
                "dense_e2e_ms": 160.0,
                "decode_ms": None,
                "dense_decode_ms": 60.0,
                "speedup_valid": True,
            },
            {
                "e2e_ms": 90.0,
                "dense_e2e_ms": None,
                "decode_ms": None,
                "dense_decode_ms": None,
                "speedup_valid": False,
            },
        ]
    )

    assert coverage["esr"] == {"available": True, "valid": 1, "total": 2, "ratio": 0.5}
    assert coverage["dsr"]["available"] is False


def test_normalize_record_exposes_dense_reference_timing_aliases():
    row = collect_metrics.normalize_record(
        {
            "pipeline_e2e_ms": 80.0,
            "pipeline_ttft_ms": 30.0,
            "prefill_ms": 20.0,
            "decode_ms": 60.0,
            "baseline_full_e2e_ms": 160.0,
            "baseline_full_ttft_ms": 60.0,
            "baseline_full_prefill_ms": 100.0,
            "baseline_full_decode_ms": 120.0,
        },
        "semantic_selection",
    )

    assert row["dense_e2e_ms"] == 160.0
    assert row["dense_ttft_ms"] == 60.0
    assert row["dense_prefill_ms"] == 100.0
    assert row["dense_decode_ms"] == 120.0


def test_compute_group_reports_speedups_only_for_paired_timing_fields():
    paired = collect_metrics.compute_group(
        [
            {
                "e2e_ms": 80.0,
                "dense_e2e_ms": 160.0,
                "decode_ms": 30.0,
                "dense_decode_ms": 60.0,
            }
        ],
        {},
    )
    unpaired = collect_metrics.compute_group([{"e2e_ms": 80.0}], {})

    assert paired["speedup"] == {"esr": 2.0, "dsr": 2.0}
    assert "speedup" not in unpaired


def test_compute_group_includes_speedup_pair_coverage():
    group = collect_metrics.compute_group(
        [{"e2e_ms": 80.0, "dense_e2e_ms": 160.0, "speedup_valid": True}],
        {},
    )

    assert group["speedup_coverage"]["esr"]["ratio"] == 1.0


def test_compute_group_preserves_external_speedup_scope():
    group = collect_metrics.compute_group(
        [
            {
                "e2e_ms": 80.0,
                "dense_e2e_ms": 160.0,
                "decode_ms": 30.0,
                "dense_decode_ms": 60.0,
                "speedup_scope": "external_reference",
                "external_reference_baseline": "vanilla_fa",
            }
        ],
        {},
    )

    assert group["speedup_scope"] == "external_reference"
    assert group["external_reference_baselines"] == ["vanilla_fa"]


def test_nested_collector_excludes_reference_sidecars_from_method_rows(tmp_path):
    run_root = tmp_path / "run"
    method_dir = run_root / "dflash"
    reference_dir = run_root / "references"
    method_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    row = {
        "method": "vanilla_hf",
        "dataset": "gov_report",
        "sample_id": "s1",
        "status": "success",
        "e2e_ms": 100.0,
    }
    (method_dir / "gov_report.jsonl").write_text(
        __import__("json").dumps({**row, "method": "dflash"}) + "\n",
        encoding="utf-8",
    )
    (reference_dir / "gov_report.jsonl").write_text(
        __import__("json").dumps(row) + "\n", encoding="utf-8"
    )

    files = collect_metrics.load_output_files(run_root, ["gov_report"])

    assert [(stem, len(rows)) for stem, _dataset, rows in files] == [
        ("dflash_gov_report", 1)
    ]


def test_speed_summary_includes_decode_only_token_throughput():
    summary = metrics.aggregate_speed(
        [{"decode_throughput_tok_s": 10.0}, {"decode_throughput_tok_s": 20.0}]
    )

    assert summary["decode_throughput_tok_s"]["mean"] == 15.0


def test_markdown_has_combined_paired_main_results_table(tmp_path):
    group = {
        "num_records": 2,
        "speed": {
            "decode_throughput_tok_s": {"mean": 50.0},
            "prefill_ms": {"mean": 11.0},
            "decode_ms": {"mean": 22.0},
        },
        "speculative": {"avg_accept_length": {"mean": 3.0}},
        "speedup": {"esr": 1.5, "dsr": 2.0},
    }
    result = {
        "outputs_dir": "outputs",
        "datasets": ["gov_report"],
        "metrics": {"gov_report": {"dflash": group}},
        "overall": {"dflash": group},
    }
    path = tmp_path / "summary.md"

    collect_metrics.write_markdown(path, result, ["gov_report"])

    report = path.read_text(encoding="utf-8")
    assert "Bảng kết quả chính (paired online)" in report
    assert "decode tok/s" in report
    assert "| dflash | 2 | 3.00 | 50.00 | 11.00 | 22.00 | 1.50 | 2.00 |" in report


def test_normalize_record_maps_native_eagle_and_gemfilter_pairs():
    eagle = collect_metrics.normalize_record(
        {
            "eagle_time": 0.08,
            "naive_time": 0.16,
            "new_tokens": 12,
            "eagle_tok_s": 150.0,
        },
        "eagle3",
    )
    gemfilter = collect_metrics.normalize_record(
        {
            "e2e_ms": 90.0,
            "base_time_s": 0.2,
        },
        "gemfilter",
    )

    assert eagle["e2e_ms"] == 80.0
    assert eagle["decode_ms"] == 80.0
    assert eagle["dense_e2e_ms"] == 160.0
    assert eagle["dense_decode_ms"] == 160.0
    assert eagle["output_tokens"] == 12
    assert eagle["throughput_tok_s"] == 150.0
    assert gemfilter["dense_e2e_ms"] == 200.0


def test_reports_include_speedup_columns(tmp_path):
    result = {
        "outputs_dir": "outputs",
        "datasets": ["xsum"],
        "metrics": {
            "xsum": {
                "method": {
                    "num_records": 1,
                    "num_reference_joined": 1,
                    "speedup": {"esr": 2.0, "dsr": 1.5},
                    "speedup_coverage": {
                        "esr": {"available": True, "valid": 1, "total": 1, "ratio": 1.0},
                        "dsr": {"available": True, "valid": 1, "total": 1, "ratio": 1.0},
                    },
                }
            }
        },
        "overall": {
            "method": {
                "num_records": 1,
                "num_reference_joined": 1,
                    "speedup": {"esr": 2.0, "dsr": 1.5},
                    "speedup_coverage": {
                        "esr": {"available": True, "valid": 1, "total": 1, "ratio": 1.0},
                        "dsr": {"available": True, "valid": 1, "total": 1, "ratio": 1.0},
                    },
            }
        },
    }

    csv_path = tmp_path / "metrics.csv"
    md_path = tmp_path / "metrics.md"
    collect_metrics.write_csv(csv_path, result, ["xsum"])
    collect_metrics.write_markdown(md_path, result, ["xsum"])

    csv_text = csv_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    assert "esr_ratio" in csv_text
    assert "esr_pair_ratio" in csv_text
    assert "2.0000" in csv_text
    assert "Speedup so với dense/reference" in md_text
    assert "ESR pairs" in md_text


def test_llmlingua_dense_reference_guard_checks_prompt_and_generation_length():
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=512),
    )

    assert _fits_model_context(400, 32, model)
    assert not _fits_model_context(500, 32, model)
    assert _fits_model_context(500, 32, SimpleNamespace(config=SimpleNamespace()))
