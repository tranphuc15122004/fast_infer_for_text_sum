from __future__ import annotations

from src.analyze.groundsync.e0_report import render_e0_report
from src.analyze.groundsync.e1_report import render_e1_report


def test_render_e0_report_exposes_coverage_and_conservative_decision() -> None:
    manifest = {
        "status": "ok",
        "experiment": "E0_target_kv_dflash_failure_map",
        "target_model": "target",
        "draft_model": "draft",
        "records_selected": 3,
        "records_excluded": 1,
        "round_rows": 9,
        "candidate_ks": [4],
        "target_load_in_8bit": False,
    }
    metrics = {
        "decision": {"status": "INCONCLUSIVE", "reason": "sparse_bucket"},
        "by_k": {
            "4": {
                "by_bucket": {
                    "4-8k": {
                        "document_count": 3,
                        "row_count": 9,
                        "mat": 0.5,
                        "survival": {"1": 0.4, "4": 0.1},
                    }
                }
            }
        },
        "context_drop": {"4": {"status": "INCONCLUSIVE", "relative_drop": None}},
    }
    report = render_e0_report(manifest, metrics)
    assert "INCONCLUSIVE" in report
    assert "records_excluded" in report
    assert "4-8k" in report
    assert "sparse_bucket" in report


def test_render_e1_report_separates_token_hidden_control_from_kv() -> None:
    report = render_e1_report(
        {"status": "ok", "feature_rows": 10, "excluded_rows": 2, "horizon": 4, "max_memory_tokens": 3, "interface_dim": 2},
        {
            "partitions": {"train": 6, "dev": 2, "test": 2},
            "representations": {
                "hidden_sequence": {
                    "status": "ok", "train_rows": 6, "test_rows": 2,
                    "ce_mean": 2.0, "acc1_by_position": [0.5] * 4,
                    "acc5_by_position": [0.9] * 4,
                    "prefix_exact_by_position": [0.5] * 4,
                    "parameter_count": 100,
                },
                "kv": {"status": "INCONCLUSIVE", "reason": "sparse"},
            },
        },
    )
    assert "hidden_sequence" in report
    assert "kv" in report
    assert "lower control" in report
