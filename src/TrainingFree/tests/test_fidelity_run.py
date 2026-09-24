import pytest

from src.TrainingFree.fidelity_run import _summarize_rows, parse_configs


def test_parse_configs_normalizes_labels_and_deduplicates() -> None:
    assert parse_configs("k16v16, K8V4, K8V4") == [
        ("K16V16", 16, 16),
        ("K8V4", 8, 4),
    ]


def test_summary_pools_target_nll_and_global_attention_error_tail() -> None:
    def candidate(*, top1: float, candidate_nll: float, dense_nll: float, errors: list[float]):
        return {
            "token_count": 1,
            "top1_agreement": top1,
            "delta_nll_percent": 0.0,
            "candidate_nll_sum": candidate_nll,
            "dense_nll_sum": dense_nll,
            "kl_mean": 0.1,
            "attention_output_error_mean": sum(errors) / len(errors),
            "attention_output_error_p99": max(errors),
            "attention_output_error_values": errors,
        }

    rows = [
        {"status": "ok", "dataset": "gov_report", "candidates": {"K8V16": candidate(top1=1.0, candidate_nll=1.1, dense_nll=1.0, errors=[0.0, 0.02])}},
        {"status": "ok", "dataset": "multi_news", "candidates": {"K8V16": candidate(top1=0.0, candidate_nll=2.2, dense_nll=2.0, errors=[0.04, 0.3])}},
    ]

    result = _summarize_rows(rows, [("K8V16", 8, 16)])["K8V16"]

    assert result["top1_agreement"] == pytest.approx(0.5)
    assert result["delta_nll_percent"] == pytest.approx(10.0)
    assert result["attention_output_error_mean"] == pytest.approx(0.09)
    assert result["attention_output_error_p99"] == pytest.approx(0.2922)


def test_parse_configs_rejects_bit_widths_outside_experiment_matrix() -> None:
    with pytest.raises(ValueError, match="precision config"):
        parse_configs("K2V16")
