from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import metrics  # noqa: E402


def test_mean_acceptance_length_averages_per_sample_values():
    records = [
        {"status": "success", "sample_id": "s1", "avg_accept_length": 2.0},
        {"status": "success", "sample_id": "s2", "acceptance_lengths": [3, 5]},
        {"status": "success", "sample_id": "s3", "accept_length": 4},
    ]

    assert metrics.mean_acceptance_length(records) == pytest.approx(10.0 / 3.0)


def test_mean_acceptance_length_ignores_failed_and_empty_traces():
    records = [
        {"status": "failed", "sample_id": "bad", "avg_accept_length": 99},
        {"status": "success", "sample_id": "empty", "acceptance_lengths": []},
        {"status": "success", "sample_id": "s1", "accept_length_list": [2, 4]},
    ]

    assert metrics.mean_acceptance_length(records) == pytest.approx(3.0)


def test_aggregate_speculative_exposes_tau_summary():
    result = metrics.aggregate_speculative(
        [
            {"status": "success", "sample_id": "s1", "avg_accept_length": 2.0},
            {"status": "success", "sample_id": "s2", "avg_accept_length": 4.0},
        ]
    )

    assert result["avg_accept_length"]["mean"] == pytest.approx(3.0)

