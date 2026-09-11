"""Tests for the numerical contract used by the KV-cache verification script."""

from __future__ import annotations

import pytest
import torch


def test_compare_hidden_states_reports_bfloat16_tolerance() -> None:
    from MR_DFlash.kv_hidden import compare_hidden_states

    reference = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32)
    candidate = reference + torch.tensor([[0.001, 0.0], [0.0, -0.002]])

    report = compare_hidden_states(reference, candidate, atol=0.01, rtol=0.0)

    assert report["shape"] == [2, 2]
    assert report["allclose"] is True
    assert report["max_abs_error"] == pytest.approx(0.002, abs=1e-7)


def test_compare_hidden_states_rejects_shape_or_value_mismatch() -> None:
    from MR_DFlash.kv_hidden import compare_hidden_states

    reference = torch.zeros(2, 3)
    value_mismatch = torch.ones(2, 3)
    shape_mismatch = torch.zeros(1, 3)

    value_report = compare_hidden_states(reference, value_mismatch, atol=1e-4, rtol=1e-4)
    shape_report = compare_hidden_states(reference, shape_mismatch, atol=1e-4, rtol=1e-4)

    assert value_report["allclose"] is False
    assert value_report["shape"] == [2, 3]
    assert shape_report["allclose"] is False
    assert shape_report["candidate_shape"] == [1, 3]
