from __future__ import annotations

import pytest

from TrainingFree.context_interventions.resilience import CampaignIntegrityError, FailureStreak, run_compute_unit


def test_compute_unit_retries_a_local_exception_and_keeps_only_latest_result():
    calls = []

    def compute():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("out of memory")
        return {"attempt": len(calls)}

    value, error = run_compute_unit(compute, label="local unit")
    assert value == {"attempt": 2}
    assert error is None


def test_compute_unit_records_persistent_local_failure_after_two_attempts():
    calls = []

    def compute():
        calls.append(1)
        raise ValueError("bad optional ranking")

    value, error = run_compute_unit(compute, label="local unit")
    assert len(calls) == 2
    assert value is None
    assert error["attempts"] == 2
    assert "bad optional ranking" in error["reason"]
    assert "ValueError" in error["traceback"]


def test_compute_unit_propagates_integrity_failure_returned_as_result():
    calls = []

    def compute():
        calls.append(1)
        return {"status": "error", "reason": "replay target cache crop mismatch: 15 != 16"}

    with pytest.raises(CampaignIntegrityError, match="cache crop mismatch"):
        run_compute_unit(
            compute,
            label="verifier unit",
            result_error=lambda result: result["reason"],
        )
    assert len(calls) == 1


def test_compute_unit_retries_a_recoverable_error_result():
    calls = []

    def compute():
        calls.append(1)
        return (
            {"status": "error", "reason": "temporary out of memory"}
            if len(calls) == 1 else {"status": "ok", "value": 7}
        )

    result, error = run_compute_unit(
        compute,
        label="recoverable result",
        result_error=lambda value: value["reason"] if value["status"] != "ok" else None,
    )
    assert result == {"status": "ok", "value": 7}
    assert error is None
    assert len(calls) == 2


@pytest.mark.parametrize("error", (
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED"),
    RuntimeError("replay target cache crop mismatch: 15 != 16"),
    OSError("No space left on device"),
    ImportError("required kernel is unavailable"),
))
def test_fatal_compute_error_propagates_without_retry(error):
    calls = []

    def compute():
        calls.append(1)
        raise error

    with pytest.raises(CampaignIntegrityError):
        run_compute_unit(compute, label="fatal unit")
    assert len(calls) == 1


def test_three_consecutive_failures_are_systemic_but_success_resets_streak():
    streak = FailureStreak()
    streak.observe(failed=True, label="first")
    streak.observe(failed=True, label="second")
    streak.observe(failed=False, label="success")
    streak.observe(failed=True, label="first again")
    streak.observe(failed=True, label="second again")
    with pytest.raises(CampaignIntegrityError, match="3 consecutive"):
        streak.observe(failed=True, label="third")


def test_operator_interrupt_is_never_swallowed():
    with pytest.raises(KeyboardInterrupt):
        run_compute_unit(lambda: (_ for _ in ()).throw(KeyboardInterrupt()), label="interrupt")
