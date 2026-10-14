from __future__ import annotations

import sys
import subprocess
import weakref
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_dflash_context_experiments as campaign


@pytest.mark.parametrize("field,value", (("max_prefixes", 3), ("repetitions", 5), ("warmups", 7), ("seed", 123)))
def test_resume_signature_rejects_changed_measurement_protocol(monkeypatch, field, value):
    monkeypatch.setattr(campaign, "_code_signature", lambda: {"sha256": "fixed-code"})
    args = campaign._parser().parse_args([])
    runtime = {"runtime_signature": {"signature": "runtime"}}
    before = campaign._cell_signature(runtime, args, "dev", "a3_rollout_pairs")
    setattr(args, field, value)
    assert campaign._cell_signature(runtime, args, "dev", "a3_rollout_pairs") != before


def test_prefix_does_not_repeat_cells_that_already_exhausted_local_retries():
    result = {"status": "ok", "a1": [], "a3_replay": [], "a2": [{
        "status": "error", "reason": "temporary cell failure",
        "compute_error": {"status": "error", "attempts": 2},
    }]}
    assert campaign._replay_result_error(result) is None


@pytest.mark.parametrize("worker_run,expected", (("run-10", 1), ("run-1", 0)))
def test_worker_detection_matches_run_id_as_an_argument(tmp_path, worker_run, expected):
    launcher = (ROOT / "scripts/run_dflash_context_A_b200.sh").read_text()
    function = launcher.split("campaign_is_running() {", 1)[1].split("launch_campaign() {", 1)[0]
    harness = 'set -Eeuo pipefail\nSELF="$3"\ncampaign_is_running() {' + function + '\ncampaign_is_running "$1" "$2"\n'
    # A live worker exposes a NUL-separated argv through /proc. Its compute
    # body is a short sleep so this regression does not import a GPU model.
    worker = subprocess.Popen([
        sys.executable, "-c", "import time; time.sleep(30)",
        str(ROOT / "scripts/run_dflash_context_experiments.py"), "--run-id", worker_run,
    ])
    try:
        result = subprocess.run([
            "bash", "-s", "--", str(tmp_path), "run-1", str(ROOT / "scripts/run_dflash_context_A_b200.sh"),
        ], input=harness, text=True, capture_output=True, timeout=10)
        assert result.returncode == expected, result.stderr
    finally:
        worker.terminate()
        worker.wait(timeout=5)


def test_nonfinite_reference_round_is_retried_before_json_checkpoint():
    assert campaign._reference_result_error({
        "parity": {"status": "pass"}, "adapter_rounds": [{"draft_ms": float("nan")}],
    }) is not None


@pytest.mark.parametrize("changed", ("parity", "rollout", "cohort"))
def test_confirm_rejects_changed_locked_dev_artifacts_before_gpu(tmp_path, monkeypatch, changed):
    import json
    import hashlib
    from TrainingFree.context_interventions import environment

    args = campaign._parser().parse_args(["--mode", "confirm", "--run-id", "run-1", "--allow-partial-confirm"])
    monkeypatch.setattr(campaign, "_load_samples", lambda args: ([], []))
    split = {"status": "complete"}
    monkeypatch.setattr(campaign, "_cohort", lambda *args, **kwargs: ([], split, {
        "phase": "confirm", "quotas": {"gov_report": 1}, "groups": [{"dataset": "gov_report"}],
    }))
    monkeypatch.setattr(campaign, "_code_signature", lambda: {"sha256": "code"})
    monkeypatch.setattr(environment, "run_preflight", lambda **kwargs: pytest.fail("GPU preflight must not be reached"))
    artifacts = {
        "parity": (tmp_path / "evidence/parity_dev.json", "dev_parity_sha256"),
        "rollout": (tmp_path / "dev/a3_rollout.jsonl", "dev_rollout_sha256"),
        "cohort": (tmp_path / "cohorts/dev.json", "dev_cohort_sha256"),
    }
    lock = {"status": "locked_for_heldout", "run_id": "run-1", "code_signature": "code",
            "selector": "previous_dflash", "budget": 2048, "refresh_period": 4}
    split_path = tmp_path / "cohorts/split_manifest.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(split))
    lock["split_manifest_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    for path, field in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
        lock[field] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "locked_config.json").write_text(json.dumps(lock))
    (tmp_path / "runtime_dev.json").write_text('{}')
    artifacts[changed][0].write_text('{"changed": true}\n')
    with pytest.raises(campaign.CampaignIntegrityError, match="locked dev"):
        campaign._run_phase(args, tmp_path, "confirm")


def test_one_bad_sample_with_many_prefixes_does_not_stop_next_sample(tmp_path, monkeypatch):
    from TrainingFree.context_adaptive import prompt
    from TrainingFree.context_adaptive.schema import read_jsonl
    from TrainingFree.context_interventions import replay

    class Ids:
        def to(self, device):
            return self

        def contiguous(self):
            return self

    monkeypatch.setattr(prompt, "prepare_prompt", lambda *args, **kwargs: (
        Ids(), SimpleNamespace(prompt_hash="hash", source_available=True),
    ))
    prefixes = [SimpleNamespace(
        round_index=i, processed_output_tokens=i * 16, prefix_hash=f"prefix-{i}",
        previous_dense_round_index=None, remaining_output_tokens=1024 - i * 16,
    ) for i in range(6)]
    monkeypatch.setattr(replay, "choose_reference_prefixes", lambda *args, **kwargs: prefixes)
    monkeypatch.setattr(campaign, "_prefix_pair", lambda *args: None)
    monkeypatch.setattr(campaign, "_code_signature", lambda: {"sha256": "code"})
    samples = [{"id": f"sample-{i}", "dataset": "gov_report", "_cad_source_group_id": f"group-{i}"}
               for i in (1, 2)]
    traces = [{
        "sample_id": sample["id"], "native_ids": [1], "adapter_ids": [1],
        "parity": {"status": "pass", "native_equals_adapter": True},
        "prompt_hash": "hash", "prompt_tokens": 4,
        "adapter_rounds": [{"type": "round", "gamma_executed": 15, "round_index": i, "accepted_candidates": 1}
                           for i in range(6)],
    } for sample in samples]
    measured = []

    def compute(args, runtime, sample, *other, **kwargs):
        measured.append(sample["id"])
        return ({"status": "baseline_mismatch", "reason": "local numerical mismatch"}
                if sample["id"] == "sample-1"
                else {"status": "ok", "a1": [], "a2": [], "a3_replay": []})

    monkeypatch.setattr(campaign, "_compute_replay_prefix", compute)
    args = campaign._parser().parse_args(["--run-id", "run-1", "--max-new-tokens", "1024"])
    campaign._run_replay(
        args, tmp_path, {"runtime_signature": {"signature": "runtime"}, "tokenizer": None,
                         "target": SimpleNamespace(device="cpu")},
        samples, traces, phase="pilot", max_new_tokens=1024, run_a3=True,
    )
    assert measured == ["sample-1"] * 6 + ["sample-2"] * 6
    assert sum(row.get("type") == "prefix_diagnostic" for row in read_jsonl(tmp_path / "pilot/prefix_diagnostics.jsonl")) == 6


def test_replay_uses_full_dflash_ids_when_ar_trajectory_diverges() -> None:
    trace = {
        "ar_ids": [10, 99, 30],
        "adapter_ids": [10, 20, 30],
        "native_ids": [10, 20, 30],
        "parity": {
            "status": "pass",
            "native_equals_adapter": True,
            "adapter_equals_ar": False,
        },
    }

    assert campaign._replay_reference_ids(trace) == [10, 20, 30]


def test_replay_rejects_trace_without_native_adapter_parity() -> None:
    trace = {
        "ar_ids": [10, 20, 30],
        "adapter_ids": [10, 20, 30],
        "native_ids": [10, 21, 30],
        "parity": {"status": "fail", "native_equals_adapter": False},
    }

    with pytest.raises(ValueError, match="native full DFlash and adapter IDs differ"):
        campaign._replay_reference_ids(trace)


def test_b200_campaign_launcher_uses_shared_config_and_python_runtime() -> None:
    launcher = (ROOT / "scripts" / "run_dflash_context_A_b200.sh").read_text(encoding="utf-8")

    assert 'source "$ROOT/scripts/common/config.sh"' in launcher
    assert "fast_infer_load_config dflash" in launcher
    assert 'source "$ROOT/scripts/common/runtime.sh" || exit 1' in launcher
    assert '"$FAST_INFER_PYTHON" - "$run_root"' in launcher
    assert "python3" not in launcher
    assert 'resume RUN_ID' in launcher
    assert 'resume)' in launcher
    assert 'phase_args+=(--resume)' in launcher
    assert 'run_phase_resumable "$run_id" pilot' in launcher
    assert "campaign.status" in launcher
    assert 'flock -n "$lock_fd"' in launcher


@pytest.mark.parametrize("status,expected_calls,expected_exit", ((2, 2, 0), (3, 1, 3), (130, 1, 130)))
def test_launcher_retries_transient_exit_but_preserves_fatal_and_interrupt(status, expected_calls, expected_exit):
    launcher = (ROOT / "scripts" / "run_dflash_context_A_b200.sh").read_text()
    recovery_function = launcher.split("run_phase_resumable() {", 1)[1].split("verify_prepare() {", 1)[0]
    harness = "set -Eeuo pipefail\nunset DCE_RESUME\nrun_phase_resumable() {" + recovery_function + '''
call_count=0
first_status="$1"
run_phase() {
  call_count=$((call_count + 1))
  printf 'call=%s resume=%s\\n' "$call_count" "${DCE_RESUME:-0}"
  if [[ "$call_count" == "1" ]]; then return "$first_status"; fi
  return 0
}
sleep() { :; }
run_phase_resumable test-run pilot
'''
    result = subprocess.run(["bash", "-s", "--", str(status)], input=harness, text=True, capture_output=True)
    calls = [line for line in result.stdout.splitlines() if line.startswith("call=")]
    assert result.returncode == expected_exit
    assert len(calls) == expected_calls
    if expected_calls == 2:
        assert calls == ["call=1 resume=0", "call=2 resume=1"]


def test_resume_does_not_skip_saved_a3_adapter_parity_failure(tmp_path, monkeypatch) -> None:
    from TrainingFree.context_interventions.artifacts import CellStore
    from TrainingFree.context_interventions import execution

    run_root = tmp_path / "run"
    sample = {
        "dataset": "gov_report",
        "_cad_source_group_id": "group-1",
        "id": "sample-1",
    }
    selector = "previous_dflash"
    store = CellStore(run_root / "pilot" / "a3_rollout", signature="sig", resume=True)
    pair_id = "|".join(("gov_report", "group-1", selector, "2048", "1", "0"))
    store.save(pair_id, {
        "status": "parity_fail", "rows": [
            {"variant": variant, "status": "ok", "matches_native_full_dflash": False}
            for variant in ("ar", "native_full_dflash", "adapter_full", selector)
        ],
    })

    monkeypatch.setattr(
        campaign,
        "_phase_store",
        lambda *_args, **_kwargs: store,
    )
    monkeypatch.setattr(campaign, "_cell_signature", lambda *_args, **_kwargs: "sig")
    monkeypatch.setattr(
        execution,
        "run_a3_comparison",
        lambda *_args, **_kwargs: pytest.fail("a saved parity failure must fail closed"),
    )
    args = SimpleNamespace(
        resume=True,
        a3_selector=selector,
        a3_budget=2048,
        run_id="run-1",
        source_anchors=4,
        recent_output=32,
    )

    with pytest.raises(RuntimeError, match="full-adapter/native-DFlash parity failed"):
        campaign._run_rollouts(
            args,
            run_root,
            runtime={},
            samples=[sample],
            traces=[{"sample_id": "sample-1"}],
            phase="pilot",
            max_new_tokens=32,
            periods=(1,),
            repetitions=1,
            warmups=0,
        )


@pytest.mark.parametrize("failure", ("baseline_mismatch", "local_exception"))
def test_replay_quarantines_bad_prefix_and_continues_to_later_prefix(tmp_path, monkeypatch, failure) -> None:
    from TrainingFree.context_interventions.artifacts import CellStore
    from TrainingFree.context_interventions import execution, replay
    from TrainingFree.context_adaptive import prompt as prompt_module

    class FakeIds:
        def to(self, _device):
            return self

        def contiguous(self):
            return self

    prefixes = [
        SimpleNamespace(
            round_index=129, processed_output_tokens=8, prefix_hash="bad-prefix",
            previous_dense_round_index=None, remaining_output_tokens=64,
        ),
        SimpleNamespace(
            round_index=238, processed_output_tokens=16, prefix_hash="good-prefix",
            previous_dense_round_index=None, remaining_output_tokens=64,
        ),
    ]
    calls = []
    stores = {}

    def phase_store(run_root, phase, kind, signature, resume):
        store = CellStore(run_root / phase / kind, signature=signature, resume=resume)
        stores[kind] = store
        return store

    monkeypatch.setattr(campaign, "_phase_store", phase_store)
    monkeypatch.setattr(campaign, "_cell_signature", lambda *_args, **_kwargs: "sig")
    monkeypatch.setattr(campaign, "_prefix_pair", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(campaign, "_prefix_at_age", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        replay,
        "choose_reference_prefixes",
        lambda *_args, **_kwargs: prefixes,
    )
    state_refs = []

    class FakeState:
        bank = SimpleNamespace(length=3)

    def rebuild_state(*_args, **_kwargs):
        assert all(ref() is None for ref in state_refs), "previous CUDA state must be released before rebuilding"
        state = FakeState()
        state_refs.append(weakref.ref(state))
        return state

    monkeypatch.setattr(replay, "rebuild_replay_state", rebuild_state)
    for name in ("run_a1_prefix", "run_a2_prefix", "run_a3_prefix"):
        monkeypatch.setattr(
            execution,
            name,
            lambda _runtime, sample, _ids, _layout, prefix, *_args, **_kwargs: calls.append(
                (sample["id"], prefix.round_index)
            ) or [],
        )
    # Keep the first replay mismatched and the second one valid.
    replay_results = iter(
        (0, 1) if failure == "baseline_mismatch"
        else (RuntimeError("temporary prefix failure"), RuntimeError("temporary prefix failure"), 1)
    )

    def replay_action(*_args, **_kwargs):
        outcome = next(replay_results)
        if isinstance(outcome, Exception):
            raise outcome
        return {"status": "ok", "accepted_candidates": outcome}

    monkeypatch.setattr(replay, "evaluate_replay_action", replay_action)

    sample = {"id": "sample-1", "dataset": "gov_report", "_cad_source_group_id": "group-1"}
    trace = {
        "sample_id": "sample-1",
        "native_ids": [10, 20],
        "adapter_ids": [10, 20],
        "parity": {"status": "pass", "native_equals_adapter": True},
        "prompt_hash": "prompt-hash",
        "prompt_tokens": 4,
        "adapter_rounds": [
            {"type": "round", "round_index": prefix.round_index, "gamma_executed": 15,
             "accepted_candidates": 1}
            for prefix in prefixes
        ],
    }
    monkeypatch.setattr(
        prompt_module,
        "prepare_prompt",
        lambda *_args, **_kwargs: (FakeIds(), SimpleNamespace(
            prompt_hash="prompt-hash", source_available=True, source_reason=None, prompt_tokens=4,
        )),
    )
    args = SimpleNamespace(
        resume=False, run_id="run-1", max_prefixes=2, source_anchors=4, recent_output=8,
        a3_selector="previous_dflash", a3_budget=2048, refresh_period=4,
    )
    run_root = tmp_path / "run"
    runtime = {
        "tokenizer": object(),
        "target": SimpleNamespace(device="cpu"),
        "draft": SimpleNamespace(layers=[object()]),
    }

    campaign._run_replay(
        args,
        run_root,
        runtime=runtime,
        samples=[sample],
        traces=[trace],
        phase="pilot",
        max_new_tokens=64,
        run_a3=True,
    )

    assert ("sample-1", 238) in calls
    assert stores["prefix_diagnostics"].rows()[0]["status"] == (
        "baseline_mismatch" if failure == "baseline_mismatch" else "error"
    )

    completed_prefix_calls = len(calls)
    args.resume = True
    if failure == "local_exception":
        monkeypatch.setattr(replay, "evaluate_replay_action", lambda *_args, **_kwargs: {
            "status": "ok", "accepted_candidates": 1,
        })
    campaign._run_replay(
        args,
        run_root,
        runtime=runtime,
        samples=[sample],
        traces=[trace],
        phase="pilot",
        max_new_tokens=64,
        run_a3=True,
    )
    assert len(calls) == completed_prefix_calls + (3 if failure == "local_exception" else 0)
    if failure == "local_exception":
        assert stores["prefix_diagnostics"].rows()[0]["status"] == "resolved"


def test_rollout_retries_returned_error_rows_once(tmp_path, monkeypatch) -> None:
    from TrainingFree.context_interventions.artifacts import CellStore
    from TrainingFree.context_interventions import execution

    store = CellStore(tmp_path / "pilot" / "a3_rollout", signature="sig", resume=False)
    monkeypatch.setattr(campaign, "_phase_store", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(campaign, "_cell_signature", lambda *_args, **_kwargs: "sig")
    calls = []

    def fake_rollout(_runtime, _sample, **_kwargs):
        calls.append(_kwargs.get("warmups"))
        rows = []
        for variant in ("ar", "native_full_dflash", "adapter_full", "previous_dflash"):
            rows.append({
                "type": "rollout",
                "variant": variant,
                "status": "error" if len(calls) == 1 and variant == "previous_dflash" else "ok",
                "matches_native_full_dflash": True,
                "measurement_attempt": len(calls),
                "wall_e2e_ms": 1.0, "counters": {"mean_accepted_candidates": 4.0},
            })
        return rows

    monkeypatch.setattr(execution, "run_a3_comparison", fake_rollout)
    args = SimpleNamespace(
        resume=False,
        a3_selector="previous_dflash",
        a3_budget=2048,
        run_id="run-1",
        source_anchors=4,
        recent_output=8,
    )
    sample = {"id": "sample-1", "dataset": "gov_report", "_cad_source_group_id": "group-1"}

    campaign._run_rollouts(
        args,
        tmp_path,
        runtime={},
        samples=[sample],
        traces=[{"sample_id": "sample-1"}],
        phase="pilot",
        max_new_tokens=32,
        periods=(1,),
        repetitions=1,
        warmups=1,
    )

    assert calls == [1, 1]
    assert len(store.rows()) == 1
    rows = store.rows()[0]["rows"]
    assert len(rows) == 4
    assert all(row["status"] == "ok" for row in rows)
    assert {row["measurement_attempt"] for row in rows} == {2}


@pytest.fixture
def rollout_case(tmp_path, monkeypatch):
    from TrainingFree.context_interventions.artifacts import CellStore

    args = SimpleNamespace(
        resume=False, a3_selector="previous_dflash", a3_budget=2048,
        run_id="run-1", source_anchors=4, recent_output=8,
    )
    samples = [
        {"id": f"sample-{i}", "dataset": "gov_report", "_cad_source_group_id": f"group-{i}"}
        for i in (1, 2)
    ]
    stores = []

    def phase_store(root, phase, kind, signature, resume):
        store = CellStore(root / phase / kind, signature=signature, resume=resume)
        stores.append(store)
        return store

    monkeypatch.setattr(campaign, "_phase_store", phase_store)
    monkeypatch.setattr(campaign, "_cell_signature", lambda *_args, **_kwargs: "sig")

    def run(traces=None):
        campaign._run_rollouts(
            args, tmp_path, {}, samples,
            [{"sample_id": row["id"]} for row in samples] if traces is None else traces,
            phase="pilot", max_new_tokens=32, periods=(4,), repetitions=1, warmups=0,
        )

    return args, samples, stores, run


def _successful_rollout_rows(attempt=1):
    return [
        {"type": "rollout", "variant": variant, "status": "ok",
         "matches_native_full_dflash": True, "attempt": attempt,
         "wall_e2e_ms": 1.0, "counters": {"mean_accepted_candidates": 4.0}}
        for variant in ("ar", "native_full_dflash", "adapter_full", "previous_dflash")
    ]


def test_rollout_rewarms_after_first_repetition_failed_before_all_arms(rollout_case, monkeypatch, tmp_path):
    from TrainingFree.context_interventions import execution

    args, samples, stores, _ = rollout_case
    calls = []

    def measure(_runtime, sample, **kwargs):
        calls.append((kwargs["repetition"], kwargs["warmups"]))
        if kwargs["repetition"] == 0:
            raise RuntimeError("allocation failed before every arm warmed")
        return _successful_rollout_rows()

    monkeypatch.setattr(execution, "run_a3_comparison", measure)
    campaign._run_rollouts(
        args, tmp_path, {}, samples[:1], [{"sample_id": samples[0]["id"]}],
        phase="pilot", max_new_tokens=32, periods=(4,), repetitions=2, warmups=1,
    )
    assert calls == [(0, 1), (0, 1), (1, 1)]
    assert sorted(row["status"] for row in stores[-1].rows()) == ["error", "ok"]


def test_resumed_missing_rollout_repetition_warms_current_process(rollout_case, monkeypatch, tmp_path):
    from TrainingFree.context_interventions import execution
    from TrainingFree.context_interventions.artifacts import CellStore

    args, samples, _, _ = rollout_case
    warmups = []
    monkeypatch.setattr(execution, "run_a3_comparison", lambda *pos, **kwargs: (
        warmups.append(kwargs["warmups"]) or _successful_rollout_rows()
    ))
    save = CellStore.save

    def interrupted_save(self, cell_id, bundle):
        save(self, cell_id, bundle)
        raise OSError("interrupted after committing first repetition")

    def run():
        campaign._run_rollouts(
            args, tmp_path, {}, samples[:1], [{"sample_id": samples[0]["id"]}],
            phase="pilot", max_new_tokens=32, periods=(4,), repetitions=2, warmups=1,
        )

    monkeypatch.setattr(CellStore, "save", interrupted_save)
    with pytest.raises(OSError):
        run()
    monkeypatch.setattr(CellStore, "save", save)
    args.resume = True
    run()
    assert warmups == [1, 1]


def test_rollout_exception_does_not_stop_next_sample_and_resume_recovers_pair(rollout_case, monkeypatch):
    from TrainingFree.context_interventions import execution

    args, samples, stores, run = rollout_case
    calls = []

    def measure(_runtime, sample, **_kwargs):
        calls.append(sample["id"])
        if sample["id"] == "sample-1" and not args.resume:
            raise RuntimeError("temporary allocation failure")
        return _successful_rollout_rows(attempt=len(calls))

    monkeypatch.setattr(execution, "run_a3_comparison", measure)
    run()
    assert calls == ["sample-1", "sample-1", "sample-2"]
    failed, succeeded = stores[-1].rows()
    assert failed["status"] == "error"
    assert all(row["status"] == "error" for row in failed["rows"])
    assert succeeded["status"] == "ok"

    args.resume = True
    run()
    assert calls == ["sample-1", "sample-1", "sample-2", "sample-1"]
    recovered, unchanged = stores[-1].rows()
    assert recovered["status"] == "ok"
    assert {row["attempt"] for row in recovered["rows"]} == {4}
    assert unchanged == succeeded


def test_rollout_atomic_pair_survives_interruption_before_jsonl_merge(rollout_case, monkeypatch):
    from TrainingFree.context_interventions import execution
    from TrainingFree.context_interventions.artifacts import CellStore

    args, samples, stores, run = rollout_case
    calls = []
    original_save = CellStore.save

    def interrupted_save(self, cell_id, row):
        original_save(self, cell_id, row)
        raise OSError("storage connection interrupted after atomic commit")

    def measure(_runtime, sample, **_kwargs):
        calls.append(sample["id"])
        return _successful_rollout_rows()

    monkeypatch.setattr(execution, "run_a3_comparison", measure)
    monkeypatch.setattr(CellStore, "save", interrupted_save)
    with pytest.raises(OSError, match="storage connection"):
        run()
    assert len(stores[-1].rows()[0]["rows"]) == 4
    monkeypatch.setattr(CellStore, "save", original_save)
    args.resume = True
    run()
    assert calls == ["sample-1", "sample-2"]
    assert len(stores[-1].rows()) == 2


def test_rollout_parity_failure_stops_once_after_saving_failed_pair(rollout_case, monkeypatch):
    from TrainingFree.context_interventions import execution

    _args, _samples, stores, run = rollout_case
    calls = []

    def measure(*_args, **_kwargs):
        calls.append(1)
        rows = _successful_rollout_rows()
        next(row for row in rows if row["variant"] == "adapter_full")["matches_native_full_dflash"] = False
        return rows

    monkeypatch.setattr(execution, "run_a3_comparison", measure)
    with pytest.raises(campaign.CampaignIntegrityError, match="parity failed"):
        run()
    assert calls == [1]
    assert stores[-1].rows()[0]["status"] == "parity_fail"


def test_phase_scoped_reference_trace_survives_dev_then_validate_resume(tmp_path, monkeypatch):
    from TrainingFree.context_interventions import execution

    args = SimpleNamespace(
        resume=True, run_id="run-1", attention_backend="flash_attention_4",
        source_anchors=4, recent_output=8,
    )
    sample = {"id": "same-sample", "dataset": "gov_report", "_cad_source_group_id": "same-group"}
    calls = []
    monkeypatch.setattr(campaign, "_code_signature", lambda: {"sha256": "code"})

    def reference(_runtime, _sample, **kwargs):
        calls.append(kwargs["max_new_tokens"])
        return {
            "input_ids": SimpleNamespace(shape=(1, 4)),
            "layout": SimpleNamespace(prompt_hash="hash", source_available=True, source_reason=None),
            "ar_ids": [1], "native_ids": [1], "adapter_ids": [1], "adapter_rounds": [], "eos_ids": set(),
            "parity": {"status": "pass", "native_equals_adapter": True, "native_equals_ar": True,
                       "adapter_equals_ar": True, "ar_output_tokens": 1},
        }

    monkeypatch.setattr(execution, "run_reference_pair", reference)
    for phase, cap in (("validate", 1024), ("pilot", 1024), ("dev", 2048), ("validate", 1024)):
        passed, traces = campaign._run_parity_cohort(
            args, tmp_path, {"runtime_signature": {"signature": "runtime"}}, [sample],
            {"phase": phase, "selected_group_count": 1}, max_new_tokens=cap,
        )
        assert passed and len(traces) == 1
    assert calls == [1024, 2048]


def test_failed_reference_is_isolated_and_does_not_discard_other_samples(tmp_path, monkeypatch):
    import json
    from TrainingFree.context_interventions import execution

    args = SimpleNamespace(
        resume=False, run_id="run-1", attention_backend="flash_attention_4", source_anchors=4, recent_output=8,
    )
    samples = [
        {"id": f"sample-{i}", "dataset": "gov_report", "_cad_source_group_id": f"group-{i}"}
        for i in (1, 2)
    ]
    calls = []
    monkeypatch.setattr(campaign, "_code_signature", lambda: {"sha256": "code"})

    def reference(_runtime, sample, **_kwargs):
        calls.append(sample["id"])
        if sample["id"] == "sample-1":
            raise RuntimeError("reference allocation failed")
        return {
            "input_ids": SimpleNamespace(shape=(1, 4)),
            "layout": SimpleNamespace(prompt_hash="hash", source_available=True, source_reason=None),
            "ar_ids": [1], "native_ids": [1], "adapter_ids": [1], "adapter_rounds": [], "eos_ids": set(),
            "parity": {"status": "pass", "native_equals_adapter": True, "native_equals_ar": True,
                       "adapter_equals_ar": True, "ar_output_tokens": 1},
        }

    monkeypatch.setattr(execution, "run_reference_pair", reference)
    allowed, traces = campaign._run_parity_cohort(
        args, tmp_path, {"runtime_signature": {"signature": "runtime"}}, samples,
        {"phase": "pilot", "selected_group_count": 2}, max_new_tokens=32,
    )
    assert allowed and [trace["sample_id"] for trace in traces] == ["sample-2"]
    assert calls == ["sample-1", "sample-1", "sample-2"]
    evidence = json.loads((tmp_path / "evidence" / "parity_pilot.json").read_text())
    assert evidence["status"] == "partial_with_errors"
    assert evidence["native_adapter_mismatch_count"] == 0


def test_missing_reference_remains_in_planned_rollout_coverage(rollout_case, monkeypatch):
    from TrainingFree.context_interventions import execution
    from TrainingFree.context_adaptive.schema import read_jsonl

    _args, _samples, stores, run = rollout_case
    monkeypatch.setattr(execution, "run_a3_comparison", lambda *_args, **_kwargs: _successful_rollout_rows())
    run(traces=[{"sample_id": "sample-2"}])
    output = read_jsonl(stores[-1].root.parent / "a3_rollout.jsonl")
    assert [row["type"] for row in output] == ["rollout"] * 4 + ["summary"]
    assert output[-1]["rows"] == 4
    assert output[-1]["planned_rows"] == 8
    assert output[-1]["configurations"][0]["status"] == "partial_with_errors"


def test_download_metadata_changes_do_not_invalidate_model_assets(tmp_path):
    from TrainingFree.context_adaptive.benchmark import _asset_signature

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"original weights")
    metadata = tmp_path / ".cache" / "download"
    metadata.mkdir(parents=True)
    cache_file = metadata / "config.json.metadata"
    cache_file.write_text("first download", encoding="utf-8")
    first = _asset_signature(tmp_path)["content_sha256"]
    cache_file.write_text("new timestamp", encoding="utf-8")
    assert _asset_signature(tmp_path)["content_sha256"] == first
    weights.write_bytes(b"changed weights")
    assert _asset_signature(tmp_path)["content_sha256"] != first


def test_ambiguous_sample_ids_are_rejected_before_trace_lookup():
    with pytest.raises(campaign.CampaignIntegrityError, match="duplicate sample ID"):
        campaign._sample_index([
            {"id": "same", "dataset": "gov_report"},
            {"id": "same", "dataset": "qmsum"},
        ])


def test_nan_latency_is_retried_and_isolated_as_failed_pair(rollout_case, monkeypatch):
    from TrainingFree.context_interventions import execution

    _args, _samples, stores, run = rollout_case
    calls = []

    def measure(_runtime, sample, **_kwargs):
        calls.append(sample["id"])
        rows = _successful_rollout_rows()
        if sample["id"] == "sample-1":
            rows[0]["wall_e2e_ms"] = float("nan")
        return rows

    monkeypatch.setattr(execution, "run_a3_comparison", measure)
    run()
    assert calls == ["sample-1", "sample-1", "sample-2"]
    assert stores[-1].rows()[0]["status"] == "error"
    failed_rows = stores[-1].rows()[0]["rows"]
    assert failed_rows[0]["wall_e2e_ms"] is None
    assert failed_rows[0]["nonfinite_fields"]
    assert stores[-1].rows()[1]["status"] == "ok"


def test_optional_period_failures_do_not_trigger_systemic_stop(rollout_case, monkeypatch, tmp_path):
    from TrainingFree.context_interventions import execution

    args, samples, stores, _run = rollout_case

    def measure(_runtime, _sample, **kwargs):
        if kwargs["refresh_period"] != 4:
            raise RuntimeError("optional refresh configuration failed")
        return _successful_rollout_rows()

    monkeypatch.setattr(execution, "run_a3_comparison", measure)
    campaign._run_rollouts(
        args, tmp_path, {}, samples, [{"sample_id": sample["id"]} for sample in samples],
        phase="pilot", max_new_tokens=32, periods=(2, 4, 8, 16), repetitions=1, warmups=0,
    )
    bundles = stores[-1].rows()
    assert len(bundles) == 8
    assert sum(row["status"] == "ok" for row in bundles) == 2
    assert sum(row["status"] == "error" for row in bundles) == 6


def test_driver_telemetry_timeout_reuses_current_preflight_for_same_gpu(tmp_path, monkeypatch):
    import json

    run_root = tmp_path / "run-1"
    run_root.mkdir()
    (run_root / "preflight.json").write_text(json.dumps({
        "nvidia_smi": ["NVIDIA B200, GPU-physical-id, 580.105.08, 42, 0 %"],
    }), encoding="utf-8")
    args = SimpleNamespace(output_root=str(tmp_path), run_id="run-1")
    calls = []

    def timeout(*_args, **_kwargs):
        calls.append(1)
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(campaign.subprocess, "run", timeout)
    assert campaign._probe_gpu_driver(args, "physical-id") == ("GPU-physical-id", "580.105.08")
    assert len(calls) == 2
    with pytest.raises(campaign.CampaignIntegrityError, match="cannot establish GPU/driver"):
        campaign._probe_gpu_driver(args, "different-GPU")


def test_relaxed_mode_tolerates_signature_mismatch_and_strict_rejects(tmp_path, monkeypatch):
    import hashlib
    import json
    from TrainingFree.context_interventions import execution

    run_root = tmp_path / "run-relaxed"
    evidence_dir = run_root / "evidence"
    evidence_dir.mkdir(parents=True)

    # 1. Test lineage recovery from previous phase parity artifact
    (evidence_dir / "parity_validate.json").write_text(json.dumps({
        "code_signature": "lineage-sig-456", "status": "pass",
    }), encoding="utf-8")
    assert campaign._saved_lineage_code_signature(run_root) == "lineage-sig-456"

    # 2. Test trace resume with mismatched signature under relaxed vs strict
    sample = {"id": "sample-1", "dataset": "gov_report", "_cad_source_group_id": "group-1"}
    trace_file = campaign._reference_trace_path(run_root, "sample-1", phase="pilot")
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps({
        "sample_id": "sample-1",
        "dataset": "gov_report",
        "_cad_source_group_id": "group-1",
        "prompt_tokens": 100,
        "trace_signature": "old-stale-sig",
        "parity": {"status": "pass", "native_equals_adapter": True, "ar_output_tokens": 10},
        "adapter_rounds": [],
    }), encoding="utf-8")

    monkeypatch.setattr(campaign, "_code_signature", lambda: {"sha256": "current-code"})
    monkeypatch.setattr(execution, "run_reference_pair", lambda *args, **kwargs: pytest.fail("Should reuse"))

    args = SimpleNamespace(
        resume=True, run_id="run-relaxed", attention_backend="flash_attention_4",
        source_anchors=4, recent_output=8, strict_signatures=False,
    )
    runtime = {"runtime_signature": {"signature": "runtime"}}
    cohort = {"phase": "pilot", "selected_group_count": 1}

    # Relaxed mode: trace is reused, no exception
    passed, traces = campaign._run_parity_cohort(
        args, run_root, runtime, [sample], cohort, max_new_tokens=1024,
    )
    assert passed and len(traces) == 1
    assert traces[0]["sample_id"] == "sample-1"

    # Strict mode: raises ValueError
    args.strict_signatures = True
    with pytest.raises(ValueError, match="resume reference trace signature mismatch"):
        campaign._run_parity_cohort(
            args, run_root, runtime, [sample], cohort, max_new_tokens=1024,
        )
