from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_dflash_speculation_B as campaign


@pytest.mark.parametrize("source_flag", ["--data-dir", "--data-file"])
def test_explicit_data_source_replaces_conflicting_master_sources(tmp_path, monkeypatch, source_flag):
    from TrainingFree.context_adaptive import benchmark

    explicit = tmp_path / "selected"
    monkeypatch.setenv("DATA_INPUT", "/old/directory")
    monkeypatch.setenv("LONG_BENCH_DATA_DIR", "/another/directory")
    monkeypatch.setenv("DATA_FILE", "/old/file.jsonl")
    monkeypatch.setenv("LONG_BENCH_DATA_FILE", "/another/file.jsonl")
    received = {}

    def load(**kwargs):
        received.update(kwargs)
        return [{"dataset": dataset} for dataset in ("gov_report", "multi_news", "qmsum")]

    monkeypatch.setattr(benchmark, "load_corpus", load)
    args = campaign._parser().parse_args([source_flag, str(explicit)])
    campaign._load_samples(args)

    assert received == {
        "data_dir": explicit if source_flag == "--data-dir" else None,
        "data_file": explicit if source_flag == "--data-file" else None,
    }


def test_two_explicit_data_sources_are_rejected():
    args = campaign._parser().parse_args(["--data-dir", "/a", "--data-file", "/b.jsonl"])
    with pytest.raises(ValueError, match="provide exactly one --data-dir/--data-file"):
        campaign._load_samples(args)


@pytest.fixture
def full_workflow(tmp_path, monkeypatch):
    args = campaign._parser().parse_args([
        "--mode", "all", "--run-id", "whole", "--output-root", str(tmp_path),
        "--exposure-manifest", "reviewed.json",
    ])
    calls = []
    state = {"dev_parity": "complete", "confirm_parity": "complete", "collection": "complete"}

    def prepare(current):
        calls.append(f"prepare:{current.cohort_phase}")
        root = campaign._run_root(current)
        root.mkdir(parents=True, exist_ok=True)
        (root / "split_manifest.json").write_text("{}", encoding="utf-8")
        if current.cohort_phase == "confirm":
            assert current.run_id == "whole_confirm"
            assert current.split_manifest == str(tmp_path / "whole" / "split_manifest.json")
        return [{"id": "one", "dataset": "qmsum"}], {}, root

    def runtime(_):
        calls.append("runtime")
        return {"b_campaign_signature": {"gpu_name": "test"}}

    def validate(current, samples, root, runtime):
        calls.append(f"B0:{current.cohort_phase}")
        evidence = root / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "parity.json").write_text(json.dumps({
            "status": state[f"{current.cohort_phase}_parity"],
        }), encoding="utf-8")
        return True

    def b3(current, samples, root, runtime, *, confirm):
        calls.append(f"B3:{current.cohort_phase}")
        if confirm:
            assert current.policy_lock == str(tmp_path / "whole" / "b3" / "locked_config.json")
        return {"status": "complete" if confirm else "locked_for_confirm"}

    def b4_analysis(current, root, runtime_signature):
        calls.append("B4-analysis")
        return {"status": "complete", "decision": "lagged_signal_supported"}

    def b4(current, samples, root, runtime, *, confirm):
        calls.append(f"B4:{current.cohort_phase}")
        if confirm:
            assert current.policy_lock == str(tmp_path / "whole" / "b4" / "locked_config.json")
        return {"status": "complete" if confirm else "locked_for_confirm"}

    monkeypatch.setattr(campaign, "_prepare", prepare)
    monkeypatch.setattr(campaign, "_runtime", runtime)
    monkeypatch.setattr(campaign, "_validate", validate)
    monkeypatch.setattr(campaign, "_b1", lambda *a: calls.append("B1/B2"))
    monkeypatch.setattr(campaign, "_b1v", lambda *a: calls.append("B1-V"))
    monkeypatch.setattr(campaign, "_analyze_without_stopping_collection", lambda *a: {"status": "complete"})
    monkeypatch.setattr(campaign, "_write_run_status_best_effort", lambda *a: {"collection_status": state["collection"]})
    monkeypatch.setattr(campaign, "_b3_campaign", b3)
    monkeypatch.setattr(campaign, "_b4_analyze", b4_analysis)
    monkeypatch.setattr(campaign, "_b4_campaign", b4)
    return args, calls, state, tmp_path / "whole" / "all_status.json"


def test_all_runs_both_eligible_branches_and_disjoint_confirm_once(full_workflow):
    args, calls, _, status_path = full_workflow
    assert campaign._run_all(args) == 0
    assert calls == [
        "prepare:dev", "runtime", "B0:dev", "B1/B2", "B1-V",
        "B3:dev", "B4-analysis", "B4:dev", "prepare:confirm", "B0:confirm",
        "B3:confirm", "B4:confirm",
    ]
    status = json.loads(status_path.read_text())
    assert status["workflow_status"] == "finished"
    assert status["collection_status"] == "complete"


@pytest.mark.parametrize("gate_as_result", [False, True])
def test_closed_research_gates_finish_without_running_confirm(full_workflow, monkeypatch, gate_as_result):
    args, calls, _, status_path = full_workflow

    def closed(*a, **kw):
        if gate_as_result:
            return {"status": "b1v_overhead_gate_closed"}
        raise RuntimeError("B3 gate closed: opportunity is low")

    monkeypatch.setattr(campaign, "_b3_campaign", closed)
    monkeypatch.setattr(campaign, "_b4_analyze", lambda *a: {"status": "inconclusive"})
    assert campaign._run_all(args) == 0
    assert "prepare:confirm" not in calls
    status = json.loads(status_path.read_text())
    assert status["collection_status"] == "complete"
    assert status["stages"]["B3-dev"]["status"] == "not_applicable"
    assert status["stages"]["B4-dev"]["status"] == "not_applicable"


def test_failed_b3_warmups_allow_b4_and_its_confirm_to_finish(full_workflow, monkeypatch):
    args, calls, _, status_path = full_workflow
    monkeypatch.setattr(campaign, "_b3_campaign", lambda *a, **kw: {"status": "warmup_incomplete"})
    assert campaign._run_all(args) == 0
    assert "B4:confirm" in calls
    assert "B3:confirm" not in calls
    status = json.loads(status_path.read_text())
    assert status["workflow_status"] == "finished"
    assert status["collection_status"] == "partial"
    assert status["stages"]["B3-dev"]["status"] == "failed"


@pytest.mark.parametrize("message", ["CUDA error: illegal memory access", "B3 policy lock signature differs"])
def test_fatal_gpu_or_identity_error_stops_other_measurements(full_workflow, monkeypatch, message):
    from TrainingFree.context_interventions.resilience import CampaignIntegrityError

    args, calls, _, status_path = full_workflow

    def fatal(*a, **kw):
        raise RuntimeError(message)

    monkeypatch.setattr(campaign, "_b3_campaign", fatal)
    with pytest.raises(CampaignIntegrityError, match=message):
        campaign._run_all(args)
    assert "B4-analysis" not in calls
    assert json.loads(status_path.read_text())["workflow_status"] == "stopped_fatal"


def test_partial_core_collection_finishes_with_dependent_phases_recorded(full_workflow):
    args, calls, state, status_path = full_workflow
    state["collection"] = "partial"
    assert campaign._run_all(args) == 0
    assert "B3:dev" not in calls
    assert "B4-analysis" not in calls
    status = json.loads(status_path.read_text())
    assert status["workflow_status"] == "finished"
    assert status["collection_status"] == "partial"
    assert status["stages"]["B3-dev"]["status"] == "prerequisites_incomplete"


def test_partial_confirm_validation_does_not_start_invalid_comparisons(full_workflow):
    args, calls, state, status_path = full_workflow
    state["confirm_parity"] = "partial"
    assert campaign._run_all(args) == 0
    assert "B3:confirm" not in calls
    assert "B4:confirm" not in calls
    status = json.loads(status_path.read_text())
    assert status["collection_status"] == "partial"
    assert status["stages"]["confirm-B0"]["status"] == "failed"


def test_unavailable_b2_analysis_does_not_block_independent_b4_branch(full_workflow, monkeypatch):
    args, calls, _, status_path = full_workflow
    monkeypatch.setattr(campaign, "_analyze_without_stopping_collection", lambda *a: {"status": "unavailable"})
    assert campaign._run_all(args) == 0
    assert "B3:dev" not in calls
    assert "B4:confirm" in calls
    assert json.loads(status_path.read_text())["collection_status"] == "partial"


def test_no_usable_b0_stops_without_collecting_invalid_replays(full_workflow, monkeypatch):
    from TrainingFree.context_interventions.resilience import CampaignIntegrityError

    args, calls, _, status_path = full_workflow
    monkeypatch.setattr(campaign, "_validate", lambda *a: False)
    with pytest.raises(CampaignIntegrityError, match="no usable reference"):
        campaign._run_all(args)
    assert "B1/B2" not in calls
    status = json.loads(status_path.read_text())
    assert status["workflow_status"] == "stopped_fatal"
    assert status["stages"]["B0"]["status"] == "failed"


def test_empty_b1v_is_not_reported_as_complete_or_open_gates(full_workflow, monkeypatch):
    args, calls, _, status_path = full_workflow
    monkeypatch.setattr(campaign, "_b1v", lambda *a: {
        "status": "failed", "reason": "no_current_paired_b1_cells",
    })
    assert campaign._run_all(args) == 0
    status = json.loads(status_path.read_text())
    assert status["stages"]["B1-V"]["status"] == "failed"
    assert status["stages"]["B3-dev"]["status"] == "prerequisites_incomplete"
    assert "B3:dev" not in calls
    assert status["collection_status"] == "partial"


def test_all_resume_loads_both_saved_campaigns(full_workflow, monkeypatch):
    args, calls, _, status_path = full_workflow
    args.resume = True
    saved_confirm = status_path.parent.parent / "whole_confirm"
    saved_confirm.mkdir()
    (saved_confirm / "campaign.json").write_text("{}", encoding="utf-8")
    prepare = campaign._prepare

    def load(current):
        calls.append(f"resume:{current.cohort_phase}")
        return prepare(current)

    monkeypatch.setattr(campaign, "_load_resume_campaign", load)
    assert campaign._run_all(args) == 0
    assert "resume:dev" in calls
    assert "resume:confirm" in calls
    assert calls.count("runtime") == 1
