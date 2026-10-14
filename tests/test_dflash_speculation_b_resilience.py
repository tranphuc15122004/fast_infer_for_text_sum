from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_dflash_speculation_B as campaign


def test_corrupt_json_checkpoint_is_archived_for_recompute(tmp_path):
    artifact = tmp_path / "traces" / "one.json"
    artifact.parent.mkdir()
    artifact.write_text("{truncated", encoding="utf-8")

    assert campaign._read_json_artifact(tmp_path, "traces", artifact) is None
    assert not artifact.exists()
    archived = list((tmp_path / "stale" / "traces").glob("one.corrupt.*.json"))
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == "{truncated"


def test_old_code_signature_cells_are_archived_not_relabelled(tmp_path):
    cell_path = tmp_path / "b1" / "cells" / "cell.json"
    cell_path.parent.mkdir(parents=True)
    old = {
        "status": "complete",
        "campaign_signature": "old",
        "actions": [{"gamma": 15}],
    }
    cell_path.write_text(json.dumps(old), encoding="utf-8")

    campaign._migrate_completed_cells(
        tmp_path,
        "b1",
        current_signature="new",
        is_complete=lambda _: True,
    )

    assert not cell_path.exists()
    archived = list((tmp_path / "stale" / "b1").glob("cell.old.*.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text(encoding="utf-8"))["campaign_signature"] == "old"


def test_incomplete_b0_trace_is_not_reused():
    trace = {
        "b0_status": "complete",
        "prompt_tokens": 8,
        "prompt_hash": "prompt",
        "sample_id": "one",
        "dataset": "qmsum",
        "source_group_id": "group",
        "ar_ids": [1],
        "native_ids": [1],
        "adapter_ids": [1],
        "eos_ids": [],
        "adapter_rounds": [],
        "parity_by_gamma": {"15": {}},
    }

    assert not campaign._complete_b0_trace(trace, (3, 7, 11, 15))


def test_matching_e2e_cell_reuses_without_resume_flag(tmp_path):
    cell_path = tmp_path / "b3" / "cells" / "dev" / "one.json"
    cell_path.parent.mkdir(parents=True)
    saved = {"status": "complete", "campaign_signature": "current"}
    cell_path.write_text(json.dumps(saved), encoding="utf-8")

    loaded = campaign._load_resume_cell(
        cell_path,
        "current",
        resume=False,
        run_root=tmp_path,
        phase="b3/cells/dev",
    )

    assert loaded == saved


def test_b1_timing_cell_can_be_complete_when_optional_b2_signal_failed():
    cell = {
        "actions": [
            {"gamma": gamma, "repetition": repetition}
            for gamma in (3, 7, 11, 15)
            for repetition in range(2)
        ],
        "accepted_by_gamma": {str(gamma): 1 for gamma in (3, 7, 11, 15)},
        "proposal_by_gamma": {str(gamma): [1] for gamma in (3, 7, 11, 15)},
        "signal": None,
        "signal_error": {"reason": "uncertainty extraction failed after retry"},
    }

    assert campaign._complete_b1_cell(cell, (3, 7, 11, 15), 2)


def test_final_run_status_failure_does_not_fail_completed_collection(tmp_path, monkeypatch, capsys):
    def fail(_run_root):
        raise OSError("optional summary write failed")

    monkeypatch.setattr(campaign, "_write_run_status", fail)
    assert campaign._write_run_status_best_effort(tmp_path) is None
    assert "collected artifacts remain available" in capsys.readouterr().err


def test_run_status_marks_missing_b2_signal_as_partial(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "parity.json").write_text(json.dumps({
        "status": "complete",
        "valid_sample_count": 1,
        "failed_sample_count": 0,
        "sample_count": 1,
    }), encoding="utf-8")
    cell_dir = tmp_path / "b1" / "cells"
    cell_dir.mkdir(parents=True)
    (cell_dir / "one.json").write_text(json.dumps({
        "status": "complete",
        "signal": None,
        "signal_error": {"reason": "optional signal extraction failed"},
    }), encoding="utf-8")

    status = campaign._write_run_status(tmp_path)

    assert status["collection_status"] == "partial"
    assert status["b2_signal_failures"] == 1
