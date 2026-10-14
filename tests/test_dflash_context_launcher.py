"""Detached launcher/lock tests; model phases are CPU process substitutes."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("launcher state did not arrive before deadline")


@pytest.fixture
def launcher(tmp_path):
    replica = tmp_path / "repo"
    (replica / "scripts/common").mkdir(parents=True)
    for name in ("scripts/run_dflash_context_A_b200.sh", "scripts/common/config.sh", "scripts/common/runtime.sh"):
        shutil.copy2(ROOT / name, replica / name)
    audit = replica / "data/dflash_context_A_audit"
    audit.mkdir(parents=True)
    (audit / "combined_exposure_manifest.json").write_text('{"review_status": "complete"}')
    (audit / "exposure_audit.json").write_text('{"status": "complete"}')
    corpus = replica / "data/corpus"
    corpus.mkdir()
    master = tmp_path / "master.env"
    master.write_text(f'DCE_DATA_DIR="{corpus}"\n')
    probe = replica / "scripts/phase_substitute.py"
    probe.write_text('''
import json, sys, time
from pathlib import Path
args = sys.argv[1:]
phase = args[args.index("--mode") + 1]
run_id = args[args.index("--run-id") + 1]
root = Path(__file__).resolve().parents[1]
run = root / "outputs/dflash_context_interventions" / run_id
run.mkdir(parents=True, exist_ok=True)
with (run / "phase_calls.jsonl").open("a") as handle:
    handle.write(json.dumps({"phase": phase, "resume": "--resume" in args}) + "\\n")
def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
if phase == "preflight":
    save(run / "preflight.json", {"status": "ready"})
elif phase == "prepare":
    save(run / "prepare_manifest.json", {"source_files": []})
    save(run / "cohorts/split_manifest.json", {"status": "complete"})
    for cohort_phase in ("pilot", "dev", "confirm"):
        save(run / "cohorts" / (cohort_phase + ".json"), {
            "status": "complete", "quotas": {"gov_report": 1, "multi_news": 1, "qmsum": 1},
            "groups": [{"dataset": name} for name in ("gov_report", "multi_news", "qmsum")],
        })
elif phase == "validate":
    (run / "validate_started").touch()
    if (root / "validate_exit").exists():
        raise SystemExit(int((root / "validate_exit").read_text()))
    if (root / "fail_validation_once").exists() and "--resume" not in args:
        raise SystemExit(2)
    deadline = time.monotonic() + 10
    while (root / "hold_validation").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
''')
    core = replica / "scripts/run_dflash_context_experiments.sh"
    core.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                    'exec "$FAST_INFER_PYTHON" "$(dirname "$0")/phase_substitute.py" "$@"\n')
    environment = dict(os.environ)
    for name in ("FAST_INFER_MASTER_LOADED", "DCE_DATA_DIR", "DCE_RESUME"):
        environment.pop(name, None)
    environment.update(FAST_INFER_MASTER_CONFIG=str(master), FAST_INFER_PYTHON=sys.executable)
    script = replica / "scripts/run_dflash_context_A_b200.sh"

    def invoke(*args):
        return subprocess.run(["bash", str(script), *args], env=environment,
                              capture_output=True, text=True, timeout=10)

    yield replica, invoke
    (replica / "hold_validation").unlink(missing_ok=True)
    for pid_file in replica.glob("outputs/dflash_context_interventions/*/campaign.pid"):
        pid = int(pid_file.read_text())
        # No kill is needed: release the phase substitute and let the EXIT
        # trap finish, including in failing tests.
        wait_until(lambda: not Path(f"/proc/{pid}").exists()
                   or Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z")


def test_start_finishes_all_phases_and_refuses_resume_of_complete_run(launcher):
    replica, invoke = launcher
    result = invoke("start")
    assert result.returncode == 0, result.stderr
    run_id = next(line.removeprefix("RUN_ID=") for line in result.stdout.splitlines() if line.startswith("RUN_ID="))
    run = replica / "outputs/dflash_context_interventions" / run_id
    wait_until(lambda: (run / "campaign.status").is_file()
               and "exit_code=0" in (run / "campaign.status").read_text())
    phases = [json.loads(line)["phase"] for line in (run / "phase_calls.jsonl").read_text().splitlines()]
    assert phases == ["preflight", "prepare", "validate", "pilot", "dev", "lock", "confirm", "report"]
    assert "campaign complete" in (run / "campaign.log").read_text()
    assert invoke("resume", run_id).returncode == 2


def test_detached_campaign_keeps_flock_and_refuses_duplicate_resume(launcher):
    replica, invoke = launcher
    (replica / "hold_validation").touch()
    result = invoke("start")
    assert result.returncode == 0, result.stderr
    run_id = next(line.removeprefix("RUN_ID=") for line in result.stdout.splitlines() if line.startswith("RUN_ID="))
    run = replica / "outputs/dflash_context_interventions" / run_id
    wait_until(lambda: (run / "validate_started").exists())
    lock = subprocess.run(["flock", "-n", str(run / "campaign.lock"), "true"], capture_output=True, timeout=5)
    assert lock.returncode == 1
    (run / "campaign.pid").unlink()  # An orphan/legacy worker still holds the lock.
    duplicate = invoke("resume", run_id)
    assert duplicate.returncode == 2
    assert "already" in duplicate.stderr
    (replica / "hold_validation").unlink()
    wait_until(lambda: "exit_code=0" in (run / "campaign.status").read_text())
    wait_until(lambda: subprocess.run(["flock", "-n", str(run / "campaign.lock"), "true"],
                                     capture_output=True, timeout=5).returncode == 0)


@pytest.mark.parametrize("exit_code", (3, 130))
def test_detached_campaign_preserves_serious_error_and_interrupt(launcher, exit_code):
    replica, invoke = launcher
    (replica / "validate_exit").write_text(str(exit_code))
    result = invoke("start")
    assert result.returncode == 0, result.stderr
    run_id = next(line.removeprefix("RUN_ID=") for line in result.stdout.splitlines() if line.startswith("RUN_ID="))
    run = replica / "outputs/dflash_context_interventions" / run_id
    wait_until(lambda: (run / "campaign.status").is_file()
               and f"exit_code={exit_code}" in (run / "campaign.status").read_text())
    phases = [json.loads(line)["phase"] for line in (run / "phase_calls.jsonl").read_text().splitlines()]
    assert phases == ["preflight", "prepare", "validate"]
    assert "campaign complete" not in (run / "campaign.log").read_text()
