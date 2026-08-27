import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import b200_smoke


def test_b200_smoke_preflight_only_records_hardware_block(tmp_path):
    output_dir = tmp_path / "b200-smoke"
    env = dict(os.environ)
    env.update(
        {
            "FAST_INFER_PYTHON": str(ROOT / ".venv/bin/python"),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HOME": str(tmp_path / "hf"),
            "TRITON_CACHE_DIR": str(tmp_path / "triton"),
            "FLASHINFER_WORKSPACE_BASE": str(tmp_path / "flashinfer"),
            "TORCH_EXTENSIONS_DIR": str(tmp_path / "torch-ext"),
        }
    )
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_b200_smoke.sh",
            "--baselines",
            "eagle3,dflash",
            "--preflight-only",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    summary_path = output_dir / "b200_smoke_summary.json"
    assert summary_path.is_file(), proc.stdout + proc.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["interpreter"]["path"] == str(ROOT / ".venv/bin/python")
    assert summary["preflight"]["status"] == "BLOCKED"
    assert summary["baselines"]["eagle3"]["status"] == "BLOCKED"
    assert summary["baselines"]["eagle3"]["reason"] == "hardware_unavailable"
    assert summary["baselines"]["dflash"]["status"] == "BLOCKED"
    assert summary["status"] == "BLOCKED"


def test_b200_smoke_continues_after_one_launcher_failure(tmp_path, monkeypatch):
    fake_root = tmp_path / "fake-root"
    fake_scripts = fake_root / "scripts"
    fake_scripts.mkdir(parents=True)
    fake_launcher = fake_scripts / "run.sh"
    fake_launcher.write_text(
        "#!/usr/bin/env bash\n"
        "[[ \"$1\" == \"eagle3\" ]] && exit 7\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_launcher.chmod(0o755)

    def ready_report(baselines, target_gpu):
        return {
            "status": "PASS",
            "target_gpu": target_gpu,
            "errors": [],
            "baselines": {
                name: {"status": "PASS", "reason": "ready"}
                for name in baselines
            },
        }

    monkeypatch.setattr(b200_smoke, "build_report", ready_report)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "b200_smoke.py",
            "--root",
            str(fake_root),
            "--baselines",
            "eagle3,dflash",
            "--output-dir",
            str(output_dir),
            "--timeout",
            "5",
        ],
    )

    assert b200_smoke.main() == 1
    summary = json.loads(
        (output_dir / "b200_smoke_summary.json").read_text(encoding="utf-8")
    )
    assert summary["baselines"]["eagle3"]["status"] == "FAIL"
    assert summary["baselines"]["dflash"]["status"] == "PASS"
    assert (output_dir / "eagle3.log").is_file()
    assert (output_dir / "dflash.log").is_file()
