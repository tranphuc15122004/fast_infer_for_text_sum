import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_b200_profile_uses_python3_and_one_sample():
    config = (ROOT / "docs/fast_infer_master.example.env").read_text(encoding="utf-8")
    assert 'FI_PYTHON="${FI_PYTHON:-python3}"' in config
    assert 'B200_MAX_SAMPLES="${B200_MAX_SAMPLES:-1}"' in config
    assert "FI_TRITON_CACHE" in config
    assert "FI_FLASHINFER_CACHE" in config


def test_preflight_emits_structured_report_without_cuda(tmp_path):
    report_path = tmp_path / "preflight.json"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/check_b200_env.py",
            "--baselines",
            "eagle3,dflash",
            "--json",
            str(report_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] in {"PASS", "BLOCKED", "FAIL"}
    assert report["interpreter"]["path"] == sys.executable
    assert report["cuda"]["available"] is False
    assert report["cuda"]["reason"] == "hardware_unavailable"
    assert report["baselines"]["eagle3"]["status"] == "BLOCKED"
    assert report["baselines"]["eagle3"]["reason"] == "hardware_unavailable"
    assert report["baselines"]["dflash"]["status"] == "BLOCKED"


def test_preflight_accepts_b200_device_and_runs_cuda_tensor_probe(tmp_path, monkeypatch):
    import check_b200_env

    class FakeTensor:
        def __add__(self, other):
            return self

        def item(self):
            return 1

    class FakeCuda:
        def synchronize(self):
            return None

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        zeros=lambda size, device: FakeTensor(),
    )
    monkeypatch.setattr(
        check_b200_env,
        "_check_torch",
        lambda: (
            {
                "available": True,
                "reason": "ok",
                "devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA B200",
                        "major": 10,
                        "minor": 0,
                        "total_memory_gb": 180.0,
                    }
                ],
            },
            fake_torch,
        ),
    )
    monkeypatch.setattr(check_b200_env, "_check_imports", lambda names: {
        name: {"ok": True, "version": "test"} for name in names
    })
    monkeypatch.setattr(check_b200_env, "_check_assets", lambda baseline: {})
    for name in check_b200_env.WRITABLE_CACHE_ENV:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    report = check_b200_env.build_report(["eagle3"], "B200")
    assert report["cuda"]["target_match"] is True
    assert report["cuda"]["reason"] == "ok"
    assert report["cuda"]["tensor_probe"]["ok"] is True
