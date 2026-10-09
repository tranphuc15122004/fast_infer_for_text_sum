"""Restore a 3.12 venv whose system-interpreter symlink now resolves elsewhere."""

import importlib.util
import json
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/repair_python312_venv.py"


def broken_venv(path):
    venv.EnvBuilder(with_pip=False, symlinks=True).create(path)
    marker = path / "lib/python3.12/site-packages/repair_probe.py"
    marker.write_text("VALUE = 'preserved'\n")
    legacy = path.parent / "legacy-python"
    legacy.write_text("#!/bin/sh\nprintf '(3, 10)\\n'\n")
    legacy.chmod(0o755)
    for name in ("python", "python3", "python3.12"):
        executable = path / "bin" / name
        executable.unlink()
        executable.symlink_to(legacy if name == "python3" else "python3")
    result = subprocess.run([str(path / "bin/python"), "-c", "import sys; print(sys.version_info[:2])"],
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() != "(3, 12)"
    return marker


def module():
    spec = importlib.util.spec_from_file_location("repair_python312", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_cli_restores_all_aliases_and_preserves_installed_packages(tmp_path):
    root = tmp_path / "phucvenv"
    marker = broken_venv(root)
    before = marker.read_bytes()
    result = subprocess.run([sys.executable, str(SCRIPT), "--venv", str(root)],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert Path(report["backup_dir"]).is_dir()
    assert (Path(report["backup_dir"]) / "python3").readlink() == tmp_path / "legacy-python"
    assert marker.read_bytes() == before
    for name in ("python", "python3", "python3.12"):
        result = subprocess.run([str(root / "bin" / name), "-I", "-c",
                                 "import sys, repair_probe; assert sys.version_info[:2] == (3,12); assert repair_probe.VALUE == 'preserved'"],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_failed_rebuild_restores_original_configuration_and_aliases(tmp_path, monkeypatch):
    root = tmp_path / "phucvenv"
    marker = broken_venv(root)
    cfg = (root / "pyvenv.cfg").read_bytes()
    links = {name: (root / "bin" / name).readlink() for name in ("python", "python3", "python3.12")}
    repair = module()

    def fail_rebuild(self, path):
        (Path(path) / "pyvenv.cfg").write_text("partial rebuild")
        raise RuntimeError("simulated rebuild failure")

    monkeypatch.setattr(venv.EnvBuilder, "create", fail_rebuild)
    with pytest.raises(RuntimeError, match="simulated"):
        repair.repair_venv(root)
    assert (root / "pyvenv.cfg").read_bytes() == cfg
    assert all((root / "bin" / name).readlink() == target for name, target in links.items())
    assert marker.read_text() == "VALUE = 'preserved'\n"


def test_refuses_other_python_minor_version_without_changing_links(tmp_path):
    root = tmp_path / "different-venv"
    broken_venv(root)
    cfg = root / "pyvenv.cfg"
    cfg.write_text("home = /usr/bin\nversion = 3.10.6\n")
    original = (root / "bin/python3").readlink()
    result = subprocess.run([sys.executable, str(SCRIPT), "--venv", str(root)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "3.12" in result.stderr
    assert (root / "bin/python3").readlink() == original
