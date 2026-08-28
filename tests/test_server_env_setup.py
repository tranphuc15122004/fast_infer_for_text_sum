from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup_server_env.py"
EXAMPLE = ROOT / "docs" / "fast_infer_master.example.env"


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    (repo / "data").mkdir()
    (repo / "docs").mkdir()
    (repo / "scripts").mkdir()
    shutil.copy2(EXAMPLE, repo / "docs" / EXAMPLE.name)
    return repo


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    repo = tmp_path / "repo"
    shared = tmp_path / "shared-data"
    env = dict(os.environ)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-dir",
            str(repo),
            "--shared-data-dir",
            str(shared),
            *args,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_init_creates_shared_master_pointer_and_dataset_links(tmp_path):
    _make_repo(tmp_path)
    result = _run(tmp_path, "--init")

    assert result.returncode == 0, result.stdout + result.stderr
    repo = tmp_path / "repo"
    shared = tmp_path / "shared-data"
    master = shared / "fast_infer_master.env"

    assert master.is_file()
    pointer_lines = (repo / "config" / "master.path").read_text(encoding="utf-8").splitlines()
    assert next(line.strip() for line in pointer_lines if line.strip() and not line.lstrip().startswith("#")) == str(master)
    for name in ("longbench_200", "representative_100"):
        link = repo / "data" / name
        assert link.is_symlink()
        assert link.resolve() == (shared / name).resolve()

    probe = subprocess.run(
        [
            "bash",
            "-c",
            'set -a; source "$1"; printf "%s\n%s\n%s\n" "$FI_PYTHON" "$LONG_BENCH_DATA_DIR" "$DATA_ROOT"',
            "bash",
            str(master),
        ],
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stdout.splitlines() == [
        "python3",
        str(shared / "longbench_200"),
        str(shared),
    ]


def test_init_does_not_overwrite_existing_master_config(tmp_path):
    _make_repo(tmp_path)
    first = _run(tmp_path, "--init")
    assert first.returncode == 0, first.stdout + first.stderr

    master = tmp_path / "shared-data" / "fast_infer_master.env"
    original = "# operator-owned config\nMODEL_TARGET=/models/operator-target\n"
    master.write_text(original, encoding="utf-8")

    second = _run(tmp_path, "--init")
    assert second.returncode == 0, second.stdout + second.stderr
    assert master.read_text(encoding="utf-8") == original


def test_check_is_read_only_when_shared_data_is_missing(tmp_path):
    _make_repo(tmp_path)
    result = _run(tmp_path, "--check", "--skip-dependencies")

    assert result.returncode != 0
    assert not (tmp_path / "shared-data").exists()
    assert not (tmp_path / "repo" / "config" / "master.path").exists()
