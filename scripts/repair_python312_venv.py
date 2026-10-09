#!/usr/bin/env python3
"""Rebind an existing Python 3.12 venv to this complete 3.12 runtime, offline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def repair_venv(root: Path) -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run this helper with a real Python 3.12 interpreter")
    root = root.expanduser().resolve(strict=True)
    cfg = root / "pyvenv.cfg"
    values = {}
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if not values.get("version", "").startswith("3.12."):
        raise ValueError("This helper only repairs venvs originally created with Python 3.12")
    if not (root / "lib/python3.12/site-packages").is_dir():
        raise ValueError("Original Python 3.12 site-packages directory is missing")
    base_executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    if base_executable.is_relative_to(root):
        raise ValueError("Use the independent runtime, not the venv being repaired")

    names = ("python", "python3", "python3.12", "activate", "activate.csh", "activate.fish", "Activate.ps1")
    paths = {name: root / "bin" / name for name in names}
    paths["pyvenv.cfg"] = cfg
    backup = Path(tempfile.mkdtemp(prefix="python_repair_backup.", dir=root))
    original = set()
    for name, path in paths.items():
        if path.exists() or path.is_symlink():
            if path.is_dir():
                raise ValueError(f"Expected a file or symlink: {path}")
            shutil.copy2(path, backup / name, follow_symlinks=False)
            original.add(name)

    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    try:
        # Remove only interpreter aliases; the builder regenerates cfg/activation.
        # Do not clear the venv or reinstall any third-party packages.
        for name in ("python", "python3", "python3.12"):
            paths[name].unlink(missing_ok=True)
        venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
        probe = (
            "import sys, ssl, ctypes, sqlite3, json; "
            "assert sys.version_info[:2] == (3,12), sys.version; "
            f"assert sys.prefix == {str(root)!r}, sys.prefix; "
            "print(json.dumps({'version': sys.version, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix}))"
        )
        result = subprocess.run([str(root / "bin/python"), "-I", "-c", probe],
                                env=environment, text=True, capture_output=True, check=True)
    except BaseException:
        for name, path in paths.items():
            path.unlink(missing_ok=True)
            if name in original:
                shutil.copy2(backup / name, path, follow_symlinks=False)
        raise

    return {"venv": str(root), "base_executable": str(base_executable),
            "backup_dir": str(backup), "probe": json.loads(result.stdout),
            "packages_reinstalled": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(repair_venv(args.venv), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
