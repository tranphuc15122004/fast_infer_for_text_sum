from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_b200_venv.sh"


def test_b200_setup_script_creates_python312_venv_and_installs_manifest() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"$PYTHON_BIN" -m venv' in source
    assert '"$VENV_PYTHON" -m pip "${PIP_ARGS[@]}"' in source
    assert "PIP_ARGS=(install" in source
    assert '"$ROOT/requirements.txt"' in source
    assert "B200_WHEELHOUSE" in source
    assert "--no-index" in source
    assert "rm -rf" not in source
