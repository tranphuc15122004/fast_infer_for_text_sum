from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_b200_venv.sh"


def test_b200_setup_script_creates_python312_venv_and_installs_manifest() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"$PYTHON_BIN" -m venv' in source
    assert '"$VENV_PYTHON" -m pip "${PIP_ARGS[@]}"' in source
    assert "PIP_ARGS=(install" in source
    assert '"$ROOT/requirements.txt"' in source
    assert "awk '!/^flash-attn==/'" in source
    assert 'install_flash_attn_b200.sh' in source
    assert "FAST_INFER_SKIP_FLASH_ATTN" in source
    assert "B200_WHEELHOUSE" in source
    assert "--no-index" in source
    assert "rm -rf" not in source


def test_flash_attn_helper_patches_only_expected_cpp_standard() -> None:
    helper = (
        SCRIPT.parent / "install_flash_attn_b200.sh"
    ).read_text(encoding="utf-8")

    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in helper
    assert 'sed -i \'s/-std=c++17/-std=c++20/g\'' in helper
    assert "--no-build-isolation" in helper
    assert "--no-index" in helper
    assert "https://" not in helper
