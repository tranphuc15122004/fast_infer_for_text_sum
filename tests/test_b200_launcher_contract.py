from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from b200_smoke import _overlay_values  # noqa: E402


def test_b200_runner_uses_shared_runtime_and_overlay():
    text = (ROOT / "scripts/run_b200_smoke.sh").read_text(encoding="utf-8")
    assert 'source "$ROOT/scripts/common/runtime.sh" || exit 1' in text
    assert 'FAST_INFER_CONFIG_OVERLAY' in text or "b200_smoke.py" in text
    assert 'FAST_INFER_PYTHON' in text


def test_all_launchers_use_shared_runtime():
    for runner in sorted((ROOT / "scripts").glob("run_*.sh")):
        text = runner.read_text(encoding="utf-8")
        assert 'source "$ROOT/scripts/common/runtime.sh" || exit 1' in text, runner


def test_b200_profile_keeps_production_python_separate_from_local_simulation():
    config = (ROOT / "config/b200.env").read_text(encoding="utf-8")
    assert 'FAST_INFER_PYTHON="${FAST_INFER_PYTHON:-python3}"' in config
    assert 'FAST_INFER_PYTHON="$PWD/.venv/bin/python"' in config
    assert 'B200_TARGET_GPU="B200"' in config
    assert 'B200_DEVICE="${B200_DEVICE:-cuda}"' in config


def test_b200_profile_lists_all_dispatcher_baselines():
    config = (ROOT / "config/b200.env").read_text(encoding="utf-8")
    expected = {
        "eagle3", "dflash", "llmlingua", "fastkv", "rocketkv", "gemfilter",
        "specprefill", "minference", "magicdec", "longspec", "specextend",
        "higoe", "semantic_selection", "flexprefill",
    }
    value = re.search(r'B200_BASELINES="\$\{B200_BASELINES:-([^}]*)\}"', config)
    assert value is not None
    listed = value.group(1).split()
    assert set(listed) == expected


def test_b200_overlays_keep_eagle_and_dflash_pairs_aligned(monkeypatch, tmp_path):
    monkeypatch.setenv("B200_TARGET_MODEL", "base-llama31")
    monkeypatch.setenv("B200_EAGLE_MODEL", "eagle-llama31")
    monkeypatch.setenv("B200_DFLASH_MODEL", "dflash-llama31")
    monkeypatch.setenv("B200_DATA_FILE", "data/sample.jsonl")
    monkeypatch.setenv("B200_SMOKE_MAX_SAMPLES", "1")
    monkeypatch.setenv("B200_SMOKE_MAX_NEW_TOKENS", "8")
    monkeypatch.setenv("B200_DEVICE", "cpu")
    generated = {
        "EAGLE_DATA_FILE": str(tmp_path / "eagle.jsonl"),
        "SPECEXTEND_INPUT_FILE": str(tmp_path / "specextend.jsonl"),
    }

    eagle = _overlay_values("eagle3", tmp_path / "eagle.out", generated)
    dflash = _overlay_values("dflash", tmp_path / "dflash.out", generated)
    assert eagle["BASE_MODEL"] == "base-llama31"
    assert eagle["EAGLE_MODEL"] == "eagle-llama31"
    assert eagle["DATA_FILE"] == generated["EAGLE_DATA_FILE"]
    assert dflash["TARGET_MODEL"] == "base-llama31"
    assert dflash["DRAFT_MODEL"] == "dflash-llama31"
    assert dflash["DATA_FILE"] == "data/sample.jsonl"
    assert eagle["MAX_SAMPLES"] == dflash["MAX_SAMPLES"] == "1"
