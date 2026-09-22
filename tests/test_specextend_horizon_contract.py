from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_horizon_runner_and_decision_note_exist():
    assert (ROOT / "scripts/runners/run_specextend_horizon.sh").is_file()
    assert (ROOT / "docs/experiments/2026-09-13_specextend_horizon_cmr_decision.md").is_file()


def test_trace_hook_is_optional_and_runtime_only():
    source = (ROOT / "externals/SpecExtend/specextend/classic/model_classic.py").read_text()
    assert "SPECEXTEND_TRACE_FILE" in source
    assert "target-future" not in source.lower()
    assert "_source_chunk_attention" in source


def test_runner_uses_official_vicuna_pair_and_stops_on_preflight():
    source = (ROOT / "scripts/runners/run_specextend_horizon.sh").read_text()
    assert "vicuna_7b" in source
    assert "govreport_4K.jsonl" in source
    assert "preflight" in source
    assert "SpecExtend Horizon-CMR blocked by preflight" in source
