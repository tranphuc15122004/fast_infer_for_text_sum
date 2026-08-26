import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "src" / "analyze" / "full_infer"
sys.path.insert(0, str(PROFILE_ROOT))

from profile_qwen3_long_summary import (  # noqa: E402
    component_ratios,
    select_source_row,
    truncate_words,
)


def test_truncate_words_preserves_exact_prefix_length():
    text = "one  two\nthree four five"

    truncated = truncate_words(text, 3)

    assert truncated == "one two three"
    assert len(truncated.split()) == 3


def test_select_source_row_chooses_smallest_document_covering_mark():
    rows = [
        {"id": "long", "document": "w " * 20},
        {"id": "short", "document": "w " * 5},
        {"id": "middle", "document": "w " * 10},
    ]

    selected = select_source_row(rows, 8)

    assert selected["id"] == "middle"


def test_component_ratios_sum_to_one_and_keep_zero_phases():
    ratios = component_ratios({"prefill_ms": 30.0, "decode_ms": 70.0})

    assert ratios["prefill_ms"] == 0.3
    assert ratios["decode_rest_ms"] == 0.7
    assert ratios["tokenize_ms"] == 0.0
    assert sum(ratios.values()) == 1.0


def test_dynamic_cache_byte_counter_reads_layer_keys_and_values():
    import torch

    from profile_qwen3_long_summary import cache_nbytes

    class Layer:
        def __init__(self):
            self.keys = torch.zeros((2, 3), dtype=torch.float16)
            self.values = torch.zeros((2, 3), dtype=torch.float16)

    class Cache:
        def __init__(self):
            self.layers = [Layer()]

    assert cache_nbytes(Cache(), torch) == 24


def test_profile_fixture_has_representative_schema():
    path = ROOT / "data/representative_100/govreport_representative.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert {"id", "document", "reference"}.issubset(row)


def test_plotter_includes_one_time_model_load_comparison(tmp_path):
    from profile_qwen3_long_summary import write_plots

    row = {
        "word_mark": 256,
        "status": "ok",
        "tokenize_ms": 1.0,
        "input_transfer_ms": 1.0,
        "prefill_ms": 10.0,
        "kv_cache_first_read_ms": 1.0,
        "decode_rest_ms": 20.0,
        "postprocess_ms": 1.0,
        "unattributed_ms": 1.0,
        "kv_cache_mb": 1.0,
        "peak_allocated_mb": 10.0,
        "peak_reserved_mb": 12.0,
        "component_ratios": component_ratios({
            "tokenize_ms": 1.0,
            "input_transfer_ms": 1.0,
            "prefill_ms": 10.0,
            "kv_cache_first_read_ms": 1.0,
            "decode_rest_ms": 20.0,
            "postprocess_ms": 1.0,
            "unattributed_ms": 1.0,
        }),
    }

    paths = write_plots([row], tmp_path, model_load_ms=1000.0)

    assert any(path.name == "model_load_vs_sample_total.png" for path in paths)


def test_full_infer_keeps_canonical_code_and_experiment_results_together():
    assert (PROFILE_ROOT / "profile_qwen3_long_summary.py").is_file()
    assert (PROFILE_ROOT / "results" / "summary.csv").is_file()
    assert (PROFILE_ROOT / "results" / "phase_time_stacked.png").is_file()


def test_profile_wrapper_runs_the_canonical_full_infer_source():
    wrapper = (ROOT / "scripts" / "run_qwen3_long_profile.sh").read_text(
        encoding="utf-8"
    )

    assert "$ROOT/src/analyze/full_infer/profile_qwen3_long_summary.py" in wrapper
