from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_specextend_defaults_to_llama31_eagle3_path():
    config = (ROOT / "docs/fast_infer_master.example.env").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_representative_100.sh").read_text(encoding="utf-8")
    adapter = (ROOT / "scripts/infer_specextend.py").read_text(encoding="utf-8")

    assert 'SPECEXTEND_SCRIPT="${SPECEXTEND_SCRIPT:-run_eagle.py}"' in config
    assert 'SPECEXTEND_MODEL_NAME="${SPECEXTEND_MODEL_NAME:-llama3_1_8b}"' in config
    assert "MODEL_TARGET" in config
    assert "MODEL_EAGLE_DRAFT" in config

    assert "BENCH_SPECEXTEND_DRAFT_MODEL" in runner
    assert 'set_env SCRIPT "run_eagle.py"' in runner
    assert 'set_env MODEL_NAME "llama3_1_8b"' in runner
    assert 'set_env DRAFT_MODEL "$(resolve_model_ref "$BENCH_SPECEXTEND_DRAFT_MODEL")"' in runner
    assert '"llama3_1_8b"' in adapter


def test_specextend_eagle_adapter_records_eagle_method():
    adapter = (ROOT / "scripts/infer_specextend.py").read_text(encoding="utf-8")
    eagle_runner = (ROOT / "externals/SpecExtend/specextend/run_eagle.py").read_text(
        encoding="utf-8"
    )

    assert "run_eagle.py" in adapter
    assert '"specextend_eagle"' in adapter
    assert "externals/EAGLE" in eagle_runner
    assert "use_eagle3=True" in eagle_runner


def test_eagle3_loader_keeps_llama_path_compatible_with_legacy_env():
    loader = (ROOT / "externals/EAGLE/eagle/model/ea_model.py").read_text(
        encoding="utf-8"
    )

    assert "try:" in loader
    assert "from .modeling_qwen3_kv" in loader
    assert "except ImportError" in loader


def test_llama31_eagle3_path_has_vram_preflight():
    runner = (ROOT / "externals/SpecExtend/specextend/run_eagle.py").read_text(
        encoding="utf-8"
    )

    assert "SPECEXTEND_MIN_GPU_MEMORY_GB" in runner
    assert "torch.cuda.get_device_properties" in runner


def test_specextend_cli_can_explicitly_disable_hybrid_attention():
    sys.path.insert(0, str(ROOT / "scripts"))
    from infer_specextend import build_parser

    parser = build_parser()
    assert parser.parse_args(["--output", "out.jsonl"]).use_specextend is True
    assert parser.parse_args(
        ["--output", "out.jsonl", "--no-use-specextend"]
    ).use_specextend is False


def test_specextend_wrapper_forwards_disabled_flag():
    wrapper = (ROOT / "scripts/run_specextend.sh").read_text(encoding="utf-8")

    assert "ARGS+=(--use-specextend)" in wrapper
    assert "ARGS+=(--no-use-specextend)" in wrapper
