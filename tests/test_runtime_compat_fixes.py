from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llama31_rope_theta_is_recovered_from_modern_config():
    from common.model_compat import ensure_rope_theta
    from transformers import LlamaConfig

    config = LlamaConfig(rope_theta=500000.0)
    # Transformers 5 stores this value only in rope_parameters, while older
    # releases may expose a top-level attribute.  Exercise the nested path on
    # both versions.
    if hasattr(config, "rope_theta"):
        delattr(config, "rope_theta")

    ensure_rope_theta(config)

    assert config.rope_theta == 500000.0


def test_dflash_adapter_registers_vendored_module_path():
    _load_script("infer_dflash.py")

    assert str(ROOT / "externals" / "dflash") in sys.path


def test_magicdec_adapter_registers_package_parent_path():
    module = _load_script("infer_magicdec.py")

    assert module.MAGICDEC_PARENT == ROOT / "externals"
    assert str(module.MAGICDEC_PARENT) in sys.path


def test_eagle_draft_attention_accepts_llama31_rope_schema():
    eagle_root = ROOT / "externals" / "EAGLE"
    if str(eagle_root) not in sys.path:
        sys.path.insert(0, str(eagle_root))

    from eagle.model.cnets import LlamaAttention
    from eagle.model.configs import EConfig
    import torch

    rope_scaling = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
        "rope_theta": 500000.0,
    }
    config = EConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=131072,
        rope_scaling=rope_scaling,
    )

    attention = LlamaAttention(config)

    assert attention._uses_llama3_rope is True
    assert attention.rotary_emb.inv_freq.numel() == 8
    output, _, _ = attention(
        torch.randn(1, 2, 128),
        position_ids=torch.arange(2).unsqueeze(0),
    )
    assert output.shape == (1, 2, 64)


def test_fafo_modules_import_without_fastchat():
    fafo_root = ROOT / "externals" / "FAFO"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fafo_root), str(ROOT / "scripts")]
    )
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pipeline.fafo.utils; import pipeline.fafo.eval_gsm8k; "
            "import pipeline.fafo.eval_mtbench",
        ],
        cwd=fafo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_fafo_llama_attention_accepts_nested_rope_theta():
    fafo_root = ROOT / "externals" / "FAFO"
    if str(fafo_root) not in sys.path:
        sys.path.insert(0, str(fafo_root))
    from pipeline.fafo.models.modeling_llama import LlamaAttention
    from transformers import LlamaConfig

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
    )
    attention = LlamaAttention(config, layer_idx=0)

    assert attention.rope_theta == 500000.0


def test_fafo_fastchat_question_loader_fallback(tmp_path):
    fafo_root = ROOT / "externals" / "FAFO"
    if str(fafo_root) not in sys.path:
        sys.path.insert(0, str(fafo_root))
    from fastchat_compat import load_questions

    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"question_id": "q1", "turns": ["hello"]}) + "\n",
        encoding="utf-8",
    )

    assert load_questions(str(path), None, None)[0]["question_id"] == "q1"


def test_specextend_has_termcolor_fallback():
    specextend_root = ROOT / "externals" / "SpecExtend" / "specextend"
    if str(specextend_root) not in sys.path:
        sys.path.insert(0, str(specextend_root))
    from optional_deps import colored

    assert colored("hello", "green") == "hello"


def test_sssd_without_fixed_datastore_is_allowed_as_prompt_only_smoke():
    from common.longbench_adapter import preflight_baseline

    result = preflight_baseline(
        "sssd",
        config={"model": "meta-llama/Llama-3.1-8B-Instruct"},
        cuda_available=True,
    )

    assert result["status"] == "aggregate_only"
    assert "empty" in result["reason"] or "without" in result["reason"]
