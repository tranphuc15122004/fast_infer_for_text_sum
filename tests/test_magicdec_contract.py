from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_magicdec_clear_kv_does_not_zero_full_cache_by_default():
    path = ROOT / "externals/MagicDec/Engine/SnapKV/backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "clear_kv"
    ]
    assert len(methods) == 1
    method = methods[0]
    assert any(
        isinstance(default, ast.Constant) and default.value is False
        for default in method.args.defaults
    )
    zero_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "zero_"
    ]
    assert len(zero_calls) >= 2
    full_cache_zero_calls = [
        node
        for node in zero_calls
        if isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr in {"kv_cache", "draft_cache"}
    ]
    assert len(full_cache_zero_calls) == 2
    assert any(isinstance(node, ast.If) for node in method.body)


def _load_magicdec_script():
    path = SCRIPTS / "infer_magicdec.py"
    spec = importlib.util.spec_from_file_location("infer_magicdec", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_magicdec_token_id_output_uses_last_token():
    module = _load_magicdec_script()
    engine_output = torch.tensor([[101, 202, 303]], dtype=torch.long)

    next_token = module._canonical_next_token(engine_output, temperature=0.0)

    assert torch.equal(next_token, torch.tensor([[303]], dtype=torch.long))


def test_magicdec_token_id_output_rejects_temperature_sampling():
    module = _load_magicdec_script()
    engine_output = torch.tensor([[101, 202, 303]], dtype=torch.long)

    with pytest.raises(ValueError, match="token IDs"):
        module._canonical_next_token(engine_output, temperature=0.7)


def test_magicdec_self_spec_warmup_has_enough_context_for_snapkv():
    module = _load_magicdec_script()
    input_ids = torch.tensor([[101, 202]], dtype=torch.long)

    warmup_ids = module._ensure_magicdec_warmup_context(
        input_ids, min_tokens=257, fill_token_id=2
    )

    assert warmup_ids.shape == (1, 257)
    assert torch.equal(warmup_ids[:, :2], input_ids)
    assert torch.equal(warmup_ids[:, 2:], torch.full((1, 255), 2))


def test_magicdec_warmup_context_does_not_truncate_long_prompt():
    module = _load_magicdec_script()
    input_ids = torch.arange(300, dtype=torch.long).reshape(1, -1)

    warmup_ids = module._ensure_magicdec_warmup_context(
        input_ids, min_tokens=257, fill_token_id=2
    )

    assert warmup_ids is input_ids


def test_magicdec_acceptance_summary_reports_mean_tau_and_draft_rate():
    module = _load_magicdec_script()

    result = module.summarize_magicdec_acceptance([1, 3, 4], gamma=4)

    assert result["avg_accept_length"] == pytest.approx(2.6667)
    assert result["acceptance_rate"] == pytest.approx(0.4167)
    assert result["rejected_draft_ratio"] == pytest.approx(0.5833)


def test_magicdec_eos_ids_match_upstream_eot_fallback():
    module = _load_magicdec_script()

    class Tokenizer:
        eos_token_id = 2
        unk_token_id = None

        @staticmethod
        def encode(value, add_special_tokens=False):
            assert value == "<|eot_id|>"
            assert add_special_tokens is False
            return [128009]

    assert module._eos_ids(Tokenizer()) == {2, 128009}


def test_magicdec_record_carries_self_spec_acceptance_and_phase_metrics():
    module = _load_magicdec_script()
    args = type(
        "Args",
        (),
        {
            "data_file": "/tmp/gov_report.jsonl",
            "model_name": "model",
            "model_pth": "/tmp/model.pth",
        },
    )()

    record = module.build_magicdec_record(
        sample={"id": "s1", "reference": "answer", "raw": {"task_type": "qa"}},
        args=args,
        input_tokens=100,
        output_tokens=8,
        text="answer",
        timing={"e2e_ms": 10.0},
        config={"self_spec": True},
        acceptance_lengths=[1, 3],
        speculative_metrics={
            "acceptance_rate": 0.25,
            "draft_latency_ms": 2.0,
            "verification_latency_ms": 4.0,
            "rejected_draft_ratio": 0.75,
        },
    )

    assert record["acceptance_lengths"] == [1, 3]
    assert record["avg_accept_length"] == pytest.approx(2.0)
    assert record["acceptance_rate"] == pytest.approx(0.25)
    assert record["draft_latency_ms"] == pytest.approx(2.0)
    assert record["verification_latency_ms"] == pytest.approx(4.0)


def test_longbench_magicdec_command_can_enable_self_spec(tmp_path):
    from common.longbench_adapter import build_adapter_command

    command = build_adapter_command(
        "magicdec",
        config={
            "python": "python3",
            "magicdec_model_pth": "/tmp/model.pth",
            "magicdec_model_name": "model",
            "magicdec_self_spec": True,
            "magicdec_gamma": 3,
            "magicdec_draft_budget": 257,
            "magicdec_window_size": 128,
        },
        data_file=tmp_path / "gov_report.jsonl",
        output=tmp_path / "out.jsonl",
        mode="full",
        max_samples=1,
        max_new_tokens=8,
    )

    assert command is not None
    assert "--self-spec" in command
    assert command[command.index("--gamma") + 1] == "3"
    assert command[command.index("--draft-budget") + 1] == "257"
