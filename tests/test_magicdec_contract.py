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
