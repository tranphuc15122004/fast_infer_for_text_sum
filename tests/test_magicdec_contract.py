from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


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
