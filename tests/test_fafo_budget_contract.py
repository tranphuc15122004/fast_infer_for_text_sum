from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAFO_ROOT = ROOT / "externals" / "FAFO"
sys.path.insert(0, str(FAFO_ROOT))


def test_fafo_budget_caps_accepted_tokens_before_append():
    from pipeline.fafo.decoding import cap_accepted_tokens

    assert cap_accepted_tokens(current_new_tokens=7, max_hit=4, budget=8) == 1
    assert cap_accepted_tokens(current_new_tokens=8, max_hit=4, budget=8) == 0
    assert cap_accepted_tokens(current_new_tokens=2, max_hit=1, budget=8) == 2
