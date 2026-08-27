"""Contract checks for the vendored DFlash GSM8K wrapper."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_dflash_gsm8k.sh"


def test_dflash_gsm8k_wrapper_exports_vendored_module_path_before_exec() -> None:
    """The module-mode child must discover the checkout without installation."""
    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    export_line = 'export PYTHONPATH="$ROOT/externals/dflash${PYTHONPATH:+:$PYTHONPATH}"'
    assert export_line in wrapper_text
    assert wrapper_text.index(export_line) < wrapper_text.index('exec "$FAST_INFER_PYTHON"')
