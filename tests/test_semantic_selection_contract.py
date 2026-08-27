"""Contract checks for the direct semantic-selection smoke fixture."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_semantic_selection.sh"


def test_direct_smoke_fixture_uses_semantic_selection_document_field() -> None:
    """The wrapper's default smoke file must match its ``document`` contract."""
    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    match = re.search(r'INPUT_FILE="\$\{INPUT_FILE:-([^}]+)\}"', wrapper_text)
    assert match, "wrapper must define a deterministic default INPUT_FILE"

    fixture = ROOT / match.group(1)
    first_record = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])
    assert "document" in first_record, (
        f"semantic-selection smoke fixture {fixture} must contain a 'document' field"
    )
