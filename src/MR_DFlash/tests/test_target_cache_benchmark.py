"""Unit tests cho helper không cần tải target model của benchmark cache."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_benchmark_removes_only_final_assistant_from_regenerated_row() -> None:
    from target_cache_benchmark import generation_messages

    messages = [
        {"role": "user", "content": "Document"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Summarize it"},
        {"role": "assistant", "content": "Target answer"},
    ]

    assert generation_messages(messages) == messages[:-1]
    assert generation_messages(messages[:-1]) == messages[:-1]


def test_benchmark_token_report_detects_exact_and_non_exact_outputs() -> None:
    from target_cache_benchmark import compare_token_ids

    exact = compare_token_ids([1, 2, 3], [1, 2, 3])
    mismatch = compare_token_ids([1, 2, 3], [1, 9, 3, 4])

    assert exact == {"exact": True, "reference_length": 3, "candidate_length": 3, "first_mismatch": None}
    assert mismatch == {
        "exact": False,
        "reference_length": 3,
        "candidate_length": 4,
        "first_mismatch": 1,
    }
