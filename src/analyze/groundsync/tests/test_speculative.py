from __future__ import annotations

import pytest
import torch

from src.analyze.groundsync.trace_speculative import (
    acceptance_record,
    build_parser,
    generate_draft_proposal,
    run_one_speculative_trace,
    select_start_positions,
)
from src.analyze.groundsync.tests.test_target import TinyTokenizer
from src.analyze.groundsync.trace_target import render_document_prompt


def test_acceptance_record_identifies_first_relative_rejection() -> None:
    result = acceptance_record([10, 20, 99], [10, 20, 30])
    assert result["accepted_len"] == 2
    assert result["first_reject_rel"] == 3
    assert result["fully_accepted"] is False


def test_acceptance_record_marks_full_acceptance_without_fake_rejection() -> None:
    result = acceptance_record([10, 20], [10, 20, 30])
    assert result == {
        "accepted_len": 2,
        "first_reject_rel": None,
        "fully_accepted": True,
    }


def test_select_start_positions_is_deterministic_and_bounded() -> None:
    assert select_start_positions(10, max_starts=3, stride=2) == [0, 2, 4]
    assert select_start_positions(3, max_starts=10, stride=2) == [0, 2]
    assert select_start_positions(10, max_starts=2, stride=2, start_offset=1) == [1, 3]
    assert select_start_positions(
        10, max_starts=10, stride=2, start_offset=1, max_new_tokens=4
    ) == [1, 3, 5]
    assert select_start_positions(
        3, max_starts=10, stride=1, max_new_tokens=4
    ) == []
    with pytest.raises(ValueError, match="stride"):
        select_start_positions(10, max_starts=2, stride=0)
    with pytest.raises(ValueError, match="max_starts"):
        select_start_positions(10, max_starts=0, stride=1)


class _SpecOutput:
    def __init__(self, input_ids):
        self.logits = torch.nn.functional.one_hot(
            torch.zeros((1, input_ids.shape[1]), dtype=torch.long), num_classes=4
        ).float() * -10.0
        self.logits[:, -1, 1] = 10.0
        self.past_key_values = object()


class _SpecModel:
    def __init__(self):
        self.seen_inputs = []

    def __call__(self, *, input_ids, **kwargs):
        self.seen_inputs.append(input_ids.detach().clone())
        return _SpecOutput(input_ids)


def test_controlled_trace_records_timing_by_block_length_when_verifier_exists() -> None:
    tokenizer = TinyTokenizer()
    rendered = render_document_prompt(tokenizer, "ab")
    target_row = {
        "generated_token_ids": [1, 1, 1],
        "attention": [
            {"nosink": [1.0, 0.0]},
            {"nosink": [1.0, 0.0]},
            {"nosink": [0.0, 1.0]},
        ],
    }
    verifier = _SpecModel()
    rows = run_one_speculative_trace(
        _SpecModel(),
        tokenizer,
        rendered,
        target_row,
        document_id="d1",
        max_k=2,
        max_starts=1,
        stride=1,
        device=torch.device("cpu"),
        verification_model=verifier,
        verification_rendered=rendered,
        verification_device=torch.device("cpu"),
    )
    assert len(rows) == 1
    assert rows[0]["fully_accepted"] is True
    assert len(rows[0]["draft_time_by_k_ms"]) == 2
    assert len(rows[0]["verification_time_by_k_ms"]) == 2
    assert rows[0]["autoregressive_time_ms"] is not None
    assert rows[0]["timing_basis"] == "measured_cached_target_check"
    # The target verifier sees a speculative block of length k in one forward,
    # rather than k sequential one-token forwards.
    assert [call.shape[1] for call in verifier.seen_inputs[1:3]] == [1, 2]


def test_draft_prefill_accepts_long_context_chunking() -> None:
    model = _SpecModel()
    generate_draft_proposal(
        model,
        torch.ones((1, 5), dtype=torch.long),
        max_new_tokens=1,
        device=torch.device("cpu"),
        prefill_chunk_size=2,
    )
    assert [call.shape[1] for call in model.seen_inputs[:3]] == [2, 2, 1]
    assert model.seen_inputs[3].shape[1] == 1


def test_standalone_speculative_parser_accepts_verification_model() -> None:
    args = build_parser().parse_args([
        "--input", "input.jsonl",
        "--target-traces", "target.jsonl",
        "--output", "output.jsonl",
        "--verification-model", "Qwen/Qwen3-4B",
    ])
    assert args.verification_model == "Qwen/Qwen3-4B"
