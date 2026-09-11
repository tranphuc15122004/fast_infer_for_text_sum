"""Tests cho batching phase regenerate, không cần load model thật."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_left_pad_keeps_each_prompt_and_attention_mask() -> None:
    from MR_DFlash.generation_batching import left_pad_prompt_ids

    input_ids, attention_mask = left_pad_prompt_ids(
        [
            torch.tensor([11, 12, 13]),
            torch.tensor([21, 22]),
        ],
        pad_token_id=0,
    )

    assert input_ids.tolist() == [[11, 12, 13], [0, 21, 22]]
    assert attention_mask.tolist() == [[1, 1, 1], [0, 1, 1]]


def test_generation_schedule_selects_batch_by_prompt_bucket() -> None:
    from MR_DFlash.generation_batching import GenerationBatchSchedule, select_generation_group

    schedule = GenerationBatchSchedule.from_payload(
        {
            "schema_version": "mr_dflash_generation_batch_profile_v1",
            "target_model_path": "target",
            "max_length": 32768,
            "requested_max_new_tokens": 2048,
            "buckets": [
                {"min_length": 1, "max_length": 8192, "selected_batch_size": 8},
                {"min_length": 8193, "max_length": 32768, "selected_batch_size": 2},
            ],
        }
    )
    items = [
        {"sample_id": "long", "prompt_tokens": 9000, "generation_budget": 2048},
        {"sample_id": "short-1", "prompt_tokens": 100, "generation_budget": 2048},
        {"sample_id": "short-2", "prompt_tokens": 200, "generation_budget": 2048},
        {"sample_id": "short-3", "prompt_tokens": 300, "generation_budget": 2048},
    ]

    group = select_generation_group(items, max_batch_size=8, schedule=schedule)

    assert [item["sample_id"] for item in group] == ["short-1", "short-2", "short-3"]


def test_generation_schedule_rejects_incompatible_metadata() -> None:
    from MR_DFlash.generation_batching import GenerationBatchSchedule

    schedule = GenerationBatchSchedule.from_payload(
        {
            "schema_version": "mr_dflash_generation_batch_profile_v1",
            "target_model_path": "target",
            "max_length": 32768,
            "requested_max_new_tokens": 2048,
            "buckets": [
                {"min_length": 1, "max_length": 32768, "selected_batch_size": 1}
            ],
        }
    )

    assert schedule.batch_size_for_length(32768) == 1


def test_generate_prepared_batch_calls_model_once_for_multiple_samples() -> None:
    from regenerate_pilot import _generate_prepared_batch
    from progress import ProgressReporter

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def decode(self, ids, skip_special_tokens=True):
            return "answer-" + str(ids[0])

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, input_ids, attention_mask, **kwargs):
            self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone(), kwargs))
            suffix = torch.tensor([[7], [8]], dtype=torch.long, device=input_ids.device)
            return torch.cat([input_ids, suffix], dim=1)

    model = FakeModel()
    items = [
        {
            "sample_id": "s1",
            "prompt_ids": torch.tensor([1, 2, 3]),
            "prompt_tokens": 3,
            "generation_budget": 1,
        },
        {
            "sample_id": "s2",
            "prompt_ids": torch.tensor([4, 5]),
            "prompt_tokens": 2,
            "generation_budget": 1,
        },
    ]

    responses = _generate_prepared_batch(
        model,
        FakeTokenizer(),
        items,
        reporter=ProgressReporter(None),
        device=torch.device("cpu"),
        args=SimpleNamespace(temperature=0.0, progress_path=None),
    )

    assert responses == ["answer-7", "answer-8"]
    assert len(model.calls) == 1
    input_ids, attention_mask, kwargs = model.calls[0]
    assert input_ids.tolist() == [[1, 2, 3], [0, 4, 5]]
    assert attention_mask.tolist() == [[1, 1, 1], [0, 1, 1]]
    assert kwargs["max_new_tokens"] == 1
    assert kwargs["use_cache"] is True
