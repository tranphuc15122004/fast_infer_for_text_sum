from __future__ import annotations

from pathlib import Path

import pytest
import torch

try:
    from Finetuning.data import (
        SummaryRecord,
        build_summary_loss_mask,
        load_summary_jsonl,
        render_summary_example,
    )
except ModuleNotFoundError as exc:  # Red phase: the adapter is not ported yet.
    _DATA_IMPORT_ERROR = exc


def _require_data_api() -> None:
    if "_DATA_IMPORT_ERROR" in globals():
        pytest.fail(f"summary data API is not implemented: {_DATA_IMPORT_ERROR}")


class FakeQwenTokenizer:
    """Small local tokenizer with Qwen-style user/assistant delimiters."""

    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    def _encode_text(self, text: str) -> list[int]:
        ids = []
        for token in text.split():
            if token not in self._vocabulary:
                self._vocabulary[token] = 200 + len(self._vocabulary)
            ids.append(self._vocabulary[token])
        return ids

    def __call__(self, text: str, *, add_special_tokens: bool = False, **_kwargs):
        del add_special_tokens
        return {"input_ids": self._encode_text(text)}

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        **_kwargs,
    ):
        assert tokenize is True
        assert return_dict is False
        ids = [100]
        for message in conversation:
            role = message["role"]
            if role == "user":
                ids.extend([1, *self._encode_text(message["content"]), 98])
            elif role == "assistant":
                ids.extend([2, 3, *self._encode_text(message["content"])])
            else:
                raise AssertionError(f"unexpected role: {role}")
        if add_generation_prompt:
            ids.extend([2, 3])
        elif conversation and conversation[-1]["role"] == "assistant":
            ids.append(self.eos_token_id)
        return ids


def test_load_summary_jsonl_preserves_unicode_and_metadata() -> None:
    _require_data_api()
    fixture = Path(__file__).parent / "fixtures" / "synthetic_summary.jsonl"

    records = load_summary_jsonl(fixture)

    assert len(records) >= 2
    assert all(isinstance(record, SummaryRecord) for record in records)
    assert any("Việt" in record.document or "Việt" in record.summary for record in records)
    assert records[0].metadata["split"] == "synthetic"


def test_summary_loss_mask_only_supervises_assistant_span() -> None:
    _require_data_api()

    mask = build_summary_loss_mask(
        torch.arange(8), assistant_start=5, assistant_end=8
    )

    assert mask.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]


def test_render_summary_example_is_qwen_compatible_and_causally_truncated() -> None:
    _require_data_api()
    tokenizer = FakeQwenTokenizer()
    record = SummaryRecord(
        id="unicode",
        document="Tài liệu rất dài",
        summary="Tóm tắt ngắn gọn",
    )

    example = render_summary_example(record, tokenizer, max_length=12)

    assert set(("input_ids", "loss_mask")).issubset(example)
    assert example["input_ids"].ndim == 1
    assert example["input_ids"].shape == example["loss_mask"].shape
    assert int(example["loss_mask"].sum()) >= 2
    assert example["loss_mask"][-1].item() == 0
    assert example["input_ids"].dtype == torch.long


def test_render_summary_example_rejects_short_supervision_after_truncation() -> None:
    _require_data_api()
    record = SummaryRecord(id="short", document="Tài liệu", summary="Một")

    with pytest.raises(ValueError, match="two consecutive supervised tokens"):
        render_summary_example(record, FakeQwenTokenizer(), max_length=32)
