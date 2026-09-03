from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from infer_syncspec import _encode_record, _format_record_text  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        text = messages[0]["content"]
        return torch.tensor([[len(text), 9, 10, 11]])

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[len(text), 12, 13]])}


def test_document_records_use_shared_summary_instruction_and_chat_template() -> None:
    raw = {"id": "x", "document": "A short document."}
    text = _format_record_text(raw)
    assert text.startswith("Summarize the following document.")
    assert _encode_record(raw, FakeTokenizer()).tolist() == [len(text), 9, 10, 11]


def test_explicit_prompt_is_not_double_wrapped() -> None:
    raw = {"prompt": "Use this exact prompt."}
    assert _format_record_text(raw) == raw["prompt"]

