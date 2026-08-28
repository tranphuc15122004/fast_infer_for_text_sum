"""Small offline fallback for the optional FastChat dependency.

FAFO only needs FastChat's model adapters for loading a Hugging Face model and
its question loader for MT-Bench.  The Llama 3.1 LongBench adapter does not
need FastChat's conversation registry, but importing FAFO used to require the
whole package unconditionally.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class _HFAdapter:
    def load_model(self, model_path: str, kwargs: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = dict(kwargs)
        if "torch_dtype" not in model_kwargs:
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer


Llama2Adapter = _HFAdapter
Llama3Adapter = _HFAdapter
QwenChatAdapter = _HFAdapter


def get_conversation_template(model_id: str):
    """Return a minimal FastChat-compatible conversation object."""

    class Conversation:
        name = model_id
        roles = ("user", "assistant")
        stop_token_ids: list[int] = []
        stop_str = None

        def __init__(self):
            self.messages: list[list[Any]] = []

        def append_message(self, role: str, message: str | None):
            self.messages.append([role, message])

        def get_prompt(self) -> str:
            return "\n".join(
                f"{role}: {message or ''}" for role, message in self.messages
            )

    return Conversation()


def load_questions(path: str, begin: int | None = None, end: int | None = None):
    rows = []
    source = Path(path)
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows[begin:end]


__all__ = [
    "Llama2Adapter",
    "Llama3Adapter",
    "QwenChatAdapter",
    "get_conversation_template",
    "load_questions",
]
