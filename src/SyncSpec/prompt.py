"""Canonical prompt/tokenization helpers shared by trajectory and inference."""

from __future__ import annotations

import hashlib

import torch


SUMMARY_INSTRUCTION = "Summarize the following document.\n\n"


def format_record_text(raw: dict) -> str:
    if raw.get("document") is not None:
        return SUMMARY_INSTRUCTION + str(raw["document"])
    if raw.get("prompt") is not None:
        return str(raw["prompt"])
    return str(raw.get("text", raw.get("content", "")))


def encode_record(raw: dict, tokenizer=None, max_input_tokens: int = 0) -> torch.Tensor:
    if "source_ids" in raw:
        ids = torch.tensor(raw["source_ids"], dtype=torch.long)
    else:
        text = format_record_text(raw)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=False,
            )
            if isinstance(ids, (tuple, list)):
                ids = ids[0]
            ids = ids[0] if ids.ndim == 2 else ids
        elif tokenizer is not None:
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            ids = encoded["input_ids"][0]
        else:
            tokens = text.split() or ["empty"]
            values = [int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 254 + 1 for token in tokens]
            ids = torch.tensor(values, dtype=torch.long)
    if max_input_tokens > 0:
        ids = ids[:max_input_tokens]
    if ids.numel() == 0:
        raise ValueError("encoded record has no input tokens")
    return ids.to(torch.long)

