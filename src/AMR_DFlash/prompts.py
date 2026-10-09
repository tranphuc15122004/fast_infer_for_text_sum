"""Shared prompt rendering for regenerated manifests and AMR capture."""

from __future__ import annotations

from typing import Any

from .artifacts import canonical_hash, sha256_text

LONG_BENCH_DATASETS = {"gov_report", "qmsum", "multi_news", "lcc", "repobench-p"}
PROMPT_POLICY = "chat_template_no_thinking_or_longbench_messages_v2"


def validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    result = []
    aliases = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant"}
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role") or message.get("from") or "").lower()
        role = aliases.get(role, role)
        content = message.get("content", message.get("value"))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a nonempty string")
        result.append({"role": role, "content": content})
    return result


def render_prompt(tokenizer: Any, sample: dict[str, Any]) -> str:
    raw = sample.get("raw") or {}
    prompt = str(sample.get("prompt") or "")
    if "messages" in raw:
        messages = validate_messages(raw["messages"])
        if messages[-1]["role"] != "user":
            raise ValueError(f"sample {sample.get('id')}: prompt messages must end with user")
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("structured AMR messages require the target tokenizer chat_template")
    else:
        if raw.get("dataset") in LONG_BENCH_DATASETS or not getattr(tokenizer, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def source_content_hash(raw: dict[str, Any], prompt: str) -> str:
    for key in ("context", "document", "text"):
        if raw.get(key):
            return sha256_text(" ".join(str(raw[key]).split()))
    if "messages" in raw:
        messages = validate_messages(raw["messages"])
        return canonical_hash([{**message, "content": " ".join(message["content"].split())}
                               for message in messages])
    return sha256_text(" ".join(prompt.split())) if prompt.strip() else ""
