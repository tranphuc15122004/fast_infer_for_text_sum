"""Local summarization records and Qwen3-compatible prompt preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class SummaryRecord:
    """The minimal local document/summary JSONL record.

    Additional JSON fields are retained in ``metadata`` so provenance can pass
    through preparation without becoming part of the model input contract.
    """

    id: str
    document: str
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("id", "document", "summary"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"SummaryRecord.{name} must be a string")


def load_summary_jsonl(
    path: str | Path,
    max_samples: int | None = None,
) -> list[SummaryRecord]:
    """Load summary records from a local UTF-8 JSONL file.

    No dataset or tokenizer lookup happens here.  Blank lines are ignored and
    malformed records fail with their line number rather than being silently
    dropped.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"summary JSONL file not found: {source}")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")

    records: list[SummaryRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_samples is not None and len(records) >= max_samples:
                break
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid summary JSONL at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"summary JSONL line {line_number} must be a JSON object"
                )
            missing = [
                key for key in ("id", "document", "summary") if key not in payload
            ]
            if missing:
                raise ValueError(
                    f"summary JSONL line {line_number} missing fields {missing}"
                )
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"id", "document", "summary"}
            }
            try:
                records.append(
                    SummaryRecord(
                        id=payload["id"],
                        document=payload["document"],
                        summary=payload["summary"],
                        metadata=metadata,
                    )
                )
            except TypeError as exc:
                raise ValueError(
                    f"summary JSONL line {line_number} has invalid field types"
                ) from exc
    return records


def _as_token_id_list(value: Any) -> list[int]:
    """Normalize list/tensor/BatchEncoding tokenizer outputs to one id row."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("tokenizer output does not contain input_ids")
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise ValueError("tokenizer output must contain one sequence")
            value = value[0]
        if value.ndim != 1:
            raise ValueError("tokenizer input_ids must be one-dimensional")
        return [int(item) for item in value.detach().cpu().tolist()]
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            if len(value) != 1:
                raise ValueError("tokenizer output must contain one sequence")
            value = value[0]
        return [int(item) for item in value]
    raise TypeError(f"unsupported tokenizer output type: {type(value)!r}")


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return _as_token_id_list(encoded)


def _template_kwargs(tokenizer: Any, chat_template: str | None) -> dict[str, Any]:
    """Select a named local template only when the tokenizer exposes a map."""

    templates = getattr(tokenizer, "chat_template", None)
    if (
        chat_template
        and isinstance(templates, Mapping)
        and chat_template in templates
    ):
        return {"chat_template": templates[chat_template]}
    return {}


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    chat_template: str | None,
) -> list[int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("supplied tokenizer does not provide apply_chat_template")
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        # Transformers 5 may return BatchEncoding by default.  The explicit
        # flag keeps the result usable by both real and fake tokenizers.
        "return_dict": False,
        **_template_kwargs(tokenizer, chat_template),
    }
    while True:
        try:
            encoded = tokenizer.apply_chat_template(messages, **kwargs)
            break
        except TypeError as exc:
            # Minimal injected tokenizers may not expose optional HF kwargs.
            # This fallback still never downloads or constructs a template.
            message = str(exc)
            removable = next(
                (
                    name
                    for name in ("return_dict", "chat_template")
                    if name in kwargs and name in message
                ),
                None,
            )
            if removable is None:
                raise
            kwargs.pop(removable)
    return _as_token_id_list(encoded)


def build_summary_loss_mask(
    input_ids: torch.Tensor,
    assistant_start: int,
    assistant_end: int,
) -> torch.Tensor:
    """Return a mask that supervises exactly ``[assistant_start, end)``."""

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        raise ValueError("input_ids must be a one-dimensional tensor")
    length = int(input_ids.shape[0])
    if not 0 <= assistant_start <= assistant_end <= length:
        raise ValueError(
            "assistant span must satisfy 0 <= start <= end <= sequence length"
        )
    mask = torch.zeros(length, dtype=torch.float32, device=input_ids.device)
    mask[assistant_start:assistant_end] = 1.0
    return mask


def _find_subsequence(sequence: list[int], subsequence: list[int], start: int) -> int | None:
    if not subsequence:
        return None
    last_start = len(sequence) - len(subsequence)
    for index in range(max(0, start), last_start + 1):
        if sequence[index : index + len(subsequence)] == subsequence:
            return index
    return None


def _tokenizer_control_ids(tokenizer: Any) -> set[int]:
    control_ids = {
        int(value)
        for value in (getattr(tokenizer, "all_special_ids", ()) or ())
    }
    for name in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "im_end_id",
        "end_of_turn_token_id",
    ):
        value = getattr(tokenizer, name, None)
        if value is not None:
            control_ids.add(int(value))
    return control_ids


def render_summary_example(
    record: SummaryRecord,
    tokenizer: Any,
    max_length: int,
    chat_template: str = "qwen3",
) -> dict[str, torch.Tensor]:
    """Render one record with the supplied local tokenizer.

    The assistant content is located by prefix-diff plus content-token
    alignment.  This works with Qwen3's normal chat template and with small
    injected tokenizers, without looking up a remote template.  The last
    retained position is never supervised because a causal next-token label is
    unavailable there.
    """

    if not isinstance(record, SummaryRecord):
        raise TypeError("record must be a SummaryRecord")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer")

    messages = [
        {"role": "user", "content": record.document},
        {"role": "assistant", "content": record.summary},
    ]
    full_ids = _apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        chat_template=chat_template,
    )
    prefix_ids = _apply_chat_template(
        tokenizer,
        messages[:1],
        add_generation_prompt=True,
        chat_template=chat_template,
    )
    summary_ids = _tokenize_text(tokenizer, record.summary)
    assistant_start = _find_subsequence(full_ids, summary_ids, len(prefix_ids))
    used_fallback = assistant_start is None
    if assistant_start is None:
        # A standalone content tokenization can differ from the tokenization
        # inside a chat-template role.  In that case, the generation prefix is
        # still a safe lower bound, but the suffix must be filtered for known
        # control tokens rather than supervising the complete assistant turn.
        assistant_start = min(len(prefix_ids), len(full_ids))
        control_ids = _tokenizer_control_ids(tokenizer)
        if assistant_start >= len(full_ids) or not control_ids:
            raise ValueError(
                "cannot locate a safe assistant content span in chat template"
            )
        assistant_end = len(full_ids)
        while assistant_end > assistant_start and full_ids[assistant_end - 1] in control_ids:
            assistant_end -= 1
    else:
        assistant_end = assistant_start + len(summary_ids)

    input_ids = torch.tensor(full_ids[:max_length], dtype=torch.long)
    clipped_end = min(assistant_end, input_ids.shape[0])
    clipped_start = min(assistant_start, input_ids.shape[0])
    loss_mask = build_summary_loss_mask(input_ids, clipped_start, clipped_end)
    if used_fallback:
        control_ids = _tokenizer_control_ids(tokenizer)
        loss_mask = loss_mask * torch.tensor(
            [
                0.0 if int(token_id) in control_ids else 1.0
                for token_id in input_ids.tolist()
            ],
            dtype=loss_mask.dtype,
            device=loss_mask.device,
        )
    if loss_mask.numel():
        # Causal LM loss shifts labels one position to the right.  Keep the
        # public mask helper a pure span builder while making rendered samples
        # safe when truncation ends inside the assistant text.
        loss_mask[-1] = 0.0
    if not any(
        bool(current) and bool(following)
        for current, following in zip(loss_mask.tolist(), loss_mask.tolist()[1:])
    ):
        raise ValueError(
            "summary example requires two consecutive supervised tokens "
            "after truncation"
        )
    return {"input_ids": input_ids, "loss_mask": loss_mask}


__all__ = [
    "SummaryRecord",
    "build_summary_loss_mask",
    "load_summary_jsonl",
    "render_summary_example",
]
