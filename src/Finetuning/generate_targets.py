"""Generate deterministic target trajectories for DFlash summarization training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from .data import DEFAULT_SUMMARY_PROMPT_TEMPLATE, iter_summary_jsonl, render_summary_prompt


def _generated_ids(output: Any, prompt_length: int) -> list[int]:
    if isinstance(output, torch.Tensor):
        value = output
    elif hasattr(output, "sequences"):
        value = output.sequences
    else:
        raise TypeError("target.generate must return token sequences")
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError("target.generate must return exactly one sequence")
    return [int(token) for token in value[0, prompt_length:].detach().cpu().tolist()]


def _trim_at_eos(token_ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return token_ids
    try:
        return token_ids[: token_ids.index(int(eos_token_id))]
    except ValueError:
        return token_ids


def generate_teacher_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tokenizer: Any,
    target: Any,
    target_model_path: str,
    max_length: int,
    max_source_tokens: int,
    max_summary_tokens: int,
    chat_template: str,
    prompt_template: str = DEFAULT_SUMMARY_PROMPT_TEMPLATE,
    device: str | torch.device,
) -> dict[str, int]:
    """Write target-generated summaries while retaining human references."""

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"teacher trajectory output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    written = 0
    rejected = 0
    device_obj = torch.device(device)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in iter_summary_jsonl(input_path):
                try:
                    prompt = render_summary_prompt(
                        record,
                        tokenizer,
                        max_length,
                        max_source_tokens=max_source_tokens,
                        max_summary_tokens=max_summary_tokens,
                        chat_template=chat_template,
                        prompt_template=prompt_template,
                    )
                    input_ids = prompt.unsqueeze(0).to(device_obj)
                    with torch.inference_mode():
                        output = target.generate(
                            input_ids,
                            do_sample=False,
                            max_new_tokens=max_summary_tokens,
                            pad_token_id=getattr(tokenizer, "pad_token_id", None),
                            eos_token_id=getattr(tokenizer, "eos_token_id", None),
                        )
                    token_ids = _trim_at_eos(
                        _generated_ids(output, input_ids.shape[1]),
                        getattr(tokenizer, "eos_token_id", None),
                    )
                    summary = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    if len(token_ids) < 2 or not summary:
                        raise ValueError("target generated fewer than two usable summary tokens")
                except (TypeError, ValueError) as exc:
                    rejected += 1
                    continue
                payload = {
                    "id": record.id,
                    "document": record.document,
                    "summary": summary,
                    "reference_summary": record.summary,
                    **dict(record.metadata),
                    "teacher": {
                        "target_model_path": str(target_model_path),
                        "chat_template": chat_template,
                        "prompt_template": prompt_template,
                        "do_sample": False,
                        "max_length": max_length,
                        "max_source_tokens": max_source_tokens,
                        "max_summary_tokens": max_summary_tokens,
                    },
                }
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
        if written == 0:
            raise ValueError("target generation produced no usable summaries")
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return {"written": written, "rejected": rejected}


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen teacher trajectories for DFlash")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-source-tokens", type=int, required=True)
    parser.add_argument("--max-summary-tokens", type=int, required=True)
    parser.add_argument("--chat-template", default="qwen3")
    parser.add_argument("--prompt-template", default=DEFAULT_SUMMARY_PROMPT_TEMPLATE)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    if args.max_source_tokens < 0 or args.max_summary_tokens < 1:
        raise ValueError("source budget must be non-negative and summary budget positive")
    if args.max_source_tokens + args.max_summary_tokens > args.max_length:
        raise ValueError("source and summary token budgets exceed --max-length")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    stats = generate_teacher_jsonl(
        args.input,
        args.output,
        tokenizer=tokenizer,
        target=target,
        target_model_path=args.target_model_path,
        max_length=args.max_length,
        max_source_tokens=args.max_source_tokens,
        max_summary_tokens=args.max_summary_tokens,
        chat_template=args.chat_template,
        prompt_template=args.prompt_template,
        device=device,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["generate_teacher_jsonl", "main"]
