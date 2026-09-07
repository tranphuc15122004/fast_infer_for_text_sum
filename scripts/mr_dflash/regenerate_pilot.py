"""Regenerate assistant trajectories bằng frozen HF target.

Có hai input mode:
* ``--target-model-path``: chạy Qwen3/Llama trực tiếp offline;
* ``--responses-jsonl``: gắn response đã sinh bởi server bên ngoài, key theo id
  (hữu ích trên server không có internet nhưng có model service).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from _common import read_jsonl, write_json, write_jsonl


def _apply_chat(tokenizer: Any, messages: List[Dict[str, str]], *, generation: bool, enable_thinking: bool = False):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": generation,
        "return_tensors": "pt",
        "return_dict": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _as_ids(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "input_ids"):
        value = value.input_ids
    return torch.as_tensor(value, dtype=torch.long)


def _truncate_prompt(tokenizer: Any, messages: List[Dict[str, str]], budget: int) -> List[Dict[str, str]]:
    """Giữ phần đầu document khi prompt vượt budget dành cho target response."""
    if int(_as_ids(_apply_chat(tokenizer, messages, generation=True)).shape[-1]) <= budget:
        return messages
    index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), None)
    if index is None:
        return messages
    content = str(messages[index].get("content", ""))
    content_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
    lo, hi = 0, len(content_ids)
    best = content_ids[:0]
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = list(messages)
        candidate[index] = {**messages[index], "content": tokenizer.decode(content_ids[:mid], skip_special_tokens=True)}
        length = int(_as_ids(_apply_chat(tokenizer, candidate, generation=True)).shape[-1])
        if length <= budget:
            best = content_ids[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    candidate = list(messages)
    candidate[index] = {**messages[index], "content": tokenizer.decode(best, skip_special_tokens=True)}
    return candidate


def _load_responses(path: str) -> Dict[str, str]:
    responses: Dict[str, str] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("id", ""))
        text = row.get("assistant", row.get("response", row.get("text", "")))
        if sample_id and text:
            responses[sample_id] = str(text)
    return responses


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate target trajectories")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--target-model-path", default=None)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--responses-jsonl", default=None)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.target_model_path) == bool(args.responses_jsonl):
        raise ValueError("chọn đúng một trong --target-model-path hoặc --responses-jsonl")
    torch.manual_seed(args.seed)
    responses = _load_responses(args.responses_jsonl) if args.responses_jsonl else {}
    tokenizer = model = None
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.target_model_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = {
            "cache_dir": args.cache_dir,
            "local_files_only": args.local_files_only,
        }
        if args.target_revision:
            load_kwargs["revision"] = args.target_revision
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, **load_kwargs)
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.torch_dtype]
        model = AutoModelForCausalLM.from_pretrained(
            args.target_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **load_kwargs,
        ).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = {str(row.get("id")) for row in read_jsonl(output)} if output.exists() else set()
    generated: List[Dict[str, Any]] = []
    stats = {"written": 0, "skipped_existing": 0, "skipped_invalid": 0}
    for index, row in enumerate(read_jsonl(args.input)):
        if args.limit is not None and stats["written"] >= args.limit:
            break
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in existing:
            if sample_id in existing:
                stats["skipped_existing"] += 1
            continue
        messages = list(row.get("conversations") or [])
        if not messages:
            stats["skipped_invalid"] += 1
            continue
        if any(str(message.get("role", "")).lower() == "assistant" for message in messages if isinstance(message, dict)):
            raise ValueError(
                f"sample {sample_id!r} đã chứa assistant response; "
                "regenerate chỉ nhận prompt-only input"
            )
        prompt_messages = messages
        assistant = responses.get(sample_id)
        prompt_ids_len = None
        if tokenizer is not None:
            budget = max(1, args.max_length - args.max_new_tokens)
            prompt_messages = _truncate_prompt(tokenizer, messages, budget)
            prompt_ids = _as_ids(_apply_chat(tokenizer, prompt_messages, generation=True)).to(device)
            if prompt_ids.ndim == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            prompt_ids_len = int(prompt_ids.shape[-1])
            if assistant is None:
                with torch.inference_mode():
                    kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
                    if args.temperature > 0:
                        kwargs.update({"do_sample": True, "temperature": args.temperature})
                    generated_ids = model.generate(prompt_ids, **kwargs)
                assistant = tokenizer.decode(generated_ids[0, prompt_ids_len:], skip_special_tokens=True).strip()
        if not assistant:
            stats["skipped_invalid"] += 1
            continue
        final_messages = prompt_messages + [{"role": "assistant", "content": assistant}]
        final_row = {
            **row,
            "conversations": final_messages,
            "metadata": {
                **(row.get("metadata") or {}),
                "generation_model": args.target_model_path or "external_response_server",
                "generation_temperature": args.temperature,
                "enable_thinking": bool(args.enable_thinking),
                "max_new_tokens": args.max_new_tokens,
                "prompt_tokens": prompt_ids_len,
            },
        }
        generated.append(final_row)
        existing.add(sample_id)
        stats["written"] += 1
        if len(generated) >= 32:
            with output.open("a", encoding="utf-8") as handle:
                for item in generated:
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            generated.clear()
            print(f"[regenerate_pilot] written={stats['written']}")
    if generated:
        with output.open("a", encoding="utf-8") as handle:
            for item in generated:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = Path(args.manifest) if args.manifest else output.with_name(output.stem + "_manifest.json")
    write_json(
        manifest_path,
        {
            "schema_version": "mr_dflash_regeneration_v1",
            "target_model": args.target_model_path or "external_response_server",
            "target_revision": args.target_revision,
            "temperature": args.temperature,
            "enable_thinking": bool(args.enable_thinking),
            "max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "seed": args.seed,
            "stats": stats,
        },
    )
    print(f"[regenerate_pilot] {stats}")


if __name__ == "__main__":
    main()
