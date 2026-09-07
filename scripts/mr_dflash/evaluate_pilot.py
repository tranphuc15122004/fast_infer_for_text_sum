"""Đánh giá reference speculative decoding trên holdout pilot.

Script dùng cùng target cho vanilla greedy và verifier. DFlash/MR output phải
exact với vanilla trước khi các số acceptance/latency được dùng trong báo cáo.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

from _common import read_jsonl


def _prompt_ids(tokenizer: Any, messages: List[Dict[str, str]]) -> torch.Tensor:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": False,
    }
    try:
        value = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        value = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(value, dict):
        value = value["input_ids"]
    return torch.as_tensor(value, dtype=torch.long).reshape(1, -1)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _length_bucket(length: int) -> str:
    if length <= 2048:
        return "0-2k"
    if length <= 4096:
        return "2-4k"
    if length <= 6144:
        return "4-6k"
    return "6-8k+"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate DFlash/MR-DFlash pilot checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="canonical regenerated holdout JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-model-path", default=None)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--exactness-check", action="store_true")
    args = parser.parse_args(argv)
    from MR_DFlash.checkpoint import warm_start_draft_model
    from MR_DFlash.inference import DFlashInferenceEngine, MRDFlashInferenceEngine
    from MR_DFlash.run_train import build_online_model, load_run_config, load_target_parts

    cfg = load_run_config(args.config)
    target_path = args.target_model_path or cfg.model.target_model_path
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    loaded = load_target_parts(
        target_path,
        cache_dir=args.cache_dir,
        torch_dtype=cfg.model.torch_dtype,
        device=device,
        local_files_only=args.local_files_only,
        keep_target_model=True,
    )
    tokenizer, target_config, embed, lm_head, target = loaded
    model = build_online_model(
        cfg,
        tokenizer=tokenizer,
        target_config=target_config,
        embed_tokens=embed,
        lm_head=lm_head,
        device=device,
    )
    warm_start_draft_model(
        model.draft_model,
        args.checkpoint,
        strategy_name=cfg.training.strategy,
    )
    engine_cls = MRDFlashInferenceEngine if cfg.model.architecture == "mr_dflash" else DFlashInferenceEngine
    engine = engine_cls(
        target,
        model.draft_model,
        mask_token_id=int(model.draft_model.spec.mask_token_id),
        device=device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    aggregate: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for index, row in enumerate(read_jsonl(args.input)):
        if args.max_samples is not None and index >= args.max_samples:
            break
        conversations = list(row.get("conversations") or [])
        if conversations and str(conversations[-1].get("role", "")).lower() == "assistant":
            conversations = conversations[:-1]
        if not conversations:
            continue
        prompt = _prompt_ids(tokenizer, conversations)
        start = time.perf_counter()
        result = engine.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        elapsed = time.perf_counter() - start
        generated_tokens = int(result.input_ids.shape[1] - prompt.shape[1])
        rounds = max(1, int(result.rounds))
        record: Dict[str, Any] = {
            "id": str(row.get("id", index)),
            "source": row.get("source"),
            "context_len": int(prompt.shape[1]),
            "context_bucket": _length_bucket(int(prompt.shape[1])),
            "generated_tokens": generated_tokens,
            "num_verify_rounds": int(result.rounds),
            "mean_accept_len": float(result.accepted_proposal_tokens / rounds),
            "accepted_proposal_tokens": int(result.accepted_proposal_tokens),
            "draft_ms": float(result.timings_s.get("draft_s", 0.0) * 1000),
            "verify_ms": float(result.timings_s.get("verify_s", 0.0) * 1000),
            "prefill_ms": float(result.timings_s.get("prefill_s", 0.0) * 1000),
            "total_ms": float(elapsed * 1000),
            "committed_tokens_per_s": float(generated_tokens / max(elapsed, 1e-9)),
        }
        if args.exactness_check:
            with torch.inference_mode():
                vanilla = target.generate(prompt.to(device), max_new_tokens=args.max_new_tokens, do_sample=False)
            record["exact_to_vanilla"] = bool(torch.equal(vanilla.cpu(), result.input_ids.cpu()))
            if not record["exact_to_vanilla"]:
                raise RuntimeError(f"exactness failed tại sample {record['id']}")
        rows.append(record)
        bucket = str(record["context_bucket"])
        for key in ("context_len", "generated_tokens", "num_verify_rounds", "mean_accept_len", "draft_ms", "verify_ms", "prefill_ms", "total_ms", "committed_tokens_per_s"):
            aggregate[bucket][key] += float(record[key])
    with output.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "type": "summary",
            "architecture": cfg.model.architecture,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "samples": len(rows),
            "by_context_bucket": {
                bucket: {key: value / len([r for r in rows if r["context_bucket"] == bucket]) for key, value in values.items()}
                for bucket, values in aggregate.items()
            },
        }
        handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[evaluate_pilot] samples={len(rows)} output={output}")


if __name__ == "__main__":
    main()
