#!/usr/bin/env python3
"""Paired GPU pilot for a cheap coverage-aware summarization prefill policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch
from rouge_score import rouge_scorer


ROOT = Path(__file__).resolve().parents[1]
INFER_FILE = ROOT / "externals" / "Sematic_selection" / "infer.py"
spec = importlib.util.spec_from_file_location("semantic_infer_coverage_probe", INFER_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {INFER_FILE}")
semantic_infer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = semantic_infer
spec.loader.exec_module(semantic_infer)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n\s*\n+")
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*(?:\s*%)?\b")
INSTRUCTION = "Summarize the following document faithfully in about 120 words. Preserve important names, numbers, and conclusions."


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def select_lead(tokenizer, document: str, budget: int) -> str:
    ids = tokenizer.encode(document, add_special_tokens=False)
    if len(ids) <= budget:
        return document
    return tokenizer.decode(ids[:budget], skip_special_tokens=False)


def select_coverage(tokenizer, document: str, budget: int, bins: int = 8) -> str:
    """Spend half the budget on the lead and distribute the rest over the source."""
    units = [s.strip() for s in SENTENCE_SPLIT.split(document) if s.strip()]
    if not units or _token_count(tokenizer, document) <= budget:
        return document
    counts = [_token_count(tokenizer, unit) for unit in units]
    lead_cap = budget // 2
    chosen: set[int] = set()
    spent = 0
    for idx, count in enumerate(counts):
        if spent + count > lead_cap:
            break
        chosen.add(idx)
        spent += count
    if not chosen:
        # A long first sentence should not consume the entire coverage budget.
        first_ids = tokenizer.encode(units[0], add_special_tokens=False)
        units[0] = tokenizer.decode(first_ids[:lead_cap], skip_special_tokens=False)
        counts[0] = _token_count(tokenizer, units[0])
        chosen.add(0)
        spent = counts[0]

    remaining = [idx for idx in range(len(units)) if idx not in chosen]
    if not remaining:
        return "\n".join(units[idx] for idx in sorted(chosen))
    # Group by cumulative token position so every region receives a chance.
    region_total = sum(counts[idx] for idx in remaining)
    groups: list[list[int]] = [[] for _ in range(bins)]
    position = 0
    for idx in remaining:
        bucket = min(bins - 1, position * bins // max(1, region_total))
        groups[bucket].append(idx)
        position += counts[idx]
    # Take the earliest sentence from each region before returning for seconds.
    depth = 0
    while any(depth < len(group) for group in groups):
        for group in groups:
            if depth >= len(group):
                continue
            idx = group[depth]
            if spent + counts[idx] <= budget:
                chosen.add(idx)
                spent += counts[idx]
        depth += 1
    selected = "\n".join(units[idx] for idx in sorted(chosen))
    ids = tokenizer.encode(selected, add_special_tokens=False)
    if len(ids) > budget:
        selected = tokenizer.decode(ids[:budget], skip_special_tokens=False)
    return selected


def _reference(row: dict) -> str:
    value = row.get("reference") or row.get("answer") or row.get("answers") or ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _number_recall(reference: str, candidate: str) -> float | None:
    needed = set(NUMBER.findall(reference.lower()))
    if not needed:
        return None
    found = set(NUMBER.findall(candidate.lower()))
    return len(needed & found) / len(needed)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tokenizer, model = semantic_infer.load_target(
        args.model,
        device=device,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    methods = ("full", "lead", "coverage")
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    # Compile/warm the common generation path before timed requests.
    semantic_infer.generate_greedy_profiled(
        model, tokenizer, "A short document about public policy.",
        device=device, max_new_tokens=8, system_prompt="", instruction=INSTRUCTION,
        disable_thinking=True,
    )
    with output.open("w", encoding="utf-8") as handle:
        for dataset in datasets:
            path = ROOT / "data" / "longbench_100_14k" / f"{dataset}.jsonl"
            rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
            rows = rows[: args.max_samples]
            for index, row in enumerate(rows):
                document = str(row.get("context") or row.get("document") or "")
                reference = _reference(row)
                sample_id = str(row.get("source_index", row.get("id", index)))
                for method in methods[index % len(methods):] + methods[:index % len(methods)]:
                    start = time.perf_counter()
                    if method == "full":
                        selected = document
                        selector_ms = 0.0
                    elif method == "lead":
                        selected = select_lead(tokenizer, document, args.budget)
                        selector_ms = (time.perf_counter() - start) * 1000.0
                    else:
                        selected = select_coverage(tokenizer, document, args.budget)
                        selector_ms = (time.perf_counter() - start) * 1000.0
                    result = semantic_infer.generate_greedy_profiled(
                        model, tokenizer, selected,
                        device=device, max_new_tokens=args.max_new_tokens,
                        system_prompt="", instruction=INSTRUCTION, disable_thinking=True,
                    )
                    scores = scorer.score(reference, result.summary) if reference else {}
                    record = {
                        "dataset": dataset, "sample_id": sample_id, "method": method,
                        "budget": args.budget, "source_tokens": _token_count(tokenizer, document),
                        "selected_tokens": result.content_tokens,
                        "selector_ms": selector_ms,
                        "prefill_ms": result.prefill_ms, "decode_ms": result.decode_ms,
                        "target_e2e_ms": result.target_request_e2e_ms,
                        "pipeline_e2e_ms": selector_ms + result.target_request_e2e_ms,
                        "pipeline_ttft_ms": selector_ms + result.target_request_ttft_ms,
                        "output_tokens": result.output_tokens, "stop_reason": result.stop_reason,
                        "rouge1_f": scores["rouge1"].fmeasure if scores else None,
                        "rouge2_f": scores["rouge2"].fmeasure if scores else None,
                        "rougeL_f": scores["rougeL"].fmeasure if scores else None,
                        "number_recall": _number_recall(reference, result.summary),
                        "summary": result.summary, "reference": reference,
                        "peak_gpu_allocated_mb": result.peak_gpu_allocated_mb,
                    }
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(json.dumps({k: record[k] for k in (
                        "dataset", "sample_id", "method", "source_tokens",
                        "selected_tokens", "pipeline_e2e_ms", "output_tokens", "rougeL_f",
                    )}), flush=True)
    summary: dict[str, dict] = {}
    for dataset in datasets:
        dense = {r["sample_id"]: r for r in records if r["dataset"] == dataset and r["method"] == "full"}
        for method in methods:
            group = [r for r in records if r["dataset"] == dataset and r["method"] == method]
            if not group:
                continue
            key = f"{dataset}/{method}"
            total_e2e = sum(r["pipeline_e2e_ms"] for r in group)
            dense_e2e = sum(dense[r["sample_id"]]["pipeline_e2e_ms"] for r in group)
            number_values = [r["number_recall"] for r in group if r["number_recall"] is not None]
            summary[key] = {
                "n": len(group), "paired_speedup_ratio_of_means": dense_e2e / total_e2e,
                "mean_pipeline_e2e_ms": statistics.fmean(r["pipeline_e2e_ms"] for r in group),
                "mean_selector_ms": statistics.fmean(r["selector_ms"] for r in group),
                "mean_prefill_ms": statistics.fmean(r["prefill_ms"] for r in group),
                "mean_decode_ms": statistics.fmean(r["decode_ms"] for r in group),
                "mean_output_tokens": statistics.fmean(r["output_tokens"] for r in group),
                "mean_rougeL_f": statistics.fmean(r["rougeL_f"] for r in group if r["rougeL_f"] is not None),
                "mean_number_recall": statistics.fmean(number_values) if number_values else None,
            }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "summary": summary}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--datasets", default="gov_report,qmsum")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--budget", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
