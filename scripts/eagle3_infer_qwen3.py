#!/usr/bin/env python3
"""Batch EAGLE3 inference with decoding statistics.

Runs EAGLE3 speculative decoding over a range of questions (and optionally
multiple choices per question) and compares it against the naive
autoregressive baseline. For every generation it records:

  * new_tokens       - accepted output tokens
  * tree_steps       - number of draft-verify iterations executed
  * acceptance_lengths - committed tokens for each draft-verify iteration
  * accept_length    - mean of acceptance_lengths for this generation
  * eagle_time       - GPU-synchronized decode-only wall time [s]
  * eagle_tok_s      - EAGLE3 throughput [tokens/s]
  * naive_time       - GPU-synchronized decode-only baseline time [s]
  * naive_tok_s      - naive throughput [tokens/s]
  * speedup          - eagle_tok_s / naive_tok_s

An aggregate summary (total tokens, mean acceptance length, mean tok/s,
decoding speedup, ...) is printed at the end and appended as a final
"summary" record to the output JSONL.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "externals" / "EAGLE"))
sys.path.insert(0, str(ROOT / "scripts"))

from eagle.model.ea_model import EaModel  # noqa: E402
from eagle.model.kv_cache import initialize_past_key_values  # noqa: E402

from common import rouge  # noqa: E402


def load_questions(question_file: Path, begin: int, end: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in question_file.read_text().splitlines()
        if line.strip()
    ]
    selected = rows[begin:end]
    if not selected:
        raise ValueError(f"No question in range [{begin}, {end}) from {question_file}")
    return selected


def build_input_ids(tokenizer, prompt: str, device) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)


def timed_generate(
    model,
    input_ids: torch.Tensor,
    temperature: float,
    max_new_tokens: int,
    total_token: int,
    spec: bool,
) -> tuple[torch.Tensor, int, int, float, list[int]]:
    """Run EAGLE (spec=True) or naive (spec=False) decoding.

    The model returns decode-only latency: prefill is excluded, matching the
    DFlash benchmark. For EAGLE, acceptance_lengths contains the number of
    output tokens committed by each draft-verify iteration (accepted draft
    tokens plus the target fallback token).
    """
    max_length = input_ids.shape[1] + max_new_tokens + total_token + 32
    if spec:
        output_ids, new_tokens, idx, elapsed, acceptance_lengths = model.eagenerate(
            input_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=True,
            return_stats=True,
        )
        steps = len(acceptance_lengths)
    else:
        output_ids, new_tokens, idx, elapsed = model.naivegenerate(
            input_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=True,
            return_stats=True,
        )
        steps = int(idx) + 1
        acceptance_lengths = []

    # Use the returned sequence as the source of truth after max-token
    # truncation, rather than relying on the internal loop counter.
    new_tokens = int(output_ids.shape[1] - input_ids.shape[1])
    return output_ids, new_tokens, steps, float(elapsed), acceptance_lengths


def decode_answer(tokenizer, output_ids: torch.Tensor, input_len: int) -> str:
    return tokenizer.decode(
        output_ids[0, input_len:],
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--eagle-model", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=1)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--total-token", type=int, default=32)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-naive", action="store_true",
                        help="Skip the naive autoregressive baseline (no speedup reported)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("EAGLE3 inference requires a visible CUDA GPU")

    questions = load_questions(
        Path(args.question_file), args.question_begin, args.question_end
    )
    print(f"Loaded {len(questions)} questions "
          f"[{args.question_begin}, {args.question_end}), "
          f"{args.num_choices} choice(s) each")

    model = EaModel.from_pretrained(
        base_model_path=args.base_model,
        ea_model_path=args.eagle_model,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # EAGLE3 builds tree tensors with .tolist(), so CPU/meta offloading
        # from device_map="auto" is not supported here. Keep both models on
        # the single Tesla T4.
        device_map={"": "cuda:0"},
        use_eagle3=True,
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    # Tokenize every prompt up front. This also gives us the longest input so
    # the shared KV cache can be sized to fit *all* questions: EAGLE3 allocates
    # the KV cache only once (on the first generate() call) and then reuses it,
    # so a cache sized for a short warmup would overflow on longer prompts.
    print("Tokenizing prompts ...")
    prompts = [q["turns"][0] for q in questions]
    all_input_ids = [build_input_ids(tokenizer, p, "cuda") for p in prompts]
    max_input_len = max(t.shape[1] for t in all_input_ids)
    kv_max_length = max_input_len + args.max_new_tokens + args.total_token + 32
    print(f"Max prompt length: {max_input_len}; KV cache max_length: {kv_max_length}")

    past_kv, past_kv_data, cur_len = initialize_past_key_values(
        model.base_model, max_length=kv_max_length
    )
    model.past_key_values = past_kv
    model.past_key_values_data = past_kv_data
    model.current_length_data = cur_len

    # Warm up CUDA kernels (cuBLAS / cuDNN autotune) so the first timed
    # generation is not inflated.
    print("Warming up ...")
    warmup_ids = build_input_ids(tokenizer, "Hello", "cuda")
    with torch.inference_mode():
        model.eagenerate(
            warmup_ids,
            temperature=args.temperature,
            max_new_tokens=8,
            max_length=warmup_ids.shape[1] + 8 + args.total_token + 32,
            log=False,
        )
        if not args.skip_naive:
            model.naivegenerate(
                warmup_ids,
                temperature=args.temperature,
                max_new_tokens=8,
                max_length=warmup_ids.shape[1] + 8 + args.total_token + 32,
                log=False,
            )
    print("Warmup done.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    raw_metrics: list[dict] = []
    with output.open("w", encoding="utf-8") as fout:
        for qi, question in enumerate(questions):
            prompt = question["turns"][0]
            qid = question.get("question_id", qi)
            for choice in range(args.num_choices):
                torch.manual_seed(choice)
                input_ids = all_input_ids[qi]
                input_len = input_ids.shape[1]

                with torch.inference_mode():
                    out_ids, new_tokens, tree_steps, eagle_time, acceptance_lengths = timed_generate(
                        model, input_ids, args.temperature,
                        args.max_new_tokens, args.total_token, spec=True,
                    )
                    answer = decode_answer(tokenizer, out_ids, input_len)

                    if not args.skip_naive:
                        _, naive_tokens, _, naive_time, _ = timed_generate(
                            model, input_ids, args.temperature,
                            args.max_new_tokens, args.total_token, spec=False,
                        )
                    else:
                        naive_tokens, naive_time = None, None

                # DFlash reports the unweighted mean of each generation's
                # per-verification acceptance lengths.
                accept_length = (
                    statistics.mean(acceptance_lengths)
                    if acceptance_lengths else 0.0
                )
                eagle_tok_s = new_tokens / eagle_time if eagle_time > 0 else 0.0
                if naive_time is not None and naive_time > 0 and naive_tokens:
                    naive_tok_s = naive_tokens / naive_time
                    speedup = eagle_tok_s / naive_tok_s if naive_tok_s > 0 else 0.0
                else:
                    naive_tok_s, speedup = None, None

                record = {
                    "question_id": qid,
                    "choice": choice,
                    "question": prompt,
                    "answer": answer,
                    "new_tokens": new_tokens,
                    "tree_steps": tree_steps,
                    "accept_length": round(accept_length, 4),
                    "acceptance_lengths": acceptance_lengths,
                    "eagle_time": round(eagle_time, 4),
                    "eagle_tok_s": round(eagle_tok_s, 2),
                    "naive_tokens": naive_tokens,
                    "naive_time": (round(naive_time, 4)
                                   if naive_time is not None else None),
                    "naive_tok_s": (round(naive_tok_s, 2)
                                    if naive_tok_s is not None else None),
                    "speedup": (round(speedup, 3)
                                if speedup is not None else None),
                    "base_model": args.base_model,
                    "eagle_model": args.eagle_model,
                }
                # ROUGE-1/2/L vs reference (nếu question file có reference/answer)
                rouge.add_rouge(
                    record, answer,
                    question.get("reference") or question.get("answer"),
                )
                records.append(record)
                raw_metrics.append({
                    "eagle_tokens": new_tokens,
                    "eagle_time": eagle_time,
                    "accept_length": accept_length,
                    "naive_tokens": naive_tokens,
                    "naive_time": naive_time,
                })
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(
                    f"[{qid}] choice={choice} tokens={new_tokens} "
                    f"steps={tree_steps} accept={accept_length:.2f} "
                    f"eagle={eagle_tok_s:.1f} tok/s ({eagle_time:.2f}s)"
                    + (f" | naive={naive_tok_s:.1f} tok/s ({naive_time:.2f}s)"
                       f" | speedup={speedup:.2f}x" if speedup is not None else "")
                )

    # ---- Aggregate statistics -------------------------------------------
    n = len(records)
    total_tokens = sum(r["new_tokens"] for r in records)
    total_steps = sum(r["tree_steps"] for r in records)
    total_eagle_time = sum(r["eagle_time"] for r in raw_metrics)
    eagle_tpot_values = [
        r["eagle_time"] / r["eagle_tokens"]
        for r in raw_metrics
        if r["eagle_tokens"] > 0 and r["eagle_time"] > 0
    ]
    eagle_tpot = statistics.mean(eagle_tpot_values) if eagle_tpot_values else 0.0
    # DFlash defines throughput as the inverse of mean per-generation TPOT,
    # rather than total tokens divided by total time.
    eagle_throughput = 1.0 / eagle_tpot if eagle_tpot > 0 else 0.0

    naive_metrics = [
        r for r in raw_metrics
        if r["naive_tokens"] is not None
        and r["naive_tokens"] > 0
        and r["naive_time"] is not None
        and r["naive_time"] > 0
    ]
    if naive_metrics and not args.skip_naive:
        total_naive_tokens = sum(r["naive_tokens"] for r in naive_metrics)
        total_naive_time = sum(r["naive_time"] for r in naive_metrics)
        naive_tpot_values = [
            r["naive_time"] / r["naive_tokens"] for r in naive_metrics
        ]
        naive_tpot = statistics.mean(naive_tpot_values)
        naive_throughput = 1.0 / naive_tpot if naive_tpot > 0 else 0.0
        # Exact DFlash definition: mean baseline TPOT / mean speculative TPOT.
        speedup = naive_tpot / eagle_tpot if eagle_tpot > 0 else 0.0
        speedups = [
            (r["naive_time"] / r["naive_tokens"])
            / (r["eagle_time"] / r["eagle_tokens"])
            for r in naive_metrics
            if r["eagle_tokens"] > 0 and r["eagle_time"] > 0
        ]
    else:
        total_naive_tokens = None
        total_naive_time, naive_throughput, speedup, speedups = 0.0, None, None, []

    eagle_tok_s_list = [
        r["eagle_tokens"] / r["eagle_time"]
        for r in raw_metrics
        if r["eagle_tokens"] > 0 and r["eagle_time"] > 0
    ]
    accept_lengths = [r["accept_length"] for r in raw_metrics]

    summary = {
        "type": "summary",
        "num_questions": len(questions),
        "num_generations": n,
        "total_tokens": total_tokens,
        "total_naive_tokens": total_naive_tokens,
        "mean_accept_length": round(
            statistics.mean(accept_lengths), 4) if accept_lengths else 0.0,
        "mean_tree_steps": round(total_steps / n, 2) if n else 0.0,
        "total_eagle_time": round(total_eagle_time, 3),
        "eagle_tok_s": round(eagle_throughput, 2),
        "mean_eagle_tok_s": round(
            statistics.mean(eagle_tok_s_list), 2) if eagle_tok_s_list else 0.0,
        "total_naive_time": (round(total_naive_time, 3)
                             if not args.skip_naive else None),
        "naive_tok_s": (round(naive_throughput, 2)
                        if naive_throughput is not None else None),
        "decoding_speedup": (round(speedup, 3) if speedup is not None else None),
        "mean_speedup": (round(statistics.mean(speedups), 3) if speedups else None),
        **rouge.aggregate_rouge(records),
    }
    with output.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print("\n" + "=" * 56)
    print("EAGLE3 Decoding Statistics")
    print("=" * 56)
    print(f"Questions            : {len(questions)}")
    print(f"Generations          : {n}")
    print(f"Total output tokens  : {total_tokens}")
    print(f"Mean acceptance len  : {summary['mean_accept_length']:.3f} "
          f"tokens/step")
    print(f"Mean tree steps      : {summary['mean_tree_steps']:.2f}")
    print("-" * 56)
    print(f"EAGLE3 throughput    : {summary['eagle_tok_s']:.2f} tok/s "
          f"({total_eagle_time:.2f}s total)")
    if naive_throughput is not None:
        print(f"Naive throughput     : {naive_throughput:.2f} tok/s "
              f"({total_naive_time:.2f}s total)")
        print(f"Decoding speedup     : {speedup:.2f}x")
    print(f"Mean per-gen EAGLE   : {summary['mean_eagle_tok_s']:.2f} tok/s")
    print("=" * 56)
    print(f"Saved {n} record(s) + summary to: {output}")


if __name__ == "__main__":
    main()
