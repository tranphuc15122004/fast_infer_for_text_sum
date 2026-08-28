#!/usr/bin/env python3
"""MagicDec verification / smoke script (dense baseline_benchmark wrapper).

Runs MagicDec's own `tests/baseline_benchmark.py` on one GPU with a small
model (e.g. TinyLlama) and short prefix so it fits a T4 16GB, and verifies
the run produces output tokens / timing lines.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

from common import io_util, verify
from common import metrics, rouge
from common.data_loader import load_records
from common.paths import ROOT

MAGICDEC = ROOT / "externals" / "MagicDec"


def _canonical_next_token(logits, temperature: float):
    scores = logits[:, -1, :]
    if temperature > 0:
        return torch.multinomial(torch.softmax(scores / temperature, dim=-1), 1)
    return scores.argmax(dim=-1, keepdim=True)


def _run_canonical(args: argparse.Namespace) -> None:
    """Run MagicDec's SnapKV engine on arbitrary canonical prompt JSONL."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("MagicDec canonical inference requires CUDA")
    from transformers import AutoTokenizer
    if str(MAGICDEC) not in sys.path:
        sys.path.insert(0, str(MAGICDEC))
    from MagicDec.Engine.SnapKV.backend import LMBackend

    records = load_records(Path(args.data_file), args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    encoded_records = []
    for sample in records:
        kwargs = {"return_tensors": "pt", "add_special_tokens": True}
        if args.max_input_tokens > 0:
            kwargs.update({"truncation": True, "max_length": args.max_input_tokens})
        input_ids = tokenizer(sample["prompt"], **kwargs).input_ids.to(device)
        encoded_records.append((sample, input_ids))
    max_input = max(int(ids.shape[1]) for _, ids in encoded_records)
    max_sequence = max(128, ((max_input + args.max_new_tokens + 127) // 128) * 128)

    load_start = time.perf_counter()
    engine = LMBackend(dtype=dtype, device="cuda:0")
    engine.load_model(Path(args.model_pth), use_tp=False, rank_group=[0])
    engine.setup_caches(max_batch_size=1, max_seq_length=max_sequence)
    torch.cuda.synchronize(device)
    model_load_ms = round((time.perf_counter() - load_start) * 1000.0, 3)

    def generate(input_ids):
        if args.reset_peak_memory:
            torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        request_start = time.perf_counter()
        prefill_start = time.perf_counter()
        logits = engine.encode(input_ids)
        torch.cuda.synchronize(device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        next_token = _canonical_next_token(logits, args.temperature)
        generated = [next_token]
        decode_start = time.perf_counter()
        eos = tokenizer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        if int(next_token[0, 0]) not in {int(value) for value in eos_ids if value is not None}:
            for _ in range(max(args.max_new_tokens - 1, 0)):
                logits = engine.inference(next_token)
                next_token = _canonical_next_token(logits, args.temperature)
                generated.append(next_token)
                if int(next_token[0, 0]) in {int(value) for value in eos_ids if value is not None}:
                    break
        torch.cuda.synchronize(device)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        e2e_ms = (time.perf_counter() - request_start) * 1000.0
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        return torch.cat([input_ids, *generated], dim=1), {
            "prefill_ms": round(prefill_ms, 3),
            "ttft_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "e2e_ms": round(e2e_ms, 3),
            "peak_memory_gb": round(peak_gb, 6),
        }

    warmup_ids = tokenizer("Hello", return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    for _ in range(max(args.warmup_runs, 0)):
        with torch.inference_mode():
            generate(warmup_ids)

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []
    with torch.inference_mode():
        for sample, input_ids in encoded_records:
            output_ids, timing = generate(input_ids)
            new_ids = output_ids[0, input_ids.shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            output_tokens = int(new_ids.shape[0])
            config = {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0),
                "dtype": str(dtype).removeprefix("torch."),
                "attention_backend": "magicdec_snapkv",
                "seed": args.seed,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "warmup_runs": args.warmup_runs,
            }
            record = build_magicdec_record(
                sample=sample,
                args=args,
                input_tokens=int(input_ids.shape[1]),
                output_tokens=output_tokens,
                text=text,
                timing={**timing, "model_load_ms": model_load_ms},
                config=config,
            )
            if sample.get("raw", {}).get("task_type") == "code_completion":
                metrics.add_code_completion(record, text, sample.get("reference"))
            else:
                rouge.add_rouge(record, text, sample.get("reference"))
            writer.add(record)
            checks += [verify.check_new_tokens(output_tokens), verify.check_output_text(text)]

    quality = (
        metrics.aggregate_code_completion(writer.records)
        if any(r.get("task_type") == "code_completion" for r in writer.records)
        else rouge.aggregate_rouge(writer.records)
    )
    writer.finalize({
        "type": "summary",
        "method": "magicdec",
        "dataset": Path(args.data_file).stem,
        "status": "success",
        "num_samples": len(records),
        "model": args.model_name,
        "model_load_ms": model_load_ms,
        **quality,
    })
    io_util.print_table([("method", "magicdec"), ("num_samples", len(records)), ("model_load_ms", model_load_ms)])
    print(f"Saved to: {args.output}")
    verify.finish("MagicDec LongBench", checks)


def build_magicdec_record(*, sample, args, input_tokens, output_tokens, text, timing, config):
    from common.benchmark_runtime import build_sample_record

    record = build_sample_record(
        method="magicdec",
        dataset=Path(args.data_file).stem,
        sample_id=sample["id"],
        model=args.model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        timing=timing,
        config=config,
        text=text,
        reference_output=sample.get("reference"),
    )
    record["task_type"] = sample.get("raw", {}).get("task_type")
    record["scope"] = "sample"
    record["status"] = "success"
    record["magicdec_model_pth"] = str(args.model_pth)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-pth", required=True, help="path to model.pth")
    parser.add_argument("--model-name", required=True, help="HF id for tokenizer")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--max-len", type=int, default=2176,
                        help="must be divisible by 128")
    parser.add_argument("--self-spec", action="store_true",
                        help="run tests/SnapKV/selfspec_benchmark.py instead of dense")
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--draft-budget", type=int, default=257)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=128,
                        help="SnapKV window; prefix/window difference divisible by 128")
    parser.add_argument("--data-file", default=None,
                        help="canonical LongBench JSONL; enables arbitrary-prompt mode")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-peak-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--use-torchrun", action="store_true",
                        help="use torchrun even for single-GPU smoke")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.prefix_len = min(args.prefix_len, 1024)
        args.max_len = ((args.prefix_len + 128) // 128) * 128
        args.num_runs = 1

    if args.data_file:
        if args.max_samples is None:
            args.max_samples = 1 if args.smoke else 200
        if args.smoke:
            args.max_new_tokens = min(args.max_new_tokens, 8)
        if args.max_samples <= 0 or args.max_new_tokens <= 0:
            raise SystemExit("--max-samples and --max-new-tokens must be positive")
        _run_canonical(args)
        return

    assert args.max_len % 128 == 0, "--max-len must be divisible by 128"

    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(MAGICDEC) + ":" + env.get("PYTHONPATH", "")
    env["ENABLE_INTRA_NODE_COMM"] = "1"

    # The benchmark subprocess runs with cwd=MAGICDEC, so the model path must
    # be absolute (a ROOT-relative path would resolve under externals/MagicDec).
    model_pth = Path(args.model_pth)
    if not model_pth.is_absolute():
        model_pth = (ROOT / model_pth).resolve()

    script = (
        "tests/SnapKV/selfspec_benchmark.py" if args.self_spec
        else "tests/baseline_benchmark.py"
    )
    launcher = ([sys.executable, "-m", "torch.distributed.run",
                 "--standalone", "--nproc_per_node=1"]
                if args.use_torchrun else [sys.executable])
    cmd = launcher + [
        script,
        "--model", str(model_pth),
        "--model_name", args.model_name,
        "--rank_group", "0",
        "--B", str(args.batch_size),
        "--prefix_len", str(args.prefix_len),
        "--max_len", str(args.max_len),
        "--printoutput",
    ]
    if args.self_spec:
        # --window_size only exists in selfspec_benchmark.py
        cmd += ["--window_size", str(args.window_size),
                "--gamma", str(args.gamma), "--draft_budget", str(args.draft_budget),
                "--benchmark"]

    print("+ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=MAGICDEC, env=env, capture_output=True, text=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    print(log[-4000:])

    writer = io_util.JsonlWriter(Path(args.output))
    record = {
        "method": "magicdec_selfspec" if args.self_spec else "magicdec_dense",
        "dataset": "pg19",
        "model": args.model_name,
        "input_tokens": args.prefix_len,
        "retained_tokens": None,
        "output_tokens": None,
        "batch_size": args.batch_size,
        "selector_latency_ms": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "e2e_ms": None,
        "throughput_tok_s": None,
        "qps": None,
        "peak_memory_gb": None,
        "returncode": proc.returncode,
        "log_tail": log[-2000:],
    }
    writer.add(record)

    checks: list[tuple[bool, str]] = [
        (proc.returncode == 0, f"benchmark process exit code = {proc.returncode}"),
        ("Throughput" in log or "Token/sec" in log or "tokens/s" in log
         or "tokens per second" in log.lower() or "Speed" in log
         or "output" in log.lower(),
         "benchmark produced timing/output lines"),
    ]
    summary = {"type": "summary", "method": record["method"],
               "returncode": proc.returncode, "num_runs": args.num_runs}
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("MagicDec", checks)


if __name__ == "__main__":
    main()
