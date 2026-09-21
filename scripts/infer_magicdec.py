#!/usr/bin/env python3
"""MagicDec verification / smoke script (dense baseline_benchmark wrapper).

Runs MagicDec's own `tests/baseline_benchmark.py` on one GPU with a small
model (e.g. TinyLlama) and short prefix so it fits a T4 16GB, and verifies
the run produces output tokens / timing lines.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

from common import io_util, verify
from common import metrics, rouge
from common.data_loader import load_records
from common.input_utils import truncate_input_ids
from common.paths import ROOT
from common.reproducibility import seed_everything

MAGICDEC = ROOT / "externals" / "MagicDec"
MAGICDEC_PARENT = ROOT / "externals"
if str(MAGICDEC_PARENT) not in sys.path:
    # ``from MagicDec.Engine...`` resolves the package from its parent.
    sys.path.insert(0, str(MAGICDEC_PARENT))


def resolve_generation_budget(max_new_tokens: int, *, warmup: bool = False) -> int:
    """Keep compile/cache warmup bounded while preserving benchmark budget."""

    budget = max(1, int(max_new_tokens))
    return min(budget, 8) if warmup else budget


def _ensure_magicdec_warmup_context(
    input_ids: torch.Tensor, *, min_tokens: int, fill_token_id: int
) -> torch.Tensor:
    """Pad a self-spec warmup prompt to SnapKV's minimum context length.

    SnapKV builds the draft cache from ``draft_budget - window_size`` past
    positions.  A short warmup prompt therefore reaches ``gen_draft_kv`` with
    a negative/undersized context even though a real LongBench prompt is long
    enough.  The returned tensor is used only for warmup; benchmark prompts
    remain untouched.
    """
    required = max(int(min_tokens), 0)
    if input_ids.ndim != 2:
        raise ValueError(
            f"MagicDec warmup input_ids must be rank 2, got rank {input_ids.ndim}"
        )
    current = int(input_ids.shape[1])
    if current >= required:
        return input_ids
    filler = torch.full(
        (int(input_ids.shape[0]), required - current),
        int(fill_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat([input_ids, filler], dim=1)


def _canonical_next_token(logits, temperature: float):
    """Return one token from either MagicDec IDs or raw model logits.

    The vendored MagicDec model applies argmax over the vocabulary and returns
    token IDs with shape ``[batch, sequence]``. Other model adapters may
    return raw logits with shape ``[batch, sequence, vocab]``.
    """
    if logits.ndim == 2:
        if temperature > 0:
            raise ValueError(
                "MagicDec returned token IDs, so temperature sampling "
                "requires a backend that exposes raw logits"
            )
        return logits[:, -1:]
    if logits.ndim != 3:
        raise ValueError(
            f"Expected MagicDec output rank 2 or raw logits rank 3; got rank {logits.ndim}"
        )

    scores = logits[:, -1, :]
    if temperature > 0:
        return torch.multinomial(torch.softmax(scores / temperature, dim=-1), 1)
    return scores.argmax(dim=-1, keepdim=True)


def summarize_magicdec_acceptance(
    acceptance_lengths: list[int], *, gamma: int, draft_tokens_proposed: int | None = None
) -> dict[str, float | None]:
    """Summarize MagicDec self-spec acceptance at the decode-step scope.

    Each ``acceptance_lengths`` entry is the number of tokens committed by one
    draft/verification round, including the bonus target token.  Therefore a
    round with no accepted draft token still has length one.  The acceptance
    rate intentionally counts only draft tokens, matching MagicDec's upstream
    ``accept_nums = accepted_draft + 1`` convention.
    """
    if not acceptance_lengths:
        return {
            "avg_accept_length": None,
            "acceptance_rate": None,
            "rejected_draft_ratio": None,
        }
    average = sum(float(value) for value in acceptance_lengths) / len(acceptance_lengths)
    proposed = (
        int(draft_tokens_proposed)
        if draft_tokens_proposed is not None
        else len(acceptance_lengths) * max(int(gamma), 0)
    )
    accepted_draft = sum(max(int(value) - 1, 0) for value in acceptance_lengths)
    acceptance_rate = accepted_draft / proposed if proposed else None
    return {
        "avg_accept_length": round(average, 4),
        "acceptance_rate": round(acceptance_rate, 4) if acceptance_rate is not None else None,
        "rejected_draft_ratio": (
            round(1.0 - acceptance_rate, 4)
            if acceptance_rate is not None
            else None
        ),
    }


def _self_spec_acceptance(
    draft_tokens: list[int], target_tokens: list[int], eos_ids: set[int]
) -> tuple[int, int]:
    """Return ``(committed_length, bonus_index)`` for one verify round."""
    accepted_draft = 0
    for draft, target in zip(draft_tokens, target_tokens):
        if draft in eos_ids or draft != target:
            break
        accepted_draft += 1
    return accepted_draft + 1, accepted_draft


def _append_bounded_tokens(
    generated: list[torch.Tensor], token_ids: torch.Tensor, remaining: int
) -> int:
    """Append at most ``remaining`` one-row tokens and return the count."""
    if remaining <= 0:
        return 0
    bounded = token_ids[:, :remaining]
    if bounded.shape[1]:
        generated.append(bounded.clone())
    return int(bounded.shape[1])


def _eos_ids(tokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    values = eos if isinstance(eos, (list, tuple, set)) else [eos]
    result = {int(value) for value in values if value is not None}
    if tokenizer.unk_token_id is not None:
        result.add(int(tokenizer.unk_token_id))
    else:
        try:
            fallback = tokenizer.encode(
                "<|eot_id|>", add_special_tokens=False
            )
            if fallback:
                result.add(int(fallback[-1]))
        except (TypeError, ValueError, IndexError):
            pass
    return result


def _generate_magicdec_self_spec(
    engine,
    input_ids: torch.Tensor,
    *,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    max_new_tokens: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Generate one prompt with MagicDec's upstream self-spec loop.

    The target model is used for both branches: ``speculate`` uses the
    compressed draft KV cache and ``verify`` evaluates the speculative block
    with the target KV cache.  The cache rollback/commit sequence mirrors
    ``externals/MagicDec/tests/SnapKV/selfspec_benchmark.py`` while keeping
    per-request output and acceptance telemetry for the LongBench schema.
    """
    gamma = int(args.gamma)
    if gamma <= 0:
        raise ValueError("MagicDec self-spec gamma must be positive")
    if int(args.draft_budget) <= int(args.window_size):
        raise ValueError("MagicDec draft budget must exceed window size")

    generation_budget = (
        resolve_generation_budget(args.max_new_tokens)
        if max_new_tokens is None
        else max(1, int(max_new_tokens))
    )
    eos_ids = _eos_ids(tokenizer)
    if args.reset_peak_memory:
        torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    request_start = time.perf_counter()
    prefill_start = time.perf_counter()
    logits = engine.encode(input_ids)
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

    next_token = _canonical_next_token(logits, args.temperature)
    generated: list[torch.Tensor] = []
    generated_count = 0
    acceptance_lengths: list[int] = []
    draft_tokens_proposed = 0
    draft_tokens_accepted = 0
    draft_latency_ms = 0.0
    verification_latency_ms = 0.0

    while generated_count < generation_budget:
        remaining = generation_budget - generated_count
        # A speculative round always commits one bonus/target token.  Limit
        # the final draft width so no token beyond max_new_tokens is ever
        # appended or counted in the public record.
        step_gamma = min(gamma, max(remaining - 1, 0))
        if step_gamma == 0:
            generated_count += _append_bounded_tokens(
                generated, next_token, remaining
            )
            break
        draft_tokens_proposed += step_gamma

        tokens_buffer = torch.zeros(
            (1, step_gamma + 1), device=device, dtype=torch.long
        )
        tokens_buffer[:, :1] = next_token

        torch.cuda.synchronize(device)
        draft_start = time.perf_counter()
        for index in range(step_gamma):
            draft_logits = engine.speculate(
                tokens_buffer[:, index : index + 1]
            )
            tokens_buffer[:, index + 1 : index + 2] = _canonical_next_token(
                draft_logits, args.temperature
            )
        torch.cuda.synchronize(device)
        draft_latency_ms += (time.perf_counter() - draft_start) * 1000.0

        torch.cuda.synchronize(device)
        verification_start = time.perf_counter()
        target_tokens = engine.verify(tokens_buffer)
        torch.cuda.synchronize(device)
        verification_latency_ms += (
            time.perf_counter() - verification_start
        ) * 1000.0

        draft_token_ids = [
            int(value) for value in tokens_buffer[0, 1:].tolist()
        ]
        target_token_ids = [
            int(value) for value in target_tokens[0].tolist()
        ]
        accept_length, bonus_index = _self_spec_acceptance(
            draft_token_ids, target_token_ids[:step_gamma], eos_ids
        )
        acceptance_lengths.append(accept_length)
        draft_tokens_accepted += max(accept_length - 1, 0)

        # verify() speculatively appended gamma+1 positions to both cache
        # views. Roll them back, then commit exactly the accepted prefix.
        rollback = step_gamma + 1
        engine.cachelens -= rollback
        engine.paged_kv_last_page_len -= rollback
        engine.draft_cachelens -= rollback
        engine.draft_paged_kv_last_page_len -= rollback
        engine.cachelens += accept_length
        engine.paged_kv_last_page_len += accept_length
        engine.draft_cachelens += accept_length
        engine.draft_paged_kv_last_page_len += accept_length

        accepted_tokens = tokens_buffer[:, :accept_length]
        generated_count += _append_bounded_tokens(
            generated, accepted_tokens, remaining
        )
        if any(int(value) in eos_ids for value in accepted_tokens[0].tolist()):
            break
        if generated_count >= generation_budget:
            break

        bonus = target_tokens[:, bonus_index : bonus_index + 1]
        if int(bonus[0, 0]) in eos_ids:
            generated_count += _append_bounded_tokens(
                generated, bonus, generation_budget - generated_count
            )
            break
        next_token = bonus

    torch.cuda.synchronize(device)
    request_end = time.perf_counter()
    e2e_ms = (request_end - request_start) * 1000.0
    decode_ms = e2e_ms - prefill_ms
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    acceptance = summarize_magicdec_acceptance(
        acceptance_lengths,
        gamma=gamma,
        draft_tokens_proposed=draft_tokens_proposed,
    )
    generated_ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty((1, 0), device=device, dtype=torch.long)
    )
    return torch.cat([input_ids, generated_ids], dim=1), {
        "prefill_ms": round(prefill_ms, 3),
        "ttft_ms": round(prefill_ms, 3),
        "decode_ms": round(max(decode_ms, 0.0), 3),
        "e2e_ms": round(e2e_ms, 3),
        "peak_memory_gb": round(peak_gb, 6),
        "acceptance_lengths": acceptance_lengths,
        "draft_latency_ms": round(draft_latency_ms, 3),
        "verification_latency_ms": round(verification_latency_ms, 3),
        "speculative_steps": len(acceptance_lengths),
        "draft_tokens_proposed": draft_tokens_proposed,
        "draft_tokens_accepted": draft_tokens_accepted,
        **acceptance,
    }


def _run_canonical(args: argparse.Namespace) -> None:
    """Run MagicDec's SnapKV engine on arbitrary canonical prompt JSONL."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("MagicDec canonical inference requires CUDA")
    seed_everything(args.seed)
    from transformers import AutoTokenizer
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
        input_ids = tokenizer(
            sample["prompt"], return_tensors="pt", add_special_tokens=True
        ).input_ids
        input_ids = truncate_input_ids(input_ids, args.max_input_tokens).to(device)
        encoded_records.append((sample, input_ids))
    max_input = max(int(ids.shape[1]) for _, ids in encoded_records)
    max_sequence = max(128, ((max_input + args.max_new_tokens + 127) // 128) * 128)

    load_start = time.perf_counter()
    engine = LMBackend(
        dtype=dtype,
        device="cuda:0",
        dec_len=args.gamma + 1 if args.self_spec else 1,
        draft_dec_len=1 if args.self_spec else None,
    )
    engine.load_model(Path(args.model_pth), use_tp=False, rank_group=[0])
    engine.setup_caches(
        max_batch_size=1,
        max_seq_length=max_sequence,
        draft_budget=args.draft_budget if args.self_spec else 0,
        window_size=args.window_size,
    )
    torch.cuda.synchronize(device)
    model_load_ms = round((time.perf_counter() - load_start) * 1000.0, 3)

    def generate_target_only(input_ids, *, max_new_tokens=None):
        generation_budget = (
            resolve_generation_budget(args.max_new_tokens)
            if max_new_tokens is None
            else max(1, int(max_new_tokens))
        )
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
            for _ in range(max(generation_budget - 1, 0)):
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

    def generate(input_ids, *, max_new_tokens=None):
        if args.self_spec:
            return _generate_magicdec_self_spec(
                engine,
                input_ids,
                tokenizer=tokenizer,
                args=args,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        return generate_target_only(input_ids, max_new_tokens=max_new_tokens)

    warmup_ids = tokenizer(
        "Hello", return_tensors="pt", add_special_tokens=True
    ).input_ids
    if args.self_spec:
        fill_token_id = next(
            (
                token_id
                for token_id in (
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                    tokenizer.unk_token_id,
                )
                if token_id is not None
            ),
            0,
        )
        warmup_ids = _ensure_magicdec_warmup_context(
            warmup_ids,
            min_tokens=max(int(args.draft_budget), int(args.window_size) + 1),
            fill_token_id=int(fill_token_id),
        )
    warmup_ids = warmup_ids.to(device)
    for _ in range(max(args.warmup_runs, 0)):
        seed_everything(args.seed)
        with torch.inference_mode():
            generate(
                warmup_ids,
                max_new_tokens=resolve_generation_budget(
                    args.max_new_tokens, warmup=True
                ),
            )

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []
    with torch.inference_mode():
        for sample, input_ids in encoded_records:
            seed_everything(args.seed)
            output_ids, timing = generate(input_ids)
            new_ids = output_ids[0, input_ids.shape[1]:]
            text = tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            output_tokens = int(new_ids.shape[0])
            config = {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0),
                "dtype": str(dtype).removeprefix("torch."),
                "attention_backend": "magicdec_snapkv",
                "self_spec": bool(args.self_spec),
                "gamma": args.gamma if args.self_spec else None,
                "draft_budget": args.draft_budget if args.self_spec else None,
                "window_size": args.window_size if args.self_spec else None,
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
                acceptance_lengths=timing.get("acceptance_lengths"),
                speculative_metrics=timing,
            )
            if sample.get("raw", {}).get("task_type") == "code_completion":
                metrics.add_code_completion(record, text, sample.get("reference"))
            else:
                rouge.add_rouge(record, text, sample.get("reference"))
                metrics.add_semantic(record, text, sample.get("reference"))
            writer.add(record)
            checks += [verify.check_new_tokens(output_tokens), verify.check_output_text(text)]

    quality = (
        metrics.aggregate_code_completion(writer.records)
        if any(r.get("task_type") == "code_completion" for r in writer.records)
        else metrics.aggregate_semantic(writer.records)
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


def build_magicdec_record(
    *,
    sample,
    args,
    input_tokens,
    output_tokens,
    text,
    timing,
    config,
    acceptance_lengths=None,
    speculative_metrics=None,
):
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
    record["magicdec_self_spec"] = bool(config.get("self_spec", False))
    record["magicdec_gamma"] = config.get("gamma")
    record["magicdec_draft_budget"] = config.get("draft_budget")
    record["magicdec_window_size"] = config.get("window_size")
    if acceptance_lengths is not None:
        record["acceptance_lengths"] = [int(value) for value in acceptance_lengths]
        record["avg_accept_length"] = (
            sum(float(value) for value in acceptance_lengths)
            / len(acceptance_lengths)
            if acceptance_lengths
            else None
        )
    else:
        record["acceptance_lengths"] = None
        record["avg_accept_length"] = None
    for key in (
        "acceptance_rate",
        "draft_latency_ms",
        "verification_latency_ms",
        "rejected_draft_ratio",
        "draft_tokens_proposed",
        "draft_tokens_accepted",
        "speculative_steps",
    ):
        record[key] = (
            speculative_metrics.get(key)
            if speculative_metrics is not None
            else None
        )
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
                        help="run SnapKV self-spec with per-sample acceptance telemetry")
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
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("LONG_BENCH_SEED", "42"))
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-peak-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--use-torchrun", action="store_true",
                        help="use torchrun even for single-GPU smoke")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_everything(args.seed)

    if args.smoke:
        args.prefix_len = min(args.prefix_len, 1024)
        args.max_len = ((args.prefix_len + 128) // 128) * 128
        args.num_runs = 1

    if args.self_spec:
        if args.gamma <= 0:
            raise SystemExit("--gamma must be positive for MagicDec self-spec")
        if args.window_size <= 0 or args.draft_budget <= args.window_size:
            raise SystemExit("--draft-budget must be greater than --window-size")
        if (args.draft_budget - 1) % 128 != 0:
            raise SystemExit("--draft-budget must satisfy (draft_budget - 1) % 128 == 0")

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
