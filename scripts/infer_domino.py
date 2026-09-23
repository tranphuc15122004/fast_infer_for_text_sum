#!/usr/bin/env python3
"""Domino Qwen3-4B LongBench adapter with online Vanilla-HF pairing."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from common import io_util, metrics, rouge, verify
from common.benchmark_runtime import build_sample_record, runtime_metadata
from common.data_loader import load_records
from common.paired_reference import attach_reference_timing, load_reference_sidecar, prompt_hash
from common.qwen3_paired import build_run_config, prepare_input_ids
from common.reproducibility import seed_everything


ROOT = Path(__file__).resolve().parents[1]
DOMINO_ROOT = ROOT / "externals" / "Domino" / "code"
if str(DOMINO_ROOT) not in sys.path:
    sys.path.insert(0, str(DOMINO_ROOT))


def normalize_draft_config_for_benchmark(config):
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    if dflash_config.get("projector_type") == "causal_v5":
        dflash_config["projector_type"] = "domino"
    if "emb_dim" not in dflash_config and getattr(config, "emb_dim", None) is not None:
        dflash_config["emb_dim"] = config.emb_dim
    if "gru_hidden_dim" not in dflash_config:
        hidden = getattr(config, "gru_hidden_dim", None) or dflash_config.get("emb_dim")
        if hidden is not None:
            dflash_config["gru_hidden_dim"] = hidden
    config.dflash_config = dflash_config
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--fixed-output-tokens", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("LONG_BENCH_SEED", "42")))
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--attention-backend", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--use-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=os.environ.get("LONG_BENCH_DEVICE", "cuda:0"))
    parser.add_argument("--run-id", default=os.environ.get("LONG_BENCH_RUN_ID"))
    parser.add_argument("--run-config-hash", default=None)
    parser.add_argument("--reference-file", default=None)
    parser.add_argument("--reference-baseline", default="vanilla_hf")
    parser.add_argument("--reference-run-id", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def _load_models(args, device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Domino inference requires CUDA")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from dflash import DFlashDraftModel

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=dtype,
        attn_implementation=args.attention_backend,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    draft_config = normalize_draft_config_for_benchmark(
        AutoConfig.from_pretrained(args.draft_model, local_files_only=args.local_files_only)
    )
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        config=draft_config,
        dtype=dtype,
        attn_implementation=args.attention_backend,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    return target, draft, tokenizer, dtype


def _run_once(draft, target, input_ids, *, max_new_tokens, temperature, block_size, fixed_output, use_bias=True):
    # Domino currently materializes stop_token_ids as an integer tensor.  A
    # sentinel outside the vocabulary disables EOS stopping without hitting
    # the empty-list dtype mismatch in torch.isin.
    stop_ids = [-1] if fixed_output else target.config.eos_token_id
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    return draft.spec_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        stop_token_ids=stop_ids,
        temperature=temperature,
        use_bias=use_bias,
        return_dict=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        args.max_samples = 1
        args.max_new_tokens = min(args.max_new_tokens, 32)
        if args.fixed_output_tokens is not None:
            args.fixed_output_tokens = min(args.fixed_output_tokens, 32)
        if args.max_input_tokens <= 0:
            args.max_input_tokens = 4096
    if args.fixed_output_tokens is not None:
        if args.fixed_output_tokens <= 0:
            raise SystemExit("--fixed-output-tokens must be positive")
        args.max_new_tokens = args.fixed_output_tokens

    seed_everything(args.seed)
    device = torch.device(args.device)
    records = load_records(Path(args.data_file), args.max_samples)
    reference_records = (
        load_reference_sidecar(Path(args.reference_file))
        if args.reference_file
        else None
    )
    load_start = time.perf_counter()
    target, draft, tokenizer, dtype = _load_models(args, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_ms = round((time.perf_counter() - load_start) * 1000.0, 3)
    block_size = int(args.block_size or draft.block_size)
    metadata = runtime_metadata()

    warmup = tokenizer("Hello", return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        for _ in range(max(args.warmup_runs, 0)):
            _run_once(
                draft, target, warmup,
                max_new_tokens=min(args.max_new_tokens, 8),
                temperature=args.temperature,
                block_size=block_size,
                fixed_output=False,
                use_bias=args.use_bias,
            )

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []
    for sample in records:
        dataset = sample.get("raw", {}).get("dataset", Path(args.data_file).stem)
        encoded = tokenizer(sample["prompt"], return_tensors="pt")
        prepared = prepare_input_ids(encoded.input_ids, max(args.max_input_tokens, 0))
        input_ids = prepared.input_ids.to(device).contiguous()
        run_config_hash = args.run_config_hash
        if run_config_hash is None:
            run_config_hash = build_run_config(
                dataset=dataset,
                target_model=args.target_model,
                input_cap=max(args.max_input_tokens, 0),
                output_tokens=args.max_new_tokens,
                seed=args.seed,
                dtype=str(dtype),
                attention_backend="paired",
                temperature=args.temperature,
            )["run_config_hash"]
        seed_everything(args.seed)
        with torch.inference_mode():
            result = _run_once(
                draft, target, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                block_size=block_size,
                fixed_output=args.fixed_output_tokens is not None,
                use_bias=args.use_bias,
            )
        output_ids = result.output_ids[0, result.num_input_tokens:]
        text = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        output_tokens = int(result.num_output_tokens)
        prefill_ms = float(result.time_to_first_token) * 1000.0
        decode_ms = float(result.time_per_output_token) * output_tokens * 1000.0
        e2e_ms = prefill_ms + decode_ms
        acceptance = [int(value) for value in result.acceptance_lengths]
        tau = sum(acceptance) / len(acceptance) if acceptance else None
        record = build_sample_record(
            method="domino",
            dataset=dataset,
            sample_id=sample["id"],
            model=args.target_model,
            input_tokens=prepared.input_tokens,
            output_tokens=output_tokens,
            timing={
                "model_load_ms": model_load_ms,
                "prefill_ms": prefill_ms,
                "ttft_ms": prefill_ms,
                "decode_ms": decode_ms,
                "e2e_ms": e2e_ms,
                "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            },
            config={
                "batch_size": 1,
                "device": str(device),
                "gpu_name": metadata.get("gpu_name"),
                "dtype": str(dtype),
                "attention_backend": args.attention_backend,
                "seed": args.seed,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "speed_output_tokens": args.fixed_output_tokens,
                "warmup_runs": args.warmup_runs,
                "original_input_tokens": prepared.original_input_tokens,
                "input_truncated": prepared.input_truncated,
                "prompt_hash": prompt_hash(prepared.input_ids),
                "run_config_hash": run_config_hash,
                "extra_metrics": {
                    "block_size": block_size,
                    "acceptance_lengths": acceptance,
                    "tau": tau,
                    "acceptance_rate": (
                        sum(max(0, value - 1) for value in acceptance)
                        / (len(acceptance) * max(block_size - 1, 1))
                        if acceptance and block_size > 1 else None
                    ),
                },
            },
            text=text,
            reference_output=sample.get("reference"),
        )
        record["run_id"] = args.run_id
        record["task_type"] = sample.get("raw", {}).get("task_type") or sample.get("task_type")
        record["acceptance_lengths"] = acceptance
        record["avg_accept_length"] = tau
        record["block_size"] = block_size
        if reference_records is not None:
            key = (str(record["dataset"]), str(record["sample_id"]), str(record["prompt_hash"]), str(record["run_config_hash"]))
            reference = reference_records.get(key) or {
                "dataset": record["dataset"], "sample_id": record["sample_id"],
                "prompt_hash": record["prompt_hash"], "run_config_hash": record["run_config_hash"],
                "status": "missing",
            }
            record = attach_reference_timing(
                record, reference,
                reference_baseline=args.reference_baseline,
                reference_run_id=args.reference_run_id,
                expected_output_tokens=args.fixed_output_tokens,
            )
        if record["task_type"] == "code_completion":
            metrics.add_code_completion(record, text, sample.get("reference"))
        else:
            rouge.add_rouge(record, text, sample.get("reference"))
            metrics.add_semantic(record, text, sample.get("reference"))
        writer.add(record)
        checks.extend((verify.check_new_tokens(output_tokens), verify.check_output_text(text)))
        print(
            f"[domino][{dataset}][{sample['id']}] input={prepared.input_tokens} "
            f"output={output_tokens} decode_tok_s={record['decode_throughput_tok_s']} "
            f"esr={record.get('esr')} dsr={record.get('dsr')}",
            flush=True,
        )

    quality = (
        metrics.aggregate_code_completion(writer.records)
        if any(r.get("task_type") == "code_completion" for r in writer.records)
        else metrics.aggregate_semantic(writer.records)
    )
    writer.finalize({
        "type": "summary", "method": "domino", "dataset": Path(args.data_file).stem,
        "run_id": args.run_id, "num_samples": len(records), "block_size": block_size,
        "model": args.target_model, "runtime": metadata, **quality,
    })
    print(f"Saved to: {args.output}")
    verify.finish("Domino", checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
