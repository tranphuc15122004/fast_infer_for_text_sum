#!/usr/bin/env python3
"""AMR-DFlash capture/train/evaluate command line for local B200 assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
for search_path in (ROOT / "src", ROOT / "scripts", ROOT / "externals" / "dflash"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from AMR_DFlash.config import load_config, resolve_env_path, resolve_run_root  # noqa: E402
from AMR_DFlash.artifacts import canonical_hash, sha256_file  # noqa: E402
from AMR_DFlash.checkpoint import contract_identity  # noqa: E402
from AMR_DFlash.inference import AMRDFlashEngine  # noqa: E402
from AMR_DFlash.pipeline import (  # noqa: E402
    _input_path,
    _split_for,
    _tokenize_prompt,
    capture_run,
    generate_candidates_for_run,
    evaluate_fixed_run,
    label_candidates,
    read_records_for_pipeline,
    select_records_for_split,
    train_compressor,
    train_selector,
    validate_document_records,
)
from AMR_DFlash.runtime import load_runtime  # noqa: E402
from AMR_DFlash.prompts import PROMPT_POLICY  # noqa: E402
from AMR_DFlash.training_data import prepare_manifest  # noqa: E402


DEFAULT_CONFIG = ROOT / "src" / "AMR_DFlash" / "configs" / "pilot.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("AMR_CONFIG", str(DEFAULT_CONFIG)))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="check offline model assets and runtime")
    preflight.add_argument("--require-b200", action="store_true")

    prepare = subparsers.add_parser("prepare-data", help="convert regenerated ShareGPT/ArXiv into an AMR pilot manifest")
    prepare.add_argument("--train-input", action="append", required=True)
    prepare.add_argument("--validation-input", action="append", required=True)
    prepare.add_argument("--holdout-input", action="append", default=[])
    prepare.add_argument("--tokenizer-path")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--train-samples", type=int, default=200)
    prepare.add_argument("--validation-samples", type=int, default=50)
    prepare.add_argument("--holdout-samples", type=int, default=0)
    prepare.add_argument("--min-input-tokens", type=int, default=4097)
    prepare.add_argument("--max-input-tokens", type=int, default=16384)
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--resume", action="store_true")

    capture = subparsers.add_parser("capture", help="capture target features and verifier-consistent states")
    _add_data_args(capture)
    capture.add_argument("--run-root")
    capture.add_argument("--resume", action="store_true")
    capture.add_argument("--max-states-per-document", type=int)

    candidates = subparsers.add_parser("candidates", help="build fixed-budget support candidates")
    candidates.add_argument("--run-root")

    label = subparsers.add_parser("label", help="label candidate supports with the exact target verifier")
    label.add_argument("--run-root")

    selector = subparsers.add_parser("train-selector", help="fit acceptance preference selector")
    selector.add_argument("--run-root")
    selector.add_argument("--checkpoint-in", default=os.environ.get("AMR_CHECKPOINT"))
    selector.add_argument("--checkpoint-out")
    selector.add_argument("--steps", type=int)

    compressor = subparsers.add_parser("train-compressor", help="fit compressed slots with verifier-logit alignment")
    compressor.add_argument("--run-root")
    compressor.add_argument("--checkpoint-in", default=os.environ.get("AMR_CHECKPOINT"), required=False)
    compressor.add_argument("--checkpoint-out")
    compressor.add_argument("--steps", type=int)

    infer = subparsers.add_parser("infer", help="run greedy AMR-DFlash or its dense/selection ablation")
    _add_data_args(infer)
    infer.add_argument("--checkpoint", default=os.environ.get("AMR_CHECKPOINT"))
    infer.add_argument("--mode", choices=("dense", "selection", "compressor", "amr"))
    infer.add_argument(
        "--split", choices=("train", "validation", "holdout", "all"), default="all",
        help="filter by explicit or stable document split before applying --max-samples",
    )
    infer.add_argument("--output")
    infer.add_argument("--disable-cost-gate", action="store_true")
    infer.add_argument("--overwrite", action="store_true")

    fixed_eval = subparsers.add_parser(
        "evaluate-fixed", help="measure acceptance on captured states with a hard memory policy"
    )
    fixed_eval.add_argument("--run-root")
    fixed_eval.add_argument(
        "--split", choices=("train", "validation", "holdout", "all"), default="validation"
    )
    fixed_eval.add_argument("--mode", choices=("dense", "selection", "compressor", "amr"))
    fixed_eval.add_argument("--checkpoint", default=os.environ.get("AMR_CHECKPOINT"))
    fixed_eval.add_argument("--output")
    fixed_eval.add_argument("--max-states", type=int)
    fixed_eval.add_argument("--overwrite", action="store_true")

    return parser


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-input-tokens", type=int)


def _preflight(config: dict[str, Any], device_name: str, *, require_b200: bool) -> dict[str, Any]:
    target_path = resolve_env_path(config, "target_model_env")
    draft_path = resolve_env_path(config, "draft_model_env")
    assert target_path is not None and draft_path is not None
    for name, path in (("target", target_path), ("draft", draft_path)):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} model directory does not exist: {path}")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{name} model config is missing: {path / 'config.json'}")
        weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
        if not weights:
            index_files = list(path.glob("*.safetensors.index.json")) + list(path.glob("*.bin.index.json"))
            if not index_files:
                raise FileNotFoundError(f"{name} model weights are missing from {path}")
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from dflash.model import DFlashDraftModel

    target_config = AutoConfig.from_pretrained(target_path, local_files_only=True)
    draft_config = AutoConfig.from_pretrained(draft_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    dflash_cfg = getattr(draft_config, "dflash_config", {}) or {}
    layer_ids = dflash_cfg.get("target_layer_ids")
    if not layer_ids:
        from dflash.model import build_target_layer_ids

        layer_ids = build_target_layer_ids(
            int(getattr(target_config, "num_hidden_layers")),
            int(getattr(draft_config, "num_hidden_layers")),
        )
    if int(getattr(draft_config, "num_hidden_layers", 0)) != 5:
        raise ValueError("B200 preflight: DFlash checkpoint must have exactly five draft layers")
    if int(getattr(draft_config, "block_size", 0)) != 16:
        raise ValueError("B200 preflight: DFlash checkpoint must have block_size=16")
    if any(int(value) < 0 or int(value) >= int(target_config.num_hidden_layers) for value in layer_ids):
        raise ValueError(f"DFlash target feature layer ids are invalid: {layer_ids}")
    if not torch.cuda.is_available():
        if require_b200 or device_name.startswith("cuda"):
            raise RuntimeError("CUDA is unavailable; run this preflight on the B200 server")
        accelerator = {"available": False, "device": "cpu-local-only"}
    else:
        capability = torch.cuda.get_device_capability(0)
        device_title = torch.cuda.get_device_name(0)
        accelerator = {
            "available": True,
            "device": device_title,
            "capability": list(capability),
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        }
        if require_b200 and capability[0] < 10:
            raise RuntimeError(f"expected B200/Blackwell capability >= 10, found {device_title} {capability}")
    input_path = None
    data_samples_checked = 0
    try:
        input_path = str(_input_path(config, None))
    except (ValueError, FileNotFoundError):
        if require_b200:
            raise ValueError(
                "B200 preflight requires AMR_DATA_MANIFEST or DATA_FILE to point to an existing JSONL"
            )
    if input_path is not None:
        checked_records = read_records_for_pipeline(Path(input_path), max_samples=None)
        validate_document_records(checked_records, config)
        if not checked_records:
            raise ValueError(f"AMR preflight input has no records: {input_path}")
        data_samples_checked = len(checked_records)
    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": accelerator,
        "target_model": str(target_path.resolve()),
        "draft_model": str(draft_path.resolve()),
        "target_feature_layers": [int(value) for value in layer_ids],
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "draft_layers": int(draft_config.num_hidden_layers),
        "block_size": int(draft_config.block_size),
        "dflash_class": f"{DFlashDraftModel.__module__}.{DFlashDraftModel.__name__}",
        "input_manifest": input_path,
        "data_samples_checked": data_samples_checked,
        "local_files_only": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _run_inference(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from common import io_util, rouge
    from common.qwen3_paired import capture_generation_output
    from common.paired_reference import prompt_hash

    source_path = _input_path(config, args.input)
    max_samples = args.max_samples
    if max_samples is None and os.environ.get("MAX_SAMPLES"):
        max_samples = int(os.environ["MAX_SAMPLES"])
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", config["data"].get("max_new_tokens", 256)))
    max_input_tokens = args.max_input_tokens
    if max_input_tokens is None:
        max_input_tokens = int(os.environ.get("MAX_INPUT_TOKENS", config["data"].get("max_input_tokens", 0)))
    if os.environ.get("SMOKE", "0") == "1":
        smoke_samples = int(os.environ.get("SMOKE_MAX_SAMPLES", "1"))
        smoke_tokens = int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "32"))
        max_samples = min(max_samples, smoke_samples) if max_samples else smoke_samples
        max_new_tokens = min(max_new_tokens, smoke_tokens)
    mode = args.mode or str(config["inference"].get("mode", "amr"))
    checkpoint = args.checkpoint
    if mode != "dense" and not checkpoint:
        raise ValueError("selection/amr inference requires --checkpoint from train-selector or train-compressor")
    records = read_records_for_pipeline(
        source_path,
        max_samples=None,
    )
    records = select_records_for_split(
        records,
        split=args.split,
        config=config,
        max_samples=max_samples,
    )
    if not records:
        raise ValueError(f"no input records found for split={args.split!r}")
    target, tokenizer, draft, memory, metadata = load_runtime(
        config,
        device=torch.device(args.device),
        checkpoint_path=checkpoint,
    )
    engine = AMRDFlashEngine(
        target,
        tokenizer,
        draft,
        memory,
        device=torch.device(args.device),
        mode=mode,
        use_cost_gate=not args.disable_cost_gate and bool(config["inference"].get("use_cost_gate", True)),
    )
    input_digest = sha256_file(source_path)
    checkpoint_digest = sha256_file(checkpoint) if checkpoint else None
    workload = {
        "input_manifest_sha256": input_digest, "target_sha256": metadata["target_fingerprint"]["sha256"],
        "dtype": metadata["dtype"], "attention_backend": metadata["attention_backend"],
        "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens,
        "split": args.split, "max_samples": max_samples, "temperature": 0.0, "batch_size": 1,
        "prompt_policy": PROMPT_POLICY,
    }
    run_config = {**workload, "mode": mode, "use_cost_gate": engine.use_cost_gate,
                  "model_metadata": contract_identity(metadata),
                  "memory_checkpoint_sha256": checkpoint_digest,
                  "seed": int(config["training"].get("seed", 17)),
                  "timing_contract": "amr_engine_v2"}
    provenance = {
        "input_manifest_sha256": input_digest, "memory_checkpoint_sha256": checkpoint_digest,
        "run_config_hash": canonical_hash(run_config), "workload_hash": canonical_hash(workload),
        "run_id": os.environ.get(str(config["output"].get("run_id_env", "AMR_RUN_ID"))) or canonical_hash(run_config)[:20],
        "seed": run_config["seed"], "run_config": run_config,
        "device": str(engine.device), "measurement_scope": "amr_dflash_batch1_engine_v2",
        "stage_timing_policy": "cuda_events_sample_boundary_sync" if engine.device.type == "cuda" else "cpu_wall_clock",
        "ttft_policy": "first_token_ready_after_feature_projection",
        "tpot_policy": "decode_ms_div_output_tokens_minus_one; null_if_no_decode_interval",
        "accept_length_policy": "avg_accept_length=raw_accepted_proposals_plus_one_before_eos_or_cap",
        "memory_build_policy": "bootstrap_selection_slots_mask_and_incremental_keys_slots_bank_updates",
    }
    output_value = args.output or os.environ.get("OUTPUT_FILE", "")
    if output_value:
        output_path = Path(output_value).expanduser()
        if not output_path.is_absolute():
            output_path = ROOT / output_path
    else:
        output_path = resolve_run_root(config, create=True) / "evaluation" / f"{mode}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_path}; pass --overwrite to replace it")
        output_path.unlink()
    writer = io_util.JsonlWriter(output_path)
    output_rows: list[dict[str, Any]] = []
    eos_value = getattr(target.config, "eos_token_id", None)
    started = time.perf_counter()
    for index, sample in enumerate(records, start=1):
        prompt, input_ids = _tokenize_prompt(
            tokenizer, sample, max_input_tokens=max_input_tokens
        )
        result = engine.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_value,
        )
        generated_ids = result.output_ids[:, input_ids.shape[1] :]
        output = capture_generation_output(tokenizer, generated_ids)
        output_count = int(generated_ids.shape[1])
        duration_s = max(result.e2e_ms / 1000.0, 1e-9)
        acceptance_rate = (
            result.accepted_tokens / result.proposed_tokens
            if result.proposed_tokens
            else None
        )
        avg_accept = (
            sum(result.acceptance_lengths) / len(result.acceptance_lengths)
            if result.acceptance_lengths
            else None
        )
        mean_accepted_proposals = (
            sum(result.accepted_proposals) / len(result.accepted_proposals)
            if result.accepted_proposals
            else None
        )
        committed_decode_tokens = sum(row["emitted_tokens"] for row in result.trace)
        decode_time_ms = result.decode_ms
        decode_throughput = (
            committed_decode_tokens / (decode_time_ms / 1000.0)
            if decode_time_ms > 0
            else None
        )
        retained = (
            sum(row["raw_tokens"] + row["slot_tokens"] for row in result.trace) / len(result.trace)
            if result.trace
            else int(input_ids.shape[1])
        )
        raw = sample.get("raw") or {}
        dataset = str(raw.get("dataset") or source_path.stem)
        record: dict[str, Any] = {
            **provenance,
            "record_type": "generation",
            "status": "success",
            "sample_id": str(sample["id"]),
            "document_id": str(sample["id"]),
            "source_document_id": str(raw.get("source_document_id") or raw.get("document_id") or raw.get("source_id") or sample["id"]),
            "split": _split_for(raw, str(sample["id"]), config),
            "prompt_hash": prompt_hash(input_ids),
            "reference_output": sample.get("reference"),
            "method": f"amr_dflash_{mode}",
            "dataset": dataset,
            "model": str(resolve_env_path(config, "target_model_env")),
            "input_tokens": int(input_ids.shape[1]),
            "retained_tokens": round(retained, 3),
            "output_tokens": output_count,
            "batch_size": 1,
            "selector_latency_ms": round(result.selector_ms, 3),
            "memory_build_latency_ms": round(result.selector_ms, 3),
            "memory_update_latency_ms": round(result.memory_update_ms, 3),
            "feature_projection_latency_ms": round(result.feature_projection_ms, 3),
            "prefill_ms": round(result.prefill_ms, 3),
            "decode_ms": round(result.decode_ms, 3),
            "ttft_ms": round(result.ttft_ms, 3),
            "tpot_ms": round(decode_time_ms / (output_count - 1), 3) if output_count > 1 else None,
            "e2e_ms": round(result.e2e_ms, 3),
            "throughput_tok_s": round(output_count / duration_s, 3),
            "qps": round(1000.0 / result.e2e_ms, 6) if result.e2e_ms else None,
            "peak_memory_gb": round(result.peak_memory_gb, 3),
            "avg_accept_length": round(avg_accept, 4) if avg_accept is not None else None,
            "mean_accepted_proposals": round(mean_accepted_proposals, 4)
            if mean_accepted_proposals is not None
            else None,
            "mean_committed_tokens_per_round": round(
                committed_decode_tokens / len(result.trace), 4
            )
            if result.trace
            else None,
            "accepted_proposals_raw_total": sum(result.accepted_proposals),
            "accepted_proposals_committed_total": result.accepted_tokens,
            "committed_tokens_total": output_count,
            "decode_committed_tokens": committed_decode_tokens,
            "verification_calls": len(result.trace),
            "decode_time_ms": round(decode_time_ms, 3),
            "decode_committed_tok_s": round(decode_throughput, 3)
            if decode_throughput is not None
            else None,
            "acceptance_rate": round(acceptance_rate, 4) if acceptance_rate is not None else None,
            "draft_latency_ms": round(result.draft_ms, 3),
            "verification_latency_ms": round(result.verification_ms, 3),
            "rejected_draft_ratio": round(1.0 - acceptance_rate, 4) if acceptance_rate is not None else None,
            "generated_token_ids": output["generated_token_ids"],
            "text": output["quality_text"],
            "full_output_text": output["full_output_text"],
            "eos_generated": output["eos_generated"],
            "mode_gate_bypassed": sum(row["bypassed"] for row in result.trace) > 0,
            "raw_budget": memory.config.raw_budget,
            "slot_budget": memory.config.num_slots if mode in {"compressor", "amr"} else 0,
            "target_fingerprint": metadata["target_fingerprint"]["sha256"],
            "draft_fingerprint": metadata["draft_fingerprint"]["sha256"],
            "state_trace": result.trace,
        }
        missing = io_util.validate_schema(record, spec=True)
        if missing:
            raise ValueError(f"AMR-DFlash generation record is missing schema fields: {missing}")
        rouge.add_rouge(record, output["quality_text"], sample.get("reference"))
        writer.add(record)
        output_rows.append(record)
        print(
            f"[amr-infer] {index}/{len(records)} id={sample.get('id')} "
            f"input={input_ids.shape[1]} output={output_count} "
            f"accept={record['avg_accept_length']} e2e={record['e2e_ms']:.1f}ms",
            flush=True,
        )
    elapsed_s = time.perf_counter() - started
    numeric_keys = (
        "input_tokens",
        "retained_tokens",
        "output_tokens",
        "selector_latency_ms",
        "memory_build_latency_ms",
        "memory_update_latency_ms",
        "feature_projection_latency_ms",
        "prefill_ms",
        "decode_ms",
        "ttft_ms",
        "tpot_ms",
        "e2e_ms",
        "throughput_tok_s",
        "avg_accept_length",
        "mean_accepted_proposals",
        "mean_committed_tokens_per_round",
        "decode_time_ms",
        "decode_committed_tok_s",
        "acceptance_rate",
        "draft_latency_ms",
        "verification_latency_ms",
        "peak_memory_gb",
    )
    summary: dict[str, Any] = {
        **provenance,
        "samples": len(output_rows),
        "mode": mode,
        "elapsed_s": round(elapsed_s, 3),
        "output": str(output_path),
        "target_fingerprint": metadata["target_fingerprint"]["sha256"],
        "draft_fingerprint": metadata["draft_fingerprint"]["sha256"],
    }
    decode_ms_total = sum(row["decode_time_ms"] for row in output_rows)
    e2e_ms_total = sum(row["e2e_ms"] for row in output_rows)
    summary["decode_committed_tokens"] = sum(row["decode_committed_tokens"] for row in output_rows)
    summary["decode_time_ms_total"] = round(decode_ms_total, 3)
    summary["decode_committed_tok_s"] = round(summary["decode_committed_tokens"] * 1000 / decode_ms_total, 3) if decode_ms_total else None
    summary["throughput_tok_s"] = round(sum(row["output_tokens"] for row in output_rows) * 1000 / e2e_ms_total, 3) if e2e_ms_total else None
    for key in numeric_keys:
        values = [float(row[key]) for row in output_rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[f"mean_{key}"] = round(sum(values) / len(values), 4)
    summary.update(rouge.aggregate_rouge(output_rows))
    writer.finalize({"record_type": "summary", **summary})
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    _apply_environment_overrides(config)
    device = torch.device(args.device)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    if args.command == "preflight":
        _preflight(config, args.device, require_b200=args.require_b200)
        return 0
    if args.command == "prepare-data":
        result = prepare_manifest(
            inputs={"train": args.train_input, "validation": args.validation_input, "holdout": args.holdout_input},
            output_dir=args.output_dir,
            tokenizer_path=args.tokenizer_path or resolve_env_path(config, "target_model_env"),
            limits={"train": args.train_samples, "validation": args.validation_samples, "holdout": args.holdout_samples},
            min_input_tokens=args.min_input_tokens, max_input_tokens=args.max_input_tokens,
            seed=args.seed, resume=args.resume,
        )
    elif args.command == "capture":
        result = capture_run(
            config,
            input_path=args.input,
            run_root=args.run_root,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            max_states_per_document=args.max_states_per_document,
            device=device,
            resume=args.resume,
        )
    elif args.command == "candidates":
        result = generate_candidates_for_run(config, run_root=args.run_root)
    elif args.command == "label":
        result = label_candidates(config, run_root=args.run_root, device=device)
    elif args.command == "train-selector":
        result = train_selector(
            config,
            run_root=args.run_root,
            checkpoint_in=args.checkpoint_in,
            checkpoint_out=args.checkpoint_out,
            steps=args.steps,
            device=device,
        )
    elif args.command == "train-compressor":
        if not args.checkpoint_in:
            raise ValueError("train-compressor requires --checkpoint-in selector checkpoint")
        result = train_compressor(
            config,
            run_root=args.run_root,
            checkpoint_in=args.checkpoint_in,
            checkpoint_out=args.checkpoint_out,
            steps=args.steps,
            device=device,
        )
    elif args.command == "infer":
        result = _run_inference(args, config)
    elif args.command == "evaluate-fixed":
        result = evaluate_fixed_run(
            config,
            run_root=args.run_root,
            split=args.split,
            mode=args.mode or str(config["inference"].get("mode", "amr")),
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            max_states=args.max_states,
            overwrite=args.overwrite,
            device=device,
        )
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def _apply_environment_overrides(config: dict[str, Any]) -> None:
    mode = os.environ.get("AMR_MODE")
    if mode:
        if mode not in {"dense", "selection", "compressor", "amr"}:
            raise ValueError("AMR_MODE must be dense, selection, compressor, or amr")
        config["inference"]["mode"] = mode
    for env_name, config_name in (
        ("AMR_RAW_BUDGET", "raw_budget"),
        ("AMR_NUM_SLOTS", "num_slots"),
        ("AMR_LOCAL_WINDOW", "local_window"),
        ("AMR_QUERY_WINDOW", "query_window"),
        ("AMR_INDEX_DIM", "index_dim"),
        ("AMR_MIN_CONTEXT_TOKENS", "min_context_tokens"),
    ):
        value = os.environ.get(env_name)
        if value:
            config["memory"][config_name] = int(value)
    from AMR_DFlash.config import memory_config

    memory_config(config)


if __name__ == "__main__":
    raise SystemExit(main())
