#!/usr/bin/env python3
"""Smoke/full adapter for the vendored FAFO implementation.

FAFO's public upstream runner is GSM8K-oriented and expects ``question`` /
``answer`` JSONL.  The adapter converts one or more unified repository records
to that input, creates an isolated FAFO config, invokes the upstream runner,
and normalizes its log metrics into the shared JSONL schema.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from common import io_util, metrics, rouge, verify
from common.data_loader import load_records
from common.paths import ROOT
from common.reproducibility import seed_everything


FAFO_ROOT = ROOT / "externals" / "FAFO"
FAFO_PIPELINE_MAIN = "pipeline/fafo/main.py"
FAFO_SMOKE_MAX_NEW_TOKENS = 8


def _config_path(kv_method: str) -> Path:
    return (
        FAFO_ROOT
        / "config/pipeline_config/fafo/gsm8k/Llama-3.1-8B-Instruct"
        / kv_method
        / "default.json"
    )


def build_pipeline_config(
    model: str,
    max_new_tokens: int,
    kv_method: str = "stream-llm",
    use_flash: bool = False,
    max_input_tokens: int = 0,
) -> dict[str, Any]:
    """Load an upstream FAFO config and apply run-specific values."""

    if kv_method not in {"stream-llm", "quest"}:
        raise ValueError(f"unsupported FAFO KV method: {kv_method}")
    config = json.loads(_config_path(kv_method).read_text(encoding="utf-8"))
    params = config["pipeline_params"]
    params["model_name"] = model
    params["n_new_tokens"] = max(1, int(max_new_tokens))
    params["use_flash"] = bool(use_flash)
    params["max_input_tokens"] = max(0, int(max_input_tokens))
    return config


def build_eval_config(dataset_path: str, max_new_tokens: int = 32) -> dict[str, Any]:
    """Build the upstream GSM8K evaluator config for a generated JSONL file."""

    return {
        "eval_params": {
            "dataset": "gsm8k",
            "dataset_path": dataset_path,
            "max_new_tokens": max(1, int(max_new_tokens)),
            "eval_metrics": ["throughput", "avg_acceptance_len"],
        },
        "management": {
            "sub_dir": {
                "input_config": "input_config/",
                "raw_results": "raw_results.json",
                "output_config": "output_config.json",
            }
        },
    }


def build_command(
    *,
    python: str,
    pipeline_config: str,
    eval_config: str,
    output_dir: str,
    exp_desc: str,
) -> list[str]:
    """Build the upstream FAFO command used by the adapter."""

    return [
        python,
        FAFO_PIPELINE_MAIN,
        "--exp_desc",
        exp_desc,
        "--pipeline_config_dir",
        pipeline_config,
        "--eval_config_dir",
        eval_config,
        "--output_folder_dir",
        output_dir,
    ]


def _resolve_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def generated_tokens_within_budget(generated_tokens: int, max_new_tokens: int) -> bool:
    """Return whether public generated tokens stayed within our budget."""

    return 0 < int(generated_tokens) <= max(1, int(max_new_tokens))


def build_fafo_sample_records(
    source_records: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    *,
    method: str,
    dataset: str,
    model: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Convert FAFO's optional per-sample sidecar into canonical records.

    ``OVERALL GEN`` is intentionally not used here: it counts internal FAFO
    lookahead work.  The sidecar is written after FAFO clamps the returned
    sequence to the public generation budget, so its output token count is the
    benchmark-visible value.
    """

    output: list[dict[str, Any]] = []
    for row in stats_rows:
        index = int(row.get("sample_index", len(output)))
        if index >= len(source_records):
            continue
        source = source_records[index]
        if str(source.get("id", "")).startswith("__fafo_warmup__"):
            continue
        output_tokens = int(row.get("output_tokens") or 0)
        e2e_ms = row.get("e2e_ms")
        decode_ms = row.get("decode_ms", e2e_ms)
        if e2e_ms is not None:
            e2e_ms = round(float(e2e_ms), 3)
        if decode_ms is not None:
            decode_ms = round(float(decode_ms), 3)
        reference = source.get("reference") or source.get("answer")
        text = row.get("text")
        record = {
            "method": method,
            "dataset": dataset,
            "task_type": source.get("raw", {}).get("task_type")
            or source.get("task_type"),
            "status": "success"
            if output_tokens > 0 and output_tokens <= max(1, int(max_new_tokens))
            else "failed",
            "scope": "sample",
            "measurement_scope": "e2e_only",
            "model": model,
            "input_tokens": row.get("input_tokens"),
            "retained_tokens": None,
            "output_tokens": output_tokens or None,
            "batch_size": 1,
            "selector_latency_ms": None,
            "ttft_ms": row.get("ttft_ms"),
            "prefill_ms": row.get("prefill_ms"),
            "decode_ms": decode_ms,
            "tpot_ms": round(decode_ms / output_tokens, 3)
            if decode_ms is not None and output_tokens > 0
            else None,
            "e2e_ms": e2e_ms,
            "throughput_tok_s": round(output_tokens / (e2e_ms / 1000.0), 3)
            if e2e_ms is not None and e2e_ms > 0 and output_tokens > 0
            else None,
            "decode_throughput_tok_s": round(output_tokens / (decode_ms / 1000.0), 3)
            if decode_ms is not None and decode_ms > 0 and output_tokens > 0
            else None,
            "qps": None,
            "peak_memory_gb": row.get("peak_memory_gb"),
            "avg_accept_length": row.get("avg_accept_length"),
            "acceptance_rate": None,
            "draft_latency_ms": None,
            "verification_latency_ms": None,
            "rejected_draft_ratio": None,
            "sample_id": source.get("id", index),
            "text": text,
            "reference_output": reference,
            "sample_index": index,
            "lookahead_tokens": row.get("lookahead_tokens"),
        }
        if text and reference:
            if record["task_type"] == "code_completion":
                metrics.add_code_completion(record, text, reference)
            else:
                rouge.add_rouge(record, text, reference)
                metrics.add_semantic(record, text, reference)
        output.append(record)
    return output


def prepare_fafo_records(
    records: list[dict[str, Any]], *, smoke: bool
) -> list[dict[str, Any]]:
    """Add one hidden compile warmup when one real request is benchmarked.

    FAFO excludes the first request only when the evaluator receives more than
    one question.  A one-sample representative run therefore used to include
    torch.compile/flex-attention setup in the measured latency, even though a
    multi-sample run did not.  Keep the warmup independent of ``smoke`` so the
    timing contract is stable for both profiles.
    """

    del smoke  # retained in the API for callers/tests that pass profile state
    if len(records) != 1:
        return records
    warmup = dict(records[0])
    warmup["id"] = f"__fafo_warmup__{records[0]['id']}"
    return [warmup, *records]


def resolve_smoke_budget(max_new_tokens: int) -> int:
    """Use the repository-wide smoke generation budget for fair pairing."""

    return min(max(1, int(max_new_tokens)), FAFO_SMOKE_MAX_NEW_TOKENS)


def _write_fafo_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(
                {
                    "question": record["prompt"],
                    "answer": str(record.get("reference") or record.get("answer") or ""),
                },
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    entries = [str(FAFO_ROOT), str(ROOT / "scripts")]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _parse_log(log: str) -> dict[str, float | int | None]:
    """Extract timing metrics from FAFO's per-step or aggregate log format."""

    time_match = re.findall(r"time:\s*([0-9.eE+-]+)", log, flags=re.IGNORECASE)
    generated_match = re.findall(
        r"generated\s+tokens\s*:\s*([0-9]+)",
        log,
        flags=re.IGNORECASE,
    )
    lookahead_match = re.findall(
        r"OVERALL\s+GEN\s*:\s*([0-9]+)", log, flags=re.IGNORECASE
    )
    stat_match = re.findall(
        r"\bSTAT\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9.eE+-]+)\s*\]",
        log,
        flags=re.IGNORECASE,
    )
    average_match = re.findall(
        r"AVERAGE\s+THROUGHPUT2\s+([0-9.eE+-]+)", log,
        flags=re.IGNORECASE,
    )
    e2e_s = float(time_match[-1]) if time_match else None
    output_tokens = int(generated_match[-1]) if generated_match else None
    lookahead_tokens = int(lookahead_match[-1]) if lookahead_match else None
    throughput = float(average_match[-1]) if average_match else None
    if stat_match:
        _, _, stat_tokens, stat_time = stat_match[-1]
        # ``OVERALL GEN`` includes the hidden compile warmup that the smoke
        # adapter prepends.  The last STAT row is the measured request and is
        # therefore the authoritative token/time pair for this record.
        output_tokens = int(stat_tokens)
        if e2e_s is None:
            e2e_s = float(stat_time)
        if throughput is None and e2e_s > 0:
            throughput = output_tokens / e2e_s
    if throughput is None and e2e_s and output_tokens is not None:
        throughput = output_tokens / e2e_s
    return {
        "e2e_s": e2e_s,
        "output_tokens": output_tokens,
        "lookahead_tokens": lookahead_tokens,
        "throughput": throughput,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("FAFO_MODEL", os.environ.get("MODEL_TARGET", "meta-llama/Llama-3.1-8B-Instruct")))
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--prompt", default="What is 2 + 2? Give the answer briefly.")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=0,
        help="truncate the upstream prompt before FAFO inference (0 = no limit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("LONG_BENCH_SEED", "42")),
    )
    parser.add_argument("--kv-method", choices=["stream-llm", "quest"], default="stream-llm")
    parser.add_argument("--use-flash", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    seed_everything(args.seed)
    if args.smoke:
        args.max_samples = 1
        args.max_new_tokens = resolve_smoke_budget(args.max_new_tokens)

    data_file = _resolve_repo_path(args.data_file)
    if data_file:
        records = load_records(data_file, args.max_samples)
    else:
        records = [{"id": "prompt", "prompt": args.prompt, "reference": None}]
    if not records:
        raise SystemExit("FAFO input contains no usable records")

    output_path = Path(args.output)
    writer = io_util.JsonlWriter(output_path)
    runtime_dir = output_path if output_path.is_absolute() else ROOT / output_path
    runtime_dir = runtime_dir.parent / f"{runtime_dir.stem}_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    process_returncode = 1
    process_log = ""
    raw_result: Any = None
    stats_file = runtime_dir / "sample_metrics.jsonl"
    stats_file.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="fafo-input-") as temp_dir:
        temp_root = Path(temp_dir)
        dataset_path = temp_root / "one_sample.jsonl"
        pipeline_path = temp_root / "pipeline.json"
        eval_path = temp_root / "eval.json"
        runtime_records = prepare_fafo_records(records, smoke=args.smoke)
        _write_fafo_dataset(dataset_path, runtime_records)
        pipeline_path.write_text(
            json.dumps(
                build_pipeline_config(
                    args.model,
                    args.max_new_tokens,
                    args.kv_method,
                    args.use_flash,
                    args.max_input_tokens,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        eval_path.write_text(
            json.dumps(
                build_eval_config(
                    str(dataset_path), max_new_tokens=args.max_new_tokens
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        command = build_command(
            python=os.environ.get("FAST_INFER_PYTHON", sys.executable),
            pipeline_config=str(pipeline_path),
            eval_config=str(eval_path),
            output_dir=str(runtime_dir),
            exp_desc=f"fafo_{args.kv_method}_{'smoke' if args.smoke else 'run'}",
        )
        print("+ " + " ".join(command))
        child_env = _runtime_env()
        child_env["FAFO_SEED"] = str(args.seed)
        child_env["FAFO_STATS_FILE"] = str(stats_file)
        proc = subprocess.run(
            command,
            cwd=FAFO_ROOT,
            env=child_env,
            capture_output=True,
            text=True,
        )
        process_returncode = proc.returncode
        process_log = (proc.stdout or "") + (proc.stderr or "")
        print(process_log[-4000:])

    raw_path = runtime_dir / "raw_results.json"
    if raw_path.is_file():
        try:
            raw_result = json.loads(raw_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw_result = None
    sidecar_rows: list[dict[str, Any]] = []
    if stats_file.is_file():
        for line in stats_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                sidecar_rows.append(row)
    parsed = _parse_log(process_log)

    # Prefer the public per-sample sidecar whenever available.  It is emitted
    # after FAFO clamps the returned sequence; the aggregate log's OVERALL GEN
    # is internal lookahead work and must never be used as output_tokens.
    sample_records = build_fafo_sample_records(
        runtime_records if "runtime_records" in locals() else records,
        sidecar_rows,
        method=f"fafo_{args.kv_method}",
        dataset=data_file.name if data_file else "prompt",
        model=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    if sample_records:
        for sample_record in sample_records:
            sample_record["raw_result"] = raw_result
            sample_record["returncode"] = process_returncode
            sample_record["warmup_injected"] = len(runtime_records) > len(records)
            sample_record["kv_method"] = args.kv_method
            sample_record["smoke"] = args.smoke
            sample_record["log_tail"] = process_log[-3000:]
            writer.add(sample_record)
        expected_samples = len(records)
        successful_samples = sum(
            1 for item in sample_records if item.get("status") == "success"
        )
        checks = [
            (process_returncode == 0, f"FAFO process exit code = {process_returncode}"),
            (len(sample_records) == expected_samples,
             f"per-sample telemetry rows = {len(sample_records)} / {expected_samples}"),
            (successful_samples == expected_samples,
             f"valid public generations = {successful_samples} / {expected_samples}"),
        ]
        summary = {
            "type": "summary",
            "method": f"fafo_{args.kv_method}",
            "returncode": process_returncode,
            "scope": "sample",
            "measurement_scope": "e2e_only",
            "num_samples": expected_samples,
            "successful_samples": successful_samples,
            "checks_passed": all(ok for ok, _ in checks),
            "runtime_dir": str(runtime_dir),
            "lookahead_tokens": parsed.get("lookahead_tokens"),
        }
        writer.finalize(summary)
        io_util.print_table(list(summary.items()))
        print(f"Saved to: {output_path}")
        verify.finish("FAFO", checks)
        return

    e2e_s = parsed["e2e_s"]
    output_tokens = parsed["output_tokens"]
    e2e_ms = round(float(e2e_s) * 1000, 3) if e2e_s is not None else None
    output_number = int(output_tokens) if output_tokens is not None else 0
    budget_ok = generated_tokens_within_budget(output_number, args.max_new_tokens)
    if output_number > args.max_new_tokens:
        print(
            f"[FAFO] public generation returned {output_number} tokens, "
            f"over requested budget {args.max_new_tokens}; marking run failed",
            file=sys.stderr,
            flush=True,
        )
    record = {
        "method": f"fafo_{args.kv_method}",
        "dataset": data_file.name if data_file else "prompt",
        "task_type": records[0].get("raw", {}).get("task_type"),
        "status": "success"
        if process_returncode == 0 and budget_ok
        else "failed",
        "scope": "sample" if len(records) == 1 else "aggregate",
        "model": args.model,
        "input_tokens": None,
        "retained_tokens": None,
        "output_tokens": output_tokens,
        "batch_size": len(records),
        "selector_latency_ms": None,
        "ttft_ms": None,
        "tpot_ms": round(e2e_ms / output_number, 3) if e2e_ms and output_number else None,
        "e2e_ms": e2e_ms,
        "throughput_tok_s": parsed["throughput"],
        "qps": round(1 / float(e2e_s), 6) if e2e_s and e2e_s > 0 else None,
        "peak_memory_gb": None,
        "avg_accept_length": None,
        "acceptance_rate": None,
        "draft_latency_ms": None,
        "verification_latency_ms": None,
        "rejected_draft_ratio": None,
        "sample_ids": [sample["id"] for sample in records],
        "sample_id": records[0]["id"] if len(records) == 1 else None,
        "warmup_injected": len(runtime_records) > len(records),
        "kv_method": args.kv_method,
        "smoke": args.smoke,
        "returncode": process_returncode,
        "raw_result": raw_result,
        "log_tail": process_log[-3000:],
    }
    writer.add(record)

    checks: list[tuple[bool, str]] = [
        (process_returncode == 0, f"FAFO process exit code = {process_returncode}"),
        (output_number > 0, f"generated tokens = {output_number} (> 0)"),
        (
            budget_ok,
            f"generated tokens = {output_number} <= budget {args.max_new_tokens}",
        ),
        (parsed["throughput"] is not None, "FAFO timing/throughput line parsed"),
    ]
    summary = {
        "type": "summary",
        "method": f"fafo_{args.kv_method}",
        "returncode": process_returncode,
        "num_samples": len(records),
        "checks_passed": all(ok for ok, _ in checks),
        "runtime_dir": str(runtime_dir),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {output_path}")
    verify.finish("FAFO", checks)


if __name__ == "__main__":
    main()
