#!/usr/bin/env python3
"""SpecExtend verification / smoke adapter.

Runs SpecExtend's classic or EAGLE path on a JSONL file and normalizes its
human-readable result into this repository's JSONL schema.  The Llama-3.1
configuration uses the EAGLE path; an EAGLE checkpoint is not a classic
``AutoModelForCausalLM`` draft model and must not be passed to ``run_classic``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import os
from pathlib import Path

from common import io_util, metrics, rouge, verify
from common.paths import ROOT
from common.reproducibility import seed_everything

SPECEXTEND = ROOT / "externals" / "SpecExtend" / "specextend"


def _load_stats_file(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def build_specextend_sample_record(
    *,
    source: dict,
    stats: dict,
    method: str,
    dataset: str,
    model: str,
    draft_model: str | None,
    script: str,
    returncode: int,
) -> dict:
    """Build one canonical record from the optional upstream telemetry row."""

    output_tokens = int(stats.get("output_tokens") or 0)
    decode_ms = stats.get("decode_ms")
    e2e_ms = stats.get("e2e_ms")
    prefill_ms = stats.get("prefill_ms")
    if decode_ms is not None:
        decode_ms = round(float(decode_ms), 3)
    if e2e_ms is not None:
        e2e_ms = round(float(e2e_ms), 3)
    if prefill_ms is not None:
        prefill_ms = round(float(prefill_ms), 3)
    full_e2e = prefill_ms is not None and e2e_ms is not None
    text = stats.get("text")
    reference = source.get("reference")
    record = {
        "method": method,
        "dataset": dataset,
        "model": model,
        "draft_model": draft_model,
        "model_name": source.get("model_name"),
        "script": script,
        "retrieval_policy": os.environ.get(
            "SPECEXTEND_RETRIEVAL_POLICY", "cmr32"
        ),
        "task_type": source.get("task_type"),
        "sample_id": source.get("id"),
        "reference_output": reference,
        "text": text,
        "scope": "sample",
        "status": "success" if returncode == 0 and output_tokens > 0 else "failed",
        "measurement_scope": "full_e2e" if full_e2e else "decode_only",
        "input_tokens": stats.get("input_tokens"),
        "retained_tokens": None,
        "output_tokens": output_tokens or None,
        "batch_size": 1,
        "selector_latency_ms": None,
        "ttft_ms": stats.get("ttft_ms", prefill_ms),
        "prefill_ms": prefill_ms,
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
        "peak_memory_gb": stats.get("peak_memory_gb"),
        "returncode": returncode,
        "accept_length_list": stats.get("accept_length_list"),
        "avg_accept_length": stats.get("avg_accept_length"),
        "cycle_count": len(stats.get("accept_length_list", []))
        if isinstance(stats.get("accept_length_list"), list)
        else None,
        "timing": stats.get("timing"),
        "trace_file": stats.get("trace_file"),
    }
    if text and reference:
        if record["task_type"] == "code_completion":
            metrics.add_code_completion(record, text, reference)
        else:
            rouge.add_rouge(record, text, reference)
            metrics.add_semantic(record, text, reference)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default="run_classic.py",
                        choices=["run_classic.py", "run_eagle.py"])
    parser.add_argument("--model-name", default="vicuna_7b",
                        choices=["vicuna_7b", "longchat_7b", "llama3_1_8b"])
    parser.add_argument("--base-model", default=None,
                        help="override base model path/id")
    parser.add_argument("--draft-model", default=None,
                        help="override draft model path/id")
    parser.add_argument("--input-file", default="data/govreport/govreport_512.jsonl")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-gen-len", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=0,
                        help="truncate each input before inference; 0 disables truncation")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("LONG_BENCH_SEED", "42")),
    )
    parser.add_argument("--use-specextend",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="enable/disable SpecExtend hybrid attention")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trace-file", default=None,
                        help="optional aggregate Horizon-CMR JSONL trace path")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    if args.smoke:
        args.max_samples = 1
        args.max_gen_len = min(args.max_gen_len, 64)
        args.warmup_runs = 0

    input_file = Path(args.input_file)
    if not input_file.is_absolute():
        # Direct SpecExtend data lives below externals/SpecExtend, while the
        # representative runner writes converted data below the repo root.
        candidates = (ROOT / input_file, SPECEXTEND / input_file)
        input_file = next((candidate for candidate in candidates if candidate.exists()),
                          candidates[0])

    cmd = [
        sys.executable, args.script,
        "--input_file", str(input_file),
        "--model_name", args.model_name,
        "--max_samples", str(args.max_samples),
        "--max_gen_len", str(args.max_gen_len),
        "--max_input_tokens", str(args.max_input_tokens),
        "--output_result_line",
    ]
    if args.use_specextend:
        cmd += ["--use_specextend"]

    print("+ " + " ".join(cmd))
    env = dict(__import__("os").environ)
    # Model paths are consumed by both SpecExtend entrypoints.  The EAGLE
    # entrypoint uses the draft path as an EAGLE checkpoint, not a classic LLM.
    if args.base_model:
        env["SPECEXTEND_BASE_MODEL"] = args.base_model
    if args.draft_model:
        env["SPECEXTEND_DRAFT_MODEL"] = args.draft_model
    if args.trace_file:
        env["SPECEXTEND_TRACE_FILE"] = str(Path(args.trace_file).resolve())
        env["SPECEXTEND_TRACE_DATASET"] = Path(args.input_file).stem
    env["SPECEXTEND_WARMUP_RUNS"] = str(args.warmup_runs)
    env["SPECEXTEND_MAX_INPUT_TOKENS"] = str(args.max_input_tokens)
    env["SPECEXTEND_SEED"] = str(args.seed)
    runtime_dir = Path(args.output).resolve().parent / (
        Path(args.output).stem + "_runtime"
    )
    stats_file = runtime_dir / "sample_metrics.jsonl"
    env["SPECEXTEND_STATS_FILE"] = str(stats_file)
    proc = subprocess.run(cmd, cwd=SPECEXTEND, env=env,
                          capture_output=True, text=True)
    stdout = proc.stdout or ""
    log = stdout + (proc.stderr or "")
    print(log[-4000:])

    # Both SpecExtend entrypoints print metrics as human-readable lines rather
    # than JSON.
    result_lines = [
        line.strip() for line in stdout.splitlines()
        if line.strip() and "Generated " in line and " tokens in " in line
    ]
    stats_results = []
    for line in stdout.splitlines():
        if line.startswith("SPECEXTEND_STATS_JSON "):
            try:
                stats_results.append(json.loads(line.split(" ", 1)[1]))
            except json.JSONDecodeError:
                continue
    stats_results = _load_stats_file(stats_file) or stats_results
    source_rows = [
        json.loads(line)
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parsed_results: list[tuple[int, float]] = []
    for line in result_lines:
        match = re.search(
            r"Generated\s+(\d+)\s+tokens\s+in\s+([0-9.eE+-]+)s", line
        )
        if match:
            parsed_results.append((int(match.group(1)), float(match.group(2))))
    got_output = bool(parsed_results or stats_results)
    writer = io_util.JsonlWriter(Path(args.output))
    method = "specextend_eagle" if args.script == "run_eagle.py" else "specextend_classic"
    if stats_results and args.script == "run_eagle.py":
        for index, measured in enumerate(stats_results):
            source_index = int(measured.get("sample_index", index))
            source = source_rows[source_index] if source_index < len(source_rows) else {}
            measured = {
                **measured,
                "trace_file": str(Path(args.trace_file).resolve())
                if args.trace_file else None,
            }
            writer.add(
                build_specextend_sample_record(
                    source=source,
                    stats=measured,
                    method=method,
                    dataset=Path(args.input_file).stem,
                    model=args.base_model or args.model_name,
                    draft_model=args.draft_model,
                    script=args.script,
                    returncode=proc.returncode,
                )
            )
    else:
        for index, (generated_tokens, elapsed_s) in enumerate(parsed_results):
            source = source_rows[index] if index < len(source_rows) else {}
            measured = stats_results[index] if index < len(stats_results) else {}
            measured = {
                **measured,
                "output_tokens": generated_tokens,
                "decode_ms": round(elapsed_s * 1000.0, 3),
                "trace_file": str(Path(args.trace_file).resolve())
                if args.trace_file else None,
            }
            writer.add(
                build_specextend_sample_record(
                    source=source,
                    stats=measured,
                    method=method,
                    dataset=Path(args.input_file).stem,
                    model=args.base_model or args.model_name,
                    draft_model=args.draft_model,
                    script=args.script,
                    returncode=proc.returncode,
                )
            )
    if not stats_results and not parsed_results:
        writer.add({
            "method": method,
            "dataset": Path(args.input_file).stem,
            "model": args.base_model or args.model_name,
            "draft_model": args.draft_model,
            "scope": "aggregate",
            "sample_ids": [row.get("id", index) for index, row in enumerate(source_rows)],
            "status": "failed",
            "reason": "SpecExtend emitted no parseable generation timing line",
            "returncode": proc.returncode,
            "result_lines": result_lines[:3],
            "log_tail": log[-2000:],
        })

    checks: list[tuple[bool, str]] = [
        (proc.returncode == 0, f"process exit code = {proc.returncode}"),
        (got_output, f"generated result line(s): {len(result_lines)}"),
    ]
    summary = {"type": "summary", "method": method,
               "returncode": proc.returncode, "got_output": got_output,
               "num_samples": len(stats_results) or len(parsed_results),
               "measurement_scope": (
                   "full_e2e" if stats_results and args.script == "run_eagle.py"
                   else "decode_only"
               )}
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("SpecExtend", checks)


if __name__ == "__main__":
    main()
