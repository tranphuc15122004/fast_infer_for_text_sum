"""Run the RECAP-KV V0 trace, simulator and report pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .collector import collect_trace, record_document
from .evaluation import EvaluationConfig, aggregate_results, evaluate_trace
from .lease_collector import collect_lease_trace
from .lease_evaluation import (
    LeaseEvaluationConfig,
    aggregate_lease_results,
    evaluate_lease_trace,
)
from .policy import RecapConfig
from .schema import validate_lease_trace_record, validate_trace_record


DEFAULT_INPUT = "data/representative_100/govreport_representative.jsonl"
DEFAULT_MODEL = "/home/tuantb/models/Qwen3-0.6B"


def load_records(paths: Sequence[str | Path], *, max_samples: int) -> list[tuple[dict[str, Any], str]]:
    result: list[tuple[dict[str, Any], str]] = []
    if not paths:
        raise ValueError("at least one input path is required")
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if max_samples > 0:
            rows = rows[:max_samples]
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index + 1} must be a JSON object")
            result.append((row, path.stem))
    if not result:
        raise ValueError("input files contain no records")
    return result


def _record_dataset(record: Mapping[str, Any], source_name: str) -> str:
    value = record.get("dataset")
    return str(value) if value else source_name


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_report(manifest: Mapping[str, Any], aggregate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# RECAP-KV V0 report",
        "",
        "## Kết luận",
        "",
        f"Scientific gate: **{aggregate.get('status', 'INCONCLUSIVE')}**.",
        "Future-use chỉ là hindsight oracle; trace này chưa triển khai physical KV eviction.",
        "",
        "## Runtime",
        "",
        f"- Model: `{manifest.get('model')}`",
        f"- Device: `{manifest.get('device')}`; CUDA: `{manifest.get('cuda_available')}`",
        f"- Samples: requested={manifest.get('requested_samples')}, ok={manifest.get('ok_samples')}, errors={manifest.get('error_samples')}",
        f"- Block size: `{manifest.get('block_size')}`; max new tokens: `{manifest.get('max_new_tokens')}`",
        "",
        "## Dataset counts",
        "",
    ]
    for dataset, count in sorted(aggregate.get("datasets", {}).items()):
        lines.append(f"- `{dataset}`: {count} valid documents")
    lines.extend(["", "## Per-sample status", ""])
    for row in rows:
        lines.append(
            f"- `{row.get('sample_id', '?')}` / `{row.get('dataset', '?')}`: "
            f"{row.get('status')} ({row.get('reason', '')})"
        )
    if aggregate.get("comparison"):
        lines.extend([
            "",
            "## RECAP vs current attention",
            "",
            "| K | Hidden-index Recall Δ | Attention-residual Recall Δ | Historical Recall Δ | Hidden-index NDCG Δ |",
            "|---:|---:|---:|---:|---:|",
        ])
        for budget, values in sorted(aggregate["comparison"].items(), key=lambda item: int(item[0])):
            recall_ablations = values["recall_at_k"].get("ablations", {})
            lines.append(
                f"| {budget} | {values['recall_at_k']['delta']:.6f} | "
                f"{recall_ablations.get('recap_attention', {}).get('delta', float('nan')):.6f} | "
                f"{recall_ablations.get('historical', {}).get('delta', float('nan')):.6f} | "
                f"{values['ndcg_at_k']['delta']:.6f} |"
            )
    return "\n".join(lines) + "\n"


def render_lease_report(
    manifest: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Render the V2 lease headroom decision without claiming physical speedup."""

    lines = [
        "# RECAP-KV V2 Source-State Lease report",
        "",
        "## Kết luận",
        "",
        f"Physical follow-up gate: **{aggregate.get('status', 'INCONCLUSIVE')}**.",
        "Lease trace chỉ đo certificate/headroom; chưa thay đổi hoặc xóa KV thật.",
        "",
        "## Runtime",
        "",
        f"- Model: `{manifest.get('model')}`",
        f"- Device: `{manifest.get('device')}`; CUDA: `{manifest.get('cuda_available')}`",
        f"- Samples: requested={manifest.get('requested_samples')}, ok={manifest.get('ok_samples')}, errors={manifest.get('error_samples')}",
        f"- Block size: `{manifest.get('block_size')}`; max new tokens: `{manifest.get('max_new_tokens')}`",
        f"- `delta_anchor`: `{manifest.get('delta_anchor')}`; `delta_cert`: `{manifest.get('delta_cert')}`",
        f"- Numerical violation tolerance: `{manifest.get('violation_tolerance')}`",
        "",
        "## Dataset counts",
        "",
    ]
    for dataset, count in sorted(aggregate.get("datasets", {}).items()):
        lines.append(f"- `{dataset}`: {count} valid documents")
    if aggregate.get("metrics"):
        metrics = aggregate["metrics"]
        lines.extend([
            "",
            "## Aggregate lease metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Median lease length | {metrics['median_lease_length']:.6f} |",
            f"| Mean lease length | {metrics['mean_lease_length']:.6f} |",
            f"| Certified cold fraction | {metrics['certified_cold_fraction']:.6f} |",
            f"| Certificate violation rate | {metrics['violation_rate']:.6f} |",
            f"| Mean actual cold mass | {metrics['mean_actual_cold_mass']:.6f} |",
            f"| Mean bound tightness | {metrics['mean_bound_tightness']:.6f} |",
            f"| Refresh rate | {metrics['refresh_rate']:.6f} |",
            f"| Source-score avoidance proxy | {metrics['source_score_avoidance']:.6f} |",
        ])
    if aggregate.get("gate"):
        lines.extend(["", "## Kill gate", "", "| Criterion | Pass |", "|---|:---:|"])
        for name, passed in aggregate["gate"].items():
            lines.append(f"| {name} | {'yes' if passed else 'no'} |")
    if aggregate.get("dataset_metrics"):
        lines.extend([
            "",
            "## Per-dataset lease metrics",
            "",
            "| Dataset | Median lease | Cold fraction | Violation rate | Refresh rate |",
            "|---|---:|---:|---:|---:|",
        ])
        for dataset, metrics in sorted(aggregate["dataset_metrics"].items()):
            lines.append(
                f"| {dataset} | {metrics['median_lease_length']:.6f} | "
                f"{metrics['certified_cold_fraction']:.6f} | "
                f"{metrics['violation_rate']:.6f} | "
                f"{metrics['refresh_rate']:.6f} |"
            )
    lines.extend(["", "## Per-sample status", ""])
    for row in rows:
        lines.append(
            f"- `{row.get('sample_id', '?')}` / `{row.get('dataset', '?')}`: "
            f"{row.get('status')} ({row.get('reason', '')})"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("recap", "lease"), default="recap")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", action="append", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--segment-tokens", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--budgets", default="1,2,4,8")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--residual-floor", type=float, default=0.1)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--delta-anchor", type=float, default=0.05)
    parser.add_argument("--delta-cert", type=float, default=0.10)
    parser.add_argument("--min-cold-fraction", type=float, default=0.20)
    parser.add_argument("--min-lease-length", type=float, default=4.0)
    return parser


def run(args: argparse.Namespace) -> int:
    import torch
    from src.analyze.groundsync.trace_target import load_local_model, render_document_prompt

    started = time.perf_counter()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_samples = 1 if args.smoke else int(args.max_samples)
    max_new_tokens = min(32, int(args.max_new_tokens)) if args.smoke else int(args.max_new_tokens)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    input_paths = list(args.input or [DEFAULT_INPUT])
    records = load_records(input_paths, max_samples=max_samples)
    budgets = tuple(int(value.strip()) for value in str(args.budgets).split(",") if value.strip())
    policy_config = RecapConfig(
        temperature=args.temperature,
        beta=args.beta,
        eta=args.eta,
        residual_floor=args.residual_floor,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    evaluation_config = EvaluationConfig(policy=policy_config, budgets=budgets)
    lease_config = LeaseEvaluationConfig(
        delta_anchor=args.delta_anchor,
        delta_cert=args.delta_cert,
        min_cold_fraction=args.min_cold_fraction,
        min_lease_length=args.min_lease_length,
    )
    manifest: dict[str, Any] = {
        "schema_version": "recap.manifest.v1",
        "experiment": args.experiment,
        "status": "running",
        "model": str(args.model),
        "device": str(args.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "input": [str(path) for path in input_paths],
        "requested_samples": len(records),
        "block_size": int(args.block_size),
        "segment_tokens": int(args.segment_tokens),
        "max_new_tokens": max_new_tokens,
        "seed": int(args.seed),
        "policy": asdict(policy_config),
    }
    if args.experiment == "lease":
        manifest.update({
            "schema_version": "recap.lease.manifest.v1",
            "delta_anchor": lease_config.delta_anchor,
            "delta_cert": lease_config.delta_cert,
            "violation_tolerance": lease_config.violation_tolerance,
            "min_cold_fraction": lease_config.min_cold_fraction,
            "min_lease_length": lease_config.min_lease_length,
        })
    trace_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    lease_trace_rows: list[dict[str, Any]] = []
    lease_evaluation_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = load_local_model(
            args.model, device=args.device, dtype=args.dtype
        )
        manifest["resolved_device"] = str(device)
        if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            manifest["cuda_device_name"] = torch.cuda.get_device_name(device)
        for index, (record, source_name) in enumerate(records):
            sample_id = str(record.get("id", index))
            dataset = _record_dataset(record, source_name)
            try:
                rendered = render_document_prompt(tokenizer, record_document(record))
                if args.experiment == "lease":
                    lease_trace = collect_lease_trace(
                        model,
                        tokenizer,
                        rendered,
                        sample_id=sample_id,
                        dataset=dataset,
                        max_new_tokens=max_new_tokens,
                        block_size=args.block_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                        delta_anchor=lease_config.delta_anchor,
                        delta_cert=lease_config.delta_cert,
                        device=device,
                    )
                    lease_trace_rows.append(validate_lease_trace_record(lease_trace))
                    lease_evaluation_rows.append(
                        evaluate_lease_trace(lease_trace, lease_config)
                    )
                else:
                    trace = collect_trace(
                        model,
                        tokenizer,
                        rendered,
                        sample_id=sample_id,
                        dataset=dataset,
                        max_new_tokens=max_new_tokens,
                        block_size=args.block_size,
                        segment_token_cap=args.segment_tokens,
                        prefill_chunk_size=args.prefill_chunk_size,
                        device=device,
                    )
                    trace_rows.append(validate_trace_record(trace))
                    evaluation_rows.append(evaluate_trace(trace, evaluation_config))
            except Exception as exc:
                schema_version = (
                    "recap.lease.trace.v1"
                    if args.experiment == "lease"
                    else "recap.trace.v1"
                )
                error_row = {
                    "schema_version": schema_version,
                    "status": "error",
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if args.experiment == "lease":
                    lease_trace_rows.append(error_row)
                    lease_evaluation_rows.append({
                        "status": "error",
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "error": error_row["error"],
                    })
                else:
                    trace_rows.append(error_row)
                    evaluation_rows.append({
                        "status": "error",
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "error": error_row["error"],
                    })
            print(f"{args.experiment}_trace {index + 1}/{len(records)} sample={sample_id}", flush=True)
    except Exception as exc:
        manifest.update({
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started,
            "ok_samples": 0,
            "error_samples": len(records),
        })
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report = render_lease_report if args.experiment == "lease" else render_report
        (output_dir / "report.md").write_text(report(manifest, {"status": "INCONCLUSIVE", "datasets": {}}, []), encoding="utf-8")
        return 2

    if args.experiment == "lease":
        aggregate = aggregate_lease_results(lease_evaluation_rows, lease_config)
        manifest.update({
            "status": "ok",
            "ok_samples": sum(row.get("status") == "ok" for row in lease_trace_rows),
            "error_samples": sum(row.get("status") == "error" for row in lease_trace_rows),
            "inconclusive_samples": sum(row.get("status") == "inconclusive" for row in lease_evaluation_rows),
            "aggregate_status": aggregate.get("status"),
            "elapsed_s": time.perf_counter() - started,
        })
        _write_jsonl(output_dir / "lease_trace.jsonl", lease_trace_rows)
        _write_jsonl(output_dir / "lease_metrics.jsonl", lease_evaluation_rows)
        (output_dir / "lease_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output_dir / "lease_report.md").write_text(
            render_lease_report(manifest, aggregate, lease_evaluation_rows),
            encoding="utf-8",
        )
        return 0 if manifest["ok_samples"] else 2

    aggregate = aggregate_results(evaluation_rows)
    manifest.update({
        "status": "ok",
        "ok_samples": sum(row.get("status") == "ok" for row in trace_rows),
        "error_samples": sum(row.get("status") == "error" for row in trace_rows),
        "inconclusive_samples": sum(row.get("status") == "inconclusive" for row in evaluation_rows),
        "aggregate_status": aggregate.get("status"),
        "elapsed_s": time.perf_counter() - started,
    })
    _write_jsonl(output_dir / "trace.jsonl", trace_rows)
    _write_jsonl(output_dir / "metrics.jsonl", evaluation_rows)
    (output_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(manifest, aggregate, evaluation_rows), encoding="utf-8")
    return 0 if manifest["ok_samples"] else 2


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
