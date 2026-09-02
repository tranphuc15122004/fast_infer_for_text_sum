"""Reproducible GroundSync experiment phases and artifact orchestration."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import grounding_horizon, js_divergence
from .report import build_hypothesis_report, write_report_artifacts
from .trace_target import (
    _record_document,
    generate_target_trace,
    load_jsonl,
    load_local_model,
    render_document_prompt,
)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("integer list must contain positive values")
    return values


def _event(output_dir: Path, phase: str, status: str, **fields: Any) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "status": status,
        **fields,
    }
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[{phase}] {status} {fields}", flush=True)


def run_synthetic_fixture(output_dir: Path, *, seed: int = 42) -> None:
    """Create a deterministic fixture that exercises all report code paths.

    Fixture rows are diagnostics for the evaluator, not evidence for a Qwen
    model claim.  The manifest labels them explicitly so they cannot be
    mistaken for model-backed results.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    target_rows: list[dict[str, Any]] = []
    speculative_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    max_k = 4
    for document_index in range(12):
        document_id = f"synthetic-{document_index:02d}"
        transition = 8 + (document_index % 5)
        attention: list[dict[str, list[float]]] = []
        entropy: list[float] = []
        for position in range(24):
            if position < transition:
                base = [0.92, 0.08]
            else:
                base = [0.08, 0.92]
            noise = (rng.random() - 0.5) * 0.02
            first = min(max(base[0] + noise, 0.01), 0.99)
            distribution = [first, 1.0 - first]
            attention.append({"raw": distribution, "nosink": distribution})
            entropy.append(0.25 if position in {transition - 1, transition} else 0.1)
        target_rows.append({
            "schema_version": "groundsync.target.v1",
            "status": "ok",
            "sample_id": document_id,
            "document_id": document_id,
            "output_tokens": len(attention),
            "generated_token_ids": list(range(1000 + document_index, 1024 + document_index)),
            "target_entropy": entropy,
            "sentence_boundary": [int(position in {transition - 1, transition}) for position in range(24)],
            "copyability": [int(position < transition) for position in range(24)],
            "attention": attention,
        })
        input_rows.append({"id": document_id, "document": f"Synthetic document {document_index}."})
        trace = [step["nosink"] for step in attention]
        for start in range(0, len(trace) - 4, 2):
            drift = js_divergence(trace[start - 1], trace[start]) if start else 0.01
            accepted = max_k if drift <= 0.2 else 0
            horizon = grounding_horizon(trace, start=start, threshold=0.2, max_horizon=4)
            confidence = [0.93, 0.9, 0.88, 0.86] if accepted else [0.42, 0.4, 0.38, 0.36]
            speculative_rows.append({
                "schema_version": "groundsync.spec.v1",
                "status": "ok",
                "document_id": document_id,
                "start_position": start,
                "max_k": max_k,
                "proposal_token_ids": list(range(10, 10 + max_k)),
                "canonical_token_ids": list(range(10, 10 + accepted)) + [999],
                "draft_confidence": confidence,
                "drift_at_start": drift,
                "accepted_len": accepted,
                "fully_accepted": accepted == max_k,
                "grounding_horizon": horizon,
            })
    _write_jsonl(output_dir / "input.jsonl", input_rows)
    _write_jsonl(output_dir / "target_traces.jsonl", target_rows)
    _write_jsonl(output_dir / "speculative_traces.jsonl", speculative_rows)
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema_version": "groundsync.run.manifest.v1",
            "mode": "synthetic_fixture",
            "seed": seed,
            "model_evidence": False,
            "target_model": None,
            "draft_model": None,
            "target_rows": len(target_rows),
            "speculative_rows": len(speculative_rows),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_analysis(
    output_dir: Path,
    *,
    threshold: float = 0.2,
    horizon_threshold: float = 0.2,
    max_horizon: int = 16,
    max_k: int = 8,
    mode: str | None = None,
) -> dict[str, Any]:
    """Analyze raw JSONL traces and write all report artifacts."""

    target_rows = _read_jsonl(output_dir / "target_traces.jsonl")
    speculative_rows = _read_jsonl(output_dir / "speculative_traces.jsonl")
    timing_rows = None
    timing_path = output_dir / "speculative_timing_traces.jsonl"
    if timing_path.exists():
        timing_rows = _read_jsonl(timing_path)
    report = build_hypothesis_report(
        target_rows,
        speculative_rows,
        timing_speculative_rows=timing_rows,
        threshold=threshold,
        horizon_threshold=horizon_threshold,
        max_horizon=max_horizon,
        max_k=max_k,
    )
    report = {
        "mode": mode or _read_manifest_mode(output_dir),
        "threshold": threshold,
        "horizon_threshold": horizon_threshold,
        "max_horizon": max_horizon,
        "max_k": max_k,
        **report,
    }
    write_report_artifacts(output_dir, report)
    return report


def _read_manifest_mode(output_dir: Path) -> str:
    path = output_dir / "run_manifest.json"
    if not path.exists():
        return "trace_analysis"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("mode", "trace_analysis"))
    except (json.JSONDecodeError, OSError):
        return "trace_analysis"


def run_target_phase(args: argparse.Namespace, output_dir: Path) -> None:
    records = load_jsonl(Path(args.input), limit=args.max_samples)
    target_path = output_dir / "target_traces.jsonl"
    started = time.perf_counter()
    try:
        model, tokenizer, device = load_local_model(
            args.model, device=args.device, dtype=args.dtype
        )
    except Exception as exc:
        rows = [
            {
                "schema_version": "groundsync.target.v1",
                "status": "unavailable",
                "sample_id": str(record.get("id", index)),
                "error": f"{type(exc).__name__}: {exc}",
            }
            for index, record in enumerate(records)
        ]
        _write_jsonl(target_path, rows)
        _event(output_dir, "target", "UNAVAILABLE", error=rows[0]["error"] if rows else "no records")
        return

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample_id = str(record.get("id", index))
        try:
            rendered = render_document_prompt(tokenizer, _record_document(record))
            rows.append(
                generate_target_trace(
                    model,
                    tokenizer,
                    rendered,
                    sample_id=sample_id,
                    document_id=sample_id,
                    max_new_tokens=args.max_new_tokens,
                    chunk_size=args.chunk_size,
                    skip_source_tokens=args.skip_source_tokens,
                    device=device,
                    prefill_chunk_size=args.prefill_chunk_size,
                    sensitivity_chunk_sizes=_parse_int_list(args.sensitivity_chunk_sizes),
                    sink_sizes=_parse_int_list(args.sink_sizes),
                )
            )
            _event(output_dir, "target", "document_ok", sample_id=sample_id, index=index)
        except Exception as exc:
            rows.append({
                "schema_version": "groundsync.target.v1",
                "status": "error",
                "sample_id": sample_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            _event(output_dir, "target", "document_error", sample_id=sample_id, error=str(exc))
    _write_jsonl(target_path, rows)
    (output_dir / "target_manifest.json").write_text(
        json.dumps({
            "schema_version": "groundsync.target.manifest.v1",
            "model": args.model,
            "device": args.device,
            "requested_samples": len(records),
            "ok_samples": sum(row.get("status") == "ok" for row in rows),
            "chunk_size": args.chunk_size,
            "skip_source_tokens": args.skip_source_tokens,
            "sensitivity_chunk_sizes": _parse_int_list(args.sensitivity_chunk_sizes),
            "sink_sizes": _parse_int_list(args.sink_sizes),
            "prefill_chunk_size": args.prefill_chunk_size,
            "elapsed_s": time.perf_counter() - started,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_speculative_phase(args: argparse.Namespace, output_dir: Path) -> None:
    from .trace_speculative import run_one_speculative_trace

    records = load_jsonl(Path(args.input), limit=args.max_samples)
    target_rows = {
        str(row.get("sample_id")): row
        for row in _read_jsonl(output_dir / "target_traces.jsonl")
        if row.get("status") == "ok"
    }
    try:
        verification_model, verification_tokenizer, verification_device = load_local_model(
            args.model, device=args.device, dtype=args.dtype
        )
        model, tokenizer, device = load_local_model(
            args.draft_model, device=args.device, dtype=args.dtype
        )
    except Exception as exc:
        rows = [{
            "schema_version": "groundsync.spec.v1",
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }]
        _write_jsonl(output_dir / "speculative_traces.jsonl", rows)
        _event(output_dir, "speculative", "UNAVAILABLE", error=rows[0]["error"])
        return

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample_id = str(record.get("id", index))
        try:
            target = target_rows[sample_id]
            rendered = render_document_prompt(tokenizer, _record_document(record))
            verification_rendered = render_document_prompt(
                verification_tokenizer, _record_document(record)
            )
            rows.extend(
                run_one_speculative_trace(
                    model,
                    tokenizer,
                    rendered,
                    target,
                    document_id=sample_id,
                    max_k=args.max_k,
                    max_starts=args.max_starts,
                    stride=args.stride,
                    device=device,
                    horizon_threshold=args.horizon_threshold,
                    verification_model=verification_model,
                    verification_rendered=verification_rendered,
                    verification_device=verification_device,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
            )
            _event(output_dir, "speculative", "document_ok", sample_id=sample_id, index=index)
        except Exception as exc:
            rows.append({
                "schema_version": "groundsync.spec.v1",
                "status": "error",
                "document_id": sample_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            _event(output_dir, "speculative", "document_error", sample_id=sample_id, error=str(exc))
    _write_jsonl(output_dir / "speculative_traces.jsonl", rows)


def _default_input() -> Path:
    root = Path(__file__).resolve().parents[3]
    candidates = [
        root / "data/representative_100/govreport_representative.jsonl",
        root / "data/representative_100/gov_report.jsonl",
        root / "data/longbench_200/gov_report.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("synthetic", "target", "speculative", "analyze", "all"), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Existing result directory for --phase analyze; never overwritten",
    )
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--input", default=str(_default_input()))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--skip-source-tokens", type=int, default=8)
    parser.add_argument("--sensitivity-chunk-sizes", default="64,128,256")
    parser.add_argument("--sink-sizes", default="4,8,16")
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--max-starts", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--horizon-threshold", type=float, default=0.2)
    parser.add_argument("--max-horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _make_output_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        output_dir = Path(args.run_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"result directory does not exist: {output_dir}")
        return output_dir
    root = Path(args.output_root)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    output_dir = root / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty result directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    if args.run_dir is not None and args.phase != "analyze":
        raise ValueError("--run-dir is supported only with --phase analyze")
    output_dir = _make_output_dir(args)
    mode = "synthetic_fixture" if args.phase == "synthetic" else "model_backed"
    if args.run_dir is None:
        (output_dir / "run_manifest.json").write_text(
            json.dumps({
                "schema_version": "groundsync.run.manifest.v1",
                "mode": mode,
                "phase": args.phase,
                "seed": args.seed,
                "model": args.model,
                "draft_model": args.draft_model,
                "input": args.input,
                "output_dir": str(output_dir),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.phase == "synthetic":
        run_synthetic_fixture(output_dir, seed=args.seed)
        run_analysis(
            output_dir,
            threshold=args.threshold,
            horizon_threshold=args.horizon_threshold,
            max_horizon=args.max_horizon,
            max_k=args.max_k,
            mode="synthetic_fixture",
        )
    elif args.phase == "target":
        run_target_phase(args, output_dir)
    elif args.phase == "speculative":
        run_speculative_phase(args, output_dir)
    elif args.phase == "analyze":
        run_analysis(
            output_dir,
            threshold=args.threshold,
            horizon_threshold=args.horizon_threshold,
            max_horizon=args.max_horizon,
            max_k=args.max_k,
        )
    else:
        run_target_phase(args, output_dir)
        run_speculative_phase(args, output_dir)
        run_analysis(
            output_dir,
            threshold=args.threshold,
            horizon_threshold=args.horizon_threshold,
            max_horizon=args.max_horizon,
            max_k=args.max_k,
            mode="model_backed",
        )
    return 0


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
