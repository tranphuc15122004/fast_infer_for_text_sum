"""CLI orchestrator for P0–P4 DFlash residual-headroom experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .alignment import compare_acceptance
from .io import join_selection_trace, read_acceptance_records, read_selection_jsonl, read_trace_jsonl, write_metrics_bundle
from .p1_task_regime import analyze_task_regimes
from .p2_coverage import analyze_coverage
from .p3_headroom import analyze_headroom
from .p4_interaction import analyze_interaction
from .plotting import plot_coverage_heatmap, plot_recovery_by_context
from .report import render_markdown_report
from .schema import SCHEMA_VERSION


def _write_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    path = output_dir / "run_manifest.json"
    if not path.exists():
        path.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase_status(metrics: Mapping[str, Any]) -> str:
    status = str(metrics.get("status", "unavailable")).lower()
    return status


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _load_base_rows(path: str | Path) -> list[dict[str, Any]]:
    return [row for row in read_trace_jsonl(path) if row.get("status") == "ok"]


def _prepare_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.trace:
        raise ValueError("--trace is required for this phase")
    rows = _load_base_rows(args.trace)
    if args.dflash2_selection:
        selections = read_selection_jsonl(args.dflash2_selection)
        rows = join_selection_trace(rows, selections)
    return rows


def _write_phase(
    output_dir: Path,
    phase: str,
    metrics: Mapping[str, Any],
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    plot: bool = False,
) -> None:
    phase_dir = output_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    report = render_markdown_report(phase, metrics)
    write_metrics_bundle(phase_dir, metrics, tables=tables, report=report)
    if plot and metrics.get("status") == "ok":
        plot_coverage_heatmap(metrics, phase_dir / "recall_heatmap.png")
    if phase == "p3" and metrics.get("status") == "ok":
        plot_recovery_by_context(metrics, phase_dir / "rho_by_context.png")


def _run_p0(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if not args.official or not args.custom:
        raise ValueError("--official and --custom are required for p0")
    official = read_acceptance_records(args.official, run_id="official")
    custom = read_acceptance_records(args.custom, run_id="custom")
    official_metadata = _read_optional_metadata(args.official_manifest)
    custom_metadata = _read_optional_metadata(args.custom_manifest)
    metrics = compare_acceptance(
        official,
        custom,
        min_blocks=args.min_blocks,
        mat_tolerance=args.mat_tolerance,
        official_metadata=official_metadata,
        custom_metadata=custom_metadata,
    )
    _write_phase(output_dir, "p0", metrics)
    return metrics


def _read_optional_metadata(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    manifest = Path(path)
    if not manifest.is_file():
        raise ValueError(f"manifest not found: {manifest}")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest}")
    return value


def run_synthetic(path: str | Path, *, documents: int = 6) -> Path:
    """Write a deterministic trace fixture with short/long coverage decay."""

    if documents < 2:
        raise ValueError("documents must be at least 2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(17)
    rows: list[dict[str, Any]] = []
    for doc_index in range(documents):
        context_length = 1024 if doc_index < documents // 2 else 9000
        sample_id = f"synthetic-{doc_index}"
        for position in range(1, 5):
            hit = position <= 2 if context_length < 2000 else position == 1
            target = 100 + position
            candidates = [target, 900 + position] if hit else [800 + position, 801 + position]
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "run_id": "synthetic",
                "sample_id": sample_id,
                "document_id": sample_id,
                "dataset": "synthetic",
                "task_regime": "canonical" if doc_index == 0 else "cnn_dm",
                "context_length": context_length,
                "context_bin": "0-2k" if context_length < 2000 else "8-16k",
                "round_index": 0,
                "draft_position": position,
                "max_depth": 4,
                "target_token_id": target,
                "target_token_source": "verifier_posterior",
                "candidate_token_ids": candidates,
                "candidate_logits": [2.0, 1.0],
                "dflash_selected_token_id": candidates[0],
                "accepted_draft_len": 2 if context_length < 2000 else 1,
                "block_size": 5,
                "native_block_size": 5,
            })
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    phases: dict[str, dict[str, Any]] = {}
    if args.phase == "p0":
        try:
            phases["p0"] = _run_p0(args, output_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            phases["p0"] = _unavailable(str(exc))
            _write_phase(output_dir, "p0", phases["p0"])
    else:
        try:
            rows = _prepare_rows(args)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            requested = ("p1", "p2", "p3", "p4") if args.phase == "all" else (args.phase,)
            for phase in requested:
                phases[phase] = _unavailable(str(exc))
                _write_phase(output_dir, phase, phases[phase])
            rows = None
        if rows is None:
            pass
        elif args.phase in {"p1", "all"}:
            phases["p1"] = analyze_task_regimes(
                rows,
                min_relative_drop=args.relative_drop_gate,
                min_documents=args.min_documents,
            )
            _write_phase(output_dir, "p1", phases["p1"])
        if rows is not None and args.phase in {"p2", "all"}:
            phases["p2"] = analyze_coverage(
                rows,
                recall_k=args.oracle_k,
                min_relative_drop=args.relative_drop_gate,
                min_documents=args.min_documents,
            )
            _write_phase(output_dir, "p2", phases["p2"], tables={"coverage": phases["p2"].get("table", [])}, plot=True)
        if rows is not None and args.phase in {"p3", "all"}:
            phases["p3"] = analyze_headroom(
                rows,
                oracle_k=args.oracle_k,
                min_blocks=args.min_blocks,
            )
            _write_phase(output_dir, "p3", phases["p3"])
        if rows is not None and args.phase in {"p4", "all"}:
            phases["p4"] = analyze_interaction(
                rows,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                min_documents=args.min_documents,
            )
            _write_phase(output_dir, "p4", phases["p4"])
    status_values = [_phase_status(metrics) for metrics in phases.values()]
    overall = "ok" if any(status == "ok" or status == "pass" for status in status_values) else "unavailable"
    bundle_status = phases["p0"].get("status", overall) if args.phase == "p0" else overall
    bundle = {
        "schema_version": "dflash_residual.analysis.v1",
        "status": bundle_status,
        "phase": args.phase,
        "phases": phases,
    }
    report_lines = [f"# DFlash residual analysis — {args.phase}", "", f"**Trạng thái tổng:** `{overall.upper()}`", ""]
    for phase, metrics in phases.items():
        report_lines.append(f"- `{phase}`: `{str(metrics.get('status', 'unavailable')).upper()}`")
    report_lines.append("")
    write_metrics_bundle(output_dir, bundle, report="\n".join(report_lines))
    _write_manifest(output_dir, {
        "schema_version": "dflash_residual.run.manifest.v1",
        "phase": args.phase,
        "trace": args.trace,
        "official": args.official,
        "custom": args.custom,
        "dflash2_selection": args.dflash2_selection,
        "status": bundle_status,
    })
    for phase, metrics in phases.items():
        print(f"phase {phase}: {str(metrics.get('status', 'unavailable')).upper()}", flush=True)
    return 0 if overall == "ok" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p0", "p1", "p2", "p3", "p4", "all"), required=True)
    parser.add_argument("--trace", default=None)
    parser.add_argument("--official", default=None)
    parser.add_argument("--custom", default=None)
    parser.add_argument("--official-manifest", default=None)
    parser.add_argument("--custom-manifest", default=None)
    parser.add_argument("--dflash2-selection", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-k", type=int, default=16)
    parser.add_argument("--min-blocks", type=int, default=5)
    parser.add_argument("--mat-tolerance", type=float, default=0.15)
    parser.add_argument("--relative-drop-gate", type=float, default=0.15)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--min-documents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
