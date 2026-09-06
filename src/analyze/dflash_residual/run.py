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
from .prefix_gap import analyze_matched_context, analyze_prefix_oracle, analyze_rank_ambiguity
from .plotting import plot_coverage_heatmap, plot_recovery_by_context
from .report import render_markdown_report
from .schema import SCHEMA_VERSION
from .source_disambiguation import (
    analyze_leave_one_dataset_out,
    analyze_leave_one_dataset_out_from_ladder_metrics,
    analyze_source_ladder,
    analyze_source_strata,
    analyze_target_near_ties,
    annotate_source_rows,
    annotate_source_phrase_rows,
    build_source_index,
)


def _write_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    path = output_dir / "run_manifest.json"
    if not path.exists():
        path.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase_status(metrics: Mapping[str, Any]) -> str:
    status = str(metrics.get("status", "unavailable")).lower()
    return status


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _prefix_tables(phase: str, metrics: Mapping[str, Any]) -> dict[str, Sequence[Mapping[str, Any]]]:
    if phase == "e1":
        rows: list[dict[str, Any]] = []
        for regime, comparison in metrics.get("pairwise", {}).items():
            rows.append({"comparison": regime, **comparison})
        return {"matched_context": rows}
    if phase == "e2":
        rows = []
        for group, group_metrics in metrics.get("groups", {}).items():
            for k, values in group_metrics.get("k_values", {}).items():
                rows.append({
                    "group": group,
                    "k": int(k),
                    "documents": group_metrics.get("documents"),
                    "blocks": group_metrics.get("blocks"),
                    "mat_d": group_metrics.get("mat_d"),
                    "mat_oracle": values.get("mat_oracle"),
                    "oracle_headroom_over_dflash": values.get("oracle_headroom_over_dflash"),
                })
        return {"prefix_oracle": rows}
    if phase == "e3":
        rows = []
        for regime, regime_metrics in metrics.get("regimes", {}).items():
            histogram = regime_metrics.get("rank_histogram", {})
            for rank in range(1, 17):
                rows.append({
                    "regime": regime,
                    "rank": rank,
                    "count": histogram.get(str(rank), 0),
                    "rows": regime_metrics.get("rows"),
                    "top16_hit_rows": regime_metrics.get("top16_hit_rows"),
                })
        return {"rank_distribution": rows}
    return {}


def _load_base_rows(path: str | Path, *, dataset_filter: str | None = None) -> list[dict[str, Any]]:
    return [
        row for row in read_trace_jsonl(path, dataset_filter=dataset_filter)
        if row.get("status") == "ok"
    ]


def _prepare_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.trace:
        raise ValueError("--trace is required for this phase")
    rows = _load_base_rows(args.trace, dataset_filter=args.dataset_filter)
    if args.dflash2_selection:
        selections = read_selection_jsonl(args.dflash2_selection)
        rows = join_selection_trace(rows, selections)
    return rows


def _load_source_records(paths: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"source record at {path}:{line_number} is not an object")
            records.append(record)
    return records


def _load_ladder_metrics(paths: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    """Load either compact E7 bundles or their single-dataset inner metrics."""

    ladder_metrics: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        dataset_name = str(payload.get("dataset", Path(path).parent.name))
        if dataset_name not in {"cnn_dm", "govreport", "multi_news"}:
            for known_name in ("cnn_dm", "govreport", "multi_news"):
                if known_name in str(path):
                    dataset_name = known_name
                    break
        if "mat_d" not in payload:
            datasets = payload.get("datasets", {})
            if isinstance(datasets, Mapping) and len(datasets) == 1:
                inner_name, inner_payload = next(iter(datasets.items()))
                if isinstance(inner_payload, Mapping):
                    payload = dict(inner_payload)
                    dataset_name = str(payload.get("dataset", dataset_name or inner_name))
        ladder_metrics[dataset_name] = payload
    return ladder_metrics


def _prepare_source_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not args.source_jsonl:
        raise ValueError("--source-jsonl is required for source-conditioned phases")
    if not args.tokenizer:
        raise ValueError("--tokenizer is required for source-conditioned phases")
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required to tokenize source records") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    relevant_tokens: dict[str, set[int]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        token_set = relevant_tokens.setdefault(sample_id, set())
        token_set.add(int(row["target_token_id"]))
        token_set.update(int(token) for token in row.get("candidate_token_ids", []))
    records = _load_source_records(args.source_jsonl)
    records = [
        record for record in records
        if str(record.get("id", record.get("sample_id", record.get("document_id")))) in relevant_tokens
    ]
    source_index = build_source_index(
        records,
        lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        token_filter_by_sample=relevant_tokens,
    )
    annotated = annotate_source_rows(rows, source_index, copy_rows=False)
    return annotate_source_phrase_rows(annotated, source_index)


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
            requested = (
                ("p1", "p2", "p3", "p4") if args.phase == "all"
                else ("e1", "e2", "e3") if args.phase == "next"
                else ("e6", "e7", "e8", "e9", "e10") if args.phase == "source-next"
                else (args.phase,)
            )
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
        if rows is not None and args.phase in {"e1", "next"}:
            phases["e1"] = analyze_matched_context(
                rows,
                context_cap=args.matched_context_cap,
                relative_drop_gate=args.matched_drop_gate,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                min_documents=args.min_documents,
            )
            _write_phase(output_dir, "e1", phases["e1"], tables=_prefix_tables("e1", phases["e1"]))
        if rows is not None and args.phase in {"e2", "next"}:
            prefix_rows = rows
            if args.prefix_context_cap is not None:
                prefix_rows = [
                    row for row in rows
                    if int(row.get("context_cap", row.get("context_length", -1))) == args.prefix_context_cap
                ]
            phases["e2"] = analyze_prefix_oracle(
                prefix_rows,
                k_values=args.prefix_k_values,
                min_documents=args.min_documents,
                context_cap=args.prefix_context_cap,
            )
            _write_phase(output_dir, "e2", phases["e2"], tables=_prefix_tables("e2", phases["e2"]))
        if rows is not None and args.phase in {"e3", "next"}:
            phases["e3"] = analyze_rank_ambiguity(
                rows,
                context_cap=args.matched_context_cap,
            )
            _write_phase(output_dir, "e3", phases["e3"], tables=_prefix_tables("e3", phases["e3"]))
        # E10 can consume compact E7 artifacts directly; do not force a
        # second source-tokenization pass when --ladder-metrics is provided.
        if rows is not None and args.phase == "e10" and args.ladder_metrics:
            ladder_metrics = _load_ladder_metrics(args.ladder_metrics)
            phases["e10"] = analyze_leave_one_dataset_out_from_ladder_metrics(
                ladder_metrics,
                lambda_values=args.source_lambda_values,
            )
            _write_phase(output_dir, "e10", phases["e10"])
        elif rows is not None and args.phase in {"e6", "e7", "e8", "e9", "e10", "source-next"}:
            try:
                source_rows = _prepare_source_rows(args, rows)
            except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
                requested = ("e6", "e7", "e8", "e9", "e10") if args.phase == "source-next" else (args.phase,)
                for phase in requested:
                    phases[phase] = _unavailable(str(exc))
                    _write_phase(output_dir, phase, phases[phase])
                source_rows = None
            if source_rows is not None:
                if args.phase in {"e6", "source-next"}:
                    phases["e6"] = analyze_source_strata(source_rows)
                    _write_phase(output_dir, "e6", phases["e6"])
                if args.phase in {"e7", "source-next"}:
                    phases["e7"] = analyze_source_ladder(source_rows, lambda_values=args.source_lambda_values)
                    if args.dataset_filter:
                        phases["e7"]["dataset"] = args.dataset_filter
                    _write_phase(output_dir, "e7", phases["e7"])
                if args.phase in {"e8", "source-next"}:
                    phases["e8"] = analyze_source_ladder(source_rows, lambda_values=args.source_lambda_values)
                    phases["e8"]["experiment"] = "E8"
                    if args.dataset_filter:
                        phases["e8"]["dataset"] = args.dataset_filter
                    _write_phase(output_dir, "e8", phases["e8"])
                if args.phase in {"e9", "source-next"}:
                    phases["e9"] = analyze_target_near_ties(source_rows, near_tie_margin=args.near_tie_margin)
                    _write_phase(output_dir, "e9", phases["e9"])
                if args.phase in {"e10", "source-next"}:
                    if args.ladder_metrics:
                        ladder_metrics = _load_ladder_metrics(args.ladder_metrics)
                        phases["e10"] = analyze_leave_one_dataset_out_from_ladder_metrics(
                            ladder_metrics,
                            lambda_values=args.source_lambda_values,
                        )
                    else:
                        phases["e10"] = analyze_leave_one_dataset_out(
                            source_rows,
                            lambda_values=args.source_lambda_values,
                        )
                    _write_phase(output_dir, "e10", phases["e10"])
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
    parser.add_argument(
        "--phase",
        choices=("p0", "p1", "p2", "p3", "p4", "e1", "e2", "e3", "e6", "e7", "e8", "e9", "e10", "next", "source-next", "all"),
        required=True,
    )
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
    parser.add_argument("--matched-context-cap", type=int, default=1024)
    parser.add_argument("--matched-drop-gate", type=float, default=0.20)
    parser.add_argument("--prefix-context-cap", type=int, default=None)
    parser.add_argument("--prefix-k-values", type=lambda value: tuple(int(item.strip()) for item in value.split(",") if item.strip()), default=(1, 4, 8, 16))
    parser.add_argument("--source-jsonl", action="append", default=[])
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--source-lambda-values", type=lambda value: tuple(float(item.strip()) for item in value.split(",") if item.strip()), default=(0.0, 0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--near-tie-margin", type=float, default=0.5)
    parser.add_argument("--dataset-filter", default=None)
    parser.add_argument("--ladder-metrics", action="append", default=[])
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
