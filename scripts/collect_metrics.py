#!/usr/bin/env python3
"""Thu thập + tổng hợp toàn bộ metric (tốc độ + semantic) từ các run baseline
trên data/representative_100.

Input : outputs/representative_100/<baseline>_<dataset>.jsonl (schema §13,
        do scripts/run_representative_100.sh sinh ra; cũng chấp nhận 1 file).
Data  : data/representative_100/<dataset>_representative.jsonl — join reference
        theo record id (doc_id/sample_id/question_id/id) để tính metric semantic.
Output: metrics_summary.json (đầy đủ) + metrics_summary.csv (bảng rộng)
        + metrics_summary.md (báo cáo đọc được) trong --outputs-dir.

Metric tốc độ (mean/median/p90/std của từng key schema §13):
  input_tokens, retained_tokens, output_tokens, selector_latency_ms, ttft_ms,
  tpot_ms, prefill_ms, decode_ms, e2e_ms, pipeline_e2e_ms,
  throughput_tok_s, qps, peak_memory_gb
  + retained_ratio, compression_ratio (suy ra) + key speculative (nếu có).

Paired speedup (ratio of means, dense/reference divided by method):
  ESR (end-to-end), DSR (decode), prefill_speedup, ttft_speedup.
  These are emitted only when both sides of the timing pair are present.

Metric semantic (mean, theo từng text key có trong record):
  ROUGE-1/2/L P/R/F, ROUGE-Lsum, BLEU-1..4, length_ratio
  (common/metrics.py; "base_" prefix cho dense baseline của GemFilter).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from common import io_util, metrics
from common.paths import ROOT

DEFAULT_OUTPUTS_DIR = ROOT / "outputs" / "representative_100"
DEFAULT_DATA_DIR = ROOT / "data" / "representative_100"

# Thứ tự các text key trong record; prefix semantic tương ứng.
TEXT_KEYS = [
    ("summary", ""),
    ("text", ""),
    ("answer", ""),
    ("gemfilter_text", ""),
    ("base_text", "base_"),
]


def load_data_index(data_dir: Path) -> tuple[list[str], dict[str, dict]]:
    """{dataset_name: {record_id: {"reference": .., "document": ..}}}."""
    datasets: list[str] = []
    index: dict[str, dict] = {}
    for f in sorted(data_dir.glob("*_representative.jsonl")):
        name = f.name.removesuffix("_representative.jsonl")
        datasets.append(name)
        by_id: dict = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if rid is not None:
                by_id[str(rid)] = {
                    "reference": rec.get("reference") or rec.get("summary") or rec.get("answer"),
                    "document": rec.get("document") or rec.get("text") or "",
                }
        index[name] = by_id
    return datasets, index


def detect_method_dataset(filename: str, datasets: list[str]) -> tuple[str | None, str | None]:
    """Từ tên file <method>_<dataset>.jsonl suy ra (method, dataset)."""
    name = filename.removesuffix(".jsonl")
    for ds in datasets:
        suffix = "_" + ds
        if name.endswith(suffix):
            return name[: -len(suffix)], ds
    return None, None


def record_id(record: dict) -> str | None:
    for key in ("doc_id", "sample_id", "question_id", "example_id", "id"):
        if record.get(key) is not None:
            return str(record[key])
    return None


def validate_completeness(
    outputs_dir: Path,
    *,
    expected_baselines: Sequence[str],
    expected_datasets: Sequence[str],
    expected_samples: int,
) -> list[str]:
    """Validate the expected representative benchmark matrix.

    A baseline may emit multiple records per source sample (for example,
    semantic-selection emits one record per selector/budget), so completeness
    is measured by unique source IDs rather than raw JSONL line count.
    Summary records are ignored.  The function returns human-readable errors
    instead of raising so callers can report every missing pair at once.
    """
    errors: list[str] = []
    for baseline in expected_baselines:
        for dataset in expected_datasets:
            path = outputs_dir / f"{baseline}_{dataset}.jsonl"
            if not path.is_file():
                errors.append(f"missing output: {path.name}")
                continue

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            records = [
                r for r in records
                if r.get("type") not in ("summary", "longspec_full")
            ]
            ids = {rid for r in records if (rid := record_id(r)) is not None}
            observed = len(ids) if ids else len(records)
            if observed != expected_samples:
                errors.append(
                    f"{path.name}: {observed}/{expected_samples} unique samples"
                )
    return errors


def _split_names(value: str | None) -> list[str]:
    """Parse comma/space-separated CLI names."""
    if not value:
        return []
    return value.replace(",", " ").split()


def normalize_record(record: dict, fallback_method: str) -> dict:
    """Map non-schema semantic-selection fields to the shared metric schema."""
    out = dict(record)
    selector = out.get("selector")

    if fallback_method == "semantic_selection" and selector:
        out["method"] = f"semantic_selection_{selector}"

    aliases = {
        "input_tokens": "original_tokens",
        "retained_tokens": "selected_tokens",
        "selector_latency_ms": "selection_total_wall_ms",
        "ttft_ms": "pipeline_ttft_ms",
        "e2e_ms": "pipeline_e2e_ms",
        "throughput_tok_s": "output_tokens_per_second",
    }
    for target, source in aliases.items():
        if out.get(target) is None and out.get(source) is not None:
            out[target] = out[source]

    # Canonical names for timings from a paired dense/reference run.  The
    # semantic-selection adapter historically called these baseline_full_*;
    # accept both spellings so old JSONL remains usable.
    dense_aliases = {
        "dense_e2e_ms": ("baseline_full_e2e_ms", "baseline_e2e_ms"),
        "dense_ttft_ms": ("baseline_full_ttft_ms", "baseline_ttft_ms"),
        "dense_prefill_ms": (
            "baseline_full_prefill_ms",
            "baseline_prefill_ms",
        ),
        "dense_decode_ms": (
            "baseline_full_decode_ms",
            "baseline_decode_ms",
        ),
    }
    for target, sources in dense_aliases.items():
        if out.get(target) is not None:
            continue
        for source in sources:
            if out.get(source) is not None:
                out[target] = out[source]
                break

    # Existing adapters already measure a dense paired run under a
    # baseline-specific name.  Normalize those names into the shared timing
    # fields used by aggregate_speedup().
    if out.get("dense_e2e_ms") is None and out.get("base_time_s") is not None:
        out["dense_e2e_ms"] = float(out["base_time_s"]) * 1000.0
    if out.get("dense_e2e_ms") is None and out.get("naive_time") is not None:
        # EAGLE's benchmark is decode-only, so naive_time is its dense decode
        # and end-to-end reference in the same measurement domain.
        out["dense_e2e_ms"] = float(out["naive_time"]) * 1000.0
    if out.get("dense_decode_ms") is None and out.get("naive_time") is not None:
        out["dense_decode_ms"] = float(out["naive_time"]) * 1000.0

    if out.get("decode_ms") is None and out.get("eagle_time") is not None:
        out["decode_ms"] = float(out["eagle_time"]) * 1000.0
    if out.get("e2e_ms") is None and out.get("eagle_time") is not None:
        out["e2e_ms"] = float(out["eagle_time"]) * 1000.0
    if out.get("output_tokens") is None and out.get("new_tokens") is not None:
        out["output_tokens"] = out["new_tokens"]
    if out.get("throughput_tok_s") is None and out.get("eagle_tok_s") is not None:
        out["throughput_tok_s"] = out["eagle_tok_s"]

    # LLMLingua reports compressor time separately from target generation.
    # Make the selector-inclusive wall-clock timing available for ESR while
    # preserving the original target-only e2e_ms field.
    if (
        out.get("pipeline_e2e_ms") is None
        and out.get("e2e_ms") is not None
        and out.get("selector_latency_ms") is not None
    ):
        out["pipeline_e2e_ms"] = (
            float(out["e2e_ms"]) + float(out["selector_latency_ms"])
        )

    if out.get("doc_id") is None and out.get("example_id") is not None:
        out["doc_id"] = out["example_id"]

    if out.get("peak_memory_gb") is None:
        peak_mb = out.get("peak_gpu_allocated_mb")
        if peak_mb is not None:
            out["peak_memory_gb"] = float(peak_mb) / 1024.0

    return out


def extract_hypotheses(record: dict) -> list[tuple[str, str]]:
    """[(prefix, text)] cho mọi text key hiện diện trong record."""
    out: list[tuple[str, str]] = []
    for key, prefix in TEXT_KEYS:
        text = record.get(key)
        if isinstance(text, str) and text.strip():
            out.append((prefix, text))
    return out


def compute_group(records: list[dict], data_index: dict) -> dict:
    """Tổng hợp speed + speculative + semantic cho một nhóm (method, dataset)."""
    speed = metrics.aggregate_speed(records)
    spec = metrics.aggregate_speculative(records)
    joined = 0
    for r in records:
        rid = record_id(r)
        ref = (
            data_index.get(rid, {}).get("reference") if rid else None
        ) or r.get("reference")
        if not ref:
            continue
        joined += 1
        for prefix, text in extract_hypotheses(r):
            metrics.add_semantic(r, text, ref, prefix=prefix)

    semantic: dict = {}
    for prefix in ("", "base_"):
        agg = metrics.aggregate_semantic(records, prefix=prefix)
        if agg:
            semantic.update(agg)

    group: dict = {
        "num_records": len(records),
        "num_reference_joined": joined,
    }
    if speed:
        group["speed"] = speed
    if spec:
        group["speculative"] = spec
    speedup = metrics.aggregate_speedup(records)
    if speedup:
        group["speedup"] = speedup
    if semantic:
        group["semantic"] = semantic
    return group


def load_output_files(outputs_dir: Path, datasets: list[str]) -> list[tuple[str, str, list[dict]]]:
    """[(filename_stem, dataset_từ_tên_file, records)] từ mọi file *.jsonl.

    Dataset suy từ tên file <baseline>_<dataset>.jsonl (chuẩn của runner);
    đây là nguồn dataset đáng tin cậy hơn record["dataset"] (nhiều baseline
    ghi hằng số "data-file").
    """
    files = sorted(outputs_dir.glob("*.jsonl")) if outputs_dir.is_dir() else [outputs_dir]
    out: list[tuple[str, str, list[dict]]] = []
    for f in files:
        rows = [
            json.loads(line)
            for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records = [r for r in rows if r.get("type") not in ("summary", "longspec_full")]
        if not records:
            continue
        _m, file_ds = detect_method_dataset(f.stem, datasets)
        out.append((f.stem, file_ds or "", records))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR,
                        help="thư mục chứa outputs/<baseline>_<dataset>.jsonl "
                             "(hoặc 1 file jsonl)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="thư mục data/representative_100 (để join reference)")
    parser.add_argument("--out", type=Path, default=None,
                        help="đường dẫn metrics_summary.json "
                             "(mặc định <outputs-dir>/metrics_summary.json)")
    parser.add_argument("--csv", type=Path, default=None, help="bảng CSV")
    parser.add_argument("--md", type=Path, default=None, help="báo cáo markdown")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--strict", action="store_true",
        help="fail before writing reports when the expected matrix is incomplete",
    )
    parser.add_argument(
        "--expected-baselines", default="",
        help="comma/space-separated baseline names required by --strict",
    )
    parser.add_argument(
        "--expected-datasets", default="",
        help="comma/space-separated dataset names required by --strict",
    )
    parser.add_argument(
        "--expected-samples", type=int, default=None,
        help="unique source samples required per (baseline, dataset)",
    )
    args = parser.parse_args()

    outputs_dir: Path = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise SystemExit(f"outputs dir not found: {outputs_dir}")

    data_dir: Path = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    datasets, data_index = load_data_index(data_dir)
    print(f"Datasets in {data_dir}: {datasets}")

    if args.strict:
        if not args.expected_baselines or not args.expected_datasets:
            raise SystemExit(
                "--strict requires --expected-baselines and --expected-datasets"
            )
        if args.expected_samples is None or args.expected_samples <= 0:
            raise SystemExit("--strict requires --expected-samples > 0")
        completeness_errors = validate_completeness(
            outputs_dir,
            expected_baselines=_split_names(args.expected_baselines),
            expected_datasets=_split_names(args.expected_datasets),
            expected_samples=args.expected_samples,
        )
        if completeness_errors:
            print("Incomplete representative benchmark:", file=sys.stderr)
            for error in completeness_errors:
                print(f"  - {error}", file=sys.stderr)
            raise SystemExit(2)

    # --- gom nhóm (method, dataset) ---------------------------------------
    # dataset ưu tiên lấy từ tên file (runner đặt tên <baseline>_<dataset>.jsonl),
    # method ưu tiên record["method"]; fallback theo tên file.
    resolved: dict[tuple[str, str], list[dict]] = {}
    for stem, file_ds, records in load_output_files(outputs_dir, datasets):
        fallback_method = detect_method_dataset(stem, datasets)[0] or stem
        for rec in records:
            rec = normalize_record(rec, fallback_method)
            method = rec.get("method") or fallback_method
            ds = file_ds or rec.get("dataset") or ""
            # chuẩn hoá dataset dạng "cnn_dailymail_representative.jsonl"
            if ds.endswith("_representative.jsonl"):
                ds = ds.removesuffix("_representative.jsonl")
            resolved.setdefault((method, ds), []).append(rec)

    if not resolved:
        print(f"No run records found under {outputs_dir} (chỉ đếm file *.jsonl top-level)")
        sys.exit(1)

    # --- tính metric cho từng nhóm -----------------------------------------
    result: dict = {
        "outputs_dir": str(outputs_dir),
        "data_dir": str(data_dir),
        "datasets": datasets,
        "baselines": sorted({m for m, _ in resolved}),
        "metrics": {},
        "overall": {},
    }

    for (method, ds), records in sorted(resolved.items()):
        index = data_index.get(ds, {})
        group = compute_group(records, index)
        result["metrics"].setdefault(ds, {})[method] = group
        if args.verbose:
            print(f"  [{ds}/{method}] records={group['num_records']} "
                  f"joined={group['num_reference_joined']}")

    # overall: gộp mọi dataset theo method
    for method in sorted({m for m, _ in resolved}):
        all_records = [r for (m, _), recs in resolved.items() if m == method for r in recs]
        index_all: dict = {}
        for ds in datasets:
            index_all.update(data_index.get(ds, {}))
        result["overall"][method] = compute_group(all_records, index_all)

    out_path = Path(args.out) if args.out else outputs_dir / "metrics_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved JSON metrics: {out_path}")

    csv_path = Path(args.csv) if args.csv else outputs_dir / "metrics_summary.csv"
    write_csv(csv_path, result, datasets)
    md_path = Path(args.md) if args.md else outputs_dir / "metrics_summary.md"
    write_markdown(md_path, result, datasets)

    print("\nTóm tắt (mean):")
    print(f"{'method':<22}{'dataset':<14}{'e2e_ms':>10}{'tok/s':>9}{'rouge2':>8}{'bleu2':>8}{'n':>5}")
    for ds in datasets:
        for method, group in sorted(result["metrics"].get(ds, {}).items()):
            sp = group.get("speed", {})
            se = group.get("semantic", {})
            print(
                f"{method:<22}{ds:<14}"
                f"{sp.get('e2e_ms', {}).get('mean', 0.0):>10.1f}"
                f"{sp.get('throughput_tok_s', {}).get('mean', 0.0):>9.1f}"
                f"{se.get('rouge2_f', 0.0):>8.4f}"
                f"{se.get('bleu2', 0.0):>8.4f}"
                f"{group.get('num_records', 0):>5}"
            )


def write_csv(path: Path, result: dict, datasets: list[str]) -> None:
    """Bảng rộng: 1 dòng / (dataset, method), cột = metric_stat."""
    order: list[tuple[str, str]] = []
    for ds in datasets:
        for method in sorted(result["metrics"].get(ds, {})):
            order.append((ds, method))
    for ds in sorted(result["metrics"].keys()):
        for method in sorted(result["metrics"][ds]):
            if (ds, method) not in order:
                order.append((ds, method))

    col_meta: list[tuple[str, str, str]] = []  # (section, key, stat)
    seen: set[str] = set()
    for ds, method in order:
        group = result["metrics"][ds][method]
        for section in ("speed", "speculative"):
            for key, agg in group.get(section, {}).items():
                for stat in ("mean", "median", "p90", "std"):
                    label = f"{key}_{stat}"
                    if label not in seen:
                        seen.add(label)
                        col_meta.append((section, key, stat))
        for key in group.get("speedup", {}):
            label = f"{key}_ratio"
            if label not in seen:
                seen.add(label)
                col_meta.append(("speedup", key, "ratio"))
        for key in group.get("semantic", {}):
            label = f"{key}_mean"
            if label not in seen:
                seen.add(label)
                col_meta.append(("semantic", key, "mean"))

    header = ["dataset", "method", "num_records", "num_reference_joined"] + [c[1] + "_" + c[2] for c in col_meta]
    lines = [",".join(header)]
    for ds, method in order:
        group = result["metrics"][ds][method]
        row = [ds, method, str(group.get("num_records", 0)), str(group.get("num_reference_joined", 0))]
        for section, key, stat in col_meta:
            if section == "semantic":
                v = group.get("semantic", {}).get(key)
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
            elif section == "speedup":
                v = group.get("speedup", {}).get(key)
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
            else:
                agg = group.get(section, {}).get(key)
                row.append(f"{agg[stat]:.4f}" if agg and stat in agg else "")
        lines.append(",".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved CSV table: {path}")


def _fmt(value, key: str = "mean") -> str:
    """Format số hoặc dict agg (speed: dict{mean,..}; semantic: float)."""
    if isinstance(value, dict):
        v = value.get(key)
    else:
        v = value
    return f"{v:.2f}" if isinstance(v, (int, float)) else "-"


def write_markdown(path: Path, result: dict, datasets: list[str]) -> None:
    md: list[str] = [
        "# Benchmark representative_100 — tổng hợp metric",
        "",
        f"- Outputs: {result['outputs_dir']}",
        f"- Datasets: {', '.join(datasets)}",
        "- Speed: mean/median/p90/std theo schema §13 (metrics_summary.json có đầy đủ).",
        "- Speedup: ratio của mean timing dense/reference chia cho mean timing method (chỉ khi có cặp ghép).",
        "- Semantic: ROUGE-1/2/L P/R/F, ROUGE-Lsum, BLEU-1..4, length ratio (mean).",
        "",
    ]

    def add_tables(md: list[str], label: str, groups: dict) -> None:
        md.append(f"## {label}")
        md.append("")
        md.append("### Tốc độ (mean)")
        md.append("")
        md.append("| method | n | input | ret% | out | sel_ms | prefill | decode | ttft | tpot | e2e | pipeline | tok/s | qps | mem |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for method, group in sorted(groups.items()):
            sp = group.get("speed", {})
            md.append(
                f"| {method} | {group.get('num_records', 0)} "
                f"| {_fmt(sp.get('input_tokens'))} | {_fmt(sp.get('retained_ratio'))} "
                f"| {_fmt(sp.get('output_tokens'))} | {_fmt(sp.get('selector_latency_ms'))} "
                f"| {_fmt(sp.get('prefill_ms'))} | {_fmt(sp.get('decode_ms'))} "
                f"| {_fmt(sp.get('ttft_ms'))} | {_fmt(sp.get('tpot_ms'))} "
                f"| {_fmt(sp.get('e2e_ms'))} | {_fmt(sp.get('pipeline_e2e_ms'))} "
                f"| {_fmt(sp.get('throughput_tok_s'))} "
                f"| {_fmt(sp.get('qps'))} | {_fmt(sp.get('peak_memory_gb'))} |"
            )
        md.append("")
        md.append("### Speedup so với dense/reference (ratio mean)")
        md.append("")
        md.append("| method | ESR | DSR | prefill | TTFT |")
        md.append("|---|---|---|---|---|")
        for method, group in sorted(groups.items()):
            su = group.get("speedup", {})
            md.append(
                f"| {method} | {_fmt(su.get('esr'))} "
                f"| {_fmt(su.get('dsr'))} | {_fmt(su.get('prefill_speedup'))} "
                f"| {_fmt(su.get('ttft_speedup'))} |"
            )
        md.append("")
        md.append("### Semantic (mean)")
        md.append("")
        md.append("| method | r1_f | r1_p | r1_r | r2_f | rL_f | rLsum_f | bleu1 | bleu2 | bleu4 | len_r |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for method, group in sorted(groups.items()):
            se = group.get("semantic", {})
            md.append(
                f"| {method} | {_fmt(se.get('rouge1_f'))} "
                f"| {_fmt(se.get('rouge1_p'))} | {_fmt(se.get('rouge1_r'))} "
                f"| {_fmt(se.get('rouge2_f'))} | {_fmt(se.get('rougeL_f'))} "
                f"| {_fmt(se.get('rougeLsum_f'))} | {_fmt(se.get('bleu1'))} "
                f"| {_fmt(se.get('bleu2'))} | {_fmt(se.get('bleu4'))} | {_fmt(se.get('length_ratio'))} |"
            )
        md.append("")

    for ds in datasets:
        groups = result["metrics"].get(ds, {})
        if groups:
            add_tables(md, f"Dataset: {ds}", groups)

    add_tables(md, "Overall (gộp mọi dataset)", result["overall"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved markdown report: {path}")


if __name__ == "__main__":
    main()
