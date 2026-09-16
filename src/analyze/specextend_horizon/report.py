"""Build a complete, honest Markdown report for a Horizon-CMR run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .horizon_oracle import analyze_records, load_jsonl
except ImportError:  # direct script execution from the experiment runner
    from analyze.specextend_horizon.horizon_oracle import analyze_records, load_jsonl


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def result_status(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path) if path.is_file() else []
    summaries = [row for row in rows if row.get("type") == "summary"]
    samples = [row for row in rows if row.get("scope") == "sample"]
    return {
        "rows": len(rows),
        "sample_rows": len(samples),
        "summaries": summaries,
        "returncodes": sorted({row.get("returncode") for row in rows if row.get("returncode") is not None}),
        "sample_records": samples,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_report(run_root: Path, level: str, preflight: Path, result: Path, trace: Path) -> str:
    pf = read_json(preflight)
    rs = result_status(result)
    trace_rows = load_jsonl(trace) if trace.is_file() else []
    oracle = analyze_records(trace_rows) if trace_rows else {}
    torch_info = pf.get("torch", {})
    target = pf.get("model_choice", {}).get("target", {})
    draft = pf.get("model_choice", {}).get("draft", {})
    run_status = "SUCCESS" if pf.get("status") == "PASS" and rs["sample_rows"] > 0 else "BLOCKED / INCOMPLETE"
    depth = oracle.get("depth_analysis", {})
    per_depth = depth.get("per_depth", {})
    depth_values = [
        (int(raw_depth), value)
        for raw_depth, value in per_depth.items()
    ]
    depth_values.sort(key=lambda item: item[0])
    missing_by_depth = [
        value.get("mean_missing_mass_fraction")
        for _, value in depth_values
        if isinstance(value.get("mean_missing_mass_fraction"), (int, float))
    ]
    missing_trend = (
        missing_by_depth[-1] - missing_by_depth[0]
        if len(missing_by_depth) >= 2 else None
    )
    correlation = depth.get("correlation_missing_mass_vs_rejection")
    h1_supported = (
        run_status == "SUCCESS"
        and len(depth_values) >= 2
        and isinstance(correlation, (int, float))
        and correlation > 0
        and isinstance(missing_trend, (int, float))
        and missing_trend > 0
    )
    hypothesis_status = "SUPPORTED" if h1_supported else "NOT SUPPORTED"

    lines = [
        f"# Báo cáo SpecExtend Horizon-CMR — {level}",
        "",
        "## Trạng thái kết luận",
        "",
        f"- **Run status:** `{run_status}`",
        f"- **H1 hypothesis:** `{hypothesis_status}`",
        f"- **Run root:** `{run_root}`",
        f"- **Preflight status:** `{pf.get('status', 'N/A')}`",
        f"- **Inference sample records:** `{rs['sample_rows']}`",
        f"- **Trace cycle records:** `{oracle.get('cycle_records', 0)}`",
        "",
        "Báo cáo không coi CPU/fallback là GPU benchmark và không suy diễn số liệu "
        "4K/8K nếu process không khởi tạo CUDA hoặc bị OOM.",
        "",
        "## 1. Câu hỏi và gate đã khóa",
        "",
        "H1 kiểm tra liệu CMR hiện tại chọn context quá myopic cho cả speculative "
        "block hay không. CMR hiện tại được đối chiếu với hindsight horizon context "
        "tạo từ target attention ở các verification queries tương lai. Target-future "
        "information chỉ được dùng offline.",
        "",
        "Gate tiếp tục: accepted-token gain >=20% hoặc additional E2E reduction >=15% "
        "so với SpecExtend. Không có gate nào được đánh giá nếu trace không có "
        "target-attention và baseline không chạy hoàn chỉnh.",
        "",
        "Kết quả H1 chỉ được coi là được hỗ trợ khi missing future mass tăng theo "
        "depth và có quan hệ dương với rejection. `Run status=SUCCESS` chỉ nói job "
        "đã chạy thành công; không đồng nghĩa H1 PASS.",
        "",
        "## 2. Cấu hình thực nghiệm",
        "",
        """| Thành phần | Cấu hình |
|---|---|
| Target | `Vicuna-7B-v1.5-16k` local snapshot |
| Draft | `Vicuna-68M` local snapshot |
| Path | upstream `classic/run_classic.py` |
| Decode | greedy, temperature 0, batch 1 |
| Precision | fp16 |
| Retrieval chunk | 32 tokens |
| Retrieval top-k | 32 chunks |
| Retrieval interval | 4 cycles |
| Seed | 42 |
| Output | 64 smoke / 128 1K / 256 pilot |
| Data | GovReport official SpecExtend JSONL |
| Pilot levels | 512, 1K, 4K, 8K; 16K chưa mở |""",
        f"- Target path tồn tại: `{target.get('exists', False)}`; weight bytes: `{target.get('weight_bytes', 'N/A')}`.",
        f"- Draft path tồn tại: `{draft.get('exists', False)}`; weight bytes: `{draft.get('weight_bytes', 'N/A')}`.",
        f"- Python: `{pf.get('python', {}).get('executable', 'N/A')}`.",
        f"- Torch: `{torch_info.get('version', 'N/A')}`, CUDA available: `{torch_info.get('cuda_available', False)}`, device count: `{torch_info.get('device_count', 0)}`.",
        "",
        "## 3. Quy trình đã thực hiện",
        "",
        "1. Kiểm tra model files, tokenizer, data levels và dependency.",
        "2. Kiểm tra CUDA trước khi khởi động model; nếu fail thì dừng inference.",
        "3. Chạy upstream classic path trước, không thay CMR.",
        "4. Hook telemetry tùy chọn ghi manifest/cycle; hook không quyết định retrieval.",
        "5. Tính overlap và projected cost offline từ JSONL trace.",
        "6. Tách số liệu measured khỏi projected/oracle và ghi gate.",
        "",
        "## 4. Kết quả preflight thực tế",
        "",
        f"- Trạng thái: `{pf.get('status', 'N/A')}`.",
        f"- Dependency: `{json.dumps(pf.get('dependencies', {}), ensure_ascii=False)}`.",
        f"- Data records: `{json.dumps(pf.get('data', {}), ensure_ascii=False)}`.",
        "",
        "## 5. Kết quả inference thực tế",
        "",
        f"- Result rows: `{rs['rows']}`; sample rows: `{rs['sample_rows']}`; return codes: `{rs['returncodes']}`.",
    ]
    if rs["sample_records"]:
        lines += ["", "| sample | input | output | decode ms | tok/s | avg accept | cycles | peak VRAM GiB |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rs["sample_records"]:
            lines.append("| {sample_id} | {input_tokens} | {output_tokens} | {decode_ms} | {throughput_tok_s} | {avg_accept_length} | {cycle_count} | {peak_memory_gb} |".format(**{key: fmt(row.get(key)) for key in ("sample_id", "input_tokens", "output_tokens", "decode_ms", "throughput_tok_s", "avg_accept_length", "cycle_count", "peak_memory_gb")}))
    else:
        lines += ["", "Chưa có sample inference thành công; không có measured acceptance/E2E metric để báo cáo."]

    lines += [
        "",
        "### Số liệu SpecExtend cũ trong repository (tham chiếu, không phải run Horizon-CMR hiện tại)",
        "",
        """| Artifact cũ | Kết quả | Cách sử dụng |
|---|---|---|
| `outputs/gpu_1sample/specextend_xsum.jsonl` | 1 sample XSum: 66 tokens / 8.73 s, average acceptance 1.750 | xác nhận classic path từng chạy trên T4; không dùng cho GovReport/H1 |
| `outputs/representative_100/specextend_cnn_dailymail.jsonl` | 1 sample: 35 tokens / 14.18 s | smoke lịch sử; không có trace |
| `outputs/representative_100/specextend_multinews.jsonl` | 1 sample: 38 tokens / 11.84 s | smoke lịch sử; không có trace |
| `outputs/representative_100/specextend_govreport.jsonl` | 4K attempt failed OOM; attempted allocation 11.96 GiB on 14.57 GiB T4 | evidence giới hạn VRAM, không phải successful baseline |""",
        "Các artifact trên được giữ riêng vì được tạo trước nhánh Horizon-CMR, "
        "dùng cấu hình/log khác và không có current-vs-horizon attention trace.",
    ]

    lines += [
        "",
        "## 6. Horizon oracle và trace",
        "",
        f"- Trace file tồn tại: `{trace.is_file()}`; tổng records: `{len(trace_rows)}`.",
        f"- Cycle records: `{oracle.get('cycle_records', 0)}`; cycles có target attention: `{oracle.get('attention_available_cycles', 0)}`.",
    ]
    overlap = oracle.get("overlap", {})
    projected = oracle.get("projected_cost", {})
    lines += [
        f"- Mean recall CMR hiện tại trong horizon top-k: `{fmt(overlap.get('mean_recall_current_in_horizon'))}`.",
        f"- Mean precision CMR so với horizon: `{fmt(overlap.get('mean_precision_current_vs_horizon'))}`.",
        f"- Mean Jaccard: `{fmt(overlap.get('mean_jaccard'))}`.",
        f"- Mean horizon/current context ratio: `{fmt(projected.get('mean_horizon_to_current_context_ratio'))}`.",
        f"- Mean projected speedup: `{fmt(projected.get('mean_projected_speedup'))}`. "
        "Projection này chỉ scale wall time theo context-token ratio; không phải "
        "actual oracle re-draft.",
        "",
        "### Acceptance curve",
        "",
    ]
    acceptance = oracle.get("acceptance", {})
    if acceptance:
        for dataset, values in acceptance.items():
            lines.append(f"- `{dataset}`: cycles `{values.get('cycles')}`, mean accepted tokens/cycle `{fmt(values.get('mean_accepted_tokens_per_cycle'))}`, survival `{json.dumps(values.get('survival', {}), ensure_ascii=False)}`.")
    else:
        lines.append("Chưa có acceptance curve vì chưa có cycle trace.")

    timing = oracle.get("timing", {})
    if timing:
        lines += [
            "",
            "## 7. H0.5 — Phân rã thời gian đo được",
            "",
            "Các giá trị dưới đây là tổng và trung bình trên cycle trace. "
            "`target verify` là forward target/tree verification; "
            "`verify + next draft` gồm acceptance bookkeeping và block draft tiếp theo; "
            "`retrieval/cache update` là cập nhật full/working draft cache. "
            "Các thành phần không được instrument không được suy diễn thành zero.",
            "",
            "| Thành phần | Tổng giây | Trung bình/cycle | Số quan sát |",
            "|---|---:|---:|---:|",
        ]
        for key, label in (
            ("cycle_time_s", "cycle wall time"),
            ("target_verify_time_s", "target verify"),
            ("verify_and_draft_time_s", "verify + next draft"),
            ("draft_time_s", "draft model"),
            ("retrieval_update_time_s", "retrieval/cache update"),
            ("cycle_overhead_time_s", "cycle overhead"),
        ):
            value = timing.get(key, {})
            lines.append(
                f"| {label} | {fmt(value.get('sum'))} | {fmt(value.get('mean'))} | {fmt(value.get('n'))} |"
            )
        lines += [
            "",
            "Amdahl ceiling chưa được gọi là một speedup thực tế: trace này mới "
            "phân rã baseline. Horizon context hiện chưa được chạy lại với một "
            "policy khác nên overlap/projected ratio không phải E2E result.",
        ]

    depth_points = sum(
        int(value.get("n", 0)) for value in depth.get("per_depth", {}).values()
    )
    lines += [
        "",
        "## 8. H1 — Horizon mismatch theo speculative depth",
        "",
        f"- Cycle có aggregate target attention: `{oracle.get('attention_available_cycles', 0)}`/{oracle.get('cycle_records', 0)}.",
        f"- Depth-points có per-depth target attention: `{depth_points}`.",
        f"- Pearson correlation giữa missing future mass và rejection: `{fmt(depth.get('correlation_missing_mass_vs_rejection'))}`.",
        "",
        "| Depth | N | CMR recall trong horizon | CMR precision | Jaccard | Missing mass | Rejection rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if depth.get("per_depth"):
        for raw_depth, value in sorted(depth["per_depth"].items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {raw_depth} | {value.get('n')} | {fmt(value.get('mean_recall_current_in_horizon'))} | "
                f"{fmt(value.get('mean_precision_current_vs_horizon'))} | {fmt(value.get('mean_jaccard'))} | "
                f"{fmt(value.get('mean_missing_mass_fraction'))} | {fmt(value.get('rejection_rate'))} |"
            )
    else:
        lines.append("| N/A | 0 | N/A | N/A | N/A | N/A | N/A |")
    lines += [
        "",
        "`Missing mass` là tổng attention của các source chunks nằm ngoài current "
        "CMR, chuẩn hóa theo source mass ở depth đó. `Rejection rate` là fraction "
        "cycle có `accepted_tokens < depth`. Các metrics H1 chỉ tính trên cycle "
        "có per-depth target attention; cycle thiếu trace không bị tính overlap bằng 0.",
    ]

    lines += [
        "",
        "## 9. Diễn giải và không diễn giải",
        "",
        "Nếu preflight/inference bị block, đây là báo cáo triển khai và kiểm chứng "
        "khả năng chạy, không phải bằng chứng H1 pass/fail. Đặc biệt không được dùng "
        "smoke cũ hoặc CPU để lấp vào kết quả 4K/8K.",
        "",
        "Trace là last-target-layer attention trên verification-tree queries. "
        "Bản có per-depth fields dùng cho H1; nó vẫn là chunk-level average trên "
        "các branching queries cùng depth, không phải actual oracle re-draft.",
        "",
        "## 10. Artifact và reproducibility",
        "",
        f"- Preflight: `{preflight}`",
        f"- Raw result: `{result}`",
        f"- Trace: `{trace}`",
        f"- Oracle summary: tạo offline từ `{trace}` bằng `horizon_oracle.py`.",
        "",
        "## 11. PASS/FAIL hiện tại",
        "",
        "- Baseline SpecExtend 4K/8K: chỉ PASS khi có measured successful samples ở đúng level.",
        f"- H1 horizon mismatch: `{hypothesis_status}`; missing-mass trend first-to-last depth = `{fmt(missing_trend)}`, Pearson missing-mass/rejection = `{fmt(correlation)}`.",
        "- H2 causal add-back, H3 simple alternatives và H4 actual oracle re-draft: "
        "`NOT RUN`, vì H1 không đạt gate sequential; không được suy diễn projected overlap "
        "thành causal hoặc E2E gain.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(args.run_root, args.level, args.preflight, args.result, args.trace), encoding="utf-8")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
