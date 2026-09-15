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
    status = "PASS" if pf.get("status") == "PASS" and rs["sample_rows"] > 0 else "BLOCKED / INCOMPLETE"

    lines = [
        f"# Báo cáo SpecExtend Horizon-CMR — {level}",
        "",
        "## Trạng thái kết luận",
        "",
        f"- **Status:** `{status}`",
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
        lines += ["", "| sample | input | output | decode ms | throughput | avg accept | cycles |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rs["sample_records"]:
            lines.append("| {sample_id} | {input_tokens} | {output_tokens} | {decode_ms} | {throughput_tok_s} | {avg_accept_length} | {cycle_count} |".format(**{key: fmt(row.get(key)) for key in ("sample_id", "input_tokens", "output_tokens", "decode_ms", "throughput_tok_s", "avg_accept_length", "cycle_count")}))
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

    lines += [
        "",
        "## 7. Diễn giải và không diễn giải",
        "",
        "Nếu preflight/inference bị block, đây là báo cáo triển khai và kiểm chứng "
        "khả năng chạy, không phải bằng chứng H1 pass/fail. Đặc biệt không được dùng "
        "smoke cũ hoặc CPU để lấp vào kết quả 4K/8K.",
        "",
        "Trace hiện tại là chunk-level aggregate của last target layer tại các "
        "retrieval checkpoints. Nó đủ để kiểm tra instrumentation và tạo oracle "
        "pilot khi có GPU, nhưng không thay thế full attention-head study.",
        "",
        "## 8. Artifact và reproducibility",
        "",
        f"- Preflight: `{preflight}`",
        f"- Raw result: `{result}`",
        f"- Trace: `{trace}`",
        f"- Oracle summary: tạo offline từ `{trace}` bằng `horizon_oracle.py`.",
        "",
        "## 9. PASS/FAIL hiện tại",
        "",
        "- Baseline SpecExtend 4K/8K: chỉ PASS khi có measured successful samples ở đúng level.",
        "- H1 horizon oracle: `NOT EVALUATED` nếu không có target attention trace hoặc không có baseline completed.",
        "- Predictor/architecture: chưa được mở.",
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
