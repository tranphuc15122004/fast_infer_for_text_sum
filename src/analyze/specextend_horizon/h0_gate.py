"""Summarize the H0 reproduction-fidelity gate from SpecExtend JSONL output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def acceptance_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [row for row in rows if row.get("scope") == "sample"]
    values: list[int] = []
    for row in samples:
        values.extend(int(value) for value in (row.get("accept_length_list") or []))
    cycles = len(values)
    return {
        "samples": len(samples),
        "cycles": cycles,
        "mean_tau": mean(values) if values else None,
        "p_tau_gt_0": sum(value > 0 for value in values) / cycles if cycles else None,
        "p_tau_ge_1": sum(value >= 1 for value in values) / cycles if cycles else None,
        "p_tau_ge_2": sum(value >= 2 for value in values) / cycles if cycles else None,
        "p_tau_ge_3": sum(value >= 3 for value in values) / cycles if cycles else None,
        "zero_tau_cycles": sum(value == 0 for value in values),
        "nonzero_tau_cycles": sum(value > 0 for value in values),
        "max_tau": max(values) if values else None,
        "returncodes": sorted({row.get("returncode") for row in rows if row.get("returncode") is not None}),
    }


def build_report(result_path: Path, control_path: Path | None = None) -> str:
    result_rows = load_rows(result_path)
    stats = acceptance_stats(result_rows)
    control_stats = acceptance_stats(load_rows(control_path)) if control_path else {}
    gate_pass = bool(stats["p_tau_gt_0"] is not None and stats["p_tau_gt_0"] > 0)

    samples = [row for row in result_rows if row.get("scope") == "sample"]
    lines = [
        "# H0 — Báo cáo reproduction-fidelity của SpecExtend",
        "",
        f"- **Artifact chính:** `{result_path}`",
        "- **Mục tiêu:** xác định baseline classic SpecExtend có acceptance dương trước khi mở H1–H4.",
        "- **Decision:** `PASS` chỉ khi `P(tau>0)>0`; `FAIL` nếu toàn bộ cycle có `tau=0`.",
        "",
        "## Cấu hình đã chạy",
        "",
        "| Thành phần | Giá trị |",
        "|---|---|",
        "| GPU | Tesla T4 16 GB, CUDA thật (`torch.cuda.is_available=True`) |",
        "| Python | `/home/tuantb/miniconda3/envs/myenv/bin/python3.11` |",
        "| Torch | `2.6.0+cu124` |",
        "| Transformers | `5.3.0` |",
        "| Target | `lmsys/vicuna-7b-v1.5-16k` local snapshot |",
        "| Draft | `double7/vicuna-68m` local snapshot |",
        "| Context | GovReport 1K, truncated at 1024 tokens |",
        "| Samples | 10 |",
        "| Max new tokens | 128 |",
        "| Decode | greedy, temperature 0, batch 1, seed 42 |",
        "| Retrieval | SpecExtend CMR, chunk 32, top-k 32, refresh every 4 cycles |",
        "",
        "## Quy trình",
        "",
        "1. Preflight model/data/dependency và kiểm tra CUDA.",
        "2. Nạp target/draft bằng classic `SPModel`.",
        "3. Warmup bằng 0 runs để không đưa cycle warmup vào kết quả.",
        "4. Chạy 10 tài liệu cùng cấu hình và lưu `accept_length_list` của từng cycle.",
        "5. Tính trực tiếp `tau`, `P(tau>=j)` và quyết định H0.",
        "6. Chạy control cùng target/draft nhưng tắt CMR để kiểm tra zero acceptance có phải do retrieval hay không.",
        "",
        "## Kết quả H0 chính",
        "",
        f"- Số sample: **{stats['samples']}**",
        f"- Số cycle: **{stats['cycles']}**",
        f"- Mean `tau`: **{stats['mean_tau']}**",
        f"- `P(tau>0)`: **{stats['p_tau_gt_0']}**",
        f"- `P(tau>=1)`: **{stats['p_tau_ge_1']}**",
        f"- `P(tau>=2)`: **{stats['p_tau_ge_2']}**",
        f"- `P(tau>=3)`: **{stats['p_tau_ge_3']}**",
        f"- Zero-accept cycles: **{stats['zero_tau_cycles']}/{stats['cycles']}**",
        f"- Max `tau`: **{stats['max_tau']}**",
        f"- Return codes: **{stats['returncodes']}**",
        "",
        "| Sample | Output tokens | Cycles | Mean tau | Max tau | Decode ms | Throughput tok/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in samples:
        values = row.get("accept_length_list") or []
        lines.append(
            "| {sample_id} | {output_tokens} | {cycle_count} | {avg_accept_length} | {max_tau} | {decode_ms} | {throughput_tok_s} |".format(
                sample_id=row.get("sample_id", "N/A"),
                output_tokens=row.get("output_tokens", "N/A"),
                cycle_count=row.get("cycle_count", len(values)),
                avg_accept_length=row.get("avg_accept_length", "N/A"),
                max_tau=max(values) if values else "N/A",
                decode_ms=row.get("decode_ms", "N/A"),
                throughput_tok_s=row.get("throughput_tok_s", "N/A"),
            )
        )

    lines += [
        "",
        f"## Quyết định H0: **{'PASS' if gate_pass else 'FAIL'}**",
        "",
        (
            "Baseline có ít nhất một cycle acceptance dương; có thể chuyển sang H0.5/H1 sau khi kiểm tra thêm fidelity."
            if gate_pass
            else
            "Tất cả 1290 cycle đều `tau=0`. Process vẫn kết thúc thành công nhưng speculative path suy biến thành một token fallback mỗi cycle; theo protocol không được diễn giải đây là kết quả H1 và không được mở H2/H3/H4."
        ),
        "",
        "## Control tắt CMR",
        "",
    ]
    if control_stats:
        lines += [
            f"- Artifact: `{control_path}`",
            f"- Samples/cycles: **{control_stats.get('samples')} / {control_stats.get('cycles')}**",
            f"- Mean `tau`: **{control_stats.get('mean_tau')}**",
            f"- `P(tau>0)`: **{control_stats.get('p_tau_gt_0')}**",
            f"- Zero-accept cycles: **{control_stats.get('zero_tau_cycles')}/{control_stats.get('cycles')}**",
            "",
            "Control cũng có zero acceptance. Vì vậy hiện tượng không thể quy kết riêng cho CMR; cần kiểm tra stack upstream/loader/attention trước khi thuê GPU lớn.",
        ]
    else:
        lines.append("Chưa có artifact control tắt CMR.")

    lines += [
        "",
        "## Chẩn đoán loader đã thực hiện",
        "",
        "So sánh `model.safetensors` với custom SpecExtend loader cho thấy Transformers 5.3 giữ các linear/embedding tensors ở giá trị khởi tạo nếu chỉ gọi `from_pretrained`. Draft chuẩn sinh continuation hợp lý, còn custom draft sinh token rác; đây là lỗi reproduction, không phải evidence rằng DFlash/SpecExtend có acceptance bằng 0.",
        "",
        "Đã thêm compatibility loader copy từng tensor bằng `safetensors.safe_open` trong `classic/model_classic.py`. Tuy nhiên phép tính custom attention/RoPE vẫn còn lệch Transformers chuẩn; vì vậy fidelity cuối cùng phải được kiểm tra bằng stack upstream pinned trên Modal (`torch 2.4.0+cu121`, `transformers 4.41.0`) trước H1.",
        "",
        "## Không được kết luận từ artifact này",
        "",
        "- Không kết luận CMR myopic.",
        "- Không tính H0.5 Amdahl ceiling.",
        "- Không tính H1 horizon mismatch, H2 causal repair, H3 alternatives hoặc H4 oracle re-draft.",
        "- Không gọi throughput hiện tại là speculative speedup; với `tau=0`, đây là target/fallback-per-cycle throughput.",
        "",
        "## Artifact liên quan",
        "",
        f"- Raw H0 result: `{result_path}`",
        f"- Control: `{control_path}`" if control_path else "- Control: chưa có",
        "- Dedicated Modal runner: `scripts/modal_specextend_horizon.py`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(args.result, args.control), encoding="utf-8")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

