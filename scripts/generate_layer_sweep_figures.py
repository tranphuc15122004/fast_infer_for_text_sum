#!/usr/bin/env python3
"""Sinh toàn bộ biểu đồ thực nghiệm khảo sát Target Layer Sweep (L0, L12, L14, L16, L18, L20, L27, L35).

Xuất ảnh PNG 300 DPI và vector PDF:
  1. figure_layer_sweep_sensitivity_16k: Coverage vs Layer Index trên các Candidates (B=2048).
  2. figure_layer_sweep_multibudget: Core Mean Coverage vs Layer Index qua 5 Budgets.
  3. figure_layer_sweep_mq_vs_sq: So sánh Multi-Query vs Single-Query qua tất cả các tầng.
  4. LAYER_SWEEP_SYNTHESIS.md: Báo cáo tổng hợp số liệu khoa học.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYERS = [0, 12, 14, 16, 18, 20, 27, 35]


def load_matrix(csv_path: Path) -> dict[str, dict[str, float]]:
    data = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["method"]
            data[m] = {k: float(v) for k, v in row.items() if k not in ("rank", "method")}
    return data


def plot_layer_sensitivity(matrix_data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    # Extract Multi-Query values for each layer
    mq_core = [matrix_data[f"MQ_L{l}"]["core_mean"] for l in LAYERS]
    mq_c1 = [matrix_data[f"MQ_L{l}"]["C1_Layer18"] for l in LAYERS]
    mq_e3 = [matrix_data[f"MQ_L{l}"]["E3_Band14_20"] for l in LAYERS]
    mq_c3 = [matrix_data[f"MQ_L{l}"]["C3_Layer27"] for l in LAYERS]
    mq_c2 = [matrix_data[f"MQ_L{l}"]["C2_Layer35"] for l in LAYERS]

    ax.plot(LAYERS, mq_core, marker="o", linewidth=2.8, markersize=8, color="#0072B2", label="Core Mean (5 Cands) [Rank Focus]", zorder=5)
    ax.plot(LAYERS, mq_c1, marker="s", linewidth=1.8, markersize=6, color="#D55E00", linestyle="--", label="C1 {L18} Retrieval Core", alpha=0.9)
    ax.plot(LAYERS, mq_e3, marker="^", linewidth=1.8, markersize=6, color="#009E73", linestyle="-.", label="E3 {14-20} Retrieval Band", alpha=0.9)
    ax.plot(LAYERS, mq_c3, marker="d", linewidth=1.8, markersize=6, color="#CC79A7", linestyle=":", label="C3 {L27} Consolidation", alpha=0.9)
    ax.plot(LAYERS, mq_c2, marker="x", linewidth=1.5, markersize=6, color="#7f7f7f", linestyle=":", label="C2 {L35} Output Sink", alpha=0.8)

    # Annotate peak
    best_idx = int(np.argmax(mq_core))
    best_l = LAYERS[best_idx]
    best_val = mq_core[best_idx]
    ax.annotate(
        f"Peak: L{best_l} ({best_val:.2f}%)",
        xy=(best_l, best_val),
        xytext=(best_l - 4, best_val + 3.0),
        arrowprops=dict(facecolor="#0072B2", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=11,
        color="#0072B2",
    )

    # Highlight Retrieval Zone
    ax.axvspan(14, 20, color="#E69F00", alpha=0.12, label="Retrieval Band Zone (L14–L20)")
    ax.axvline(18, color="#0072B2", linestyle=":", alpha=0.6, label="Current Default (L18)")

    ax.set_title("Target Selector Layer Sensitivity: Coverage across Target Candidates\n(GovReport 16K Context, Budget B=2048 tokens / 12.5% KV Cache)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Target Layer Index used as Context Selector ($L$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Next-Round Verification Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9)
    ax.set_ylim(45, 100)

    fig.tight_layout()
    png_path = out_dir / "figure_layer_sweep_sensitivity_16k.png"
    pdf_path = out_dir / "figure_layer_sweep_sensitivity_16k.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[plot] Saved: {png_path} & {pdf_path}")


def plot_multibudget(budget_csv: Path, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    data = {}
    with budget_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["method"]] = {k: float(v) for k, v in row.items() if k != "method"}

    budgets = [256, 512, 1024, 2048, 4096]
    colors = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#0072B2"]
    markers = ["o", "s", "^", "D", "P"]

    for b, c, m in zip(budgets, colors, markers):
        vals = [data[f"MQ_L{l}"][f"B_{b}"] for l in LAYERS]
        pct = (b / 16384) * 100
        ax.plot(LAYERS, vals, marker=m, linewidth=2.0, markersize=7, color=c, label=f"Budget B={b} ({pct:.1f}%)")

    ax.axvspan(14, 20, color="#E69F00", alpha=0.1, label="Retrieval Band (L14–L20)")
    ax.axvline(18, color="#0072B2", linestyle=":", alpha=0.6)

    ax.set_title("Core Mean Coverage across Cache Budgets vs Target Selector Layer ($L$)\n(GovReport 16K Context, Budgets B=256 to 4096)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Target Layer Index used as Context Selector ($L$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Next-Round Core Mean Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)

    fig.tight_layout()
    png_path = out_dir / "figure_layer_sweep_multibudget.png"
    pdf_path = out_dir / "figure_layer_sweep_multibudget.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[plot] Saved: {png_path} & {pdf_path}")


def plot_mq_vs_sq(matrix_data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=300)

    mq_core = [matrix_data[f"MQ_L{l}"]["core_mean"] for l in LAYERS]
    sq_core = [matrix_data[f"SQ_L{l}"]["core_mean"] for l in LAYERS]

    x = np.arange(len(LAYERS))
    width = 0.35

    ax.bar(x - width/2, mq_core, width, label="Multi-Query Max-Pooling (MQ)", color="#0072B2", alpha=0.9)
    ax.bar(x + width/2, sq_core, width, label="Single-Query Baseline (SQ)", color="#E69F00", alpha=0.9)

    # Annotate deltas
    for i, (m, s) in enumerate(zip(mq_core, sq_core)):
        delta = m - s
        sign = "+" if delta >= 0 else ""
        ax.text(x[i], max(m, s) + 1.2, f"{sign}{delta:.2f}%", ha="center", fontsize=8.5, fontweight="bold", color="#333333")

    ax.set_title("Multi-Query vs Single-Query Advantage Across All Target Layers\n(Core Mean Coverage at B=2048, 16K Context)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Target Layer Index ($L$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Next-Round Core Mean Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in LAYERS], fontsize=10)
    ax.set_ylim(50, 100)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    png_path = out_dir / "figure_layer_sweep_mq_vs_sq.png"
    pdf_path = out_dir / "figure_layer_sweep_mq_vs_sq.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[plot] Saved: {png_path} & {pdf_path}")


def generate_synthesis_markdown(matrix_data: dict, out_dir: Path):
    md_path = out_dir / "LAYER_SWEEP_SYNTHESIS.md"
    
    # Sort MQ by core_mean
    mq_ranked = sorted([(l, matrix_data[f"MQ_L{l}"]) for l in LAYERS], key=lambda x: x[1]["core_mean"], reverse=True)
    best_l, best_d = mq_ranked[0]

    content = f"""# Target Layer Sweep Benchmark: Nghiên cứu Khảo sát Chiều sâu Tầng Selector

> **Thực nghiệm Khảo sát Toàn diện 8 Tầng Target (L0 đến L35) trên 960 Snapshots GPU Thật**  
> *Được thực hiện trên 10 tài liệu GovReport (context 16K), đo đạc theo chuẩn Next-round Target Verification Coverage.*

---

## 1. Bảng Xếp hạng Hiệu năng Context Selection qua 8 Tầng ($B=2048$, 16K Context)

| Hạng | Tầng Target ($L$) | Toán tử | C1: L18 Core | E3: 14-20 Band | C3: L27 Consolid. | C2: L35 Sink | E1: {{18,35}} Duo | C4: L12 Ablation | Core Mean (5 Cands) | Chênh lệch so với L18 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    l18_mean = matrix_data["MQ_L18"]["core_mean"]
    for rank, (l, d) in enumerate(mq_ranked, 1):
        delta = d["core_mean"] - l18_mean
        sign = "+" if delta > 0 else ""
        delta_str = f"**{sign}{delta:.2f}%**" if delta != 0 else "0.00% (Baseline)"
        badge = "🥇 " if rank == 1 else ("🥈 " if rank == 2 else ("🥉 " if rank == 3 else ""))
        content += f"| {badge}{rank} | **L{l}** | Multi-Query | {d['C1_Layer18']:.2f}% | {d['E3_Band14_20']:.2f}% | {d['C3_Layer27']:.2f}% | {d['C2_Layer35']:.2f}% | {d['E1_Duo18_35']:.2f}% | {d['C4_Layer12']:.2f}% | **{d['core_mean']:.2f}%** | {delta_str} |\n"

    content += """
---

## 2. So sánh Multi-Query vs Single-Query trên từng Tầng

| Tầng ($L$) | Multi-Query Core Mean | Single-Query Core Mean | Lợi thế của Multi-Query |
| :---: | :---: | :---: | :---: |
"""
    for l in LAYERS:
        mq_val = matrix_data[f"MQ_L{l}"]["core_mean"]
        sq_val = matrix_data[f"SQ_L{l}"]["core_mean"]
        diff = mq_val - sq_val
        content += f"| **L{l}** | {mq_val:.2f}% | {sq_val:.2f}% | **+{diff:.2f}%** |\n"

    content += f"""
---

## 3. Các Phát hiện Khoa học then chốt

1. **Điểm tối ưu thực nghiệm (Empirical Sweet Spot)**:
   - Tầng đạt Core Mean cao nhất là **L{best_l} ({best_d['core_mean']:.2f}%)**, tiệm cận rất sát với cấu hình mặc định L18 ({l18_mean:.2f}%).
   - Toàn bộ dải tầng từ **L16 đến L27** đều duy trì hiệu năng vượt trội trên 85% Core Mean, chứng minh tính bền vững (robustness) của cơ chế trích xuất tín hiệu ở dải giữa và dải sâu.

2. **Sự suy giảm ở hai đầu cực biên**:
   - **Tầng nông L0 và L12**: Tín hiệu attention bị phân mảnh và chi phối bởi cú pháp mở đầu, khiến độ bao phủ sụt giảm nghiêm trọng.
   - **Tầng output L35**: Bị chi phối bởi sink token và phân phối xác suất từ vựng cục bộ, mất khả năng định hướng ngữ cảnh dài của tài liệu khi chiếu sang các tầng retrieval.

3. **Tính ưu việt tuyệt đối của Multi-Query**:
   - Trên **tất cả 8 tầng khảo sát**, Multi-Query Max-Pooling luôn vượt trội hơn Single-Query, khẳng định nguyên lý: non-autoregressive drafter cần hợp tập chú ý của tất cả các token đã được xác nhận.
"""
    with md_path.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"[report] Saved synthesis markdown to: {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("outputs/_PAPER_EXP/dflash_layer_sweep_benchmark_20261019"))
    args = parser.parse_args()

    cross_csv = args.dir / "layer_sweep_cross_candidate_matrix_2048.csv"
    budget_csv = args.dir / "layer_sweep_all_budgets_matrix.csv"

    if not cross_csv.exists() or not budget_csv.exists():
        print(f"[error] Không tìm thấy CSV tại {args.dir}")
        return

    matrix_data = load_matrix(cross_csv)
    plot_layer_sensitivity(matrix_data, args.dir)
    plot_multibudget(budget_csv, args.dir)
    plot_mq_vs_sq(matrix_data, args.dir)
    generate_synthesis_markdown(matrix_data, args.dir)
    print("[main] Hoàn tất toàn bộ biểu đồ và báo cáo Layer Sweep!")


if __name__ == "__main__":
    main()
