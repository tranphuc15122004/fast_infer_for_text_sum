#!/usr/bin/env python3
"""Vẽ toàn bộ hình ảnh và ma trận cho 6 Selectors (M0 - M5) và 6 Candidate Layers.
Chuẩn Paper khoa học: 300 DPI PNG và Vector PDF (Type 42 fonts).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs/_PAPER_EXP/dflash_sota_selection_benchmark/m0_m5_matrix"
OUTPUT_DIR = DATA_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHODS = [
    "M0_SingleQuery",
    "M1_P1_MultiQuery",
    "M2_SnapKV_Cluster",
    "M3_RocketKV_MaxPool",
    "M4_DuoEnsemble_18_35",
    "M5_DeepSink_L35",
]

LABELS = [
    "M0: Single Parent Q",
    "M1: Multi-Q Max (Đề xuất)",
    "M2: SnapKV (Chunk 128)",
    "M3: RocketKV (Conv64)",
    "M4: Duo-Ensemble (18+35)",
    "M5: Deep Sink (L35 Pure)",
]

COLORS = [
    "#000000",  # M0: Black
    "#0072B2",  # M1: Blue
    "#E69F00",  # M2: Orange
    "#56B4E9",  # M3: Sky blue
    "#009E73",  # M4: Green
    "#D55E00",  # M5: Vermillion
]

CANDIDATES = ["C0_L0", "C4_L12", "C1_L18", "C3_L27", "C2_L35", "E1_L18_L35"]
CLABELS = ["L0 (Syntax)", "L12 (Early)", "L18 (Retrieval)", "L27 (Deep)", "L35 (Decision)", "L18 + L35 (Duo)"]


def load_data():
    csv_path = DATA_DIR / "sota_comparison_summary.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    index = {}
    for r in rows:
        key = (
            int(r["context_cap"]),
            int(r["budget"]),
            r["method"],
            r["candidate"],
            r["evaluation_scope"],
        )
        index[key] = {k: float(v) if k not in ("method", "candidate", "evaluation_scope") else v for k, v in r.items()}
    return index


def plot_figure1_target_coverage(index):
    """Figure 1: Target Coverage curves across budgets at 16K."""
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    budgets = [256, 512, 1024, 2048, 4096]

    for i, m in enumerate(METHODS):
        y = [100 * index[16384, b, m, "E1_L18_L35", "next_round"]["mean_coverage"] for b in budgets]
        lo = [100 * index[16384, b, m, "E1_L18_L35", "next_round"]["coverage_ci_low"] for b in budgets]
        hi = [100 * index[16384, b, m, "E1_L18_L35", "next_round"]["coverage_ci_high"] for b in budgets]
        yerr = np.maximum(0, [np.array(y) - np.array(lo), np.array(hi) - np.array(y)])

        linewidth = 2.5 if m in ("M1_P1_MultiQuery", "M4_DuoEnsemble_18_35") else 1.5
        linestyle = "-" if m != "M0_SingleQuery" else "--"
        ax.errorbar(
            budgets, y, yerr=yerr, fmt="o" + linestyle, color=COLORS[i],
            label=LABELS[i], linewidth=linewidth, capsize=3, markersize=5
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets, [f"{b:,}" for b in budgets])
    ax.set_ylim(60, 100)
    ax.set_xlabel("Working Set Budget $B$ (tokens)")
    ax.set_ylabel("Next-Round Target Attention Coverage (%)")
    ax.set_title("Figure 1: Target Verification Coverage across Budgets (16K Context, GovReport 10 Docs)")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.savefig(OUTPUT_DIR / "figure1_target_coverage_all_sota.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "figure1_target_coverage_all_sota.pdf")
    plt.close(fig)
    print("Saved Figure 1")


def plot_figure2_draft_coverage(index):
    """Figure 2: Draft Coverage scaling across context lengths at B=2048."""
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    caps = [3072, 5120, 8192, 16384]

    for i, m in enumerate(METHODS):
        y = [100 * index[c, 2048, m, "E1_L18_L35", "next_round"]["mean_draft_coverage"] for c in caps]
        ax.plot(caps, y, "o-", color=COLORS[i], label=LABELS[i], linewidth=2, markersize=6)

    ax.set_xscale("log", base=2)
    ax.set_xticks(caps, [f"{c:,}" for c in caps])
    ax.set_ylim(35, 100)
    ax.set_xlabel("Context Length $N$ (tokens)")
    ax.set_ylabel("DFlash Draft Attention Coverage (%)")
    ax.set_title("Figure 2: Drafter Attention Alignment across Context Scaling ($B=2048$)")
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.savefig(OUTPUT_DIR / "figure2_draft_coverage_all_sota.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "figure2_draft_coverage_all_sota.pdf")
    plt.close(fig)
    print("Saved Figure 2")


def plot_figure3_multimetric(index):
    """Figure 3: Multi-metric comparison at 16K, B=2048."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")

    # Panel 1: Target Coverage
    ax = axes[0, 0]
    vals = [100 * index[16384, 2048, m, "E1_L18_L35", "next_round"]["mean_coverage"] for m in METHODS]
    ax.bar(range(len(METHODS)), vals, color=COLORS)
    ax.set_xticks(range(len(METHODS)), LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(60, 100)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("(a) Target Coverage (L18+L35)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for j, v in enumerate(vals):
        ax.text(j, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8)

    # Panel 2: Draft Coverage
    ax = axes[0, 1]
    vals = [100 * index[16384, 2048, m, "E1_L18_L35", "next_round"]["mean_draft_coverage"] for m in METHODS]
    ax.bar(range(len(METHODS)), vals, color=COLORS)
    ax.set_xticks(range(len(METHODS)), LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(30, 60)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("(b) DFlash Drafter Coverage")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for j, v in enumerate(vals):
        ax.text(j, v + 0.6, f"{v:.1f}%", ha="center", fontsize=8)

    # Panel 3: Spatial Continuity
    ax = axes[1, 0]
    vals = [100 * index[16384, 2048, m, "E1_L18_L35", "next_round"]["mean_spatial_continuity"] for m in METHODS]
    ax.bar(range(len(METHODS)), vals, color=COLORS)
    ax.set_xticks(range(len(METHODS)), LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Adjacent Token Pairs (%)")
    ax.set_title("(c) Spatial Continuity (Memory Coalescing Proxy)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for j, v in enumerate(vals):
        ax.text(j, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)

    # Panel 4: Layer 18 vs Layer 35 Contrast
    ax = axes[1, 1]
    l18_vals = [100 * index[16384, 2048, m, "C1_L18", "next_round"]["mean_coverage"] for m in METHODS]
    l35_vals = [100 * index[16384, 2048, m, "C2_L35", "next_round"]["mean_coverage"] for m in METHODS]
    x = np.arange(len(METHODS))
    width = 0.38
    ax.bar(x - width / 2, l18_vals, width, label="Target L18 (Retrieval)", color="#0072B2")
    ax.bar(x + width / 2, l35_vals, width, label="Target L35 (Decision)", color="#D55E00")
    ax.set_xticks(x, LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(40, 100)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("(d) Retrieval (L18) vs. Logits Decision (L35)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Figure 3: Multi-metric Comparison at 16K Context ($B=2048$, Next-Round)", fontsize=14)
    fig.savefig(OUTPUT_DIR / "figure3_multimetric_comparison_16k.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "figure3_multimetric_comparison_16k.pdf")
    plt.close(fig)
    print("Saved Figure 3")


def plot_figure4_tradeoff(index):
    """Figure 4: Continuity vs Coverage Pareto trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for i, m in enumerate(METHODS):
        x = 100 * index[16384, 2048, m, "E1_L18_L35", "next_round"]["mean_spatial_continuity"]
        y = 100 * index[16384, 2048, m, "E1_L18_L35", "next_round"]["mean_coverage"]
        size = 140 if m in ("M1_P1_MultiQuery", "M2_SnapKV_Cluster", "M4_DuoEnsemble_18_35") else 90
        ax.scatter(x, y, color=COLORS[i], s=size, label=LABELS[i], zorder=4)
        ax.annotate(
            LABELS[i].split(":")[0], (x, y), xytext=(6, 4), textcoords="offset points",
            fontweight="bold" if m == "M1_P1_MultiQuery" else "normal"
        )

    ax.set_xlim(35, 105)
    ax.set_ylim(70, 92)
    ax.set_xlabel("Spatial Continuity (% adjacent token pairs in mask)")
    ax.set_ylabel("Next-Round Target Attention Coverage (%)")
    ax.set_title("Figure 4: Trade-off between Spatial Continuity and Attention Coverage (16K, $B=2048$)")
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.savefig(OUTPUT_DIR / "figure4_tradeoff_continuity_coverage.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "figure4_tradeoff_continuity_coverage.pdf")
    plt.close(fig)
    print("Saved Figure 4")


def plot_heatmap_matrix(index):
    """Figure Heatmap: 6 Selectors x 6 Candidate Layers at 16K, B=2048."""
    mat = np.zeros((len(METHODS), len(CANDIDATES)))
    for i, m in enumerate(METHODS):
        for j, c in enumerate(CANDIDATES):
            mat[i, j] = 100 * index[16384, 2048, m, c, "next_round"]["mean_coverage"]

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    cmap = plt.colormaps["viridis"].copy()
    im = ax.imshow(mat, cmap=cmap, vmin=25, vmax=95, aspect="auto")

    ax.set_xticks(range(len(CANDIDATES)), CLABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(METHODS)), LABELS, fontsize=9)
    ax.set_title("Complete $6 \\times 6$ Evaluation Matrix: 6 Context Selectors vs. 6 Target Verification Layers\n(Context 16K, Budget $B=2048$, Next-Round Coverage %)", fontsize=11)

    for (i, j), val in np.ndenumerate(mat):
        color = "white" if val < 65 else "black"
        ax.text(j, i, f"{val:.2f}%", ha="center", va="center", color=color, fontweight="bold" if j == 2 else "normal")

    fig.colorbar(im, ax=ax, label="Next-Round Attention Coverage (%)")
    fig.savefig(OUTPUT_DIR / "figure_matrix_heatmap_16k.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "figure_matrix_heatmap_16k.pdf")
    plt.close(fig)
    print("Saved Heatmap Matrix")


def plot_layer_profiles(index):
    """Figure Layer Profile: Single layer distribution across L0, L12, L18, L27, L35."""
    single_cands = ["C0_L0", "C4_L12", "C1_L18", "C3_L27", "C2_L35"]
    layers_x = [0, 12, 18, 27, 35]

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for i, m in enumerate(METHODS):
        y = [100 * index[16384, 2048, m, c, "next_round"]["mean_coverage"] for c in single_cands]
        linewidth = 2.5 if m in ("M1_P1_MultiQuery", "M5_DeepSink_L35", "M4_DuoEnsemble_18_35") else 1.5
        ax.plot(layers_x, y, "o-", color=COLORS[i], label=LABELS[i], linewidth=linewidth, markersize=6)

    ax.set_xticks(layers_x, ["Layer 0\n(Syntax/Sink)", "Layer 12\n(Early)", "Layer 18\n(Retrieval)", "Layer 27\n(Deep)", "Layer 35\n(Logits Decision)"])
    ax.set_ylim(20, 100)
    ax.set_xlabel("Target Model Verification Layer Depth")
    ax.set_ylabel("Attention Coverage (%)")
    ax.set_title("Target Verification Layer Depth Profile: SOTA Selectors vs. Layer Sensitivity (16K, $B=2048$)")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.savefig(OUTPUT_DIR / "figure_matrix_layer_profile_16k.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "figure_matrix_layer_profile_16k.pdf")
    plt.close(fig)
    print("Saved Layer Profile")


def main():
    print("[figures] Đang tải dữ liệu...")
    index = load_data()
    print("[figures] Đang vẽ các biểu đồ Paper...")
    plot_figure1_target_coverage(index)
    plot_figure2_draft_coverage(index)
    plot_figure3_multimetric(index)
    plot_figure4_tradeoff(index)
    plot_heatmap_matrix(index)
    plot_layer_profiles(index)
    print(f"[figures] Hoàn tất toàn bộ hình ảnh tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
