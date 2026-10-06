#!/usr/bin/env python3
"""Vẽ cache-only và mass tổng khi giữ nguyên 16 draft-block key."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dflash-cache-analysis-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#777777"]


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mass_axis(ax, denominator):
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel(denominator)
    ax.grid(axis="y", alpha=.18)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Folder context_cache_analysis chứa summary.json")
    args = parser.parse_args()
    data = json.loads((args.input / "summary.json").read_text())
    caps = data["caps"]
    figures = args.input / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "legend.frameon": False, "legend.fontsize": 8.5,
                         "axes.titlesize": 11, "lines.linewidth": 1.7})
    x = np.arange(len(caps))
    labels = [f"{c['context_cap']//1024}K" for c in caps]
    strategies = [("top", "Top-K cache rải rác"), ("best_window", "Cửa sổ cache tốt nhất"),
                  ("edges", "K/2 đầu + K/2 cuối cache"), ("head", "K key đầu cache"),
                  ("middle", "K key giữa cache")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for row, budget in enumerate([1024, 4096]):
        for column, prefix in enumerate(["", "retained_total_"]):
            ax = axes[row, column]
            for index, (field, name) in enumerate(strategies):
                ax.plot(x, [c["mean"][f"{prefix}{field}_{budget}"] for c in caps],
                        marker="o", color=COLORS[index], label=name)
            ax.set_xticks(x, labels)
            ax.set_xlabel("Prompt ban đầu; cache gồm prompt + output đã commit")
            if column == 0:
                title = f"Budget {budget//1024}K cache · bỏ 16 key block, chuẩn hóa cache"
                denominator = "Coverage / cache mass"
            else:
                title = f"{budget//1024}K cache + 16 key block luôn giữ"
                denominator = "Mass giữ lại / tổng mass gốc"
            ax.set_title(title)
            mass_axis(ax, denominator)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, -.10), ncol=3)
    fig.suptitle("DFlash · Nén cache, giữ nguyên draft block · hai mẫu số được ghi riêng")
    save(fig, figures, "cache_selection_coverage")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for index, cap in enumerate(caps):
        m = cap["mean"]
        budgets = data["budgets"]
        axes[0].plot(np.asarray(budgets)/1024, [m[f"top_{k}"] for k in budgets],
                     color=COLORS[index], marker="o", label=labels[index])
        fractions = data["top_fraction_budgets"]
        axes[1].plot([0.] + fractions,
                     [0.] + [m[f"top_fraction_{round(f*100)}pct"] for f in fractions],
                     color=COLORS[index], marker="o", label=labels[index])
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(np.asarray(data["budgets"])/1024, ["0,25", "0,5", "1", "2", "4", "8"])
    axes[0].set_xlabel("Budget cache Top-K (1K = 1.024 key; thang log₂)")
    axes[0].axhline(.9, color=".5", linewidth=.8, linestyle="--")
    axes[0].axhline(.95, color=".7", linewidth=.8, linestyle=":")
    axes[0].set_title("Coverage cache với budget tuyệt đối")
    axes[0].legend(title="Prompt ban đầu")
    axes[1].plot([0, 1], [0, 1], color=".5", linestyle="--", label="Cache phân bố đều")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel("Tỷ lệ cache key có weight cao nhất được giữ")
    axes[1].set_title("Coverage cache với budget theo tỷ lệ")
    axes[1].legend(loc="lower right")
    for ax in axes:
        mass_axis(ax, "Coverage / cache mass sau khi bỏ block")
    fig.suptitle("DFlash · Độ tập trung của cache · loại 16 block key khỏi phép đo")
    save(fig, figures, "cache_concentration")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained", sharey=True)
    phases = [("first", "Lượt đầu"), ("middle", "Lượt giữa"), ("last", "Lượt cuối")]
    ymax = max(c["phase_means"][phase][f"position_bin_{i}"]
               for c in caps for phase, _ in phases for i in range(10))*1.12
    for ax, cap, label in zip(axes.flat, caps, labels):
        for index, (phase, name) in enumerate(phases):
            values = [cap["phase_means"][phase][f"position_bin_{i}"] for i in range(10)]
            if abs(sum(values)-1)>1e-6:
                raise ValueError("Cache bins không cộng đủ 100% cache mass")
            ax.plot(np.arange(5, 100, 10), values, color=COLORS[index], marker="o", label=name)
        ax.axhline(.1, color=".5", linestyle="--", linewidth=.8, label="Cache đều: 10% mỗi bin")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, ymax)
        ax.set_title(f"Prompt {label}; không có 16 key block")
        ax.set_xlabel("Vị trí tương đối trong context cache (%)")
        ax.set_ylabel("Mass bin / cache mass")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.18)
    axes[0, 0].legend(loc="upper center", fontsize=8)
    fig.suptitle("DFlash · Vị trí trong cache: prompt + output đã commit · tổng bin = 100% cache mass")
    save(fig, figures, "cache_spatial_profiles")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for budget, color in [(1024, COLORS[0]), (4096, COLORS[1])]:
        name = f"{budget//1024}K"
        axes[0].plot(x, [c["adjacent_rounds"][f"overlap_{budget}"] for c in caps],
                     color=color, marker="o", label=f"Top-{name} cache: hai lượt liền kề")
        axes[0].plot(x, [c["phase_pairs"]["middle_to_last"][f"overlap_{budget}"] for c in caps],
                     color=color, marker="s", linestyle="--", label=f"Top-{name} cache: giữa → cuối")
        axes[1].plot(x, [c["phase_pairs"]["middle_to_last"][f"current_top_retained_total_{budget}"] for c in caps],
                     color=color, marker="o", label=f"Top-{name} cache chọn lại + block")
        axes[1].plot(x, [c["phase_pairs"]["middle_to_last"][f"old_selection_retained_total_{budget}"] for c in caps],
                     color=color, marker="s", linestyle="--", label=f"Top-{name} cache từ lượt giữa + block")
    axes[0].set_title("Tập cache key ưu tiên thay đổi")
    axes[0].set_ylabel("Key chung / số cache key chọn ở lượt trước")
    axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Mass giữ tại lượt cuối; luôn giữ 16 key block")
    mass_axis(axes[1], "Mass giữ lại / tổng mass gốc")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Prompt ban đầu")
        ax.grid(axis="y", alpha=.18)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("DFlash · Phân tích thời gian chỉ trên cache · đánh giá giữ lại trên 100% mass gốc")
    save(fig, figures, "cache_temporal_selection")
    saved = (figures / "gen_fig_context_cache.py").resolve()
    if Path(__file__).resolve() != saved:
        shutil.copyfile(Path(__file__), saved)
    print(f"[cache-plot] Đã tạo 4 PNG + 4 PDF: {figures}")


if __name__ == "__main__":
    main()
