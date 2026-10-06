#!/usr/bin/env python3
"""Vẽ báo cáo DFlash với cùng mẫu số 100% attention mass trên mọi key."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dflash-full-context-mpl")
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


def format_mass_axis(ax):
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Tỷ lệ trên tổng attention mass")
    ax.grid(axis="y", alpha=.18)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Folder full_context_analysis chứa summary.json")
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    strategies = [("top", "Top-K rải rác"), ("best_window", "Cửa sổ liên tiếp tốt nhất"),
                  ("edges", "K/2 đầu + K/2 cuối"), ("head", "K key đầu"),
                  ("middle", "K key ở giữa")]
    for ax, budget in zip(axes, [1024, 4096]):
        for index, (field, name) in enumerate(strategies):
            values = [c["mean"][f"{field}_{budget}"] for c in caps]
            ax.plot(x, values, marker="o", color=COLORS[index], label=name)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Prompt ban đầu; attention trên mọi key")
        ax.set_title(f"Cùng budget {budget//1024}K key")
        format_mass_axis(ax)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, -.14), ncol=3)
    fig.suptitle("DFlash · Coverage chọn key trên 100% attention mass · mean 10 instance")
    save(fig, figures, "selection_strategies_full_context")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for index, cap in enumerate(caps):
        means = cap["mean"]
        budgets = data["budgets"]
        axes[0].plot(np.asarray(budgets)/1024, [means[f"top_{k}"] for k in budgets],
                     marker="o", color=COLORS[index], label=labels[index])
        fractions = data["top_fraction_budgets"]
        axes[1].plot([0.] + fractions,
                     [0.] + [means[f"top_fraction_{round(f*100)}pct"] for f in fractions],
                     marker="o", color=COLORS[index], label=labels[index])
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Budget Top-K (1K = 1.024 key; thang log₂)")
    axes[0].set_xticks(np.asarray(data["budgets"])/1024, ["0,25", "0,5", "1", "2", "4", "8"])
    axes[0].set_title("Budget tuyệt đối")
    axes[0].axhline(.9, color=".5", linestyle="--", linewidth=.8)
    axes[0].axhline(.95, color=".7", linestyle=":", linewidth=.8)
    axes[0].text(8.05, .9, "90%", va="center", fontsize=8)
    axes[0].text(8.05, .95, "95%", va="center", fontsize=8)
    axes[1].plot([0, 1], [0, 1], color=".5", linestyle="--", label="Phân bố đều")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel("Tỷ lệ key có weight cao nhất được giữ")
    axes[1].set_title("Budget theo tỷ lệ tổng số key")
    for ax in axes:
        format_mass_axis(ax)
    axes[0].legend(title="Prompt ban đầu")
    axes[1].legend(loc="lower right")
    fig.suptitle("DFlash · Mức tập trung và phần mass ngoài Top-K · mọi key")
    save(fig, figures, "concentration_full_context")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained", sharey=True)
    phase_names = [("first", "Lượt đầu"), ("middle", "Lượt giữa"), ("last", "Lượt cuối")]
    ymax = max(c["phase_means"][phase][f"position_bin_{i}"]
               for c in caps for phase, _ in phase_names for i in range(10)) * 1.12
    for ax, cap, label in zip(axes.flat, caps, labels):
        for index, (phase, name) in enumerate(phase_names):
            values = [cap["phase_means"][phase][f"position_bin_{i}"] for i in range(10)]
            if abs(sum(values) - 1.) > 1e-6:
                raise ValueError("Spatial bins không chứa đủ 100% mass")
            ax.plot(np.arange(5, 100, 10), values, marker="o", color=COLORS[index], label=name)
        ax.axhline(.1, color=".5", linestyle="--", linewidth=.8, label="Phân bố đều: 10% mỗi bin")
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Vị trí tương đối trong toàn bộ sequence (%)")
        ax.set_ylabel("Mass của bin chứa 10% số key")
        ax.set_title(f"Prompt {label}")
        ax.grid(axis="y", alpha=.18)
    axes[0, 0].legend(loc="upper center", fontsize=8)
    fig.suptitle("DFlash · Phân bố theo vị trí trên toàn bộ context · tổng 10 bin = 100%")
    save(fig, figures, "spatial_profiles_full_context")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for budget, color in [(1024, COLORS[0]), (4096, COLORS[1])]:
        name = f"{budget//1024}K"
        axes[0].plot(x, [c["adjacent_rounds"][f"overlap_{budget}"] for c in caps],
                     color=color, marker="o", label=f"Top-{name}: hai lượt liền kề")
        axes[0].plot(x, [c["phase_pairs"]["middle_to_last"][f"overlap_{budget}"] for c in caps],
                     color=color, marker="s", linestyle="--", label=f"Top-{name}: lượt giữa → cuối")
        axes[1].plot(x, [c["phase_pairs"]["middle_to_last"][f"current_top_mass_{budget}"] for c in caps],
                     color=color, marker="o", label=f"Top-{name} chọn lại tại lượt cuối")
        axes[1].plot(x, [c["phase_pairs"]["middle_to_last"][f"old_selection_mass_{budget}"] for c in caps],
                     color=color, marker="s", linestyle="--", label=f"Top-{name} từ lượt giữa, đọc ở cuối")
    axes[0].set_title("Mức giữ nguyên tập vị trí")
    axes[0].set_ylabel("Vị trí chung / số key đã chọn ở lượt trước")
    axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Coverage tại lượt cuối")
    format_mass_axis(axes[1])
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Độ dài prompt ban đầu")
        ax.grid(axis="y", alpha=.18)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("DFlash · Tập key ưu tiên thay đổi khi sinh · current block đối chiếu theo slot")
    save(fig, figures, "temporal_selection_full_context")
    source_script = Path(__file__).resolve()
    saved_script = (figures / "gen_fig_full_context.py").resolve()
    if source_script != saved_script:
        shutil.copyfile(source_script, saved_script)
    print(f"[full-context-plot] Đã tạo 4 PNG + 4 PDF: {figures}")


if __name__ == "__main__":
    main()
