#!/usr/bin/env python3
"""Vẽ độ giống/khác nhau giữa attention target và DFlash."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dflash-target-comparison-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

TARGET_COLORS = ["#0072B2", "#D55E00", "#009E73"]
DRAFT_COLORS = ["#CC79A7", "#777777", "#56B4E9", "#E69F00", "#332288"]


def save(fig, directory: Path, name: str):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def matrix(cap: dict, metric: str) -> np.ndarray:
    return np.asarray([[pair["mean"][metric] for pair in target_row]
                       for target_row in cap["pairwise"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.input / "analysis"
    data = json.loads((analysis / "summary.json").read_text())
    records = [json.loads(line) for line in (analysis / "round_metrics.jsonl").read_text().splitlines()
               if json.loads(line).get("type") == "target_dflash_attention_pair_analysis"]
    figures = analysis / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.titlesize": 10, "legend.frameon": False})
    caps = data["caps"]
    target_ids, draft_ids = data["target_layers"], data["draft_layers"]

    fig, axes = plt.subplots(2, len(caps), figsize=(4.0*len(caps), 6.2),
                             layout="constrained", squeeze=False)
    for col, cap in enumerate(caps):
        for row, metric, label in [
            (0, "js_divergence_full_context", "JS · toàn bộ key / 100% mass"),
            (1, "js_divergence_cache_conditional", "JS · cache đã chuẩn hóa riêng"),
        ]:
            ax = axes[row, col]
            values = matrix(cap, metric)
            image = ax.imshow(values, vmin=0, vmax=1, cmap="magma", aspect="auto")
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center",
                            color="white" if values[y, x] < .55 else "black", fontsize=8)
            ax.set_xticks(range(len(draft_ids)), [f"D{i}" for i in draft_ids])
            ax.set_yticks(range(len(target_ids)), [f"T{i}" for i in target_ids])
            ax.set_title(f"{cap['context_cap']//1024}K · {label}")
            if col == 0:
                ax.set_ylabel("Target layer")
            if row == 1:
                ax.set_xlabel("DFlash layer")
    fig.colorbar(image, ax=axes, label="Jensen–Shannon divergence (log₂; 0 giống, 1 khác)", shrink=.82)
    fig.suptitle("Attention target và DFlash · ghép query dự đoán cùng proposal")
    save(fig, figures, "attention_js_matrices")

    fig, axes = plt.subplots(2, len(caps), figsize=(4.0*len(caps), 6.2),
                             layout="constrained", squeeze=False)
    for col, cap in enumerate(caps):
        for row, budget in enumerate([1024, 4096]):
            ax = axes[row, col]
            values = matrix(cap, f"top_{budget}_cache_overlap_fraction")
            image = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center",
                            color="white" if values[y, x] < .55 else "black", fontsize=8)
            ax.set_xticks(range(len(draft_ids)), [f"D{i}" for i in draft_ids])
            ax.set_yticks(range(len(target_ids)), [f"T{i}" for i in target_ids])
            ax.set_title(f"{cap['context_cap']//1024}K · giao Top-{budget} cache")
            if col == 0:
                ax.set_ylabel("Target layer")
            if row == 1:
                ax.set_xlabel("DFlash layer")
    fig.colorbar(image, ax=axes, label="Số vị trí chung / budget cache", shrink=.82)
    fig.suptitle("Mức trùng vị trí cache được xếp cao bởi hai mô hình")
    save(fig, figures, "attention_topk_overlap")

    fig, axes = plt.subplots(1, len(caps), figsize=(4.0*len(caps), 4.8),
                             sharey=True, squeeze=False)
    for ax, cap in zip(axes.flat, caps):
        layers = cap["layer_means"]
        model_labels = [f"D{i}" for i in draft_ids] + [f"T{i}" for i in target_ids]
        model_values = [layers["draft"][str(i)] for i in draft_ids] + [
            layers["target"][str(i)] for i in target_ids]
        x = np.arange(len(model_labels))
        prompt = np.asarray([v["prompt_mass"] for v in model_values])
        output = np.asarray([v["committed_output_mass"] for v in model_values])
        block = np.asarray([v["current_block_mass"] for v in model_values])
        ax.bar(x, prompt, color="#0072B2", label="Prompt")
        ax.bar(x, output, bottom=prompt, color="#E69F00", label="Output đã commit")
        ax.bar(x, block, bottom=prompt+output, color="#CC79A7", label="Block hiện tại")
        ax.set_xticks(x, model_labels, rotation=45)
        ax.set_title(f"{cap['context_cap']//1024}K")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.18)
        ax.set_xlabel("Mô hình / layer")
    axes[0, 0].set_ylabel("Attention mass / tổng mass gốc")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .035))
    fig.suptitle("Attention trên prompt, output đã commit và block 16 token")
    fig.subplots_adjust(left=.07, right=.99, top=.86, bottom=.24, wspace=.13)
    save(fig, figures, "attention_mass_by_scope")

    bins_by_key = {}
    for record in records:
        cap, sample = record["context_cap"], record["sample_id"]
        for model in ["draft", "target"]:
            for layer, values in record[f"{model}_layers"].items():
                bins_by_key.setdefault((cap, sample, model, int(layer)), []).append(
                    np.asarray(values["position_bins_full_context"], dtype=np.float64))
    columns = max(1, math.ceil(len(caps) / 2))
    rows_count = math.ceil(len(caps) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(5.5*columns, 3.25*rows_count),
                             layout="constrained", sharey=True, squeeze=False)
    for ax, cap in zip(axes.flat, caps):
        phase = cap["context_cap"]
        for index, layer in enumerate(draft_ids):
            docs = [np.mean(v, axis=0) for (c, _, model, idx), v in bins_by_key.items()
                    if c == phase and model == "draft" and idx == layer]
            mean = np.mean(docs, axis=0)
            ax.plot(np.arange(5, 100, 10), mean, marker=".", linewidth=1.2,
                    color=DRAFT_COLORS[index], alpha=.8, label=f"DFlash {layer}")
        for index, layer in enumerate(target_ids):
            docs = [np.mean(v, axis=0) for (c, _, model, idx), v in bins_by_key.items()
                    if c == phase and model == "target" and idx == layer]
            mean = np.mean(docs, axis=0)
            ax.plot(np.arange(5, 100, 10), mean, marker="o", linewidth=2.1,
                    color=TARGET_COLORS[index], label=f"Target {layer}")
        ax.axhline(.1, color=".5", linestyle="--", linewidth=.8, label="Mass đều")
        ax.set_title(f"Prompt {cap['context_cap']//1024}K")
        ax.set_xlabel("Vị trí tương đối trên toàn key scope (%)")
        ax.set_ylabel("Mass bin / tổng mass")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.18)
    axes.flat[0].legend(ncol=2, fontsize=7, loc="upper left")
    fig.suptitle(f"Phân bố vị trí trên 100% key scope · mean {data['caps'][0]['instances']} instance")
    save(fig, figures, "attention_position_profiles")


    saved = (figures / "gen_fig_target_comparison.py").resolve()
    if Path(__file__).resolve() != saved:
        shutil.copyfile(Path(__file__), saved)
    print(f"[plot] Đã tạo 4 PNG + 4 PDF tại {figures}")


if __name__ == "__main__":
    main()
