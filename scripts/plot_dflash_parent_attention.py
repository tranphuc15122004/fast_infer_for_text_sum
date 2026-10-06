#!/usr/bin/env python3
"""Vẽ histogram và heatmap cho parent target query và DFlash lượt kế tiếp."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dflash-parent-attention-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def save(fig, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def bins_full_scope(vector: np.ndarray, prompt: int, prefix: int,
                    output_bins: int) -> list[float]:
    vector = np.asarray(vector, dtype=np.float64)
    vector = vector / vector.sum()
    values = [float(vector[start:min(start + 1024, prompt)].sum())
              for start in range(0, prompt, 1024)]
    for index in range(output_bins):
        start = prompt + index * 128
        stop = min(start + 128, prefix)
        values.append(float(vector[start:stop].sum()) if start < prefix else 0.0)
    values.extend([float(vector[prefix]), float(vector[prefix + 1:prefix + 16].sum())])
    if abs(sum(values) - 1.0) > 2e-5:
        raise ValueError(f"Histogram không phủ đủ 100% mass: {sum(values)}")
    return values


def plot_histograms(run: Path, summary: dict, figures: Path) -> None:
    pattern = re.compile(r"sample_(\d+)_context_(\d+)_(first_transition|middle|last)\.npz$")
    snapshots = {}
    for path in (run / "parent_transition_snapshots").glob("*.npz"):
        match = pattern.fullmatch(path.name)
        if match:
            sample, cap, phase = match.groups()
            snapshots[(int(cap), phase, sample)] = path

    caps = [int(row["context_cap"]) for row in summary["caps"]]
    phases = ["first_transition", "middle", "last"]
    phase_title = {
        "first_transition": "Chuyển tiếp đầu · DFlash lượt 2",
        "middle": "Chuyển tiếp giữa", "last": "Chuyển tiếp cuối",
    }
    series = ["DFlash · mean 5 layer", "Parent target · L0",
              "Parent target · L18", "Parent target · L35"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    prepared = {}

    for cap in caps:
        expected_docs = next(int(row["instances"]) for row in summary["caps"]
                             if row["context_cap"] == cap)
        for phase in phases:
            paths = sorted((sample, path) for (current_cap, current_phase, sample), path in snapshots.items()
                           if current_cap == cap and current_phase == phase)
            if len(paths) != expected_docs:
                raise ValueError(f"Thiếu parent snapshots: cap={cap}, phase={phase}, n={len(paths)}")
            docs = []
            output_bin_count = 0
            for sample, path in paths:
                with np.load(path) as archive:
                    draft = archive["draft"].astype(np.float64)
                    parent = archive["parent_target"].astype(np.float64)
                    prefix = int(archive["prefix_length"])
                    prompt = int(archive["prompt_length"])
                    if draft.shape != (5, prefix + 16) or parent.shape != (3, prefix):
                        raise ValueError(f"Parent snapshot shape sai: {path.name}")
                    output_bin_count = max(output_bin_count,
                                           math.ceil((prefix - prompt) / 128))
                    docs.append((sample, draft, parent, prompt, prefix))

            labels = [f"P:{start//1024}–{min(start+1024, cap)/1024:g}K"
                      for start in range(0, cap, 1024)]
            labels += [f"O:{i*128}–{(i+1)*128}" for i in range(output_bin_count)]
            labels += ["Anchor mới", "15 mask"]
            doc_profiles = []
            for _, draft, parent, prompt, prefix in docs:
                draft = draft / draft.sum(axis=1, keepdims=True)
                draft_mean = draft.mean(axis=0)
                # Parent row is defined on exactly the next DFlash cache. Pad
                # the newly created anchor and 15 mask keys with zero mass.
                rows = [bins_full_scope(draft_mean, prompt, prefix, output_bin_count)]
                for layer in range(parent.shape[0]):
                    padded_parent = np.zeros(prefix + 16, dtype=np.float64)
                    padded_parent[:prefix] = parent[layer] / parent[layer].sum()
                    rows.append(
                        bins_full_scope(padded_parent, prompt, prefix, output_bin_count))
                doc_profiles.append(rows)
            means = np.asarray(doc_profiles, dtype=np.float64).reshape(expected_docs, 4, -1).mean(axis=0)
            if means.shape[1] != len(labels) or np.max(np.abs(means.sum(axis=1) - 1.0)) > 2e-5:
                raise ValueError(f"Histogram mass/shape sai: {cap=} {phase=}")
            prepared[(cap, phase)] = (labels, means,
                                      math.ceil(cap / 1024), output_bin_count)

    fig, axes = plt.subplots(len(caps), len(phases),
                             figsize=(7.7 * len(phases), 3.8 * len(caps)),
                             sharey=True, squeeze=False)
    width = .84 / len(series)
    for row, cap in enumerate(caps):
        for col, phase in enumerate(phases):
            ax = axes[row, col]
            labels, means, prompt_bins, output_bins = prepared[(cap, phase)]
            x = np.arange(len(labels))
            for index, (name, color) in enumerate(zip(series, colors)):
                offset = (index - (len(series) - 1) / 2) * width
                ax.bar(x + offset, means[index], width=width * .9,
                       color=color, label=name, linewidth=0)
            if output_bins:
                ax.axvline(prompt_bins - .5, color=".45", linestyle="--", linewidth=.8)
                ax.axvline(prompt_bins + output_bins - .5, color=".45", linestyle="--", linewidth=.8)
            ax.set_xticks(x, labels, rotation=55, ha="right", rotation_mode="anchor", fontsize=6.5)
            ax.set_title(f"{cap//1024}K · {phase_title[phase]} · 10 instance")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel("Attention mass / tổng mass gốc")
            if row == len(caps) - 1:
                ax.set_xlabel("Vùng key của DFlash lượt kế tiếp")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4,
               bbox_to_anchor=(.5, .025), frameon=False)
    fig.suptitle("Target parent attention và DFlash ở lượt kế tiếp · toàn bộ 100% mass",
                 y=.99)
    fig.text(.5, .006,
             "P: chunk prompt 1K · O: chunk output 128 · Target parent có 0 mass trên anchor/mask mới",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.055, right=.995, top=.955, bottom=.12,
                        hspace=.37, wspace=.12)
    save(fig, figures, "parent_attention_histograms_fullmass")


def plot_matrices(summary: dict, figures: Path) -> None:
    caps = summary["parent_attention"]["all_transitions"]
    target_ids = summary["target_layers"]
    draft_ids = summary["draft_layers"]
    metrics = [
        ("js_divergence_full_context", "JS toàn scope / 100% mass", "magma", 1.0),
        ("js_divergence_cache_conditional", "JS sau khi chuẩn hóa cache", "magma", 1.0),
        ("dflash_full_mass_on_parent_top_1024_cache",
         "DFlash mass trên Parent Top-1K", "viridis", 1.0),
    ]
    fig, axes = plt.subplots(len(metrics), len(caps),
                             figsize=(4.0 * len(caps), 8.8),
                             layout="constrained", squeeze=False)
    for row, (metric, title, cmap, vmax) in enumerate(metrics):
        for col, cap in enumerate(caps):
            matrix = np.asarray([
                [pair["mean"][metric] for pair in target_row]
                for target_row in cap["all_transitions"]["pairwise"]
            ], dtype=np.float64)
            ax = axes[row, col]
            image = ax.imshow(matrix, vmin=0, vmax=vmax, cmap=cmap, aspect="auto")
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    ax.text(x, y, f"{matrix[y,x]:.2f}", ha="center", va="center",
                            color="white" if matrix[y, x] < .55 else "black", fontsize=8)
            ax.set_xticks(range(len(draft_ids)), [f"D{i}" for i in draft_ids])
            ax.set_yticks(range(len(target_ids)), [f"T{i}" for i in target_ids])
            ax.set_title(f"{cap['context_cap']//1024}K · {title}")
            if col == 0:
                ax.set_ylabel("Target parent layer")
            if row == len(metrics) - 1:
                ax.set_xlabel("DFlash layer kế tiếp")
        fig.colorbar(image, ax=axes[row, :], shrink=.8)
    fig.suptitle("Parent target query ở lượt r so với DFlash lượt r+1")
    save(fig, figures, "parent_attention_comparison_matrices")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.input / "analysis"
    summary = json.loads((analysis / "summary.json").read_text())
    if "parent_attention" not in summary:
        raise ValueError("Run chưa thu thập parent attention")
    figures = analysis / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.titlesize": 9, "legend.frameon": False})
    plot_histograms(args.input, summary, figures)
    plot_matrices(summary, figures)
    print(f"[plot] Đã tạo histogram và heatmap parent attention tại {figures}")


if __name__ == "__main__":
    main()
