#!/usr/bin/env python3
"""Vẽ histogram và heatmap của probe DFlash; không cần tải model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dflash-attention-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0072B2", "#D55E00", "#009E73"]


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


PART_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
PART_NAMES = ["Prompt ban đầu", "Output đã sinh", "Anchor hiện tại", "15 mask trong draft"]


def full_layout(cap, output_bins):
    prompt_bins = (cap + 1023) // 1024
    labels = [f"P:{i}–{i+1}K" for i in range(prompt_bins)]
    labels += [f"O:{i*128}–{(i+1)*128}" for i in range(output_bins)]
    labels += ["Anchor", "Mask"]
    colors = ([PART_COLORS[0]] * prompt_bins + [PART_COLORS[1]] * output_bins
              + PART_COLORS[2:])
    return prompt_bins, labels, colors


def full_values(record, output_bins):
    generated = record["generated_bins_absolute"]
    values = np.asarray(record["prompt_bins_absolute"] + generated
                        + [0.] * (output_bins - len(generated))
                        + [record["draft_anchor_mass"], record["draft_mask_mass"]])
    if not np.isfinite(values).all() or abs(values.sum() - record["mass_sum"]) > 1e-5:
        raise ValueError("Hình chưa bao gồm đủ attention mass")
    return values


def style_full_axis(ax, cap, output_bins):
    prompt_bins, labels, _ = full_layout(cap, output_bins)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=6)
    ax.axvline(prompt_bins - .5, color=".4", linewidth=.6, linestyle="--")
    ax.axvline(prompt_bins + output_bins - .5, color=".4", linewidth=.6, linestyle="--")


def plot_full_mass(folder, rows):
    """Mỗi lượt gồm 100% key mass; tổng hợp cho mỗi document trọng số bằng nhau."""
    from matplotlib.patches import Patch
    caps = sorted({r["context_cap"] for r in rows})
    sample_ids = list(dict.fromkeys(r["sample_id"] for r in rows))
    runs = {(sample, cap): sorted(
        [r for r in rows if r["sample_id"] == sample and r["context_cap"] == cap],
        key=lambda r: r["round"]) for sample in sample_ids for cap in caps}
    if any(not records for records in runs.values()):
        raise ValueError("Có instance thiếu một mức context")
    output_bins = max(1, max(len(r["generated_bins_absolute"]) for r in rows))
    for record in rows:
        full_values(record, output_bins)
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "axes.titlesize": 10})
    legend = [Patch(color=color, label=name) for color, name in zip(PART_COLORS, PART_NAMES)]
    vmax = max(float(full_values(r, output_bins).max()) for r in rows)
    n = len(caps)
    phase_names = ["Lượt đầu", "Lượt giữa", "Lượt cuối"]
    for sample_index, sample in enumerate(sample_ids):
        destination = figures / f"sample_{sample_index:02d}"
        destination.mkdir(exist_ok=True)
        fig, axes = plt.subplots(n, 3, figsize=(14, 3 * n), squeeze=False, sharey=True)
        for index, cap in enumerate(caps):
            records = runs[sample, cap]
            choices = [records[0], records[len(records)//2], records[-1]]
            _, _, colors = full_layout(cap, output_bins)
            for column, record in enumerate(choices):
                ax = axes[index, column]
                values = full_values(record, output_bins)
                ax.bar(range(len(values)), values, color=colors)
                ax.set_ylim(0, min(1., vmax * 1.15))
                style_full_axis(ax, cap, output_bins)
                ax.set_title(f"{cap//1024}K · lượt {record['round']} · output +{record['output_offset']}")
                ax.text(.98, .96, f"Tổng {record['mass_sum']:.2%}\n"
                        f"Top-1K toàn bộ: {record['top_1k_all_coverage']:.1%}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8)
                if column == 0:
                    ax.set_ylabel("Attention mass tuyệt đối")
        fig.suptitle(f"DFlash · {sample}\nP: prompt (vùng 1K), O: output đã sinh (vùng 128 token)")
        fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, .95))
        save(fig, destination, "attention_histograms_full")

        fig, axes = plt.subplots(n, 1, figsize=(13, 2.8 * n), squeeze=False, layout="constrained")
        for index, cap in enumerate(caps):
            ax = axes[index, 0]
            records = runs[sample, cap]
            matrix = np.stack([full_values(r, output_bins) for r in records])
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
            style_full_axis(ax, cap, output_bins)
            ticks = np.unique(np.linspace(0, len(records)-1, min(7, len(records))).astype(int))
            ax.set_yticks(ticks, [f"{records[t]['round']} (+{records[t]['output_offset']})" for t in ticks])
            ax.set_ylabel("Lượt (output offset)")
            ax.set_title(f"{cap//1024}K · {len(records)} lượt · mọi hàng cộng đủ ≈100%")
            fig.colorbar(im, ax=ax, label="Mass tuyệt đối", fraction=.03, pad=.02)
        fig.suptitle(f"DFlash · toàn bộ attention qua lượt draft · {sample}")
        save(fig, destination, "attention_round_heatmaps_full")

    # Mean of first/middle/last round across documents; equal document weighting.
    fig, axes = plt.subplots(n, 3, figsize=(14, 3 * n), squeeze=False, sharey=True)
    for index, cap in enumerate(caps):
        _, _, colors = full_layout(cap, output_bins)
        for column, phase in enumerate(phase_names):
            selected = [records[[0, len(records)//2, -1][column]]
                        for sample in sample_ids for records in [runs[sample, cap]]]
            matrix = np.stack([full_values(r, output_bins) for r in selected])
            mean = matrix.mean(0)
            ax = axes[index, column]
            ax.bar(range(len(mean)), mean, color=colors)
            ax.set_ylim(0, min(1., vmax * 1.15))
            style_full_axis(ax, cap, output_bins)
            ax.set_title(f"{cap//1024}K · {phase} · {len(sample_ids)} instance")
            ax.text(.98, .96, f"Tổng {mean.sum():.2%}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8)
            if column == 0:
                ax.set_ylabel("Mean attention mass tuyệt đối")
    fig.suptitle("DFlash · histogram toàn bộ attention · mỗi document có trọng số bằng nhau")
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .95))
    save(fig, figures, "attention_histograms_full_mean")

    fields = ["prompt_mass", "generated_mass", "draft_anchor_mass", "draft_mask_mass"]
    grid = np.linspace(0, 1, 64)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.3 * n), squeeze=False, layout="constrained")
    for index, cap in enumerate(caps):
        curves = []
        for sample in sample_ids:
            records = runs[sample, cap]
            offsets = np.asarray([r["output_offset"] for r in records], dtype=float)
            progress = offsets / max(1., offsets[-1])
            curves.append([np.interp(grid, progress, [r[field] for r in records]) for field in fields])
        mean = np.asarray(curves).mean(0)
        ax = axes[index, 0]
        ax.stackplot(grid * 100, mean, colors=PART_COLORS, labels=PART_NAMES)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 100)
        ax.set_title(f"Context {cap//1024}K")
        ax.set_ylabel("Attention mass")
        ax.set_xlabel("Tiến độ output: từ lượt draft đầu tới lượt cuối (%)")
    axes[0, 0].legend(loc="upper left", ncol=4, fontsize=8)
    fig.suptitle(f"DFlash · prompt + output đã sinh + draft block ≈100% · mean {len(sample_ids)} instance")
    save(fig, figures, "attention_mass_parts_progress")

    # Distribution of per-document means, rather than giving long trajectories more weight.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout="constrained")
    metrics = ["top_1k_all_coverage", "top_1k_prompt_coverage", "generated_mass"]
    titles = ["Top-1K: mọi key", "Top-1K: trong prompt", "Mass lên output đã sinh"]
    stats_fields = list(dict.fromkeys(fields + metrics + ["draft_block_mass",
        "tokens_for_90pct_all_mass", "tokens_for_95pct_all_mass"]))
    aggregate = []
    for cap in caps:
        per_doc = [{field: float(np.mean([r[field] for r in runs[sample, cap]]))
                    for field in stats_fields}
                   for sample in sample_ids]
        result = {"context_cap": cap, "sample_count": len(sample_ids),
                  "draft_rounds": sum(len(runs[sample, cap]) for sample in sample_ids)}
        for field in stats_fields:
            values = [d[field] for d in per_doc]
            result[field] = {"mean": float(np.mean(values)), "min": float(np.min(values)),
                             "max": float(np.max(values)), "per_document": values}
        result["phase_means"] = {
            phase: {field: float(np.mean([
                runs[sample, cap][[0, len(runs[sample, cap])//2, -1][index]][field]
                for sample in sample_ids])) for field in stats_fields}
            for index, phase in enumerate(["first", "middle", "last"])}
        aggregate.append(result)
    for ax, field, title in zip(axes, metrics, titles):
        data = [a[field]["per_document"] for a in aggregate]
        ax.boxplot(data, tick_labels=[f"{cap//1024}K" for cap in caps], widths=.45)
        for index, values in enumerate(data, 1):
            ax.scatter(index + np.linspace(-.1, .1, len(values)), values, s=14,
                       color=PART_COLORS[0], alpha=.65, zorder=3)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Tỷ lệ attention mass")
        ax.grid(axis="y", alpha=.2)
    save(fig, figures, "attention_concentration_instances")
    (folder / "aggregate_full_mass.json").write_text(json.dumps({
        "weighting": "mean over rounds within document, then equal mean across documents",
        "sample_ids": sample_ids, "caps": aggregate,
        "max_mass_sum_error": max(abs(r['mass_sum']-1.) for r in rows),
        "output_bins_for_figures": output_bins,
    }, indent=2, ensure_ascii=False))
    print(f"[plot] {len(sample_ids)} instance, {len(rows)} lượt; full-mass PNG/PDF: {figures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory chứa attention.jsonl")
    args = parser.parse_args()
    folder = Path(args.input)
    rows = [json.loads(line) for line in (folder / "attention.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row.get("type") == "draft_attention"]
    if not rows:
        raise ValueError("Không có lượt draft để vẽ")
    if "generated_bins_absolute" in rows[0]:
        plot_full_mass(folder, rows)
        return
    groups = {cap: [r for r in rows if r["context_cap"] == cap]
              for cap in sorted({r["context_cap"] for r in rows})}
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "axes.titlesize": 10})
    n = len(groups)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.5 * n), squeeze=False, sharey=True)
    max_mass = max(max(r["prompt_bins_absolute"]) for r in rows) * 1.12
    for index, (cap, records) in enumerate(groups.items()):
        choices = [records[0], records[len(records) // 2], records[-1]]
        for column, record in enumerate(choices):
            ax = axes[index, column]
            values = np.asarray(record["prompt_bins_absolute"])
            ax.bar(np.arange(len(values)), values, color=COLORS[column], width=.82)
            ax.set_ylim(0, max_mass)
            ax.set_xticks(np.arange(len(values)))
            ax.set_xticklabels([str(i) for i in range(len(values))], fontsize=7)
            ax.set_title(f"{cap//1024}K · lượt {record['round']} · output +{record['output_offset']}")
            ax.text(.98, .94, f"Prompt mass {record['prompt_mass']:.1%}\n"
                    f"Top-1K coverage {record['top_1k_prompt_coverage']:.1%}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.set_xlabel("Vùng 1K trong initial prompt (từ đầu)")
            if column == 0:
                ax.set_ylabel("Attention mass tuyệt đối")
    fig.suptitle("DFlash: attention mass theo vị trí prompt và lượt draft", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, figures, "attention_histograms")

    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n), squeeze=False,
                             layout="constrained")
    vmax = max(max(r["prompt_bins_normalized"]) for r in rows)
    for index, (cap, records) in enumerate(groups.items()):
        ax = axes[index, 0]
        matrix = np.asarray([r["prompt_bins_normalized"] for r in records])
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels([f"{i}–{i+1}K" for i in range(matrix.shape[1])])
        ax.set_yticks(range(len(records)))
        ax.set_yticklabels([f"{r['round']} (+{r['output_offset']})" for r in records])
        ax.set_ylabel("Lượt (output offset)")
        ax.set_title(f"Context {cap//1024}K · prompt mass "
                     f"{min(r['prompt_mass'] for r in records):.1%}–"
                     f"{max(r['prompt_mass'] for r in records):.1%}")
        fig.colorbar(im, ax=ax, label="Mass điều kiện trên prompt", fraction=.03, pad=.02)
    fig.suptitle("DFlash: attention theo lượt draft — cùng thang màu", fontsize=13)
    save(fig, figures, "attention_round_heatmaps")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    for cap, records in groups.items():
        rounds = [r["round"] for r in records]
        axes[0].plot(rounds, [r["top_1k_prompt_coverage"] for r in records],
                     marker="o", label=f"{cap//1024}K")
        axes[1].plot(rounds, [r["prompt_mass"] for r in records],
                     marker="o", label=f"{cap//1024}K")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_xlabel("Lượt draft")
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel("Tỷ lệ prompt mass trong Top-1K token")
    axes[1].set_ylabel("Attention mass tuyệt đối dành cho prompt")
    save(fig, figures, "attention_concentration")
    print(f"[plot] PNG/PDF: {figures}")


if __name__ == "__main__":
    main()
