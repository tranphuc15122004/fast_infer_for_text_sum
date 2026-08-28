"""
Usage:
python itl_plot.py --root ./final_runs/4xH100_8B/results --out ./final_runs/4xH100_8B/plots --only_batches 1 4 8 16
python itl_plot.py --root ./final_runs/4xh200_with_new_sssd_datastore/data/results/ --out ./final_runs/4xh200_with_new_sssd_datastore/plots --only_batches 1 4 8 16 32 48 --model 70B
python itl_plot.py --root ./final_runs/h200x8_with_new_sssd_datastore/data/results/ --out ./final_runs/h200x8_with_new_sssd_datastore/plots --only_batches 1 4 8 16 --model 70B
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

# ---------------- Config / Style ----------------

# Fixed legend order for DISPLAY LABELS (with PIA removed and NGRAM -> Lookahead)
METHOD_ORDER_LABELS = [
    "Autoregressive",
    "PLD",
    "REST",
    "Lookahead",
    "EAGLE2",
    "EAGLE3",
    "SSSD",
]
COLOR_ORDER_LABELS = [
    "Autoregressive",
    "EAGLE3",
    "SSSD",
    "Lookahead",
    "PLD",
    "EAGLE2",
    "REST",
]

# Methods to drop entirely
DROP_METHODS = {"PIA"}

# Map raw method tokens from filenames -> display labels
RENAME_LABEL = {
    "NGRAM": "Lookahead",
    "ngram": "Lookahead",
    "autoreg": "Autoregressive",
    "autoregressive": "Autoregressive",
    "eagle2": "EAGLE2",
    "eagle3": "EAGLE3",
    "sssd": "SSSD",
    "rest": "REST",
    "pld": "PLD",
    # keep others as-is unless listed here
}

# Filename pattern: pd_<method>_<dataset>_bs_<batch>_<timestamp>.json
FNAME_RE = re.compile(r"^pd_([^_]+)_(.+?)_bs_(\d+)_\d{8}.*\.json$")

# ---------------- Args ----------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root folder to scan (recursively) for JSON files",
    )
    ap.add_argument(
        "--out", type=str, default="analysis", help="Where to save the plots"
    )
    ap.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional: only plot these datasets",
    )
    ap.add_argument(
        "--only_batches",
        type=int,
        nargs="*",
        default=None,
        help="Optional: only plot these batch sizes",
    )
    ap.add_argument(
        "--baseline_label",
        type=str,
        default="Autoregressive",
        help="Display label name for baseline (default: Autoregressive)",
    )
    ap.add_argument("--img_dpi", type=int, default=300)
    ap.add_argument(
        "--bar_size",
        type=float,
        nargs=2,
        default=(18.0, 8.0),
        help="Bar figure size (width height)",
    )
    ap.add_argument(
        "--line_size",
        type=float,
        nargs=2,
        default=(9.0, 8.0),
        help="Line figure size (width height)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional suffix in filenames (e.g., model size)",
    )
    return ap.parse_args()


# ---------------- IO / Parsing ----------------


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_parts_from_filename(filename: str) -> Tuple[str, str, int]:
    m = FNAME_RE.match(filename)
    if not m:
        raise ValueError(f"Unexpected filename: {filename}")
    method_raw, dataset, bs_str = m.groups()
    return method_raw, dataset, int(bs_str)


def collect_itl_stats(
    root: str, datasets_filter: Optional[List[str]] = None
) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
    """
    Returns:
      data[dataset][batch_size][method_raw] = {
          "median_itl_ms": float,
          "mean_itl_ms": float (optional),
          "std_itl_ms": float (optional),
          "se_itl_ms": float (optional, from std_itl_ms / sqrt(N)),
          "output_throughput": float
      }
    Where:
      N = total_output_tokens - completed
    """
    data: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root folder not found: {root}")

    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            try:
                method_raw, dataset, bs = extract_parts_from_filename(fn)
            except Exception:
                continue  # not our pattern

            if datasets_filter and dataset not in datasets_filter:
                continue

            fpath = os.path.join(dirpath, fn)
            try:
                js = read_json(fpath)
            except Exception as e:
                print(f"Failed to read {fpath}: {e}")
                continue

            key_itl_med = "median_itl_ms"
            key_itl_mean = "mean_itl_ms"
            key_itl_std = "std_itl_ms"
            key_thr = "output_throughput"
            key_total_tokens = "total_output_tokens"
            key_completed = "completed"

            if key_itl_med not in js:
                print(f"Missing '{key_itl_med}' in {fpath}, skipping.")
                continue
            if key_thr not in js:
                print(f"Missing '{key_thr}' in {fpath}, skipping.")
                continue

            itl_med_val = js[key_itl_med]
            thr_val = js[key_thr]

            if not isinstance(itl_med_val, (int, float)):
                print(f"'{key_itl_med}' not numeric in {fpath}, skipping.")
                continue
            if not isinstance(thr_val, (int, float)):
                print(f"'{key_thr}' not numeric in {fpath}, skipping.")
                continue

            stats: Dict[str, float] = {
                "median_itl_ms": float(itl_med_val),
                "output_throughput": float(thr_val),
            }

            # mean is optional
            mean_val = js.get(key_itl_mean, None)
            if mean_val is not None:
                if isinstance(mean_val, (int, float)):
                    stats["mean_itl_ms"] = float(mean_val)
                else:
                    print(
                        f"'{key_itl_mean}' not numeric in {fpath}, ignoring mean ITL for this file."
                    )

            # std is optional, and where possible we compute SE from it
            std_val = js.get(key_itl_std, None)
            if std_val is not None:
                if isinstance(std_val, (int, float)):
                    stats["std_itl_ms"] = float(std_val)

                    # Try to compute standard error: SE = std / sqrt(N),
                    # with N = total_output_tokens - completed
                    total_tokens = js.get(key_total_tokens, None)
                    completed = js.get(key_completed, None)
                    if isinstance(total_tokens, (int, float)) and isinstance(
                        completed, (int, float)
                    ):
                        N = int(total_tokens) - int(completed)
                        if N > 0:
                            se_itl = float(std_val) / np.sqrt(N)
                            stats["se_itl_ms"] = float(se_itl)
                        else:
                            print(f"N <= 0 when computing SE in {fpath}, skipping SE.")
                    else:
                        # If we cannot compute N, we just don't store SE
                        pass
                else:
                    print(
                        f"'{key_itl_std}' not numeric in {fpath}, ignoring std/SE ITL for this file."
                    )

            data[dataset][bs][method_raw] = stats

    return data


# ---------------- Labeling / Ordering ----------------


def label_for_method(method_raw: str) -> Optional[str]:
    if method_raw in DROP_METHODS:
        return None
    return RENAME_LABEL.get(
        method_raw, RENAME_LABEL.get(method_raw.lower(), method_raw)
    )


def order_labels(present_labels: List[str]) -> List[str]:
    ordered_present = [m for m in METHOD_ORDER_LABELS if m in present_labels]
    extras = sorted([m for m in present_labels if m not in METHOD_ORDER_LABELS])
    return ordered_present + extras


def sns_default_palette_dict() -> Dict[str, tuple]:
    """Assign Seaborn 'deep' colors to methods in COLOR_ORDER_LABELS order,
    skipping the 5th color (index 4) because it's too close to the 1st."""
    # Ask for one extra color so we can throw one away
    base_pal = sns.color_palette("deep", n_colors=len(COLOR_ORDER_LABELS) + 1)

    # Drop the 5th color (index 4)
    filtered_pal = [c for i, c in enumerate(base_pal) if i != 4]

    # Trim in case we ever have fewer labels than colors
    filtered_pal = filtered_pal[: len(COLOR_ORDER_LABELS)]

    return {lab: col for lab, col in zip(COLOR_ORDER_LABELS, filtered_pal)}


# ---------------- Data transforms ----------------


def _baseline_itl_for_batch(
    m2stats: Dict[str, Dict[str, float]], baseline_label: str, metric_key: str
) -> Optional[float]:
    """
    m2stats[method_raw] = {...}
    metric_key is "median_itl_ms" or "mean_itl_ms"
    """
    for method_raw, stats in m2stats.items():
        lab = label_for_method(method_raw)
        if lab is None:
            continue
        if lab == baseline_label:
            itl = stats.get(metric_key, None)
            if itl is None:
                continue
            return float(itl)
    return None


def build_speedup_df(
    ds_map: Dict[int, Dict[str, Dict[str, float]]],
    baseline_label: str,
    metric_key: str,
    only_batches: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Uses metric_key ("median_itl_ms" or "mean_itl_ms") to compute speed-ups vs the baseline.
    """
    rows = []
    for bs, m2stats in sorted(ds_map.items()):
        if only_batches and bs not in only_batches:
            continue

        base = _baseline_itl_for_batch(m2stats, baseline_label, metric_key=metric_key)
        if base is None or base <= 0:
            print(
                f"Warning: baseline '{baseline_label}' missing/invalid for {metric_key} at batch {bs}; skipping that batch."
            )
            continue

        for method_raw, stats in m2stats.items():
            label = label_for_method(method_raw)
            if label is None:
                continue

            itl = stats.get(metric_key, None)
            if itl is None or itl <= 0:
                continue

            rows.append(
                {
                    "Batch size": bs,
                    "Method label": label,
                    "Speed-up (x)": float(base) / float(itl),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Method label"] = pd.Categorical(
        df["Method label"],
        categories=order_labels(df["Method label"].unique()),
        ordered=True,
    )
    df["Batch size"] = pd.Categorical(
        df["Batch size"], categories=sorted(df["Batch size"].unique()), ordered=True
    )
    return df


def build_mean_itl_df(
    ds_map: Dict[int, Dict[str, Dict[str, float]]],
    only_batches: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame for plotting mean_itl_ms (+ standard error) vs output_throughput.
    Standard error is computed as:
        SE = std_itl_ms / sqrt(total_output_tokens - completed)
    and stored in 'se_itl_ms' by collect_itl_stats.
    """
    rows = []
    for bs, m2stats in sorted(ds_map.items()):
        if only_batches and bs not in only_batches:
            continue
        for method_raw, stats in m2stats.items():
            label = label_for_method(method_raw)
            if label is None:
                continue
            itl_mean = stats.get("mean_itl_ms", None)
            thr = stats.get("output_throughput", None)
            se = stats.get("se_itl_ms", None)  # may be None if we couldn't compute it
            if itl_mean is None or thr is None:
                continue
            rows.append(
                {
                    "Batch size": bs,
                    "Method label": label,
                    "Mean ITL (ms)": float(itl_mean),
                    "StdErr ITL (ms)": float(se) if se is not None else None,
                    "Output throughput": float(thr),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Method label"] = pd.Categorical(
        df["Method label"],
        categories=order_labels(df["Method label"].unique()),
        ordered=True,
    )
    df["Batch size"] = pd.Categorical(
        df["Batch size"], categories=sorted(df["Batch size"].unique()), ordered=True
    )
    return df


# ---------------- Plotting ----------------


def _apply_eagle_hatching(ax):
    """
    Apply thick hatching to EAGLE bars.
    """
    try:
        handles, labels = ax.get_legend_handles_labels()
        rgba_to_label = {
            tuple(h.get_facecolor()): lab for h, lab in zip(handles, labels)
        }

        for rect in ax.patches:
            lab = rgba_to_label.get(tuple(rect.get_facecolor()))
            if lab in ("EAGLE2", "EAGLE3"):
                rect.set_hatch("/")
                rect.set_edgecolor("white")
                rect.set_linewidth(2.0)
                # Thick hatch where supported
                if hasattr(rect, "set_hatch_linewidth"):
                    rect.set_hatch_linewidth(4.0)
    except Exception as e:
        print(f"Note: could not apply hatching reliably ({e}).")


def _plot_dataset_speedups_bars_generic(
    dataset_name: str,
    df: pd.DataFrame,
    out_dir: str,
    figsize=(18, 8),
    dpi: int = 300,
    model_name: Optional[str] = None,
    filename_suffix: str = "_itl_speedup_bar",
    y_label: str = "Speed-up (× vs Autoregressive)",
):
    if df.empty:
        print(
            f"No plottable data for dataset '{dataset_name}' (bars, suffix {filename_suffix})."
        )
        return

    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("talk")

    palette_dict = sns_default_palette_dict()
    present = [
        lab for lab in METHOD_ORDER_LABELS if lab in df["Method label"].cat.categories
    ]

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x="Batch size",
        y="Speed-up (x)",
        hue="Method label",
        estimator="mean",
        errorbar=None,
        palette=palette_dict,
        hue_order=present,
    )

    # Value labels
    for p in ax.patches:
        h = p.get_height()
        if h and h > 0:
            ax.annotate(
                f"{h:.2f}x",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="center",
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=10,
            )

    # Thick hatching for EAGLE
    _apply_eagle_hatching(ax)

    ymax = max(1.0, df["Speed-up (x)"].max())
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlabel("Batch size", fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    # No grid in bar plots
    ax.grid(False)
    sns.despine(left=True, bottom=True)
    ax.tick_params(axis="y", left=False)
    plt.tight_layout()

    suffix = f"_{model_name}" if model_name else ""
    out_path = os.path.join(out_dir, f"{dataset_name}{suffix}{filename_suffix}.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {out_path}")


def plot_dataset_speedups_bars_mean(
    dataset_name: str,
    df: pd.DataFrame,
    out_dir: str,
    figsize=(18, 8),
    dpi: int = 300,
    model_name: Optional[str] = None,
):
    """
    Mean-ITL-based speed-up (new plot).
    """
    _plot_dataset_speedups_bars_generic(
        dataset_name=dataset_name,
        df=df,
        out_dir=out_dir,
        figsize=figsize,
        dpi=dpi,
        model_name=model_name,
        filename_suffix="_mean_itl_speedup_bar",
        y_label="Speed-up (× vs Autoregressive, mean ITL)",
    )


def _lineplot_mean_itl_vs_throughput_with_se(
    df: pd.DataFrame,
    dataset_name: str,
    out_dir: str,
    figsize=(9, 8),
    dpi: int = 300,
    model_name: Optional[str] = None,
    filename_suffix: str = "_mean_itl_vs_throughput_line",
):
    """
    Mean ITL vs throughput: colored lines + shaded ± standard error band.
    Standard error column is 'StdErr ITL (ms)'.
    """
    if df.empty:
        print(
            f"No plottable data for dataset '{dataset_name}' (mean ITL vs throughput)."
        )
        return

    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("talk")

    palette_dict = sns_default_palette_dict()

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # One line (plus optional SE band) per method
    methods = order_labels(df["Method label"].unique().tolist())
    for method in methods:
        method_df = df[df["Method label"] == method].copy()
        if method_df.empty:
            continue
        method_df = method_df.sort_values("Batch size")

        x = method_df["Output throughput"].values
        y = method_df["Mean ITL (ms)"].values
        se = method_df["StdErr ITL (ms)"].values  # may contain NaNs / None

        color = palette_dict.get(method, None)
        if color is None:
            # fallback if method not in COLOR_ORDER_LABELS
            color = next(iter(palette_dict.values()))

        # Line
        ax.plot(x, y, label=method, color=color, marker="o")

        # # Shaded SE band (only where SE is finite)
        # se = np.array(se, dtype=float)
        # if np.any(np.isfinite(se)):
        #     # Replace non-finite SE with 0 so we at least draw band for valid points
        #     se_clean = np.where(np.isfinite(se), se, 0.0)
        #     y_lower = y - se_clean
        #     y_upper = y + se_clean
        #     ax.fill_between(x, y_lower, y_upper, color=color, alpha=0.2)

        for xi, yi, bs in zip(
            x,
            y,
            method_df["Batch size"].values,
        ):
            ax.annotate(
                str(int(bs)),
                (xi, yi),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=16,
            )

    eagle_colors = {
        tuple(to_rgba(palette_dict[m]))
        for m in ("EAGLE2", "EAGLE3")
        if m in palette_dict
    }

    for line in ax.lines:
        line.set_linewidth(3.0)
        col = tuple(to_rgba(line.get_color()))
        if col in eagle_colors:
            line.set_linestyle("--")

    ax.set_ylim(0, 40)
    ax.set_xlim(0, None)
    ax.grid(True, axis="both")
    ax.minorticks_on()
    ax.grid(True, which="major", axis="both", linewidth=1.2, alpha=0.8)
    ax.grid(True, which="minor", axis="both", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Output throughput (= 1/cost) [toks/s]", fontsize=18)
    ax.set_ylabel("Mean inter-token latency [ms]", fontsize=18)
    legend1 = ax.legend(loc="upper left", title=None)

    # Style legend: one entry per method
    style_handles = [
        Line2D([0], [0], color="black", lw=3, linestyle="-"),
        Line2D([0], [0], color="black", lw=3, linestyle="--"),
    ]
    style_labels = ["Training-free", "Trained"]

    ax.legend(
        handles=style_handles,
        labels=style_labels,
        loc="lower right",
    )
    ax.add_artist(legend1)
    # sns.despine()
    plt.tight_layout()

    suffix = f"_{model_name}" if model_name else ""
    out_path = os.path.join(out_dir, f"{dataset_name}{suffix}{filename_suffix}.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {out_path}")


def plot_dataset_mean_itl_lines(
    dataset_name: str,
    df: pd.DataFrame,
    out_dir: str,
    figsize=(9, 8),
    dpi: int = 300,
    model_name: Optional[str] = None,
):
    _lineplot_mean_itl_vs_throughput_with_se(
        df=df,
        dataset_name=dataset_name,
        out_dir=out_dir,
        figsize=figsize,
        dpi=dpi,
        model_name=model_name,
    )


# ---------------- Main ----------------


def main():
    args = parse_args()
    all_data = collect_itl_stats(args.root, datasets_filter=args.datasets)
    if not all_data:
        print(
            "No datasets found or no valid JSON files with the required keys. Nothing to plot."
        )
        return

    os.makedirs(args.out, exist_ok=True)

    for dataset, bs_map in sorted(all_data.items()):
        # Bars: mean-based speedups vs Autoregressive (new plot)
        df_speed_mean = build_speedup_df(
            bs_map,
            baseline_label=args.baseline_label,
            metric_key="mean_itl_ms",
            only_batches=args.only_batches,
        )
        plot_dataset_speedups_bars_mean(
            dataset_name=dataset,
            df=df_speed_mean,
            out_dir=args.out,
            figsize=tuple(args.bar_size),
            dpi=args.img_dpi,
            model_name=args.model,
        )

        # Lines: mean ITL vs throughput with **standard error** shading
        df_mean = build_mean_itl_df(bs_map, only_batches=args.only_batches)
        plot_dataset_mean_itl_lines(
            dataset_name=dataset,
            df=df_mean,
            out_dir=args.out,
            figsize=tuple(args.line_size),
            dpi=args.img_dpi,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
