"""
python plot_speedups_from_json_average.py \
  --root final_runs/reasoning/evaluation \
  --out final_runs/reasoning/plots \
  --baseline Autoregressive \
  --model deepseek \
  --only_batches 1 4 8 16 32 48 64
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

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

# Map raw method names -> display labels
RENAME_LABEL = {
    "NGRAM": "Lookahead"
}  # everything else keeps its name unless listed here


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="data/evaluation",
        help="Root folder that contains dataset subfolders",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data/analysis",
        help="Where to save the plots (two PNGs per dataset)",
    )
    ap.add_argument(
        "--baseline", type=str, default="Autoregressive", help="Baseline method name"
    )
    ap.add_argument(
        "--only_batches",
        type=int,
        nargs="*",
        default=None,
        help="Optional filter: only plot these batch sizes (ints)",
    )
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--img_dpi", type=int, default=300)
    ap.add_argument("--bar_width", type=float, default=18.0)
    ap.add_argument("--bar_height", type=float, default=8.0)
    ap.add_argument("--line_width", type=float, default=9.0)
    ap.add_argument("--line_height", type=float, default=8.0)
    return ap.parse_args()


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_output_throughput(js: dict, path: str) -> float | None:
    """
    Extract a single numeric output_throughput value from the JSON.
    - If top-level 'output_throughput' exists, use that.
    - Else, if 'results' is a list of runs, average their 'output_throughput'.
    Returns None if nothing valid is found.
    """
    # Old format: top-level scalar
    if "output_throughput" in js:
        tp = js["output_throughput"]
        if isinstance(tp, (int, float)):
            return float(tp)
        else:
            print(f"'output_throughput' not numeric in {path}, skipping.")
            return None

    # New format: list of results
    if "results" in js and isinstance(js["results"], list):
        vals = []
        for i, res in enumerate(js["results"]):
            if not isinstance(res, dict):
                continue
            tp = res.get("output_throughput")
            if isinstance(tp, (int, float)):
                vals.append(float(tp))
            else:
                # quietly skip non-numeric entries
                continue

        if not vals:
            print(
                f"No numeric 'output_throughput' values found in 'results' for {path}, skipping."
            )
            return None

        avg_tp = sum(vals) / len(vals)
        return avg_tp

    print(f"Missing 'output_throughput' or 'results' in {path}, skipping.")
    return None


def extract_parts(filename: str, dataset_name: str) -> Tuple[str, int]:
    assert filename.endswith(".json"), f"Not a JSON: {filename}"
    stem = filename[:-5]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    try:
        bs = int(parts[-1])
    except ValueError:
        raise ValueError(
            f"Batch size must be an integer at end of filename: {filename}"
        )
    method = "_".join(parts[:-2])  # allow underscores in method
    return method, bs


def collect_throughputs(root: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Returns:
      data[dataset][bs][method] = output_throughput

    For each JSON:
      - method, bs come from the filename (via extract_parts)
      - output_throughput is:
          * top-level 'output_throughput' if present and numeric, OR
          * the average of 'output_throughput' across entries in js['results'].
    """
    data: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root folder not found: {root}")

    for dataset in sorted(os.listdir(root)):
        ds_dir = os.path.join(root, dataset)
        if not os.path.isdir(ds_dir):
            continue

        for fn in os.listdir(ds_dir):
            if not fn.endswith(".json"):
                continue
            try:
                method, bs = extract_parts(fn, dataset)
            except Exception as e:
                print(f"Skipping {fn}: {e}")
                continue

            fpath = os.path.join(ds_dir, fn)
            try:
                js = read_json(fpath)
            except Exception as e:
                print(f"Failed to read {fpath}: {e}")
                continue

            tp = extract_output_throughput(js, fpath)
            if tp is None:
                continue

            data[dataset][bs][method] = tp

    return data


def label_for_method(method_raw: str) -> str:
    """Apply drop/rename policy and return the display label (or None if dropped)."""
    if method_raw in DROP_METHODS:
        return None
    return RENAME_LABEL.get(method_raw, method_raw)


def order_labels(present_labels: List[str]) -> List[str]:
    ordered_present = [m for m in METHOD_ORDER_LABELS if m in present_labels]
    extras = sorted([m for m in present_labels if m not in METHOD_ORDER_LABELS])
    return ordered_present + extras


def build_speedup_df(
    ds_map: Dict[int, Dict[str, float]],
    baseline: str,
    only_batches: List[int] | None = None,
) -> pd.DataFrame:
    rows = []
    for bs, m2t in sorted(ds_map.items()):
        if only_batches and bs not in only_batches:
            continue

        # For baseline, use its RAW name (no rename/drop while locating it)
        if baseline not in m2t:
            print(
                f"Warning: baseline '{baseline}' not found for batch size {bs}; skipping that batch."
            )
            continue
        base_tp = m2t[baseline]
        if base_tp <= 0:
            print(
                f"Warning: baseline throughput <= 0 for batch size {bs}; skipping that batch."
            )
            continue

        for method_raw, tp in m2t.items():
            label = label_for_method(method_raw)
            if label is None:
                continue  # dropped method (e.g., PIA)
            rows.append(
                {"Batch size": bs, "Method label": label, "Speed-up (x)": tp / base_tp}
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


def build_latency_df(
    ds_map: Dict[int, Dict[str, float]], only_batches: List[int] | None = None
) -> pd.DataFrame:
    """
    Return columns:
      Method label, Batch size, Output throughput, Latency
    where Latency = batch_size / output_throughput.
    """
    rows = []
    for bs, m2t in sorted(ds_map.items()):
        if only_batches and bs not in only_batches:
            continue
        if bs <= 0:
            continue
        for method_raw, tp in m2t.items():
            label = label_for_method(method_raw)
            if label is None:
                continue  # dropped
            if tp <= 0:
                continue
            rows.append(
                {
                    "Method label": label,
                    "Batch size": bs,
                    "Batch size (n)": bs,  # numeric copy for annotation
                    "Output throughput": tp,
                    "Latency": (bs / tp) * 1000,
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


# ---------- NEW: stable default Seaborn palette (label -> color) ----------
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


def plot_dataset_speedups(
    dataset_name: str,
    df: pd.DataFrame,
    out_dir: str,
    figsize=(14, 8),
    dpi: int = 300,
    model_name: str = None,
):
    if df.empty:
        print(f"No plottable data for dataset '{dataset_name}' (speedups).")
        return

    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Stable palette & order limited to labels actually present
    palette_dict = sns_default_palette_dict()
    present = [
        lab for lab in METHOD_ORDER_LABELS if lab in df["Method label"].cat.categories
    ]

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x="Batch size",
        y="Speed-up (x)",
        hue="Method label",  # use display labels
        estimator="mean",
        errorbar=None,
        palette=palette_dict,  # default sns colors pinned to labels
        hue_order=present,  # keep order and colors stable
    )

    # Label bars with numeric values
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

    # Apply striped hatching to EAGLE methods (keep their colors; add stripes)
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
                rect.set_linewidth(2.5)
                rect.set_hatch_linewidth(3)
    except Exception as e:
        # Non-fatal; just continue without hatching if mapping fails
        print(f"Note: could not apply hatching reliably ({e}).")

    ymax = max(1.0, df["Speed-up (x)"].max())
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlabel("Batch size", fontsize=18)
    ax.set_ylabel("Speed-up (x)", fontsize=18)

    ax.grid(False)
    sns.despine(left=True, bottom=True)
    ax.tick_params(axis="y", left=False)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{dataset_name}_{model_name}_speedups.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    plt.close()
    print(f"Saved: {out_path}")


def plot_latency_vs_throughput(
    dataset_name: str,
    df: pd.DataFrame,
    out_dir: str,
    figsize=(8, 8),
    dpi: int = 300,
    model_name: str = None,
):
    """
    Line plot:
      - x: Output throughput
      - y: Latency (= batch_size / output_throughput)
      - One line per Method (display label), with markers at batch-size points.
      - Each point annotated with the batch size number.
    """
    if df.empty:
        print(
            f"No plottable data for dataset '{dataset_name}' (latency vs throughput)."
        )
        return

    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=figsize)
    plot_df = df.sort_values(["Method label", "Batch size"])

    # Stable palette & order for present labels
    palette_dict = sns_default_palette_dict()
    present = [
        lab
        for lab in METHOD_ORDER_LABELS
        if lab in plot_df["Method label"].cat.categories
    ]

    ax = sns.lineplot(
        data=plot_df,
        x="Output throughput",
        y="Latency",
        hue="Method label",
        marker="o",
        sort=False,  # already sorted
        palette=palette_dict,  # default sns colors
        hue_order=present,
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

    # Annotate each point with its batch size
    for _, row in plot_df.iterrows():
        ax.annotate(
            str(int(row["Batch size (n)"])),
            (row["Output throughput"], row["Latency"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=16,
        )

    ax.set_xlabel("Throughput = 1/cost [toks/s]", fontsize=18)
    ax.set_ylabel("Latency [ms/tok]", fontsize=18)

    # Denser grid (major + minor) on both axes and keep bottom spine
    ax.minorticks_on()
    ax.grid(True, which="major", axis="both", linewidth=1.2, alpha=0.8)
    ax.grid(True, which="minor", axis="both", linewidth=0.8, alpha=0.4)
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
        loc="upper right",
    )
    ax.add_artist(legend1)
    ax.set_xlim(0)
    ax.set_ylim(0, 60)
    # sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    plt.tight_layout()

    out_path = os.path.join(
        out_dir, f"{dataset_name}_{model_name}_latency_vs_throughput.png"
    )
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    all_data = collect_throughputs(args.root)
    if not all_data:
        print(
            "No datasets found or no valid JSON files with 'output_throughput'. Nothing to plot."
        )
        return

    os.makedirs(args.out, exist_ok=True)

    for dataset, bs_map in sorted(all_data.items()):
        # 1) Speedup bars
        df_speed = build_speedup_df(
            bs_map, baseline=args.baseline, only_batches=args.only_batches
        )
        plot_dataset_speedups(
            dataset_name=dataset,
            df=df_speed,
            out_dir=args.out,
            figsize=(args.bar_width, args.bar_height),
            dpi=args.img_dpi,
            model_name=args.model,
        )

        # 2) Latency vs throughput lines (with batch-size labels)
        df_lat = build_latency_df(bs_map, only_batches=args.only_batches)
        plot_latency_vs_throughput(
            dataset_name=dataset,
            df=df_lat,
            out_dir=args.out,
            figsize=(args.line_width, args.line_height),
            dpi=args.img_dpi,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
