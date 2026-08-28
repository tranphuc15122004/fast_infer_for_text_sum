import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator

# Use seaborn whitegrid style
sns.set_theme(style="whitegrid", font_scale=1.3)

# Optional: fine-tune matplotlib font sizes
plt.rcParams.update(
    {
        "font.size": 14,  # base font size
        "axes.titlesize": 16,  # figure titles
        "axes.labelsize": 14,  # x / y labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

# === CONFIG ===
BASE_DIR = "final_runs/multilingual_rerun/"
os.makedirs(BASE_DIR + "plots", exist_ok=True)

# Multiple evaluation roots; same structure & filenames inside each
EVAL_ROOT_DIRS = [
    "final_runs/multilingual_rerun/data_multilingual/evaluation",
    "final_runs/multilingual_rerun/data_multilingual2/evaluation",
    "final_runs/multilingual_rerun/run1/evaluation",
    "final_runs/multilingual_rerun/run2/evaluation",
    "final_runs/multilingual_rerun/run3/evaluation",
]
EVAL_ROOT_DIR_NOT_GENERATED_SSSD = (
    "final_runs/multilingual_rerun/data_multilingual_sssd_from_hf/evaluation"
)
PLOT_LOOKAHEAD = False

BATCH_SIZE = 1
SPEEDUP_PLOTS_YLIM = 2.21

# These will be computed dynamically in main()
english_sssd_at_max = None
all_languages_on_english = None

# Map language name -> dataset folder / dataset prefix in filenames
LANG_TO_DATASET = {
    "English": "mt-bench",
    "Italian": "mt-bench-it",
    "Japanese": "mt-bench-jp",
    "Indonesian": "mt-bench-id",
}

# Dictionary for the *actual* datastore sizes (in tokens) for each language.
ACTUAL_LAST_TOKENS = {
    "English": 148_917_188 + 134_934_953 + 192_743_039,
    "Japanese": 64_137_884 + 12_064_801,
    "Indonesian": 66_498_607 + 11_544_930,
    "Italian": 65_215_906 + 13_239_763,
}

ACTUAL_LAST_TOKENS_HF_DATA = {
    "English": 235_434_100,
    "Japanese": 31_239_400,
    "Indonesian": 38_766_300,
    "Italian": 38_279_400,
}

# === HELPERS ===

SIZE_TO_TOKENS = {
    "none": 0,
    "100k": 100_000,
    "1m": 1_000_000,
    "10m": 10_000_000,
    "100m": 100_000_000,
    "1g": 1_000_000_000,
}

CUSTOM_COLORS = [sns.color_palette()[i] for i in range(len(LANG_TO_DATASET))]

# To plot the 0 in log scale
FAKE_ZERO_TOKENS = 10_000
BREAK_LEFT = FAKE_ZERO_TOKENS * 2.4
BREAK_RIGHT = FAKE_ZERO_TOKENS * 4.3


def parse_ds_size_from_filename(path):
    """
    Extract datastore size string from filename, e.g.
    *_ds-100k.json -> '100k'
    *_ds-none.json -> 'none'
    *_ds-1g.json   -> '1g'
    Returns (size_str, tokens) or (None, None) if not found.
    """
    filename = os.path.basename(path)
    m = re.search(r"ds-([0-9]+k|[0-9]+m|[0-9]+g|none)", filename)
    if not m:
        return None, None
    size_str = m.group(1)
    tokens = SIZE_TO_TOKENS.get(size_str)
    return size_str, tokens


def load_mean_output_throughput(json_path):
    """
    Load JSON and return the mean 'output_throughput' across all results.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        return None
    vals = [r["output_throughput"] for r in results if "output_throughput" in r]
    if not vals:
        return None
    return float(np.mean(vals))


def load_mean_acceptance_length(json_path):
    """
    Load JSON and return the mean 'extra_metrics.avg_acceptance_length'
    across all results.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        return None

    vals = []
    for r in results:
        extra = r.get("extra_metrics", {})
        if "avg_acceptance_length" in extra:
            vals.append(extra["avg_acceptance_length"])

    if not vals:
        return None
    return float(np.mean(vals))


def compute_speedup_for_file(
    dataset_name, target_filename, batch_size, eval_root_dirs=None
):
    """
    Compute the *average* speedup vs autoregressive baseline for a specific file,
    aggregating across multiple evaluation roots.

    - dataset_name: e.g. 'mt-bench'
    - target_filename: e.g. 'SSSD_mt-bench_bs1_ds-1g.json'
    """
    if eval_root_dirs is None:
        eval_root_dirs = EVAL_ROOT_DIRS

    speedups = []

    for root in eval_root_dirs:
        eval_dir = os.path.join(root, dataset_name)

        # Autoregressive baseline for this root
        autoreg_pattern = os.path.join(
            eval_dir, f"Autoregressive_{dataset_name}_bs{batch_size}.json"
        )
        autoreg_files = glob.glob(autoreg_pattern)
        if not autoreg_files:
            print(f"[WARN] No autoregressive file for dataset {dataset_name} in {root}")
            continue

        baseline_throughput = load_mean_output_throughput(autoreg_files[0])
        if baseline_throughput is None or baseline_throughput <= 0:
            print(
                f"[WARN] Failed to read baseline throughput for {dataset_name} in {root}"
            )
            continue

        target_path = os.path.join(eval_dir, target_filename)
        if not os.path.exists(target_path):
            print(f"[WARN] Target file not found: {target_path}")
            continue

        target_thr = load_mean_output_throughput(target_path)
        if target_thr is None:
            print(f"[WARN] Failed to read throughput for {target_path}")
            continue

        speedups.append(target_thr / baseline_throughput)

    if not speedups:
        return None

    return float(np.mean(speedups))


def collect_language_data(dataset_name, batch_size, mixed=False, eval_root_dirs=None):
    """
    Collect speedup vs datastore size for a given dataset (language), averaging
    across multiple evaluation roots.

    mixed=False -> monolingual SSSD (filenames NOT containing 'en-plus'/'english_plus')
    mixed=True  -> mixed-with-English SSSD (filenames containing 'en-plus'/'english_plus')

    Returns:
      x_tokens: list of datastore sizes (tokens)
      speedups: list of *averaged* SSSD speedups
      eagle_speedup: averaged scalar for EAGLE3 (or None if missing)
      eagle_speedup_sl32: averaged scalar for EAGLE3 speclen32 (or None if missing)
      ngram_speedup: averaged scalar for NGRAM (or None if missing)
    """
    if eval_root_dirs is None:
        eval_root_dirs = EVAL_ROOT_DIRS

    sssd_by_tokens = {}  # tokens -> [speedup, speedup, ...]
    eagle_list = []
    eagle_sl32_list = []
    ngram_list = []

    for root in eval_root_dirs:
        eval_dir = os.path.join(root, dataset_name)

        # Autoregressive baseline in this root
        autoreg_pattern = os.path.join(
            eval_dir, f"Autoregressive_{dataset_name}_bs{batch_size}.json"
        )
        autoreg_files = glob.glob(autoreg_pattern)
        if not autoreg_files:
            print(f"[WARN] No autoregressive file for dataset {dataset_name} in {root}")
            continue

        baseline_throughput = load_mean_output_throughput(autoreg_files[0])
        if baseline_throughput is None or baseline_throughput <= 0:
            print(
                f"[WARN] Failed to read baseline throughput for {dataset_name} in {root}"
            )
            continue

        # EAGLE3 (normal)
        eagle_pattern = os.path.join(
            eval_dir, f"EAGLE3_{dataset_name}_bs{batch_size}.json"
        )
        eagle_files = glob.glob(eagle_pattern)
        if eagle_files:
            eagle_thr = load_mean_output_throughput(eagle_files[0])
            if eagle_thr is not None and baseline_throughput > 0:
                eagle_list.append(eagle_thr / baseline_throughput)

        # EAGLE3 speclen 32
        eagle_pattern = os.path.join(
            eval_dir, f"EAGLE3_{dataset_name}_bs{batch_size}_speclen32.json"
        )
        eagle_files = glob.glob(eagle_pattern)
        if eagle_files:
            eagle_thr = load_mean_output_throughput(eagle_files[0])
            if eagle_thr is not None and baseline_throughput > 0:
                eagle_sl32_list.append(eagle_thr / baseline_throughput)

        # NGRAM
        ngram_pattern = os.path.join(
            eval_dir, f"NGRAM_{dataset_name}_bs{batch_size}.json"
        )
        ngram_files = glob.glob(ngram_pattern)
        if ngram_files:
            ngram_thr = load_mean_output_throughput(ngram_files[0])
            if ngram_thr is not None and baseline_throughput > 0:
                ngram_list.append(ngram_thr / baseline_throughput)

        # SSSD
        sssd_pattern = os.path.join(
            eval_dir, f"SSSD_{dataset_name}_bs{batch_size}_*.json"
        )
        sssd_files = glob.glob(sssd_pattern)

        for path in sssd_files:
            filename = os.path.basename(path)
            has_mixed_tag = ("en-plus" in filename) or ("english_plus" in filename)

            # Filter by mixed flag
            if mixed and not has_mixed_tag:
                continue
            if (not mixed) and has_mixed_tag:
                continue

            size_str, tokens = parse_ds_size_from_filename(path)
            if size_str is None or tokens is None:
                continue  # skip any SSSD file without a ds-* pattern

            thr = load_mean_output_throughput(path)
            if thr is None or baseline_throughput <= 0:
                continue

            speedup = thr / baseline_throughput
            sssd_by_tokens.setdefault(tokens, []).append(speedup)

    if not sssd_by_tokens and not eagle_list and not ngram_list:
        return [], [], None, None, None

    # Average SSSD per datastore size
    tokens_sorted = sorted(sssd_by_tokens.keys())
    x_tokens = tokens_sorted
    speedups = [float(np.mean(sssd_by_tokens[t])) for t in tokens_sorted]

    eagle_speedup = float(np.mean(eagle_list)) if eagle_list else None
    eagle_speedup_sl32 = float(np.mean(eagle_sl32_list)) if eagle_sl32_list else None
    ngram_speedup = float(np.mean(ngram_list)) if ngram_list else None

    return x_tokens, speedups, eagle_speedup, eagle_speedup_sl32, ngram_speedup


def collect_language_acceptance_data(
    dataset_name, batch_size, mixed=False, eval_root_dirs=None, load_sssd_hf=False
):
    """
    Collect avg_acceptance_length vs datastore size for a given dataset (language),
    averaging across multiple evaluation roots.

    mixed=False -> monolingual SSSD
    mixed=True  -> mixed-with-English SSSD

    Returns:
      x_tokens: list of datastore sizes (tokens)
      accs: list of *averaged* avg_acceptance_length for SSSD
      eagle_acc: averaged scalar for EAGLE3 (or None if missing)
      eagle_acc_sl32: averaged scalar for EAGLE3 speclen32 (or None if missing)
      ngram_acc: averaged scalar for NGRAM (or None if missing)
    """
    if eval_root_dirs is None:
        eval_root_dirs = EVAL_ROOT_DIRS

    sssd_acc_by_tokens = {}  # tokens -> [acc, acc, ...]
    eagle_acc_list = []
    eagle_acc_sl32_list = []
    ngram_acc_list = []
    sssd_hf_acc_by_tokens = {}

    for root in eval_root_dirs:
        eval_dir = os.path.join(root, dataset_name)

        # EAGLE3
        eagle_pattern = os.path.join(
            eval_dir, f"EAGLE3_{dataset_name}_bs{batch_size}.json"
        )
        eagle_files = glob.glob(eagle_pattern)
        if eagle_files:
            acc = load_mean_acceptance_length(eagle_files[0])
            if acc is not None:
                eagle_acc_list.append(acc)

        # EAGLE3 speclen 32
        eagle_pattern = os.path.join(
            eval_dir, f"EAGLE3_{dataset_name}_bs{batch_size}_speclen32.json"
        )
        eagle_files = glob.glob(eagle_pattern)
        if eagle_files:
            acc = load_mean_acceptance_length(eagle_files[0])
            if acc is not None:
                eagle_acc_sl32_list.append(acc)

        # NGRAM
        ngram_pattern = os.path.join(
            eval_dir, f"NGRAM_{dataset_name}_bs{batch_size}.json"
        )
        ngram_files = glob.glob(ngram_pattern)
        if ngram_files:
            acc = load_mean_acceptance_length(ngram_files[0])
            if acc is not None:
                ngram_acc_list.append(acc)

        # SSSD
        sssd_pattern = os.path.join(
            eval_dir, f"SSSD_{dataset_name}_bs{batch_size}_*.json"
        )
        sssd_files = glob.glob(sssd_pattern)

        for path in sssd_files:
            filename = os.path.basename(path)
            has_mixed_tag = ("en-plus" in filename) or ("english_plus" in filename)

            if mixed and not has_mixed_tag:
                continue
            if (not mixed) and has_mixed_tag:
                continue

            size_str, tokens = parse_ds_size_from_filename(path)
            if size_str is None or tokens is None:
                continue  # skip any SSSD file without a ds-* pattern

            acc = load_mean_acceptance_length(path)
            if acc is None:
                continue

            sssd_acc_by_tokens.setdefault(tokens, []).append(acc)

    # SSSD from hf data (not generated)
    if load_sssd_hf:
        eval_dir = os.path.join(EVAL_ROOT_DIR_NOT_GENERATED_SSSD, dataset_name)
        sssd_pattern = os.path.join(
            eval_dir, f"SSSD_{dataset_name}_bs{batch_size}_*.json"
        )
        sssd_files = glob.glob(sssd_pattern)

        for path in sssd_files:
            filename = os.path.basename(path)
            has_mixed_tag = ("en-plus" in filename) or ("english_plus" in filename)

            if mixed and not has_mixed_tag:
                continue
            if (not mixed) and has_mixed_tag:
                continue

            size_str, tokens = parse_ds_size_from_filename(path)
            if size_str is None or tokens is None:
                continue  # skip any SSSD file without a ds-* pattern

            acc = load_mean_acceptance_length(path)
            if acc is None:
                continue

            sssd_hf_acc_by_tokens.setdefault(tokens, []).append(acc)

    if not sssd_acc_by_tokens and not eagle_acc_list and not ngram_acc_list:
        return [], [], None, None, None, []

    tokens_sorted = sorted(sssd_acc_by_tokens.keys())
    x_tokens = tokens_sorted
    accs = [float(np.mean(sssd_acc_by_tokens[t])) for t in tokens_sorted]

    eagle_acc = float(np.mean(eagle_acc_list)) if eagle_acc_list else None
    eagle_acc_sl32 = (
        float(np.mean(eagle_acc_sl32_list)) if eagle_acc_sl32_list else None
    )
    ngram_acc = float(np.mean(ngram_acc_list)) if ngram_acc_list else None

    if load_sssd_hf:
        tokens_sorted = sorted(sssd_hf_acc_by_tokens.keys())
        x_tokens = tokens_sorted
        accs_hf = [float(np.mean(sssd_hf_acc_by_tokens[t])) for t in tokens_sorted]
    else:
        accs_hf = None

    return x_tokens, accs, eagle_acc, eagle_acc_sl32, ngram_acc, accs_hf


def add_log_minor_grid_from(ax, start=100_000):
    """
    Add logarithmic minor grid lines on the x-axis, but only for x >= start.
    Keeps major ticks (powers of 10) as usual and does not touch your fake zero.
    """
    ax.set_xscale("log")

    # Major ticks at powers of 10
    ax.xaxis.set_major_locator(LogLocator(base=10.0))

    # Figure out x-range actually used in the plot
    xmin, xmax = ax.get_xlim()

    # Determine exponents to cover [max(start, xmin), xmax]
    lower = max(start, xmin)
    if lower <= 0:
        lower = start
    min_exp = int(np.floor(np.log10(lower)))
    max_exp = int(np.ceil(np.log10(xmax)))

    minor_ticks = []
    for exp in range(min_exp, max_exp + 1):
        decade = 10.0**exp
        for k in range(2, 10):  # 2*10^exp ... 9*10^exp
            val = decade * k
            if val < start or val < xmin or val > xmax:
                continue
            minor_ticks.append(val)

    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))

    # Grid: keep major grid (powers of 10), and add lighter minor grid
    ax.grid(True, which="major", axis="x", linewidth=1.2, alpha=1.0)
    ax.grid(True, which="minor", axis="x", linewidth=0.8, alpha=0.8)


def zero_log_formatter(x, pos):
    """
    Formatter for the x-axis (tokens).

    - Tick at FAKE_ZERO_TOKENS is shown as '0'
    - Other ticks shown as powers of 10, e.g. 10^4, 10^5, ...
    """
    if np.isclose(x, FAKE_ZERO_TOKENS):
        return "0"

    # Show other ticks as powers of 10
    if x <= 0:
        return ""
    exp = int(np.round(np.log10(x)))
    return rf"$10^{exp}$"


def add_xaxis_break(
    ax, x_left, x_right, y=0.0, halfheight=0.02, halfwidth=0.01, lw=1.3, color="0.7"
):
    """
    Draw diagonal break marks on the x-axis.

    x_left/x_right: DATA coords
    y, halfheight, halfwidth: AXES fraction
    """

    # helper: data x -> axes fraction x
    def data_x_to_axes_frac(x):
        return ax.transAxes.inverted().transform(ax.transData.transform((x, 0)))[0]

    for xb in (x_left, x_right):
        xa = data_x_to_axes_frac(xb)

        ax.plot(
            [xa - halfwidth, xa + halfwidth],
            [y - halfheight, y + halfheight],
            transform=ax.transAxes,  # <-- key change
            color=color,
            lw=lw,
            solid_capstyle="butt",
            clip_on=False,
            zorder=60,
        )


def mask_xaxis_spine(ax, x_left, x_right, height=0.03):
    """
    Hide a portion of the bottom x-axis spine between x_left and x_right
    by covering it with a background-colored rectangle.

    height is in axes fraction.
    """
    import matplotlib.patches as patches

    # Convert data x -> axes fraction
    x0 = ax.transAxes.inverted().transform(ax.transData.transform((x_left, 0)))[0]
    x1 = ax.transAxes.inverted().transform(ax.transData.transform((x_right, 0)))[0]

    rect = patches.Rectangle(
        (x0, -height / 2),
        x1 - x0,
        height,
        transform=ax.transAxes,
        facecolor=ax.figure.get_facecolor(),
        edgecolor="none",
        clip_on=False,
        zorder=9,  # below break markers, above spine
    )

    ax.add_patch(rect)


# === MAIN PLOTTING ===


def main():
    global english_sssd_at_max, all_languages_on_english

    languages = list(LANG_TO_DATASET.keys())
    datasets = [LANG_TO_DATASET[lang] for lang in languages]

    colors = CUSTOM_COLORS

    # Compute english_sssd_at_max from the 1g SSSD file vs autoregressive
    english_sssd_at_max = compute_speedup_for_file(
        dataset_name=LANG_TO_DATASET["English"],
        target_filename=f"SSSD_{LANG_TO_DATASET['English']}_bs{BATCH_SIZE}_ds-1g.json",
        batch_size=BATCH_SIZE,
        eval_root_dirs=EVAL_ROOT_DIRS,
    )

    # Compute all_languages_on_english from the all_languages SSSD file vs autoregressive
    all_languages_on_english = compute_speedup_for_file(
        dataset_name=LANG_TO_DATASET["English"],
        target_filename=f"SSSD_{LANG_TO_DATASET['English']}_bs{BATCH_SIZE}_all_languages.json",
        batch_size=BATCH_SIZE,
        eval_root_dirs=EVAL_ROOT_DIRS,
    )

    # -------- Plot 1: Individual languages (monolingual datastores, throughput) --------
    fig1, ax1 = plt.subplots(figsize=(6, 5))

    # First pass: collect all data to determine global x-range
    lang_data = []

    for i, (lang, dataset_name) in enumerate(zip(languages, datasets)):
        (
            x_tokens,
            speedups,
            eagle_speedup,
            eagle_speedup_sl32,
            ngram_speedup,
        ) = collect_language_data(
            dataset_name, BATCH_SIZE, mixed=False, eval_root_dirs=EVAL_ROOT_DIRS
        )

        if len(x_tokens) == 0:
            print(f"[INFO] No monolingual SSSD data for {lang}")
            continue

        x_tokens = np.array(x_tokens, dtype=float)
        speedups = np.array(speedups, dtype=float)

        # Replace the last x-coordinate with the actual datastore size if provided
        actual_last = ACTUAL_LAST_TOKENS.get(lang)
        if actual_last is not None:
            x_tokens[-1] = actual_last

        zero_mask = x_tokens == 0
        x_tokens[zero_mask] = FAKE_ZERO_TOKENS

        x_min = float(x_tokens.min())
        x_max = float(x_tokens.max())

        lang_data.append(
            dict(
                lang=lang,
                color=colors[i],
                x_tokens=x_tokens,
                speedups=speedups,
                eagle=eagle_speedup,
                eagle_sl32=eagle_speedup_sl32,
                ngram=ngram_speedup,
                x_min=x_min,
                x_max=x_max,
            )
        )

    if not lang_data:
        print("[WARN] No monolingual data found at all.")
    else:
        # Global x-range for all horizontal lines
        global_x_min = min(d["x_min"] for d in lang_data)
        global_x_max = max(d["x_max"] for d in lang_data)

        # Second pass: actually plot
        for d in lang_data:
            c = d["color"]

            # SSSD line
            ax1.plot(
                d["x_tokens"],
                d["speedups"],
                marker="o",
                linestyle="-",
                lw=2.5,
                color=c,
                zorder=3,
            )

            # EAGLE horizontal lines, all using the SAME xmin/xmax
            if d["eagle"] is not None:
                ax1.hlines(
                    y=d["eagle"],
                    xmin=global_x_min,
                    xmax=global_x_max,
                    lw=2.5,
                    linestyle=(0, (3, 1)),
                    color=c,
                    zorder=2,
                    alpha=0.7,
                )

            if d["eagle_sl32"] is not None:
                ax1.hlines(
                    y=d["eagle_sl32"],
                    xmin=global_x_min,
                    xmax=global_x_max,
                    lw=2.5,
                    linestyle=(0, (4, 2, 1, 2)),
                    color=c,
                    zorder=2,
                    alpha=0.7,
                )

            # Lookahead (NGRAM) horizontal "segment" between fake 0 and 100k
            if PLOT_LOOKAHEAD and d["ngram"] is not None:
                xs_ngram = np.logspace(
                    np.log10(FAKE_ZERO_TOKENS), np.log10(100_000), num=30
                )
                ys_ngram = np.full(xs_ngram.shape, d["ngram"])
                ax1.plot(
                    xs_ngram,
                    ys_ngram,
                    lw=2.5,
                    linestyle=(0, (1, 1)),
                    color=c,
                    zorder=2,
                )

    ax1.set_xlabel("Datastore Size [tokens]")
    ax1.set_ylabel("Throughput increase")
    ax1.set_xscale("log")
    add_log_minor_grid_from(ax1, start=100_000)
    ax1.xaxis.set_major_formatter(FuncFormatter(zero_log_formatter))
    mask_xaxis_spine(ax1, BREAK_LEFT, BREAK_RIGHT, height=0.005)
    add_xaxis_break(ax1, BREAK_LEFT, BREAK_RIGHT, y=0.0)
    ax1.set_ylim(top=SPEEDUP_PLOTS_YLIM)

    # Color legend: one entry per language/dataset
    color_handles = [Line2D([0], [0], color=d["color"], lw=2.5) for d in lang_data]
    color_labels = [d["lang"] for d in lang_data]

    # Style legend: one entry per method
    style_handles = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", marker="o"),
        Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (3, 1)), alpha=0.7),
        Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (4, 2, 1, 2)), alpha=0.7),
    ]
    style_labels = ["SSSD", "EAGLE3", "EAGLE3 sl32"]

    if PLOT_LOOKAHEAD:
        style_handles.append(
            Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (1, 1)))
        )
        style_labels.append("Lookahead")

    legend1 = ax1.legend(
        handles=color_handles,
        labels=color_labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 0),
        borderaxespad=0.5,
    )
    ax1.add_artist(legend1)

    legend2 = ax1.legend(
        handles=style_handles,
        labels=style_labels,
        loc="lower right",
        bbox_to_anchor=(0.69, 0),
        borderaxespad=0.5,
    )

    fig1.tight_layout()
    fig1.savefig(BASE_DIR + "plots/multilingual.png", dpi=300)

    # -------- Plot 2: Languages mixed with English (throughput) --------
    fig2, ax2 = plt.subplots(figsize=(6, 5))

    # Add results on English (only if computed successfully)
    if english_sssd_at_max is not None:
        ax2.scatter(
            FAKE_ZERO_TOKENS,
            english_sssd_at_max,
            color=CUSTOM_COLORS[languages.index("English")],
            marker="o",
        )
        en_toks_int = round(ACTUAL_LAST_TOKENS["English"] / 1e6)
        ax2.text(
            FAKE_ZERO_TOKENS * 1.10,
            english_sssd_at_max,
            f"English only,\n{en_toks_int}×$10^6$ toks",
            va="center",
            ha="left",
            fontsize=12,
        )

    tot_other_langs_toks = 0
    for key, value in ACTUAL_LAST_TOKENS.items():
        if key != "English":
            tot_other_langs_toks += value

    if all_languages_on_english is not None:
        ax2.scatter(
            tot_other_langs_toks,
            all_languages_on_english,
            color=CUSTOM_COLORS[languages.index("English")],
            marker="o",
        )
        ax2.text(
            tot_other_langs_toks * 0.1,
            all_languages_on_english + 0.01,
            "All languages\n  on English",
            va="center",
            ha="left",
            fontsize=12,
        )

    lang_data_mixed = []

    for i, (lang, dataset_name) in enumerate(zip(languages, datasets)):
        if lang == "English":
            continue

        (
            x_tokens,
            speedups,
            eagle_speedup,
            eagle_speedup_sl32,
            ngram_speedup,
        ) = collect_language_data(
            dataset_name, BATCH_SIZE, mixed=True, eval_root_dirs=EVAL_ROOT_DIRS
        )

        if len(x_tokens) == 0:
            print(f"[INFO] No mixed (English+{lang}) SSSD data")
            continue

        x_tokens = np.array(x_tokens, dtype=float)
        speedups = np.array(speedups, dtype=float)

        actual_last = ACTUAL_LAST_TOKENS.get(lang)
        if actual_last is not None:
            x_tokens[-1] = actual_last

        zero_mask = x_tokens == 0
        x_tokens[zero_mask] = FAKE_ZERO_TOKENS

        x_min = float(x_tokens.min())
        x_max = float(x_tokens.max())

        lang_data_mixed.append(
            dict(
                lang=lang,
                color=CUSTOM_COLORS[i],
                x_tokens=x_tokens,
                speedups=speedups,
                eagle=eagle_speedup,
                eagle_sl32=eagle_speedup_sl32,
                x_min=x_min,
                x_max=x_max,
            )
        )

    if not lang_data_mixed:
        print("[WARN] No mixed-with-English data found.")
    else:
        global_x_min = min(d["x_min"] for d in lang_data_mixed)
        global_x_max = max(d["x_max"] for d in lang_data_mixed)

        for d in lang_data_mixed:
            c = d["color"]

            # SSSD mixed line
            ax2.plot(
                d["x_tokens"],
                d["speedups"],
                marker="o",
                linestyle="-",
                lw=2.5,
                color=c,
                zorder=3,
            )

            # EAGLE horizontal lines with common x-range
            if d["eagle"] is not None:
                ax2.hlines(
                    y=d["eagle"],
                    xmin=global_x_min,
                    xmax=tot_other_langs_toks,
                    lw=2.5,
                    linestyle=(0, (3, 1)),
                    color=c,
                    zorder=2,
                    alpha=0.7,
                )

            if d["eagle_sl32"] is not None:
                ax2.hlines(
                    y=d["eagle_sl32"],
                    xmin=global_x_min,
                    xmax=tot_other_langs_toks,
                    lw=2.5,
                    linestyle=(0, (4, 2, 1, 2)),
                    color=c,
                    zorder=2,
                    alpha=0.7,
                )

    ax2.set_xlabel("Additional tokens on top of the English datastore")
    ax2.set_ylabel("Throughput increase")
    ax2.set_xscale("log")
    add_log_minor_grid_from(ax2, start=100_000)
    ax2.xaxis.set_major_formatter(FuncFormatter(zero_log_formatter))
    mask_xaxis_spine(ax2, BREAK_LEFT, BREAK_RIGHT, height=0.005)
    add_xaxis_break(ax2, BREAK_LEFT, BREAK_RIGHT, y=0.0)
    ax2.set_ylim(top=SPEEDUP_PLOTS_YLIM)

    color_handles_mixed = [
        Line2D([0], [0], color=d["color"], lw=2.5) for d in lang_data_mixed
    ]
    color_labels_mixed = [f"{d['lang']}" for d in lang_data_mixed]

    style_handles_mixed = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", marker="o"),
        Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (3, 1)), alpha=0.7),
        Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (4, 2, 1, 2)), alpha=0.7),
    ]
    style_labels_mixed = ["SSSD", "EAGLE3", "EAGLE3 sl32"]

    legend1 = ax2.legend(
        handles=color_handles_mixed,
        labels=color_labels_mixed,
        loc="lower right",
        bbox_to_anchor=(1.0, 0),
        borderaxespad=0.5,
    )
    ax2.add_artist(legend1)

    legend2 = ax2.legend(
        handles=style_handles_mixed,
        labels=style_labels_mixed,
        loc="lower right",
        bbox_to_anchor=(0.69, 0),
        borderaxespad=0.5,
    )

    fig2.tight_layout()
    fig2.savefig(BASE_DIR + "plots/multilingual_with_english.png", dpi=300)

    # -------- Plot 3: Individual languages (monolingual datastores, acceptance length) --------
    fig3, ax3 = plt.subplots(figsize=(6, 5))

    lang_data_acc = []

    for i, (lang, dataset_name) in enumerate(zip(languages, datasets)):
        x_tokens, accs, eagle_acc, eagle_acc_sl32, ngram_acc, accs_hf = (
            collect_language_acceptance_data(
                dataset_name,
                BATCH_SIZE,
                mixed=False,
                eval_root_dirs=EVAL_ROOT_DIRS,
                load_sssd_hf=True,
            )
        )

        if len(x_tokens) == 0:
            print(f"[INFO] No monolingual SSSD acceptance data for {lang}")
            continue

        x_tokens = np.array(x_tokens, dtype=float)
        accs = np.array(accs, dtype=float)
        accs_hf = np.array(accs_hf, dtype=float)

        actual_last = ACTUAL_LAST_TOKENS.get(lang)
        if actual_last is not None:
            x_tokens[-1] = actual_last

        zero_mask = x_tokens == 0
        x_tokens[zero_mask] = FAKE_ZERO_TOKENS

        # Data coming from hf comes in different amount, so has a different number of tokens for the last datastore
        x_tokens_hf = x_tokens.copy()
        actual_last = ACTUAL_LAST_TOKENS_HF_DATA.get(lang)
        if actual_last is not None:
            x_tokens_hf[-1] = actual_last
        zero_mask = x_tokens_hf == 0
        x_tokens_hf[zero_mask] = FAKE_ZERO_TOKENS

        x_min = float(x_tokens.min())
        x_max = float(x_tokens.max())

        lang_data_acc.append(
            dict(
                lang=lang,
                color=CUSTOM_COLORS[i],
                x_tokens=x_tokens,
                accs=accs,
                eagle=eagle_acc,
                eagle_sl32=eagle_acc_sl32,
                ngram=ngram_acc,
                x_min=x_min,
                x_max=x_max,
                x_tokens_hf=x_tokens_hf,
                accs_hf=accs_hf,
            )
        )

    if not lang_data_acc:
        print("[WARN] No monolingual acceptance data found at all.")
    else:
        global_x_min = min(d["x_min"] for d in lang_data_acc)
        global_x_max = max(d["x_max"] for d in lang_data_acc)

        for d in lang_data_acc:
            c = d["color"]

            # SSSD line (acceptance length)
            ax3.plot(
                d["x_tokens"],
                d["accs"],
                marker="o",
                linestyle="-",
                lw=2.5,
                color=c,
                zorder=3,
            )

            ax3.plot(
                d["x_tokens_hf"],
                d["accs_hf"],
                marker="D",
                ms=4,
                # linestyle="-",
                linestyle=(0, (1, 1)),
                lw=2.5,
                color=c,
                zorder=3,
            )

            # EAGLE horizontal lines with same x-range
            if d["eagle"] is not None:
                ax3.hlines(
                    y=d["eagle"],
                    xmin=global_x_min,
                    xmax=global_x_max,
                    lw=2.5,
                    linestyle=(0, (3, 1)),
                    color=c,
                    zorder=2,
                    alpha=0.7,
                )

            if d["eagle_sl32"] is not None:
                ax3.hlines(
                    y=d["eagle_sl32"],
                    xmin=global_x_min,
                    xmax=global_x_max,
                    lw=2.5,
                    linestyle=(0, (4, 2, 1, 2)),
                    color=c,
                    zorder=2,
                    alpha=0.7,
                )

            if PLOT_LOOKAHEAD and d["ngram"] is not None:
                xs_ngram = np.logspace(
                    np.log10(FAKE_ZERO_TOKENS), np.log10(100_000), num=30
                )
                ys_ngram = np.full(xs_ngram.shape, d["ngram"])
                ax3.plot(
                    xs_ngram,
                    ys_ngram,
                    lw=2.5,
                    linestyle=(0, (1, 1)),
                    color=c,
                    zorder=2,
                )

    ax3.set_xlabel("Datastore Size [tokens]")
    ax3.set_ylabel("Average acceptance length")
    ax3.set_xscale("log")
    add_log_minor_grid_from(ax3, start=100_000)
    mask_xaxis_spine(ax3, BREAK_LEFT, BREAK_RIGHT, height=0.005)
    add_xaxis_break(ax3, BREAK_LEFT, BREAK_RIGHT, y=0.0)
    ax3.xaxis.set_major_formatter(FuncFormatter(zero_log_formatter))

    color_handles_acc = [
        Line2D([0], [0], color=d["color"], lw=2.5) for d in lang_data_acc
    ]
    color_labels_acc = [d["lang"] for d in lang_data_acc]

    style_handles_acc = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", marker="o"),
        Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (3, 1)), alpha=0.7),
        Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (4, 2, 1, 2)), alpha=0.7),
    ]
    style_labels_acc = ["SSSD", "EAGLE3", "EAGLE3 sl32"]
    if PLOT_LOOKAHEAD:
        style_handles_acc.append(
            Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (1, 1)))
        )
        style_labels_acc.append("Lookahead")

    # For SSSD with HF data
    style_handles_acc.append(
        Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (1, 1)), marker="D", ms=4)
    )
    style_labels_acc.append("SSSD (original data)")

    legend1 = ax3.legend(
        handles=color_handles_acc,
        labels=color_labels_acc,
        loc="lower right",
        bbox_to_anchor=(1.0, 0),
        borderaxespad=0.5,
    )
    ax3.add_artist(legend1)

    legend2 = ax3.legend(
        handles=style_handles_acc,
        labels=style_labels_acc,
        loc="upper left",
        # bbox_to_anchor=(0.69, 0),
        # borderaxespad=0.5,
    )

    fig3.tight_layout()
    fig3.savefig(BASE_DIR + "plots/multilingual_acceptance.png", dpi=300)

    # -------- Plot 4: Languages mixed with English (acceptance length) --------
    fig4, ax4 = plt.subplots(figsize=(6, 5))

    lang_data_acc_mixed = []

    for i, (lang, dataset_name) in enumerate(zip(languages, datasets)):
        if lang == "English":
            continue

        x_tokens, accs, eagle_acc, eagle_acc_sl32, ngram_acc, _ = (
            collect_language_acceptance_data(
                dataset_name,
                BATCH_SIZE,
                mixed=True,
                eval_root_dirs=EVAL_ROOT_DIRS,
                load_sssd_hf=False,
            )
        )

        if len(x_tokens) == 0:
            print(f"[INFO] No mixed (English+{lang}) SSSD acceptance data")
            continue

        x_tokens = np.array(x_tokens, dtype=float)
        accs = np.array(accs, dtype=float)

        actual_last = ACTUAL_LAST_TOKENS.get(lang)
        if actual_last is not None:
            x_tokens[-1] = actual_last

        zero_mask = x_tokens == 0
        x_tokens[zero_mask] = FAKE_ZERO_TOKENS

        x_min = float(x_tokens.min())
        x_max = float(x_tokens.max())

        lang_data_acc_mixed.append(
            dict(
                lang=lang,
                color=CUSTOM_COLORS[i],
                x_tokens=x_tokens,
                accs=accs,
                eagle=eagle_acc,
                eagle_sl32=eagle_acc_sl32,
                x_min=x_min,
                x_max=x_max,
            )
        )

    if not lang_data_acc_mixed:
        print("[WARN] No mixed-with-English acceptance data found.")
    else:
        global_x_min = min(d["x_min"] for d in lang_data_acc_mixed)
        global_x_max = max(d["x_max"] for d in lang_data_acc_mixed)

        for d in lang_data_acc_mixed:
            c = d["color"]

            # SSSD mixed line
            ax4.plot(
                d["x_tokens"],
                d["accs"],
                marker="o",
                linestyle="-",
                lw=2.5,
                color=c,
                zorder=3,
            )

            # EAGLE horizontal lines with common x-range
            if d["eagle"] is not None:
                ax4.hlines(
                    y=d["eagle"],
                    xmin=global_x_min,
                    xmax=global_x_max,
                    lw=2.5,
                    linestyle=(0, (3, 1)),
                    color=c,
                    zorder=2,
                    alpha=0.7,
                )

            if d["eagle_sl32"] is not None:
                ax4.hlines(
                    y=d["eagle_sl32"],
                    xmin=global_x_min,
                    xmax=global_x_max,
                    lw=2.5,
                    linestyle=(0, (4, 2, 1, 2)),
                    color=c,
                    zorder=2,
                    alpha=0.7,
                )

    ax4.set_xlabel("Datastore Size [tokens]")
    ax4.set_ylabel("Average acceptance length")
    ax4.set_xscale("log")
    add_log_minor_grid_from(ax4, start=100_000)
    ax4.xaxis.set_major_formatter(FuncFormatter(zero_log_formatter))
    mask_xaxis_spine(ax4, BREAK_LEFT, BREAK_RIGHT, height=0.005)
    add_xaxis_break(ax4, BREAK_LEFT, BREAK_RIGHT, y=0.0)

    color_handles_acc_mixed = [
        Line2D([0], [0], color=d["color"], lw=2.5) for d in lang_data_acc_mixed
    ]
    color_labels_acc_mixed = [f"{d['lang']}" for d in lang_data_acc_mixed]

    style_handles_acc_mixed = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", marker="o"),
        Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (3, 1)), alpha=0.7),
        Line2D([0], [0], color="black", lw=2.5, linestyle=(0, (4, 2, 1, 2)), alpha=0.7),
    ]
    style_labels_acc_mixed = ["SSSD", "EAGLE3", "EAGLE3 sl32"]

    legend1 = ax4.legend(
        handles=color_handles_acc_mixed,
        labels=color_labels_acc_mixed,
        loc="lower right",
        bbox_to_anchor=(1.0, 0),
        borderaxespad=0.5,
    )
    ax4.add_artist(legend1)

    legend2 = ax4.legend(
        handles=style_handles_acc_mixed,
        labels=style_labels_acc_mixed,
        loc="lower right",
        bbox_to_anchor=(0.69, 0),
        borderaxespad=0.5,
    )

    fig4.tight_layout()
    fig4.savefig(BASE_DIR + "plots/multilingual_with_english_acceptance.png", dpi=300)


if __name__ == "__main__":
    main()
