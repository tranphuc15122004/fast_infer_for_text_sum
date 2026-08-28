"""Plots speedups from the collected latencies of different methods
and batch sizes in a bar chart.
"""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sglang.utils import read_json


def get_number_at_end(s):
    match = re.search(r"(\d+)$", s)
    if match:
        return int(match.group(1))
    return None


def construct_result_df(
    data: dict, only_batch_sizes: list | None = None, baseline: str = "Autoregressive"
) -> tuple[pd.DataFrame, str]:
    eval_data = data["evaluation"]
    num_benchmarks = len(eval_data.keys())
    assert (
        num_benchmarks == 1
    ), f"Expected exactly one benchmark in the data. Found {num_benchmarks} benchmarks."
    benchmark_name = next(iter(eval_data))
    benchmark_data = eval_data[benchmark_name]
    algorithms = list(benchmark_data.keys())
    algorithms.remove(baseline)
    algorithms.insert(0, baseline)  # Ensure baseline is first
    batch_sizes = list(benchmark_data[algorithms[0]].keys())

    out_dict = {"Batch size": [], "Method": [], "Speed-up (x)": []}
    baseline_latencies = {
        k: benchmark_data[baseline][k]["total_latency"]
        for k in benchmark_data[baseline]
    }

    for b in batch_sizes:
        if only_batch_sizes and get_number_at_end(b) not in only_batch_sizes:
            continue
        for a in algorithms:
            out_dict["Batch size"].append(get_number_at_end(b))
            out_dict["Method"].append(a)
            out_dict["Speed-up (x)"].append(
                baseline_latencies[b] / benchmark_data[a][b]["total_latency"]
            )
    return pd.DataFrame(out_dict), benchmark_name


def plot_speedups(df: pd.DataFrame, benchmark_name: str, out_dir: str):
    # Set the style and context for the plot
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Create the figure and axes for the plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df, x="Batch size", y="Speed-up (x)", hue="Method")
    # hue_order=['Autoregressive', 'EAGLE', 'EAGLE-3', 'SSSD'])

    # Add the value labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        if (
            height > 0
        ):  # Only annotate non-zero heights (sns seems to generate some empty bars)
            ax.annotate(
                f"{p.get_height():.2f}x",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=10,
            )

    # Add labelling
    ax.set_xlabel("Batch size", fontsize=16)
    ax.set_ylabel("Speed-up (x)", fontsize=16)
    ax.set_ylim(0.5, df["Speed-up (x)"].max() * 1.2)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(
        axis="y", left=False, labelleft=False
    )  # Hide y-axis ticks and labels
    ax.set_title(f"Speed-up Comparison for {benchmark_name}", fontsize=20)
    # Remove the grid and spines for a cleaner look
    ax.grid(False)
    sns.despine(left=True, bottom=True)

    ax.legend()

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    os.makedirs(out_dir, exist_ok=True)

    # Display the plot
    plt.savefig(f"{out_dir}/speedup_comparison.png", dpi=300)


if __name__ == "__main__":
    # Change as needed
    result_path = "data/collected_results.json"
    out_dir = "data/analysis/h100"
    data = read_json(result_path)
    df, benchmark_name = construct_result_df(
        data, only_batch_sizes=[1, 4, 8, 16, 32, 48, 64]
    )
    plot_speedups(df, benchmark_name, out_dir)
    print("Speedup plot saved successfully.")
