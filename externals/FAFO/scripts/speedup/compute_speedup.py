"""Read baseline + FAFO output_config.json files and print speedup = fafo/baseline.

Usage:
    python scripts/speedup/compute_speedup.py <out_root> [dataset ...]

Expects, for each dataset, the layout produced by run_speedup.sh:
    <out_root>/<dataset>/baseline/output_config.json
    <out_root>/<dataset>/fafo/output_config.json
"""
import json
import os
import sys


def load_throughputs(path):
    """Return (avg_throughput_1, avg_throughput_2, avg_compression_ratio) or None."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        cfg = json.load(f)
    pr = cfg.get("eval_results", {}).get("processed_results", {})
    return (
        pr.get("avg_throughput_1"),
        pr.get("avg_throughput_2"),
        pr.get("avg_compression_ratio"),
    )


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else "results/speedup"
    datasets = sys.argv[2:] or ["gsm8k", "humaneval", "mtbench"]

    header = f"{'dataset':<12}{'baseline tok/s':>16}{'fafo tok/s':>14}{'speedup':>10}{'fafo compr':>12}"
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for ds in datasets:
        base = load_throughputs(os.path.join(out_root, ds, "baseline", "output_config.json"))
        fafo = load_throughputs(os.path.join(out_root, ds, "fafo", "output_config.json"))
        if base is None or fafo is None:
            missing = "baseline" if base is None else "fafo"
            print(f"{ds:<12}  [missing {missing} output_config.json]")
            continue
        # Use avg_throughput_2 (overall tokens / total time) as the headline metric.
        b_tp = base[1]
        f_tp = fafo[1]
        speedup = (f_tp / b_tp) if (b_tp and f_tp) else float("nan")
        compr = fafo[2]
        print(f"{ds:<12}{b_tp:>16.2f}{f_tp:>14.2f}{speedup:>10.2f}x{compr:>11.2f}")
        rows.append({
            "dataset": ds,
            "baseline_throughput": b_tp,
            "fafo_throughput": f_tp,
            "speedup": speedup,
            "fafo_compression_ratio": compr,
        })

    print()
    summary_path = os.path.join(out_root, "speedup_summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"summary written to {summary_path}\n")


if __name__ == "__main__":
    main()
