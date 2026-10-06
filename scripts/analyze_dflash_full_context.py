#!/usr/bin/env python3
"""Phân tích attention DFlash trên mọi key, luôn dùng tổng mass làm mẫu số."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.io_util import JsonlWriter

BUDGETS = [256, 512, 1024, 2048, 4096, 8192]
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 1.0]


def canonical_ids(indices, prefix):
    """Prefix giữ absolute position; current block được đối chiếu theo slot."""
    return np.where(indices < prefix, indices, -(indices - prefix + 1))


def compare_selection(old, new, probabilities, prefix):
    common = len(np.intersect1d(old, new, assume_unique=True)) / len(old)
    positions = np.where(old < 0, prefix - old - 1, old)
    # Prefix tăng theo decoding; mọi key đã commit trước đó vẫn tồn tại.
    retained = float(probabilities[positions].sum())
    return common, retained


def summarize_vector(probabilities):
    n = len(probabilities)
    order = np.argsort(probabilities, kind="stable")[::-1]
    cumulative_rank = np.cumsum(probabilities[order])
    cumulative_position = np.concatenate(([0.0], np.cumsum(probabilities)))
    entropy = float(-np.sum(probabilities[probabilities > 0] *
                            np.log(probabilities[probabilities > 0])))
    metrics = {
        "key_tokens": n,
        "entropy_normalized": entropy / np.log(n),
        "effective_keys_entropy": float(np.exp(entropy)),
        "effective_keys_participation": float(1.0 / np.sum(probabilities ** 2)),
        "top16": float(cumulative_rank[min(16, n) - 1]),
        "top64": float(cumulative_rank[min(64, n) - 1]),
    }
    for target in [90, 95, 99]:
        count = min(n, int(np.searchsorted(cumulative_rank, target / 100)) + 1)
        metrics[f"keys_for_{target}pct"] = count
        metrics[f"fraction_keys_for_{target}pct"] = count / n
    selections = {}
    for budget in BUDGETS:
        width = min(budget, n)
        windows = cumulative_position[width:] - cumulative_position[:-width]
        start = int(np.argmax(windows))
        middle_start = (n - width) // 2
        head_width = width // 2
        tail_width = width - head_width
        metrics.update({
            f"top_{budget}": float(cumulative_rank[width - 1]),
            f"uniform_{budget}": width / n,
            f"best_window_{budget}": float(windows[start]),
            f"best_window_{budget}_start_fraction": start / max(1, n - width),
            f"head_{budget}": float(probabilities[:width].sum()),
            f"tail_{budget}": float(probabilities[-width:].sum()),
            f"middle_{budget}": float(probabilities[middle_start:middle_start + width].sum()),
            f"edges_{budget}": float(probabilities[:head_width].sum() +
                                     probabilities[-tail_width:].sum()),
        })
        if budget in [1024, 4096]:
            selections[budget] = order[:width]
    for fraction in FRACTIONS:
        width = min(n, int(np.ceil(fraction * n)))
        metrics[f"top_fraction_{round(fraction * 100)}pct"] = float(cumulative_rank[width - 1])
    boundaries = np.linspace(0, n, 11, dtype=int)
    for index in range(10):
        metrics[f"position_bin_{index}"] = float(probabilities[boundaries[index]:boundaries[index + 1]].sum())
    metrics["edges_20pct"] = metrics["position_bin_0"] + metrics["position_bin_9"]
    return metrics, selections


def aggregate_scalar(rows):
    fields = sorted(set.intersection(*(set(row) for row in rows)))
    return {field: float(np.mean([row[field] for row in rows])) for field in fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.input / "full_context_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    source = args.input / "attention.jsonl"
    original = [row for line in source.read_text().splitlines()
                if (row := json.loads(line)).get("type") == "draft_attention"]
    grouped = defaultdict(list)
    for row in original:
        grouped[row["sample_id"], row["context_cap"]].append(row)
    phase_indices = {}
    for key, rows in grouped.items():
        phase_indices[key] = {0: "first", len(rows) // 2: "middle", len(rows) - 1: "last"}
    round_destination = destination / "round_metrics.jsonl"
    round_destination.write_text("")
    writer = JsonlWriter(round_destination)
    document_metrics = {}
    phase_bank = {}
    transition_metrics = {}
    max_top1k_error = 0.0
    max_mass_error = 0.0
    processed = 0
    for key, rows in grouped.items():
        sample, cap = key
        values, transitions = [], []
        phases, previous = {}, None
        for index, record in enumerate(rows):
            with np.load(args.input / record["vectors_file"]) as archive:
                layers = archive["per_layer_attention"]
            if layers.shape != (5, record["key_tokens"]) or not np.isfinite(layers).all() or (layers < 0).any():
                raise ValueError(f"Vector không hợp lệ: {record['vectors_file']}")
            vector = layers.mean(axis=0).astype(np.float64)
            total = float(vector.sum())
            max_mass_error = max(max_mass_error, abs(total - 1.0))
            p = vector / total
            metrics, selections = summarize_vector(p)
            max_top1k_error = max(max_top1k_error, abs(metrics["top_1024"] - record["top_1k_all_coverage"]))
            if max_top1k_error > 1e-6:
                raise ValueError("Coverage Top-1K không khớp JSONL gốc")
            prefix = record["prefix_tokens"]
            metrics.update({
                "prompt_mass": float(p[:cap].sum()),
                "output_mass": float(p[cap:prefix].sum()),
                "anchor_mass": float(p[prefix]),
                "mask_mass": float(p[prefix + 1:].sum()),
                "output_tokens_at_draft": prefix - cap,
                "accepted_proposals": record["accepted_proposals"],
            })
            current_sets = {}
            for budget, indices in selections.items():
                current_sets[budget] = canonical_ids(indices, prefix)
                layer_total = layers.sum(axis=1, dtype=np.float64)
                coverage = layers[:, indices].sum(axis=1, dtype=np.float64) / layer_total
                for layer, value in enumerate(coverage):
                    metrics[f"shared_top_{budget}_layer_{layer}"] = float(value)
                metrics[f"shared_top_{budget}_minimum_layer"] = float(coverage.min())
                if previous is not None:
                    overlap, retained = compare_selection(previous[budget], current_sets[budget], p, prefix)
                    regret = metrics[f"top_{budget}"] - retained
                    if regret < -1e-8:
                        raise ValueError("Tập chọn cũ vượt oracle Top-K")
                    if budget == 1024:
                        transition = {}
                    transition.update({f"overlap_{budget}": overlap,
                                       f"previous_selection_mass_{budget}": retained,
                                       f"current_top_mass_{budget}": metrics[f"top_{budget}"],
                                       f"mass_gap_{budget}": regret})
            if previous is not None:
                transitions.append(transition)
            previous = current_sets
            values.append(metrics)
            if index in phase_indices[key]:
                phases[phase_indices[key][index]] = {
                    "metrics": metrics, "selection_ids": current_sets,
                    "probabilities": p, "prefix": prefix,
                    "round": record["round"], "output_offset": record["output_offset"],
                }
            writer.add({"type": "attention_full_context_analysis", "method": "dflash_full_context_analysis",
                        "dataset": "gov_report", "sample_id": sample, "context_cap": cap,
                        "round": record["round"], "vectors_file": record["vectors_file"], "metrics": metrics})
            processed += 1
            if processed % 500 == 0 or processed == len(original):
                print(f"[full-context] Đã đọc {processed}/{len(original)} NPZ", flush=True)
        document_metrics[key] = aggregate_scalar(values)
        transition_metrics[key] = aggregate_scalar(transitions)
        phase_bank[key] = phases

    caps = []
    flat = []
    for cap in sorted({cap for _, cap in grouped}):
        keys = [key for key in grouped if key[1] == cap]
        documents = [{"sample_id": key[0], "mean": document_metrics[key],
                      "adjacent_rounds": transition_metrics[key]} for key in keys]
        means = aggregate_scalar([d["mean"] for d in documents])
        ranges = {field: {"min": min(d["mean"][field] for d in documents),
                          "max": max(d["mean"][field] for d in documents)} for field in means}
        phase_means = {tag: aggregate_scalar([phase_bank[key][tag]["metrics"] for key in keys])
                       for tag in ["first", "middle", "last"]}
        pairs = {}
        pair_documents = {}
        for earlier, later in [("first", "middle"), ("first", "last"), ("middle", "last")]:
            comparisons = []
            for key in keys:
                old, new = phase_bank[key][earlier], phase_bank[key][later]
                metrics = {}
                for budget in [1024, 4096]:
                    overlap, retained = compare_selection(old["selection_ids"][budget],
                        new["selection_ids"][budget], new["probabilities"], new["prefix"])
                    metrics.update({f"overlap_{budget}": overlap,
                                    f"old_selection_mass_{budget}": retained,
                                    f"current_top_mass_{budget}": new["metrics"][f"top_{budget}"],
                                    f"mass_gap_{budget}": new["metrics"][f"top_{budget}"] - retained})
                comparisons.append({"sample_id": key[0], "metrics": metrics})
            name = f"{earlier}_to_{later}"
            pairs[name] = aggregate_scalar([d["metrics"] for d in comparisons])
            pair_documents[name] = comparisons
        caps.append({"context_cap": cap, "documents": len(keys),
                     "draft_rounds": sum(len(grouped[key]) for key in keys),
                     "mean": means, "document_range": ranges, "per_document": documents,
                     "phase_means": phase_means, "phase_pairs": pairs,
                     "phase_pair_documents": pair_documents,
                     "adjacent_rounds": aggregate_scalar([transition_metrics[key] for key in keys])})
        flat.append({"context_cap": cap, **means})

    manifest = json.loads((args.input / "manifest.json").read_text())
    result = {
        "status": "success", "source_run": args.input.name, "source_directory": str(args.input),
        "source_jsonl_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256((args.input / "manifest.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "collector_sha256": manifest["collector_sha256"],
        "no_new_inference": True, "denominator": "total attention mass over every key, normalized to 1 per draft",
        "query_aggregation": manifest["query_aggregation"],
        "budgets": BUDGETS, "top_fraction_budgets": FRACTIONS,
        "weighting": "mean over rounds within document, then equal mean over 10 documents",
        "phase_definition": "first, record len(records)//2, and final record of each trajectory",
        "key_alignment": "Committed prefix keys use absolute sequence position; live block uses slot 0..15. Matching slots do not imply identical hidden states.",
        "overlap_definition": "intersection size divided by earlier selection size; this is not Jaccard",
        "fixed_budget_strategies": "Top-K, best contiguous window, head, tail, middle, and half-budget head plus half-budget tail; all use min(K, N) keys",
        "checks": {"processed_npz": processed, "max_mass_error": max_mass_error,
                   "max_recomputed_top1k_error": max_top1k_error},
        "caps": caps,
    }
    (destination / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with (destination / "summary.csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        csv_writer.writeheader()
        csv_writer.writerows(flat)
    writer.finalize({"type": "summary", "method": "dflash_full_context_analysis",
                     "status": "success", "draft_rounds": processed, "sample_count": len(manifest["samples"]),
                     "denominator": "all keys", "summary_file": "summary.json", "checks": result["checks"]})
    print(f"[full-context] Kết quả: {destination / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
