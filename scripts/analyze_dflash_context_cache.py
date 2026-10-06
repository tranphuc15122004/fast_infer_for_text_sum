#!/usr/bin/env python3
"""Bỏ 16 draft-block key khỏi phân tích cache; báo thêm tổng mass khi giữ block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_dflash_full_context import BUDGETS, FRACTIONS, aggregate_scalar, summarize_vector
from common.io_util import JsonlWriter

STRATEGIES = ["top", "best_window", "head", "tail", "middle", "edges"]


def compare(old_indices, new_indices, cache_probabilities):
    overlap = len(np.intersect1d(old_indices, new_indices, assume_unique=True)) / len(old_indices)
    return overlap, float(cache_probabilities[old_indices].sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.input / "context_cache_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    source = args.input / "attention.jsonl"
    records = [row for line in source.read_text().splitlines()
               if (row := json.loads(line)).get("type") == "draft_attention"]
    groups = defaultdict(list)
    for row in records:
        groups[row["sample_id"], row["context_cap"]].append(row)
    per_document, adjacent_documents, phases_by_document = {}, {}, {}
    round_output = destination / "round_metrics.jsonl"
    round_output.write_text("")
    writer = JsonlWriter(round_output)
    processed, max_decomposition_error = 0, 0.0
    manifest = json.loads((args.input / "manifest.json").read_text())
    if manifest["block_size"] != 16:
        raise ValueError("Phân tích này cần draft block đúng 16 vị trí")
    for key, rows in groups.items():
        sample, cap = key
        collected, transitions, phases = [], [], {}
        phase_indices = {0: "first", len(rows)//2: "middle", len(rows)-1: "last"}
        previous = None
        for index, row in enumerate(rows):
            with np.load(args.input / row["vectors_file"]) as archive:
                layers = archive["per_layer_attention"]
            prefix = row["prefix_tokens"]
            if layers.shape != (5, prefix + 16) or not np.isfinite(layers).all() or (layers < 0).any():
                raise ValueError(f"Vector không hợp lệ: {row['vectors_file']}")
            full_vector = layers.mean(0).astype(np.float64)
            full = full_vector / full_vector.sum()
            block_mass = float(full[prefix:].sum())
            cache_mass = float(full[:prefix].sum())
            if not 0 < cache_mass <= 1:
                raise ValueError("Cache mass phải dương")
            cache = full[:prefix] / cache_mass
            metrics, selections = summarize_vector(cache)
            metrics.update({"block_mass_total": block_mass, "cache_mass_total": cache_mass,
                            "full_key_tokens": prefix + 16,
                            "output_tokens_at_draft": prefix - cap,
                            "accepted_proposals": row["accepted_proposals"]})
            for strategy in STRATEGIES:
                for budget in BUDGETS:
                    metric = f"{strategy}_{budget}"
                    metrics[f"selected_cache_share_total_{metric}"] = cache_mass * metrics[metric]
                    metrics[f"retained_total_{metric}"] = block_mass + cache_mass * metrics[metric]
            cache_order = np.argsort(cache, kind="stable")[::-1]
            cumulative_absolute = np.cumsum(full[:prefix][cache_order])
            for target in [90, 95, 99]:
                needed_mass = max(0.0, target / 100 - block_mass)
                needed_keys = min(prefix, int(np.searchsorted(cumulative_absolute, needed_mass)) + 1) if needed_mass > 0 else 0
                metrics[f"cache_keys_for_{target}pct_total_with_block"] = needed_keys
                metrics[f"all_keys_for_{target}pct_total_with_block"] = needed_keys + 16
            layer_total = layers.sum(axis=1, dtype=np.float64)
            layer_cache_total = layers[:, :prefix].sum(axis=1, dtype=np.float64)
            layer_block = layers[:, prefix:].sum(axis=1, dtype=np.float64)
            for budget, indices in selections.items():
                selected = layers[:, indices].sum(axis=1, dtype=np.float64)
                cache_coverage = selected / layer_cache_total
                retained_total = (layer_block + selected) / layer_total
                for layer in range(5):
                    metrics[f"shared_top_{budget}_layer_{layer}"] = float(cache_coverage[layer])
                    metrics[f"retained_total_top_{budget}_layer_{layer}"] = float(retained_total[layer])
                metrics[f"shared_top_{budget}_minimum_layer"] = float(cache_coverage.min())
                metrics[f"retained_total_top_{budget}_minimum_layer"] = float(retained_total.min())
                if previous is not None:
                    overlap, old_cache_mass = compare(previous[budget], indices, cache)
                    if budget == 1024:
                        transition = {}
                    transition.update({f"overlap_{budget}": overlap,
                                       f"previous_selection_cache_mass_{budget}": old_cache_mass,
                                       f"current_top_cache_mass_{budget}": metrics[f"top_{budget}"],
                                       f"cache_mass_gap_{budget}": metrics[f"top_{budget}"] - old_cache_mass,
                                       f"previous_selection_retained_total_{budget}": block_mass + cache_mass * old_cache_mass,
                                       f"current_top_retained_total_{budget}": metrics[f"retained_total_top_{budget}"],
                                       f"total_mass_gap_{budget}": cache_mass * (metrics[f"top_{budget}"] - old_cache_mass)})
            if previous is not None:
                transitions.append(transition)
            previous = selections
            max_decomposition_error = max(max_decomposition_error, abs(cache_mass + block_mass - 1))
            collected.append(metrics)
            if index in phase_indices:
                phases[phase_indices[index]] = {"metrics": metrics, "selection_ids": selections,
                                               "cache_probabilities": cache, "cache_mass": cache_mass,
                                               "block_mass": block_mass}
            writer.add({"type": "attention_context_cache_analysis", "method": "dflash_context_cache_analysis",
                        "dataset": "gov_report", "sample_id": sample, "context_cap": cap,
                        "round": row["round"], "vectors_file": row["vectors_file"], "metrics": metrics})
            processed += 1
            if processed % 500 == 0 or processed == len(records):
                print(f"[context-cache] Đã đọc {processed}/{len(records)} NPZ", flush=True)
        per_document[key] = aggregate_scalar(collected)
        adjacent_documents[key] = aggregate_scalar(transitions)
        phases_by_document[key] = phases

    caps, flat = [], []
    for cap in sorted({cap for _, cap in groups}):
        keys = [key for key in groups if key[1] == cap]
        documents = [{"sample_id": key[0], "mean": per_document[key],
                      "adjacent_rounds": adjacent_documents[key]} for key in keys]
        means = aggregate_scalar([d["mean"] for d in documents])
        ranges = {f: {"min": min(d["mean"][f] for d in documents),
                      "max": max(d["mean"][f] for d in documents)} for f in means}
        phase_means = {tag: aggregate_scalar([phases_by_document[key][tag]["metrics"] for key in keys])
                       for tag in ["first", "middle", "last"]}
        pairs, pair_documents = {}, {}
        for earlier, later in [("first", "middle"), ("first", "last"), ("middle", "last")]:
            comparisons = []
            for key in keys:
                old, new = phases_by_document[key][earlier], phases_by_document[key][later]
                comparison = {}
                for budget in [1024, 4096]:
                    overlap, old_mass = compare(old["selection_ids"][budget], new["selection_ids"][budget], new["cache_probabilities"])
                    current_mass = new["metrics"][f"top_{budget}"]
                    comparison.update({f"overlap_{budget}": overlap,
                                       f"old_selection_cache_mass_{budget}": old_mass,
                                       f"current_top_cache_mass_{budget}": current_mass,
                                       f"cache_mass_gap_{budget}": current_mass - old_mass,
                                       f"old_selection_retained_total_{budget}": new["block_mass"] + new["cache_mass"] * old_mass,
                                       f"current_top_retained_total_{budget}": new["block_mass"] + new["cache_mass"] * current_mass,
                                       f"total_mass_gap_{budget}": new["cache_mass"] * (current_mass - old_mass)})
                comparisons.append({"sample_id": key[0], "metrics": comparison})
            name = f"{earlier}_to_{later}"
            pairs[name] = aggregate_scalar([d["metrics"] for d in comparisons])
            pair_documents[name] = comparisons
        caps.append({"context_cap": cap, "documents": len(keys),
                     "draft_rounds": sum(len(groups[key]) for key in keys),
                     "mean": means, "document_range": ranges, "per_document": documents,
                     "phase_means": phase_means, "phase_pairs": pairs,
                     "phase_pair_documents": pair_documents,
                     "adjacent_rounds": aggregate_scalar([adjacent_documents[key] for key in keys])})
        flat.append({"context_cap": cap, **means})
    base_script = Path(__file__).with_name("analyze_dflash_full_context.py")
    summary = {"status": "success", "source_run": args.input.name,
               "source_directory": str(args.input), "no_new_inference": True,
               "source_jsonl_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "source_manifest_sha256": hashlib.sha256((args.input / "manifest.json").read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "base_analysis_script_sha256": hashlib.sha256(base_script.read_bytes()).hexdigest(),
               "collector_sha256": manifest["collector_sha256"],
               "scope": "prompt + committed output prefix; exclude exactly the final 16 live draft-block keys: anchor + 15 masks",
               "default_method": "Keep all 16 draft-block keys and queries; compress draft context KV derived from target features. Target verifier KV is unchanged.",
               "cache_denominator": "sum of original attention on context-cache keys, after mean layers/heads/queries",
               "total_denominator": "original mass over all keys, including the 16-key block",
               "retained_total_formula": "block_mass + cache_mass * cache_coverage, evaluated per draft before averaging",
               "layer_order": "Average layer vectors first, then condition on cache. Per-layer cache coverage uses its own cache denominator and need not average to the conditioned aggregate.",
               "normalization_note": "Post-processing of original dense attention; no second forward without block keys was run.",
               "budgets": BUDGETS, "top_fraction_budgets": FRACTIONS,
               "budget_note": "K cache keys plus 16 retained live block keys; old all-key Top-K has a different total count",
               "weighting": "mean over rounds within document, then equal mean over 10 documents",
               "overlap_definition": "intersection divided by earlier selected cache key count; cache IDs are persistent absolute prefix positions",
               "checks": {"processed_npz": processed, "max_decomposition_error": max_decomposition_error},
               "caps": caps}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (destination / "summary.csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        csv_writer.writeheader()
        csv_writer.writerows(flat)
    writer.finalize({"type": "summary", "method": "dflash_context_cache_analysis", "status": "success",
                     "draft_rounds": processed, "excluded_live_block_keys": 16,
                     "summary_file": "summary.json", "checks": summary["checks"]})
    print(f"[context-cache] Kết quả: {destination / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
