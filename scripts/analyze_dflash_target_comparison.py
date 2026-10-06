#!/usr/bin/env python3
"""Tổng hợp phép so attention DFlash/target trên các lượt draft thật."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.io_util import JsonlWriter

PAIR_METRICS = [
    "js_divergence_full_context",
    "js_divergence_cache_conditional",
    "top_1024_cache_overlap_fraction",
    "top_4096_cache_overlap_fraction",
    "target_total_mass_on_dflash_top_1024_cache",
    "dflash_total_mass_on_target_top_1024_cache",
]
PARENT_PAIR_METRICS = [
    "js_divergence_full_context",
    "js_divergence_cache_conditional",
    "top_1024_cache_overlap_fraction",
    "top_4096_cache_overlap_fraction",
    "parent_attention_mass_on_dflash_top_1024_cache",
    "dflash_full_mass_on_parent_top_1024_cache",
    "dflash_cache_conditional_mass_on_parent_top_1024_cache",
    "parent_top_1024_own_mass",
    "dflash_current_block_mass_full_denominator",
]
LAYER_METRICS = [
    "total_mass", "prompt_mass", "committed_output_mass", "context_cache_mass",
    "current_block_mass",
]


def mean_rows(rows: list[dict]) -> dict:
    fields = sorted(set.intersection(*(set(row) for row in rows)))
    return {key: float(np.mean([row[key] for row in rows])) for key in fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input / "attention_comparison.jsonl"
    destination = args.output or args.input / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.input / "manifest.json").read_text())
    run_summary = json.loads((args.input / "summary.json").read_text())
    if run_summary.get("status") != "success":
        raise ValueError("Run Modal chưa thành công")
    if (run_summary["exact_greedy_replay_runs"] != run_summary["run_count"] or
            run_summary["exact_round_count_runs"] != run_summary["run_count"]):
        raise ValueError("Có trajectory không khớp run gốc; dừng tổng hợp để giữ cặp so sánh")
    records = [json.loads(line) for line in source.read_text().splitlines()
               if (row := json.loads(line)).get("type") == "draft_attention"]
    target_ids = manifest["target_layers"]
    draft_ids = list(range(manifest["draft_layers"]))
    pair_values, layer_values = defaultdict(list), defaultdict(list)
    parent_pair_values, parent_bonus_pair_values = defaultdict(list), defaultdict(list)
    prior_rounds = {}
    per_round = destination / "round_metrics.jsonl"
    per_round.write_text("")
    writer = JsonlWriter(per_round)
    max_full_mass_error = 0.0
    max_future_block_mass = 0.0
    for row in records:
        comparison = row.get("target_dflash_comparison")
        if comparison is None:
            raise ValueError(f"Thiếu attention comparison ở {row.get('vectors_file')}")
        if comparison["prefix_length"] != row["prefix_tokens"]:
            raise ValueError("Prefix mismatch giữa collector và phép so sánh")
        future_mass = float(comparison["target_max_future_block_mass"])
        if future_mass > 1e-6:
            raise ValueError(f"Target verifier có mass vào key tương lai: {future_mass}")
        max_future_block_mass = max(max_future_block_mass, future_mass)
        if len(comparison["pairwise"]) != len(target_ids) * manifest["draft_layers"]:
            raise ValueError("Thiếu cặp layer target/DFlash")
        round_layers = {"target": {}, "draft": {}}
        for model_name, layers in [("target", comparison["target_layers"]),
                                   ("draft", comparison["draft_layers"])]:
            expected = target_ids if model_name == "target" else draft_ids
            if [layer["layer"] for layer in layers] != expected:
                raise ValueError(f"Layer index không khớp manifest: {model_name}")
            for layer in layers:
                metric_row = {key: float(layer[key]) for key in LAYER_METRICS}
                if abs(metric_row["total_mass"] - 1.0) > 2e-3:
                    raise ValueError(f"Attention {model_name} không cộng 100%: {metric_row['total_mass']}")
                max_full_mass_error = max(max_full_mass_error, abs(metric_row["total_mass"] - 1.0))
                layer_values[(row["sample_id"], row["context_cap"],
                              model_name, layer["layer"])].append(metric_row)
                round_layers[model_name][str(layer["layer"])] = {
                    key: layer[key] for key in [*LAYER_METRICS, "position_bins_full_context"]
                }
        round_pairs = []
        for pair in comparison["pairwise"]:
            target_layer, draft_layer = pair["target_layer"], pair["draft_layer"]
            metric_row = {key: float(pair["metrics"][key]) for key in PAIR_METRICS}
            pair_values[(row["sample_id"], row["context_cap"], target_layer, draft_layer)].append(metric_row)
            round_pairs.append({"target_layer": target_layer, "draft_layer": draft_layer,
                                "metrics": metric_row})
        round_record = {
            "type": "target_dflash_attention_pair_analysis",
            "sample_id": row["sample_id"], "context_cap": row["context_cap"],
            "round": row["round"], "prefix_tokens": row["prefix_tokens"],
            "output_offset": row["output_offset"],
            "target_layers": round_layers["target"], "draft_layers": round_layers["draft"],
            "pairwise": round_pairs,
        }

        parent_comparison = row.get("parent_attention_comparison")
        if parent_comparison is not None:
            previous = prior_rounds.get((row["sample_id"], row["context_cap"], row["round"] - 1))
            if previous is None:
                raise ValueError("Parent attention không có record verify ngay trước")
            expected_query = int(previous["accepted_proposals"])
            expected_prefix = int(previous["prefix_tokens"]) + expected_query + 1
            if (parent_comparison["parent_query_index"] != expected_query or
                    parent_comparison["shared_cache_length"] != row["prefix_tokens"] or
                    parent_comparison["shared_cache_length"] != expected_prefix or
                    parent_comparison["draft_round"] != row["round"] or
                    parent_comparison["parent_round"] != previous["round"]):
                raise ValueError("Ghép lệch target query sinh anchor và DFlash round tiếp theo")
            expected_role = "bonus" if expected_query == 15 else "correction"
            if parent_comparison["parent_token_role"] != expected_role:
                raise ValueError("Sai nhãn bonus/correction của parent target query")
            role_rows = []
            for pair in parent_comparison["pairs"]:
                target_layer, draft_layer = pair["target_layer"], pair["draft_layer"]
                metrics = {key: float(pair["metrics"][key]) for key in PARENT_PAIR_METRICS}
                if any(not 0.0 <= value <= 1.0 for value in metrics.values()):
                    raise ValueError("Parent comparison có metric ngoài [0,1]")
                key = (row["sample_id"], row["context_cap"], target_layer, draft_layer)
                parent_pair_values[key].append(metrics)
                if expected_role == "bonus":
                    parent_bonus_pair_values[key].append(metrics)
                role_rows.append({"target_layer": target_layer, "draft_layer": draft_layer,
                                  "metrics": metrics})
            if len(role_rows) != len(target_ids) * manifest["draft_layers"]:
                raise ValueError("Parent comparison thiếu cặp layer")
            round_record["parent_attention"] = {
                "parent_round": parent_comparison["parent_round"],
                "draft_round": parent_comparison["draft_round"],
                "parent_query_index": expected_query,
                "parent_token_role": expected_role,
                "accepted_proposals": expected_query,
                "shared_cache_length": parent_comparison["shared_cache_length"],
                "pairwise": role_rows,
            }
        writer.add(round_record)
        prior_rounds[(row["sample_id"], row["context_cap"], row["round"])] = row

    cap_summaries, flat_rows = [], []
    for cap in manifest["caps"]:
        cap_records = [row for row in records if row["context_cap"] == cap]
        if len(cap_records) != sum(r["draft_rounds"] for r in run_summary["runs"]
                                   if r["context_cap"] == cap):
            raise ValueError(f"Thiếu lượt draft tại cap={cap}")
        matrix = []
        for target_layer in target_ids:
            row = []
            for draft_layer in draft_ids:
                selected = [(doc, values) for (doc, current_cap, t, d), values in pair_values.items()
                            if current_cap == cap and t == target_layer and d == draft_layer]
                docs = {doc: mean_rows(values) for doc, values in selected}
                if len(docs) != manifest["sample_count"]:
                    raise ValueError(f"Cần đủ document cho cặp layer {target_layer}/{draft_layer}")
                mean = mean_rows(list(docs.values()))
                ranges = {key: {"min": min(v[key] for v in docs.values()),
                                "max": max(v[key] for v in docs.values())}
                          for key in PAIR_METRICS}
                row.append({"target_layer": target_layer, "draft_layer": draft_layer,
                            "mean": mean, "document_range": ranges,
                            "per_document": docs})
                flat_rows.append({"context_cap": cap, "target_layer": target_layer,
                                  "draft_layer": draft_layer, **mean})
            matrix.append(row)
        layer_means = {model: {} for model in ["target", "draft"]}
        for model, layer_ids in [("target", target_ids), ("draft", draft_ids)]:
            for layer_id in layer_ids:
                selected = [(doc, values) for (doc, current_cap, name, idx), values in layer_values.items()
                            if current_cap == cap and name == model and idx == layer_id]
                docs = {doc: mean_rows(values) for doc, values in selected}
                if len(docs) != manifest["sample_count"]:
                    raise ValueError(f"Thiếu model layer summaries: {model} {layer_id} {cap}")
                layer_means[model][str(layer_id)] = mean_rows(list(docs.values()))
        cap_summaries.append({
            "context_cap": cap, "instances": manifest["sample_count"],
            "draft_rounds": sum(1 for row in records if row["context_cap"] == cap),
            "target_layer_ids": target_ids, "draft_layer_ids": draft_ids,
            "pairwise": matrix, "layer_means": layer_means,
        })

    parent_summaries = []
    for cap in manifest["caps"]:
        cap_rows = [row for row in records if row["context_cap"] == cap]
        transition_rows = [row for row in cap_rows if row.get("parent_attention_comparison")]
        bonus_rows = [row for row in transition_rows
                      if row["parent_attention_comparison"]["parent_token_role"] == "bonus"]
        matrices = {}
        for label, value_map in [("all_transitions", parent_pair_values),
                                 ("literal_bonus_only", parent_bonus_pair_values)]:
            matrix = []
            for target_layer in target_ids:
                matrix_row = []
                for draft_layer in draft_ids:
                    selected = [(doc, values) for (doc, current_cap, t, d), values in value_map.items()
                                if current_cap == cap and t == target_layer and d == draft_layer]
                    docs = {doc: mean_rows(values) for doc, values in selected}
                    mean = mean_rows(list(docs.values())) if docs else None
                    matrix_row.append({
                        "target_layer": target_layer, "draft_layer": draft_layer,
                        "documents": len(docs), "mean": mean,
                        "per_document": docs,
                    })
                matrix.append(matrix_row)
            present = [cell["mean"] for row in matrix for cell in row if cell["mean"] is not None]
            overall = mean_rows(present) if present else None
            matrices[label] = {"pairwise": matrix, "mean_over_layer_pairs": overall}
        parent_summaries.append({
            "context_cap": cap,
            "instances": manifest["sample_count"],
            "transitions": len(transition_rows),
            "literal_bonus_transitions": len(bonus_rows),
            **matrices,
        })

    parent_transition_count = sum(row["transitions"] for row in parent_summaries)
    parent_bonus_count = sum(row["literal_bonus_transitions"] for row in parent_summaries)
    expected_parent_transitions = sum(max(0, row["draft_rounds"] - 1)
                                      for row in run_summary["runs"])
    if manifest.get("capture_parent_attention"):
        if parent_transition_count != expected_parent_transitions:
            raise ValueError(
                f"Parent transition count {parent_transition_count} != expected {expected_parent_transitions}"
            )
        if parent_transition_count != run_summary["parent_attention_transitions"]:
            raise ValueError("Parent transition count lệch run summary")
        if parent_bonus_count != run_summary["parent_attention_bonus_transitions"]:
            raise ValueError("Literal bonus transition count lệch run summary")
    summary = {
        "status": "success", "source_run": manifest["source_run"],
        "source_attention_run_manifest_sha256": manifest["source_run_manifest_sha256"],
        "comparison_manifest_sha256": hashlib.sha256((args.input / "manifest.json").read_bytes()).hexdigest(),
        "source_jsonl_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_layers": target_ids, "draft_layers": draft_ids,
        "primary_metric": manifest["primary_metric"],
        "query_alignment": manifest["query_alignment"],
        "scope": (
            "Parent comparison pads the previous target next-anchor query with zero mass on the next 16-key DFlash block; the full distribution then covers prompt + committed output + current block. Cache-conditional metrics are secondary."
            if manifest.get("capture_parent_attention") else
            "Full probability distributions over prompt + committed output + 16 current-block keys; cache-only conditional metrics are secondary."
        ),
        "weighting": "Average rounds within each document, then equal mean over the 10 documents.",
        "checks": {"records": len(records), "all_100pct_distributions_passed": True,
                   "max_mean_distribution_mass_error": max_full_mass_error,
                   "all_40_greedy_outputs_exact": True,
                   "all_40_draft_round_counts_exact": True,
                   "max_target_mass_on_future_verifier_keys": max_future_block_mass,
                   "all_document_layer_pairs_present": True,
                   "parent_attention_transitions": parent_transition_count,
                   "literal_bonus_transitions": parent_bonus_count},
        "caps": cap_summaries,
        "parent_attention": {
            "alignment": "Target query q=accepted_proposals in verifier round r predicts the next round anchor; compare that row with DFlash attention in round r+1 over the shared cache, padding the new 16-key block with zero target mass for full-scope metrics.",
            "all_transitions": parent_summaries,
        },
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    import csv
    with (destination / "summary.csv").open("w", newline="") as handle:
        fieldnames = list(flat_rows[0])
        output = csv.DictWriter(handle, fieldnames=fieldnames)
        output.writeheader()
        output.writerows(flat_rows)
    parent_flat_rows = []
    for cap in parent_summaries:
        for role in ["all_transitions", "literal_bonus_only"]:
            for target_row in cap[role]["pairwise"]:
                for pair in target_row:
                    if pair["mean"] is not None:
                        parent_flat_rows.append({
                            "context_cap": cap["context_cap"], "subset": role,
                            "target_layer": pair["target_layer"],
                            "draft_layer": pair["draft_layer"],
                            "documents": pair["documents"], **pair["mean"],
                        })
    with (destination / "parent_summary.csv").open("w", newline="") as handle:
        fieldnames = list(parent_flat_rows[0]) if parent_flat_rows else ["context_cap", "subset"]
        output = csv.DictWriter(handle, fieldnames=fieldnames)
        output.writeheader()
        output.writerows(parent_flat_rows)
    writer.finalize({"type": "summary", "status": "success", "records": len(records),
                     "summary_file": "summary.json", "checks": summary["checks"]})
    print(f"[analysis] records={len(records)}; saved {destination / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
