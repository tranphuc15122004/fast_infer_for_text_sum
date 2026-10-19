#!/usr/bin/env python3
"""Đánh giá chi tiết Sweep qua các Target Attention Layer làm Context Selector trên 960 snapshots GPU thật.

Khảo sát 8 tầng Target có sẵn: L0, L12, L14, L16, L18, L20, L27, L35.
Mỗi tầng được đánh giá theo 2 chế độ toán tử:
  1. Multi-Query Max-Pooling (MQ): s[i] = max_{r in accepted} a_L[r, i]
  2. Single-Query (SQ): s[i] = a_L[accepted, i]

Mục tiêu:
  - Tìm ra tầng Target Attention tối ưu nhất cho Context Selection.
  - Kiểm tra xem Layer 18 có phải là đỉnh (peak) hay Layer 20 / 27 vượt trội hơn.
  - Phân tích độ nhạy cảm của tầng (Layer Sensitivity Analysis).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from TrainingFree.context_interventions.selector_evaluation import (
    normalized, select_chunks, spatial_continuity
)

BUDGETS = (256, 512, 1024, 2048, 4096)
CONTEXT_CAPS = (3072, 5120, 8192, 16384)
LAYERS = [0, 12, 14, 16, 18, 20, 27, 35]

CANDIDATES = {
    "C1_Layer18": (18,),
    "E3_Band14_20": (14, 16, 18, 20),
    "C2_Layer35": (35,),
    "C3_Layer27": (27,),
    "E1_Duo18_35": (18, 35),
    "C4_Layer12": (12,),
    "E2_TriAnchor": (18, 27, 35),
    "C0_L0": (0,),
}

CORE_CANDIDATES = ["C1_Layer18", "E3_Band14_20", "C2_Layer35", "C3_Layer27", "E1_Duo18_35"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 20261019) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    mean = float(arr.mean())
    ci_low = float(np.quantile(boot_means, 0.025))
    ci_high = float(np.quantile(boot_means, 0.975))
    return mean, ci_low, ci_high


def extract_layer_selectors(snapshot: dict, prefix: int) -> dict[str, np.ndarray]:
    val = snapshot["target_query_attention"]  # Shape: (8 layers, 16 queries, prefix)
    layer_ids = list(map(int, snapshot["target_layer_ids"]))
    accepted = int(snapshot["accepted_candidates"])

    scores = {}
    for lyr in LAYERS:
        if lyr not in layer_ids:
            continue
        idx = layer_ids.index(lyr)
        
        # Multi-Query Max Pooling across accepted queries
        mq_vec = val[idx, : accepted + 1].max(axis=0)[:prefix].astype(np.float64)
        scores[f"MQ_L{lyr}"] = normalized(mq_vec)
        
        # Single-Query (last accepted query)
        sq_vec = val[idx, accepted][:prefix].astype(np.float64)
        scores[f"SQ_L{lyr}"] = normalized(sq_vec)
        
    return scores


def get_target_reference(snapshot: dict, layer_tuple: tuple[int, ...], prefix: int) -> np.ndarray:
    layer_ids = list(map(int, snapshot["target_layer_ids"]))
    vecs = []
    for l in layer_tuple:
        idx = layer_ids.index(l)
        row = snapshot["target"][idx][:prefix].astype(np.float64)
        vecs.append(normalized(row))
    return np.mean(vecs, axis=0)


def evaluate_snapshots(snapshot_dir: Path, output_dir: Path) -> dict:
    paths = sorted(snapshot_dir.glob("*_round_*.npz"))
    print(f"[evaluate] Tìm thấy {len(paths)} snapshots tại {snapshot_dir}")
    if not paths:
        raise ValueError(f"Không có snapshot nào tại {snapshot_dir}")

    groups = defaultdict(list)
    for p in paths:
        name = p.name
        prefix = name.split("_round_")[0]
        groups[prefix].append(p)

    output_dir.mkdir(parents=True, exist_ok=True)
    accum = defaultdict(lambda: np.zeros(5, dtype=np.float64))
    pair_count = 0
    source_files = []

    methods = [f"MQ_L{l}" for l in LAYERS] + [f"SQ_L{l}" for l in LAYERS]

    for group_name, flist in sorted(groups.items()):
        flist.sort()
        for i in range(len(flist)):
            curr_p = flist[i]
            source_files.append({"file": curr_p.name, "sha256": sha(curr_p)})
            curr_snap = np.load(curr_p)
            following_snap = np.load(flist[i + 1]) if i + 1 < len(flist) else None

            sample_id = str(curr_snap.get("sample_id", group_name.split("_context_")[0]))
            cap = int(curr_snap.get("context_cap", group_name.split("_context_")[1]))
            n = int(curr_snap["prefix_length"])

            selector_scores = extract_layer_selectors(curr_snap, n)

            references = [("same_round_diagnostic", curr_snap)]
            if following_snap is not None:
                if int(following_snap["round"]) == int(curr_snap["round"]) + 1:
                    references.append(("next_round", following_snap))
                    pair_count += 1

            for budget in BUDGETS:
                if budget > n:
                    continue

                masks = {
                    m: select_chunks(selector_scores[m], budget, chunk_size=1, sink=16, recent=32)
                    for m in methods
                }
                continuities = {m: spatial_continuity(idx) for m, idx in masks.items()}

                for scope, ref_snap in references:
                    draft_vec = np.mean([normalized(row[:n]) for row in ref_snap["draft"]], axis=0)
                    target_refs = {c: get_target_reference(ref_snap, layers, n) for c, layers in CANDIDATES.items()}

                    for method, idx in masks.items():
                        draft_cov = float(draft_vec[idx].sum())
                        continuity = continuities[method]
                        sel_tokens = len(idx)

                        for cand, gt in target_refs.items():
                            cov = float(gt[idx].sum())
                            key = (sample_id, cap, budget, method, cand, scope)
                            accum[key] += [cov, draft_cov, continuity, sel_tokens, 1]

        print(f"[evaluate] Hoàn tất nhóm: {group_name} ({len(flist)} snapshots)", flush=True)

    # Document-level aggregation
    doc_rows = []
    fields = ("sample_id", "context_cap", "budget", "method", "candidate", "evaluation_scope")
    for key, v in sorted(accum.items()):
        doc_rows.append(
            dict(zip(fields, key))
            | {
                "coverage": v[0] / v[4],
                "draft_coverage": v[1] / v[4],
                "spatial_continuity": v[2] / v[4],
                "selected_tokens": v[3] / v[4],
                "rounds": int(v[4]),
            }
        )

    # Macro-level aggregation across documents
    macro_accum = defaultdict(list)
    for r in doc_rows:
        mkey = (r["context_cap"], r["budget"], r["method"], r["candidate"], r["evaluation_scope"])
        macro_accum[mkey].append((r["coverage"], r["draft_coverage"], r["spatial_continuity"], r["selected_tokens"]))

    macro_rows = []
    mfields = ("context_cap", "budget", "method", "candidate", "evaluation_scope")
    for mkey, values in sorted(macro_accum.items()):
        covs = [v[0] for v in values]
        draft_covs = [v[1] for v in values]
        conts = [v[2] for v in values]
        tokens = [v[3] for v in values]
        mean_cov, ci_low, ci_high = bootstrap_ci(covs)

        macro_rows.append(
            dict(zip(mfields, mkey))
            | {
                "coverage_mean": mean_cov,
                "coverage_ci_low": ci_low,
                "coverage_ci_high": ci_high,
                "draft_coverage_mean": float(np.mean(draft_covs)),
                "spatial_continuity_mean": float(np.mean(conts)),
                "selected_tokens_mean": float(np.mean(tokens)),
                "doc_count": len(covs),
            }
        )

    # Export main CSV
    csv_path = output_dir / "layer_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(macro_rows[0].keys()))
        writer.writeheader()
        writer.writerows(macro_rows)

    # Build 16K, next_round cross-candidate table for B=2048
    matrix_16k = {}
    for r in macro_rows:
        if r["context_cap"] == 16384 and r["budget"] == 2048 and r["evaluation_scope"] == "next_round":
            matrix_16k[(r["method"], r["candidate"])] = r["coverage_mean"]
            matrix_16k[(r["method"], "_continuity")] = r["spatial_continuity_mean"]

    matrix_rows = []
    for m in methods:
        row = {"method": m}
        core_vals = []
        for c in CANDIDATES.keys():
            val = matrix_16k.get((m, c), 0.0) * 100
            row[c] = val
            if c in CORE_CANDIDATES:
                core_vals.append(val)
        row["core_mean"] = float(np.mean(core_vals)) if core_vals else 0.0
        row["spatial_continuity"] = matrix_16k.get((m, "_continuity"), 0.0) * 100
        matrix_rows.append(row)

    matrix_rows.sort(key=lambda x: x["core_mean"], reverse=True)
    for rank, r in enumerate(matrix_rows, 1):
        r["rank"] = rank

    cross_matrix_path = output_dir / "layer_sweep_cross_candidate_matrix_2048.csv"
    with cross_matrix_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "method", "core_mean", "spatial_continuity"] + list(CANDIDATES.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)

    # Build multi-budget core mean table for 16K
    budget_matrix = defaultdict(dict)
    for r in macro_rows:
        if r["context_cap"] == 16384 and r["evaluation_scope"] == "next_round" and r["candidate"] in CORE_CANDIDATES:
            budget_matrix[(r["method"], r["budget"])][r["candidate"]] = r["coverage_mean"]

    all_budgets_rows = []
    for m in methods:
        brow = {"method": m}
        for b in BUDGETS:
            cvals = [budget_matrix[(m, b)].get(c, 0.0) * 100 for c in CORE_CANDIDATES]
            brow[f"B_{b}"] = float(np.mean(cvals)) if cvals else 0.0
        all_budgets_rows.append(brow)

    all_budgets_rows.sort(key=lambda x: x["B_2048"], reverse=True)
    all_budgets_path = output_dir / "layer_sweep_all_budgets_matrix.csv"
    with all_budgets_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method"] + [f"B_{b}" for b in BUDGETS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_budgets_rows)

    # Save metadata report
    report = {
        "status": "success",
        "snapshots_processed": len(paths),
        "documents": len(groups),
        "pairs_evaluated": pair_count,
        "layers_evaluated": LAYERS,
        "budgets": list(BUDGETS),
        "context_caps": list(CONTEXT_CAPS),
        "matrix_2048_top_ranks": matrix_rows[:10],
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[evaluate] Xong! Kết quả lưu tại: {output_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Layer Sweep Evaluation")
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Thư mục chứa các file snapshot .npz")
    parser.add_argument("--output-dir", type=Path, required=True, help="Thư mục xuất CSV và report")
    args = parser.parse_args()

    evaluate_snapshots(args.snapshot_dir, args.output_dir)


if __name__ == "__main__":
    main()
