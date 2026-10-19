#!/usr/bin/env python3
"""Đánh giá toàn diện 6 Selector (M0 - M5) CHUẨN NGUYÊN TÁC (KHÔNG PHA LẪN) trên 7 Target Attention Candidates.

Selectors (M0 - M5):
  - M0_SingleQuery: FastKV (Single Parent Query) tại Layer 18 (Last accepted query)
  - M1_P1_MultiQuery: DFlash-MQ (Proposed) - Multi-Query Max Pooling tại Layer 18
  - M2_SnapKV_Cluster: SnapKV (1D-AvgPool k=7) trên Observation Window attention gốc (ĐỘC LẬP VỚI M1)
  - M3_RocketKV_MaxPool: RocketKV (1D-MaxPool k=7) trên Observation Window attention gốc (ĐỘC LẬP VỚI M1)
  - M4_DuoEnsemble_18_35: Duo-Ensemble (18+35) - Cân bằng 50% L18 và 50% L35
  - M5_DeepSink_L35: Deep Sink (Pure Layer 35) - Last accepted query tại Layer 35

Target Candidate Attention Layers (Khóa theo Heatmap cũ figure_matrix_heatmap_16k.png):
  - C1_Layer18: Layer 18 (Middle Retrieval Core)
  - E3_Band14_20: Layers 14, 16, 18, 20 (Retrieval Band)
  - C2_Layer35: Layer 35 (Output Decision & Sink)
  - C3_Layer27: Layer 27 (Late-Middle Consolidation)
  - E1_Duo18_35: Layers 18, 35 (Duo-Ensemble)
  - C4_Layer12: Layer 12 (Early-Middle Core)
  - E2_TriAnchor: Layers 18, 27, 35 (Tri-Anchor)
  - C0_L0: Layer 0 (Syntax / Initial Sink)
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
    normalized, select_chunks, spatial_continuity, pool_keys
)

BUDGETS = (256, 512, 1024, 2048, 4096)
CONTEXT_CAPS = (3072, 5120, 8192, 16384)

METHODS = [
    "M4_DuoEnsemble_18_35",
    "M1_P1_MultiQuery",
    "M2_SnapKV_Cluster",
    "M3_RocketKV_MaxPool",
    "M0_SingleQuery",
    "M5_DeepSink_L35",
]

METHOD_LABELS = {
    "M4_DuoEnsemble_18_35": "M4: Duo-Ensemble (18+35)",
    "M1_P1_MultiQuery": "M1: DFlash-MQ (Proposed)",
    "M2_SnapKV_Cluster": "M2: SnapKV (1D-AvgPool)",
    "M3_RocketKV_MaxPool": "M3: RocketKV (1D-MaxPool)",
    "M0_SingleQuery": "M0: FastKV (Single-Q)",
    "M5_DeepSink_L35": "M5: Deep Sink (L35 Pure)",
}

METHOD_COLORS = {
    "M4_DuoEnsemble_18_35": "#009E73",  # Green
    "M1_P1_MultiQuery": "#0072B2",      # Blue (Proposed)
    "M2_SnapKV_Cluster": "#E69F00",     # Orange
    "M3_RocketKV_MaxPool": "#56B4E9",   # Sky blue
    "M0_SingleQuery": "#000000",        # Black
    "M5_DeepSink_L35": "#D55E00",       # Vermillion
}

# Khóa 7 candidates chính xác theo thứ tự và định nghĩa của figure_matrix_heatmap_16k.png
CANDIDATES = {
    "C1_Layer18": (18,),
    "E3_Band14_20": (14, 16, 18, 20),
    "C2_Layer35": (35,),
    "C3_Layer27": (27,),
    "E1_Duo18_35": (18, 35),
    "C4_Layer12": (12,),
    "E2_TriAnchor": (18, 27, 35),
    "C0_L0": (0,),  # Tầng đầu cú pháp bổ trợ
}

CANDIDATE_ORDER = [
    "C1_Layer18",
    "E3_Band14_20",
    "C2_Layer35",
    "C3_Layer27",
    "E1_Duo18_35",
    "C4_Layer12",
    "E2_TriAnchor",
]

CANDIDATE_LABELS = {
    "C1_Layer18": "C1 {L18}",
    "E3_Band14_20": "E3 {14-20}",
    "C2_Layer35": "C2 {L35}",
    "C3_Layer27": "C3 {L27}",
    "E1_Duo18_35": "E1 {18,35}",
    "C4_Layer12": "C4 {L12}",
    "E2_TriAnchor": "E2 {18,27,35}",
    "C0_L0": "C0 {L0}",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 20261019):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = arr[rng.integers(len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    mean = float(arr.mean())
    ci_low = float(np.quantile(boot_means, 0.025))
    ci_high = float(np.quantile(boot_means, 0.975))
    return mean, ci_low, ci_high


def extract_faithful_selectors(snapshot, prefix: int):
    """Tính toán 6 Selectors (M0 - M5) CHUẨN NGUYÊN TÁC, HOÀN TOÀN ĐỘC LẬP.
    
    Đặc biệt: M2 (SnapKV) và M3 (RocketKV) được tính từ attention gốc của Observation Window,
    TUYỆT ĐỐI KHÔNG lấy vector điểm số s_M1 làm đầu vào.
    """
    val = snapshot["target_query_attention"]  # Shape: (8 layers, 16 queries, prefix)
    layer_ids = list(map(int, snapshot["target_layer_ids"]))
    accepted = int(snapshot["accepted_candidates"])

    idx_18 = layer_ids.index(18)
    idx_35 = layer_ids.index(35)

    # Observation Window attention tại L18 và L35 (trung bình các query trong window hiện tại)
    obs_18 = val[idx_18, : accepted + 1].mean(axis=0)[:prefix].astype(np.float64)
    obs_35 = val[idx_35, : accepted + 1].mean(axis=0)[:prefix].astype(np.float64)

    # M0: FastKV (Single Parent Query) - query cuối cùng được chấp nhận tại L18
    m0_score = normalized(val[idx_18, accepted][:prefix].astype(np.float64))

    # M1: DFlash-MQ (Đề xuất của ta) - Max Pooling qua accepted queries tại L18
    m1_score = normalized(val[idx_18, : accepted + 1].max(axis=0)[:prefix].astype(np.float64))

    # M2: SnapKV (1D-AvgPool k=7) trên attention gốc của Observation Window (KHÔNG dùng m1_score!)
    m2_score = normalized(pool_keys(obs_18, mode="mean", kernel=7))

    # M3: RocketKV (1D-MaxPool k=7) trên attention gốc của Observation Window (KHÔNG dùng m1_score!)
    m3_score = normalized(pool_keys(obs_18, mode="max", kernel=7))

    # M4: Duo-Ensemble (18+35) - Cân bằng 50% L18 và 50% L35 attention gốc
    m4_score = normalized(0.5 * normalized(obs_18) + 0.5 * normalized(obs_35))

    # M5: Deep Sink (Pure Layer 35) - query cuối cùng tại Layer 35
    m5_score = normalized(val[idx_35, accepted][:prefix].astype(np.float64))

    return {
        "M0_SingleQuery": m0_score,
        "M1_P1_MultiQuery": m1_score,
        "M2_SnapKV_Cluster": m2_score,
        "M3_RocketKV_MaxPool": m3_score,
        "M4_DuoEnsemble_18_35": m4_score,
        "M5_DeepSink_L35": m5_score,
    }


def get_target_reference(snapshot, layers: tuple[int, ...], prefix: int) -> np.ndarray:
    target_data = snapshot["target"]
    layer_ids = list(map(int, snapshot["target_layer_ids"]))
    vectors = [normalized(target_data[layer_ids.index(l), :prefix]) for l in layers]
    return np.mean(vectors, axis=0)


def evaluate_snapshots(snapshot_dir: Path, output_dir: Path):
    manifest_path = snapshot_dir.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = sorted(snapshot_dir.glob("*_round_*.npz"))
    if not files:
        raise ValueError(f"Không tìm thấy snapshots tại {snapshot_dir}")

    print(f"[evaluate] Tìm thấy {len(files)} snapshot files...")
    groups = defaultdict(list)
    for p in files:
        prefix_key = p.name.split("_round_")[0]
        groups[prefix_key].append(p)

    output_dir.mkdir(parents=True, exist_ok=True)

    accum = defaultdict(lambda: np.zeros(5, dtype=np.float64))
    pair_count = 0
    source_files = []

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

            # Trích xuất 6 selector scores hoàn toàn độc lập, không pha lẫn
            selector_scores = extract_faithful_selectors(curr_snap, n)

            references = [("same_round_diagnostic", curr_snap)]
            if following_snap is not None:
                if int(following_snap["round"]) == int(curr_snap["round"]) + 1:
                    references.append(("next_round", following_snap))
                    pair_count += 1

            for budget in BUDGETS:
                if budget > n:
                    continue

                # Tạo masks cho 6 selectors (với sink 16 và recent 32)
                masks = {
                    m: select_chunks(selector_scores[m], budget, chunk_size=1, sink=16, recent=32)
                    for m in METHODS
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
    macro_groups = defaultdict(list)
    macro_fields = ("context_cap", "budget", "method", "candidate", "evaluation_scope")
    for r in doc_rows:
        macro_groups[tuple(r[f] for f in macro_fields)].append(r)

    results = []
    for key, rows in sorted(macro_groups.items()):
        record = dict(zip(macro_fields, key)) | {
            "documents": len(rows),
            "snapshots": sum(r["rounds"] for r in rows),
        }
        for metric in ("coverage", "draft_coverage", "spatial_continuity", "selected_tokens"):
            m_val, low, high = bootstrap_ci([r[metric] for r in rows])
            record[f"mean_{metric}"] = m_val
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        results.append(record)

    report = {
        "schema_version": "phase-a.faithful-matrix.v1",
        "status": "ok",
        "phase": "A",
        "unique_documents": len({r["sample_id"] for r in doc_rows}),
        "unique_snapshots": len(files),
        "next_round_pairs": pair_count,
        "methods": METHODS,
        "candidates": list(CANDIDATES.keys()),
        "budgets": list(BUDGETS),
        "context_caps": list(CONTEXT_CAPS),
        "results": results,
    }

    # Write files
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    # Summary CSV
    csv_rows = []
    for r in results:
        csv_rows.append({
            "context_cap": r["context_cap"],
            "budget": r["budget"],
            "method": r["method"],
            "candidate": r["candidate"],
            "evaluation_scope": r["evaluation_scope"],
            "documents": r["documents"],
            "mean_coverage": r["mean_coverage"],
            "coverage_ci_low": r["coverage_ci_low"],
            "coverage_ci_high": r["coverage_ci_high"],
            "mean_draft_coverage": r["mean_draft_coverage"],
            "mean_spatial_continuity": r["mean_spatial_continuity"],
            "mean_selected_tokens": r["mean_selected_tokens"],
        })
    with (output_dir / "sota_comparison_summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    print(f"[evaluate] Hoàn tất xuất kết quả: {len(results)} dòng macro tại {output_dir}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate_snapshots(args.snapshots, args.output)
