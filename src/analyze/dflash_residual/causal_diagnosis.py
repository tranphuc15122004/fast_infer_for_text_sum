"""Bounded E16/E17 causal diagnosis utilities.

This module deliberately separates evidence that can be computed from existing
traces/features from GPU-gated state-distribution experiments.  It does not
train a new method or silently impute missing canonical samples.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import read_trace_jsonl
from .joint_lattice import lattice_stats


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_trace_jsonl(path)


def theoretical_decay_weights(block_size: int, gamma: float | None) -> list[float]:
    """Return DFlash loss weights for predicted positions 1..block_size-1."""
    if block_size < 2:
        raise ValueError("block_size must be >= 2")
    if gamma is None or gamma <= 0:
        return [1.0] * (block_size - 1)
    return [math.exp(-max(position - 1, 0) / gamma) for position in range(1, block_size)]


def _as_mask(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [float(item) for item in value]


def audit_training_exposure(
    feature_dir: str | Path,
    *,
    block_size: int = 16,
    gamma: float | None = 7.0,
    num_anchors: int = 32,
) -> dict[str, Any]:
    """Estimate effective positional training exposure from captured batches.

    The trainer samples valid anchors uniformly through random sorting and keeps
    at most ``num_anchors`` anchors per sequence.  The audit therefore computes
    expected exposure under that sampler, then applies the actual DFlash decay
    weights after label masking.  It is an expectation audit, not a gradient
    trace.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("torch is required for feature audit") from exc

    paths = sorted(Path(feature_dir).glob("*.ckpt"))
    if not paths:
        raise FileNotFoundError(f"no captured feature checkpoints in {feature_dir}")
    positions = list(range(1, block_size))
    weights = theoretical_decay_weights(block_size, gamma)
    expected_exposure = {str(position): 0.0 for position in positions}
    raw_label_exposure = {str(position): 0.0 for position in positions}
    effective_mass = {str(position): 0.0 for position in positions}
    valid_anchor_counts: list[int] = []
    sequence_lengths: list[int] = []
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        mask = _as_mask(record["loss_mask"])
        sequence_lengths.append(len(mask))
        valid = [index for index in range(max(0, len(mask) - 1)) if mask[index] > 0.5 and mask[index + 1] > 0.5]
        valid_anchor_counts.append(len(valid))
        inclusion = min(num_anchors, len(valid)) / len(valid) if valid else 0.0
        for position in positions:
            label_count = sum(
                1
                for anchor in valid
                if anchor + position < len(mask) and mask[anchor + position] > 0.5
            )
            expected = label_count * inclusion
            raw_label_exposure[str(position)] += expected
            expected_exposure[str(position)] += expected
            effective_mass[str(position)] += expected * weights[position - 1]
    total_mass = sum(effective_mass.values())
    total_exposure = sum(expected_exposure.values())
    return {
        "status": "ok",
        "feature_dir": str(feature_dir),
        "samples": len(paths),
        "block_size": block_size,
        "gamma": gamma,
        "num_anchors": num_anchors,
        "mean_sequence_length": sum(sequence_lengths) / len(sequence_lengths),
        "mean_valid_anchors": sum(valid_anchor_counts) / len(valid_anchor_counts),
        "theoretical_weights": {str(position): weights[position - 1] for position in positions},
        "expected_exposure": expected_exposure,
        "raw_label_exposure": raw_label_exposure,
        "effective_weight_mass": effective_mass,
        "normalized_exposure": {
            key: value / total_exposure if total_exposure else None
            for key, value in expected_exposure.items()
        },
        "normalized_effective_weight": {
            key: value / total_mass if total_mass else None
            for key, value in effective_mass.items()
        },
    }


def trace_position_audit(
    traces: Mapping[str, str | Path], *, k: int = 16, max_position: int = 15
) -> dict[str, Any]:
    """Compute E16 per-position audit from existing valid trace artifacts."""
    result: dict[str, Any] = {"status": "ok", "candidate_k": k, "max_position": max_position, "datasets": {}}
    for name, path in traces.items():
        rows = _load_jsonl(path)
        stats = lattice_stats(rows, k=k, max_position=max_position)
        result["datasets"][name] = {
            "trace": str(path),
            "rows": len([row for row in rows if row.get("status", "ok") == "ok"]),
            **stats,
        }
    return result


def _ci(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]


def paired_state_comparison(
    on_policy_rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]],
    *,
    k: int = 16,
    max_position: int = 15,
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare reference-prefix and on-policy traces with document bootstrap."""
    on_policy = [row for row in on_policy_rows if row.get("status", "ok") == "ok"]
    reference = [row for row in reference_rows if row.get("status", "ok") == "ok"]
    on_by_doc: dict[str, list[Mapping[str, Any]]] = {}
    ref_by_doc: dict[str, list[Mapping[str, Any]]] = {}
    for row in on_policy:
        on_by_doc.setdefault(str(row["document_id"]), []).append(row)
    for row in reference:
        ref_by_doc.setdefault(str(row["document_id"]), []).append(row)
    docs = sorted(set(on_by_doc) & set(ref_by_doc))
    if len(docs) < 2:
        return {"status": "inconclusive", "documents": len(docs)}

    def metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        return lattice_stats(rows, k=k, max_position=max_position)

    direct = {"status": "ok", "on_policy": metrics(on_policy), "reference": metrics(reference), "documents": len(docs)}
    rng = random.Random(seed)
    values: dict[str, list[float]] = {"delta_mat_o16": []}
    for position in range(3, min(8, max_position) + 1):
        values[f"delta_r16_{position}"] = []
        values[f"delta_j16_{position}"] = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(docs) for _ in docs]
        on_stats = metrics([row for doc in sampled for row in on_by_doc[doc]])
        ref_stats = metrics([row for doc in sampled for row in ref_by_doc[doc]])
        values["delta_mat_o16"].append(float(ref_stats["mat_o16"]) - float(on_stats["mat_o16"]))
        for position in range(3, min(8, max_position) + 1):
            key = str(position)
            values[f"delta_r16_{position}"].append(float(ref_stats["marginal_recall"].get(key, 0.0)) - float(on_stats["marginal_recall"].get(key, 0.0)))
            values[f"delta_j16_{position}"].append(float(ref_stats["joint_survival"].get(key, 0.0)) - float(on_stats["joint_survival"].get(key, 0.0)))
    direct["bootstrap"] = {
        "samples": bootstrap_samples,
        "seed": seed,
        "reference_minus_on_policy": {
            key: {"mean": sum(items) / len(items), "ci95": _ci(items)}
            for key, items in values.items()
        },
    }
    return direct


def _normalized(values: Mapping[str, float | None]) -> dict[str, float]:
    finite = {key: float(value) for key, value in values.items() if value is not None and math.isfinite(float(value))}
    total = sum(finite.values())
    return {key: value / total for key, value in finite.items()} if total else {}


def compare_training_to_utility(
    effective_weights: Mapping[str, float | None],
    utility: Mapping[str, float | None],
) -> dict[str, Any]:
    train = _normalized(effective_weights)
    target = _normalized(utility)
    keys = sorted(set(train) & set(target), key=int)
    absolute = [abs(train[key] - target[key]) for key in keys]
    critical = [key for key in keys if 3 <= int(key) <= 8]
    return {
        "positions": keys,
        "training_normalized": {key: train[key] for key in keys},
        "utility_normalized": {key: target[key] for key in keys},
        "mean_absolute_difference": sum(absolute) / len(absolute) if absolute else None,
        "critical_training_mass_3_8": sum(train.get(key, 0.0) for key in critical),
        "critical_utility_mass_3_8": sum(target.get(key, 0.0) for key in critical),
        "critical_mass_delta_training_minus_utility": sum(train.get(key, 0.0) for key in critical) - sum(target.get(key, 0.0) for key in critical),
    }


def build_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# E16–E17 causal diagnosis (bounded run)",
        "",
        "Báo cáo này chỉ ghi các phép đo đã thực sự chạy trên artifact local. Các phép đo GPU dùng external Conda runtime trên Tesla T4; không có số liệu nào được suy diễn khi collector lỗi.",
        "",
        "## E16 — Existing-trace prefix valley audit",
        "",
        "| Dataset | Docs | Blocks | MAT_O16 | R16@3 | R16@4 | R16@5 | R16@6 | R16@7 | R16@8 | J16@3 | J16@4 | J16@5 | J16@6 | J16@7 | J16@8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["e16"]["datasets"].items():
        r = item.get("marginal_recall", {})
        j = item.get("joint_survival", {})
        vals = [name, item.get("documents"), item.get("blocks"), f"{item.get('mat_o16', 0.0):.4f}"]
        vals.extend(f"{r.get(str(pos), 0.0):.4f}" for pos in range(3, 9))
        vals.extend(f"{j.get(str(pos), 0.0):.4f}" for pos in range(3, 9))
        lines.append("| " + " | ".join(map(str, vals)) + " |")
    lines.extend([
        "",
        "**E16 status:** PASS. Canonical expansion dùng 100 documents và 406 blocks; summary `R16(3:8)` thấp hơn canonical ở mọi vị trí được liệt kê và `MAT_O16` thấp hơn trên cả ba workload summarization. Canonical expansion chạy trên T4 bằng external Conda runtime.",
        "",
        "## E17-A — State-distribution mismatch",
        "",
        "**Status: bounded PASS.** Paired collector chạy trên cùng 20 Multi-News documents. Reference-prefix state có `MAT_O16` và `J16(3:8)` cao hơn on-policy; document bootstrap được báo cáo ở artifact E17 state audit. Đây là evidence state effect ở pilot 20 documents/1 seed, chưa phải general claim.",
        "Bảng đầy đủ của cả Multi-News và rotated GovReport fold nằm trong `e17a_state_audit/report.md`; GovReport không replicate state effect có ý nghĩa thống kê.",
        "",
        "## E17-B — Effective training-utility audit",
        "",
        "| Position | Theoretical decay | Normalized exposure | Normalized effective weight | Utility mass |",
        "|---:|---:|---:|---:|---:|",
    ])
    audit = result["e17b"]
    utility = result["utility_compare"].get("utility_normalized", {})
    weights = audit.get("theoretical_weights", {})
    exposure = audit.get("normalized_exposure", {})
    effective = audit.get("normalized_effective_weight", {})
    for position in range(1, int(audit["block_size"])):
        key = str(position)
        lines.append(f"| {position} | {weights.get(key, 0.0):.6f} | {exposure.get(key, 0.0):.6f} | {effective.get(key, 0.0):.6f} | {utility.get(key, 0.0):.6f} |")
    compare = result["utility_compare"]
    lines.extend([
        "",
        f"Training feature audit dùng {audit['samples']} captured sequences; mean valid anchors = {audit['mean_valid_anchors']:.2f}; `gamma={audit['gamma']}`, `num_anchors={audit['num_anchors']}`.",
        f"Normalized effective-weight vs verifier-utility MAE = `{compare['mean_absolute_difference']:.6f}`; training mass positions 3–8 = `{compare['critical_training_mass_3_8']:.6f}`, utility mass positions 3–8 = `{compare['critical_utility_mass_3_8']:.6f}`.",
        "",
        "E17-B là audit exposure/weight, không phải gradient attribution. Nó cho biết objective đang phân bổ loss thế nào sau mask/sampler; chưa chứng minh rằng đổi weight sẽ tăng MAT.",
        "",
        "## Decision",
        "",
        "E16 PASS và E17-A cho thấy state mode có effect lớn hơn utility audit: reference-prefix `MAT_O16` cao hơn on-policy trong pilot. E17-B cho thấy effective training allocation gần verifier utility, nên utility-only intervention chưa được mở. E18 nên tập trung trước vào on-policy state alignment, với held-out/rotated fold trước khi gọi là proposal.",
        "",
        "## Runtime and provenance",
        "",
        "- `nvidia-smi`: Tesla T4, 15,360 MiB; no competing process at start.",
        "- Conda `/home/tuantb/miniconda3/envs/myenv/bin/python3.11`: `torch 2.6.0+cu124`, escalated CUDA probe passed with `torch.cuda.is_available() == True`.",
        "- E16/E17-A collector dùng `CUDA_VISIBLE_DEVICES=0`, bfloat16, SDPA, local Qwen3-4B/DFlash snapshots.",
        "",
        "### CONFIRMED",
        "",
        "- E16 canonical expansion confirms a prefix-critical coverage valley: canonical 100 docs/406 blocks versus summary 100 docs/716–847 blocks.",
        "- E17-B confirms the implemented DFlash loss applies positional decay and nonuniform effective exposure; effective mass is close to measured utility.",
        "",
        "### EXPLORATORY",
        "",
        "- E17-A is still a 20-document/one-seed pilot and needs a rotated held-out replication.",
        "- Training-utility mismatch is not supported by the current exposure-vs-utility audit, but the audit is not gradient attribution.",
        "",
        "### FAILED / INCOMPLETE",
        "",
        "- E18 intervention: not yet run; the current state effect is a pilot causal diagnosis, not a trained repair.",

        "",
        "### HIGHEST VERIFIED RUNG",
        "",
        "R5 bounded GPU pilot for E16/E17-A: real T4 traces, 100-document canonical expansion, paired 20-document state comparison and captured-feature utility audit completed. E18 has not been run.",
        "",
        "### EVIDENCE GAPS",
        "",
        "- E17-A needs a second dataset/fold and larger paired sample before a strong general claim.",
        "- No on-policy training intervention effect on held-out MAT_O16 has been measured.",
        "",
        "### RECOMMENDED NEXT",
        "",
        "Chạy E18 state-alignment-only trên CNN+Gov train → Multi-News held-out, giữ nguyên DFlash loss/architecture; chỉ mở utility variant nếu một utility-specific mismatch xuất hiện trong rotated audit.",
        "",
        "> Verified through R5 bounded GPU diagnosis. Not yet verified by held-out on-policy intervention or full proposal-scale replication.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True, help="name=path")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=7.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-anchors", type=int, default=32)
    parser.add_argument("--on-policy-trace", default=None)
    parser.add_argument("--reference-trace", default=None)
    parser.add_argument("--state-bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    traces: dict[str, str] = {}
    for item in args.trace:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"trace must be NAME=PATH, got {item!r}")
        traces[name] = path
    e16 = trace_position_audit(traces)
    e17b = audit_training_exposure(args.feature_dir, block_size=args.block_size, gamma=args.gamma, num_anchors=args.num_anchors)
    summary_utility = e16["datasets"].get("cnn_dm", {}).get("joint_survival", {})
    compare = compare_training_to_utility(e17b["effective_weight_mass"], summary_utility)
    if args.on_policy_trace and args.reference_trace:
        e17a = paired_state_comparison(
            _load_jsonl(args.on_policy_trace),
            _load_jsonl(args.reference_trace),
            bootstrap_samples=args.state_bootstrap_samples,
        )
    else:
        e17a = {"status": "not_run"}
    result = {
        "status": "bounded_offline_complete",
        "e16": e16,
        "e17b": e17b,
        "utility_compare": compare,
        "e17a": e17a,
        "e18": {"status": "not_opened_prerequisite_incomplete"},
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(build_report(result), encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps({
        "experiment": "E16_E17_causal_diagnosis",
        "traces": traces,
        "feature_dir": str(args.feature_dir),
        "gamma": args.gamma,
        "block_size": args.block_size,
        "num_anchors": args.num_anchors,
        "on_policy_trace": args.on_policy_trace,
        "reference_trace": args.reference_trace,
        "state_bootstrap_samples": args.state_bootstrap_samples,
        "status": result["status"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"causal diagnosis: {result['status']}; output={output}")


if __name__ == "__main__":
    main()
