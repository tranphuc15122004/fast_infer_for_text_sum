#!/usr/bin/env python3
"""Summarize a retrieved length-bin Context-Adaptive DFlash Modal pilot."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASELINE = "dflash_full_fixed"
AR = "ar"
VARIANTS = (
    BASELINE,
    "a_only",
    "b_only_history",
    "b_only_entropy",
    "independent_ab",
    "joint_no_entropy",
    "joint_no_source_relevance",
    "joint",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _paired_geomean_speedup(
    rows: list[dict[str, Any]],
    baseline_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[float | None, int, int]:
    logs_by_dataset: dict[str, list[float]] = defaultdict(list)
    exact = 0
    pair_count = 0
    for row in rows:
        key = (str(row["dataset"]), str(row["sample_id"]), str(row["repetition"]))
        base = baseline_by_key.get(key)
        if base is None:
            continue
        baseline_ms = float(base["e2e_ms"])
        candidate_ms = float(row["e2e_ms"])
        if baseline_ms <= 0 or candidate_ms <= 0:
            continue
        logs_by_dataset[str(row["dataset"])].append(math.log(baseline_ms / candidate_ms))
        pair_count += 1
        if row.get("output_ids_hash") == base.get("output_ids_hash"):
            exact += 1
    dataset_log_means = [sum(values) / len(values) for values in logs_by_dataset.values() if values]
    speedup = math.exp(sum(dataset_log_means) / len(dataset_log_means)) if dataset_log_means else None
    return speedup, pair_count, exact


def _load(root: Path, run_id: str) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], dict[tuple[str, str, str], str]]:
    manifest_path = root / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bin_by_sample = {
        (str(row["dataset"]), str(row["id"])): str(row["length_bin_name"])
        for row in manifest["samples"]
    }
    run_root = root / "runs" / run_id
    requests: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in sorted(run_root.glob("dev/*/online/rep_*/requests.jsonl")):
        rows = _read_jsonl(path)
        for row in rows:
            if row.get("type") != "sample" or row.get("status") != "ok":
                continue
            key = (
                str(row["variant"]), str(row["dataset"]), str(row["sample_id"]), str(row["repetition"])
            )
            row["length_bin_name"] = bin_by_sample[(key[1], key[2])]
            requests[key] = row
    return requests, bin_by_sample


def analyze(root: Path, run_id: str) -> dict[str, Any]:
    requests, bin_by_sample = _load(root, run_id)
    run_root = root / "runs" / run_id
    hardware_path = run_root / "evidence/modal_hardware.json"
    prereg_path = run_root / "evidence/preregistration.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8")) if hardware_path.is_file() else {}
    preregistration = json.loads(prereg_path.read_text(encoding="utf-8")) if prereg_path.is_file() else {}
    summary_metadata = {
        "run_id": run_id,
        "gpu_name": hardware.get("gpu_name", "unknown GPU"),
        "gpu_capability": hardware.get("gpu_capability"),
        "gpu_memory_bytes": hardware.get("gpu_memory_bytes"),
        "python": hardware.get("python", "unknown"),
        "torch": hardware.get("torch", "unknown"),
        "torch_cuda": hardware.get("torch_cuda", "unknown"),
        "transformers": hardware.get("runtime_versions", {}).get("transformers", "unknown"),
        "flash_attn_4": hardware.get("runtime_versions", {}).get("flash-attn-4", "unknown"),
        "attention_backend": hardware.get("attention_backend_requested", "unknown"),
        "max_new_tokens": preregistration.get("max_new_tokens", 128),
        "output_scope": preregistration.get("output_scope", "fixed_tokens"),
        "calibration_documents": preregistration.get("calibration_documents", 30),
        "dev_documents": preregistration.get("dev_documents", 15),
        "repetitions": preregistration.get("repetitions", 3),
        "warmup_runs": preregistration.get("warmup_runs", 1),
        "variant_order": hardware.get("variant_order", preregistration.get("variant_order", [])),
        "fa4_kernel_probe_ms": hardware.get("fa4_kernel_probe_ms"),
    }
    bins = [f"bin{i}_{label}" for i, label in enumerate(("00k_02k", "02k_04k", "04k_08k", "08k_12k", "12k_16k"), start=1)]
    variants = [AR, *VARIANTS]
    baseline_rows_by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (variant, _, _, _), row in requests.items():
        if variant == BASELINE:
            baseline_rows_by_bin[row["length_bin_name"]].append(row)

    summary: dict[str, Any] = {"bins": {}, "variants": list(VARIANTS), "baseline": BASELINE, "ar_control": AR}
    for bin_name in bins:
        bin_summary: dict[str, Any] = {"variants": {}}
        for variant in variants:
            rows = [
                row for (name, _, _, _), row in requests.items()
                if name == variant and row["length_bin_name"] == bin_name
            ]
            base_rows = baseline_rows_by_bin[bin_name]
            base_by_key = {
                (str(row["dataset"]), str(row["sample_id"]), str(row["repetition"])): row
                for row in base_rows
            }
            if variant == AR:
                dflash_speedup, pair_count, exact_vs_dflash = _paired_geomean_speedup(rows, base_by_key)
            else:
                dflash_speedup, pair_count, exact_vs_dflash = _paired_geomean_speedup(rows, base_by_key)

            ar_rows = [
                candidate for (name, _, _, _), candidate in requests.items()
                if name == AR and candidate["length_bin_name"] == bin_name
            ]
            ar_by_key = {
                (str(row["dataset"]), str(row["sample_id"]), str(row["repetition"])): row
                for row in ar_rows
            }
            _, ar_pair_count, exact_vs_ar = _paired_geomean_speedup(rows, ar_by_key)
            input_tokens = [float(row["input_tokens"]) for row in rows if row.get("input_tokens") is not None]
            accept = [float(row["avg_accept_length"]) for row in rows if row.get("avg_accept_length") is not None]
            acceptance_rate = [float(row["acceptance_rate"]) for row in rows if row.get("acceptance_rate") is not None]
            throughput = [float(row["throughput_tok_s"]) for row in rows if row.get("throughput_tok_s") is not None]
            e2e = [float(row["e2e_ms"]) for row in rows if row.get("e2e_ms") is not None]
            bin_summary["variants"][variant] = {
                "requests": len(rows),
                "unique_dev_documents": len({str(row["sample_id"]) for row in rows}),
                "mean_prompt_tokens": _mean(input_tokens),
                "min_prompt_tokens": min(input_tokens) if input_tokens else None,
                "max_prompt_tokens": max(input_tokens) if input_tokens else None,
                "mean_acceptance_length": _mean(accept),
                "mean_acceptance_rate": _mean(acceptance_rate),
                "mean_tok_s": _mean(throughput),
                "mean_e2e_ms": _mean(e2e),
                "median_e2e_ms": statistics.median(e2e) if e2e else None,
                "speedup_vs_dflash_macro_geomean": dflash_speedup,
                "paired_requests_vs_dflash": pair_count,
                "token_id_exact_vs_dflash": exact_vs_dflash,
                "token_id_exact_vs_ar": exact_vs_ar,
                "paired_requests_vs_ar": ar_pair_count,
            }

        # Action/budget use helps explain whether the adaptive policy really activated.
        action_by_variant: dict[str, Any] = {}
        for variant in VARIANTS:
            round_rows: list[dict[str, Any]] = []
            for path in sorted((root / "runs" / run_id / "dev" / variant / "online").glob("rep_*/rounds.jsonl")):
                round_rows.extend(
                    row for row in _read_jsonl(path)
                    if row.get("type") == "round"
                    and row.get("status") == "ok"
                    and bin_by_sample.get((str(row["dataset"]), str(row["sample_id"]))) == bin_name
                )
            budgets = Counter(str(row.get("requested_budget")) for row in round_rows)
            gammas = Counter(int(row.get("gamma_executed", 0)) for row in round_rows)
            fallback = Counter(str(row.get("fallback_reason")) for row in round_rows if row.get("fallback_reason"))
            action_by_variant[variant] = {
                "rounds": len(round_rows),
                "budget_counts": dict(sorted(budgets.items())),
                "budget_1024_fraction": budgets.get("1024", 0) / len(round_rows) if round_rows else None,
                "gamma_counts": dict(sorted(gammas.items())),
                "mean_gamma": _mean([float(row["gamma_executed"]) for row in round_rows]),
                "fallback_reasons": dict(sorted(fallback.items())),
            }
        bin_summary["action_usage"] = action_by_variant
        dflash_tok_s = bin_summary["variants"][BASELINE]["mean_tok_s"]
        ar_tok_s = bin_summary["variants"][AR]["mean_tok_s"]
        for row in bin_summary["variants"].values():
            row["throughput_speedup_vs_dflash"] = (
                row["mean_tok_s"] / dflash_tok_s
                if row["mean_tok_s"] is not None and dflash_tok_s
                else None
            )
            row["throughput_speedup_vs_ar"] = (
                row["mean_tok_s"] / ar_tok_s
                if row["mean_tok_s"] is not None and ar_tok_s
                else None
            )
        summary["bins"][bin_name] = bin_summary

    # Pooled summary follows the same paired macro-geometric estimator, across datasets.
    pooled: dict[str, Any] = {}
    all_bins = set(bins)
    for variant in variants:
        rows = [row for (name, _, _, _), row in requests.items() if name == variant and row["length_bin_name"] in all_bins]
        base_by_key = {
            (str(row["dataset"]), str(row["sample_id"]), str(row["repetition"])): row
            for row in requests.values() if row["variant"] == BASELINE
        }
        speedup, pairs, exact_vs_dflash = _paired_geomean_speedup(rows, base_by_key)
        ar_by_key = {
            (str(row["dataset"]), str(row["sample_id"]), str(row["repetition"])): row
            for row in requests.values() if row["variant"] == AR
        }
        _, ar_pairs, exact_vs_ar = _paired_geomean_speedup(rows, ar_by_key)
        acceptance = [float(row["avg_accept_length"]) for row in rows if row.get("avg_accept_length") is not None]
        tok_s = [float(row["throughput_tok_s"]) for row in rows if row.get("throughput_tok_s") is not None]
        e2e = [float(row["e2e_ms"]) for row in rows if row.get("e2e_ms") is not None]
        pooled[variant] = {
            "requests": len(rows),
            "mean_acceptance_length": _mean(acceptance),
            "mean_tok_s": _mean(tok_s),
            "mean_e2e_ms": _mean(e2e),
            "speedup_vs_dflash_macro_geomean": speedup,
            "paired_requests_vs_dflash": pairs,
            "token_id_exact_vs_dflash": exact_vs_dflash,
            "token_id_exact_vs_ar": exact_vs_ar,
            "paired_requests_vs_ar": ar_pairs,
        }
    dflash_tok_s = pooled[BASELINE]["mean_tok_s"]
    ar_tok_s = pooled[AR]["mean_tok_s"]
    for row in pooled.values():
        row["throughput_speedup_vs_dflash"] = (
            row["mean_tok_s"] / dflash_tok_s
            if row["mean_tok_s"] is not None and dflash_tok_s
            else None
        )
        row["throughput_speedup_vs_ar"] = (
            row["mean_tok_s"] / ar_tok_s
            if row["mean_tok_s"] is not None and ar_tok_s
            else None
        )
    summary["pooled"] = pooled

    comparison = json.loads((run_root / "dev/online/comparison.json").read_text())
    summary["report_gates"] = {
        "report_gate_status": comparison.get("gate_status"),
        "provenance_mixed": comparison.get("provenance_mixed"),
        "expected_requests": comparison.get("expected_requests"),
    }
    summary["experiment_metadata"] = summary_metadata
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    metadata = summary.get("experiment_metadata", {})
    gpu_name = metadata.get("gpu_name", "unknown GPU")
    capability = metadata.get("gpu_capability")
    gpu_text = gpu_name + (f" (compute capability {'.'.join(map(str, capability))})" if capability else "")
    runtime_text = (
        f"Python {metadata.get('python')}; Torch {metadata.get('torch')} + CUDA {metadata.get('torch_cuda')}; "
        f"Transformers {metadata.get('transformers')}; FlashAttention-4 {metadata.get('flash_attn_4')}; "
        f"backend `{metadata.get('attention_backend')}`"
    )
    output_text = (
        f"max-new-tokens {metadata.get('max_new_tokens')}, `{metadata.get('output_scope')}`, "
        f"{metadata.get('repetitions')} repetitions and {metadata.get('warmup_runs')} warmups"
    )
    schedule_text = metadata.get("variant_order")
    schedule_text = ", ".join(schedule_text) if schedule_text else "thứ tự không được ghi trong metadata"
    lines = [
        "# Kết quả pilot Context-Adaptive DFlash theo length bin",
        "",
        "Speedup throughput được tính bằng `mean tok/s(variant) / mean tok/s(baseline)`; `tok/s` và mean acceptance length là trung bình request-level. Cột E2E speedup được tính riêng bằng paired geometric macro mean theo dataset của `E2E(DFlash full fixed) / E2E(variant)`.",
        "",
        "## Thiết lập thực nghiệm",
        "",
        f"- Chạy trên Modal {gpu_text}; Qwen3-4B target + Qwen3-4B-DFlash-b16 draft, BF16. Runtime: {runtime_text}.",
        f"- Dữ liệu từ `data/length_bins/`, chỉ gồm summarization (`gov_report`, `multi_news`, `qmsum`); loại code-completion. Split: {metadata.get('calibration_documents')} calibration và {metadata.get('dev_documents')} dev documents, chia theo source group; seed 42.",
        f"- Dev: 3 documents/bin × 3 repetitions = 9 request/variant/bin; greedy, {output_text}. Calibration dùng checkpoints output 0/64, context budget 1024/full và gamma 3/7/15.",
        "- Variants: AR, DFlash full-context gamma 15, `a_only`, hai B-only (history/entropy), `independent_ab`, `joint_no_entropy`, `joint_no_source_relevance`, và `joint`.",
        f"- Tổng cộng {summary['report_gates'].get('expected_requests')} request dev theo manifest; variant-block order: {schedule_text}. Các variant chạy tuần tự nên timing vẫn có thể chịu ảnh hưởng clock/order.",
        f"- Mỗi bin chỉ có 3 document độc lập và output cap {metadata.get('max_new_tokens')} token; đây là pilot, chưa phải heldout/natural-EOS benchmark.",
        "",
        "## Pooled dev",
        "",
        "| Variant | Mean acceptance length | Mean tok/s | tok/s speedup vs DFlash | tok/s speedup vs AR | Mean E2E (ms) | E2E speedup vs DFlash | Exact token IDs vs DFlash | Exact token IDs vs AR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in (BASELINE, "a_only", "b_only_history", "b_only_entropy", "independent_ab", "joint_no_entropy", "joint_no_source_relevance", "joint", AR):
        row = summary["pooled"][variant]
        e2e_speedup = "—" if row["speedup_vs_dflash_macro_geomean"] is None else f"{row['speedup_vs_dflash_macro_geomean']:.3f}×"
        accept = "—" if row["mean_acceptance_length"] is None else f"{row['mean_acceptance_length']:.3f}"
        speedup_dflash = "—" if row["throughput_speedup_vs_dflash"] is None else f"{row['throughput_speedup_vs_dflash']:.3f}×"
        speedup_ar = "—" if row["throughput_speedup_vs_ar"] is None else f"{row['throughput_speedup_vs_ar']:.3f}×"
        lines.append(
            f"| {variant} | {accept} | {row['mean_tok_s']:.2f} | {speedup_dflash} | {speedup_ar} | {row['mean_e2e_ms']:.1f} | {e2e_speedup} | {row['token_id_exact_vs_dflash']}/{row['paired_requests_vs_dflash']} | {row['token_id_exact_vs_ar']}/{row['paired_requests_vs_ar']} |"
        )
    lines.extend([
        "",
        "## Theo length bin",
        "",
        "| Length bin | Variant | n requests | Mean prompt tok | Mean acceptance length | Mean tok/s | tok/s speedup vs DFlash | tok/s speedup vs AR | Mean E2E (ms) | E2E speedup vs DFlash | Exact vs DFlash | Exact vs AR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for bin_name, bin_data in summary["bins"].items():
        for variant in (BASELINE, "a_only", "b_only_history", "b_only_entropy", "independent_ab", "joint_no_entropy", "joint_no_source_relevance", "joint", AR):
            row = bin_data["variants"][variant]
            e2e_speedup = "—" if row["speedup_vs_dflash_macro_geomean"] is None else f"{row['speedup_vs_dflash_macro_geomean']:.3f}×"
            speedup_dflash = "—" if row["throughput_speedup_vs_dflash"] is None else f"{row['throughput_speedup_vs_dflash']:.3f}×"
            speedup_ar = "—" if row["throughput_speedup_vs_ar"] is None else f"{row['throughput_speedup_vs_ar']:.3f}×"
            prompt_mean = "—" if row["mean_prompt_tokens"] is None else f"{row['mean_prompt_tokens']:.0f}"
            accept = "—" if row["mean_acceptance_length"] is None else f"{row['mean_acceptance_length']:.3f}"
            tok_s = "—" if row["mean_tok_s"] is None else f"{row['mean_tok_s']:.2f}"
            e2e = "—" if row["mean_e2e_ms"] is None else f"{row['mean_e2e_ms']:.1f}"
            lines.append(
                f"| {bin_name} | {variant} | {row['requests']} | {prompt_mean} | {accept} | {tok_s} | {speedup_dflash} | {speedup_ar} | {e2e} | {e2e_speedup} | {row['token_id_exact_vs_dflash']}/{row['paired_requests_vs_dflash']} | {row['token_id_exact_vs_ar']}/{row['paired_requests_vs_ar']} |"
            )
    lines.extend([
        "",
        "## Mức sử dụng action của `joint`",
        "",
        "| Length bin | Rounds | Budget 1024 | Budget full | Gamma 3 | Gamma 7 | Gamma 15 | Mean gamma |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for bin_name, bin_data in summary["bins"].items():
        row = bin_data["action_usage"]["joint"]
        counts = row["budget_counts"]
        gammas = row["gamma_counts"]
        n = max(row["rounds"], 1)
        mean_gamma = "—" if row["mean_gamma"] is None else f"{row['mean_gamma']:.2f}"
        lines.append(
            f"| {bin_name} | {row['rounds']} | {counts.get('1024', 0)} ({counts.get('1024', 0)/n:.1%}) | {counts.get('full', 0)} ({counts.get('full', 0)/n:.1%}) | {gammas.get(3, 0)} | {gammas.get(7, 0)} | {gammas.get(15, 0)} | {mean_gamma} |"
        )
    lines.extend([
        "",
        "## Diễn giải và giới hạn",
        "",
        "- Mỗi bin có 3 dev documents × 3 repetitions; output cố định 128 token. Đây là pilot để so sánh theo độ dài, không phải heldout/natural-EOS headline.",
        "- Exact-token columns đối chiếu trực tiếp output IDs. Reporter đánh dấu `headline_valid=false` với AR khi output mismatch; speedup so AR chỉ mang tính mô tả.",
        "- Nếu một ablation khớp token IDs với DFlash nhưng DFlash không khớp AR, nó giữ nguyên output của DFlash; correctness của shared DFlash path vẫn cần được sửa/audit.",
        "- Acceptance length cao hơn không tự đảm bảo throughput speedup; xem đồng thời tok/s, E2E latency và overhead/action usage.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/cadflash_lengthbins_modal_20261009/retrieved"))
    parser.add_argument("--run-id", default="cadflash_lengthbins_modal_20261009_001")
    args = parser.parse_args()
    result = analyze(args.root, args.run_id)
    json_path = args.root.parent / "lengthbin_metrics.json"
    markdown_path = args.root.parent / "lengthbin_results.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(markdown_path)
    print(json_path)


if __name__ == "__main__":
    main()
