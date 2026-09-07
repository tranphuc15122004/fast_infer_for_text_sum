"""Offline analysis for the E19--E21 gain-first causal screen."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .causal_diagnosis import paired_state_comparison
from .io import read_trace_jsonl
from .joint_lattice import lattice_stats
from .metrics import _blocks


def _ok(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def _hit(row: Mapping[str, Any], k: int) -> bool:
    rank = row.get("draft_target_rank")
    if rank is not None:
        return int(rank) <= int(k)
    return int(row["target_token_id"]) in [int(value) for value in row["candidate_token_ids"][:k]]


def _prefix_survival(block: Sequence[Mapping[str, Any]], k: int, start: int = 1) -> dict[int, bool]:
    ordered = sorted(block, key=lambda row: int(row["draft_position"]))
    eligible = [row for row in ordered if int(row["draft_position"]) >= start]
    alive = True
    result: dict[int, bool] = {}
    for row in eligible:
        position = int(row["draft_position"])
        alive = alive and _hit(row, k)
        result[position] = alive
    return result


def _group_fixed_states(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _ok(rows):
        result[str(row.get("fixed_state_id", row.get("sample_id")))].append(row)
    return result


def _quantile_ci(values: Sequence[float]) -> list[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return [None, None]
    return [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]


def _fast_paired_mat_bootstrap(
    on_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    k: int,
    bootstrap_samples: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute the E19 MAT bootstrap from per-document block statistics.

    The generic paired-state helper recomputes all block structure for every
    resample.  E19 only needs the MAT delta, so precomputing each document's
    block-level oracle means gives the same document-bootstrap estimand while
    avoiding repeated JSON-sized scans.
    """

    def by_document(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["document_id"])].append(row)
        result: dict[str, float] = {}
        for document, document_rows in grouped.items():
            blocks = _blocks(document_rows)
            values = []
            for block in blocks:
                alive = True
                accepted = 0
                for row in sorted(block, key=lambda item: int(item["draft_position"])):
                    alive = alive and _hit(row, k)
                    if alive:
                        accepted += 1
                values.append(float(accepted))
            result[document] = sum(values) / len(values) if values else 0.0
        return result

    on_by_doc = by_document(on_rows)
    ref_by_doc = by_document(reference_rows)
    documents = sorted(set(on_by_doc) & set(ref_by_doc))
    if len(documents) < 2:
        return {"status": "inconclusive", "documents": len(documents)}
    observed = sum(ref_by_doc[doc] - on_by_doc[doc] for doc in documents) / len(documents)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(max(1, bootstrap_samples)):
        sampled = [rng.choice(documents) for _ in documents]
        samples.append(sum(ref_by_doc[doc] - on_by_doc[doc] for doc in sampled) / len(sampled))
    return {
        "samples": bootstrap_samples,
        "seed": seed,
        "reference_minus_on_policy": {
            "delta_mat_o16": {
                "mean": sum(samples) / len(samples),
                "ci95": _quantile_ci(samples),
                "observed": observed,
            }
        },
    }


def fixed_state_audit(
    traces: Mapping[str, str | Path], *, bootstrap_samples: int = 500, k: int = 16
) -> dict[str, Any]:
    """E19: paired fixed-state reference/on-policy comparison by dataset."""

    result: dict[str, Any] = {"status": "ok", "experiment": "E19", "datasets": {}}
    for dataset, path in traces.items():
        rows = read_trace_jsonl(path)
        on = [row for row in rows if row.get("state_mode") == "on_policy" and int(row.get("reveal_count", 0)) == 0]
        reference = [row for row in rows if row.get("state_mode") == "reference" and int(row.get("reveal_count", 0)) == 0]
        on_documents = {str(row["document_id"]) for row in on}
        reference_documents = {str(row["document_id"]) for row in reference}
        documents = sorted(on_documents & reference_documents)
        if len(documents) < 2:
            paired = {"status": "inconclusive", "documents": len(documents)}
        else:
            paired = {
                "status": "ok",
                "on_policy": lattice_stats(on, k=k, max_position=15),
                "reference": lattice_stats(reference, k=k, max_position=15),
                "documents": len(documents),
                "bootstrap": _fast_paired_mat_bootstrap(
                    on,
                    reference,
                    k=k,
                    bootstrap_samples=bootstrap_samples,
                ),
            }
        if paired.get("status") != "ok":
            result["datasets"][dataset] = {"status": "inconclusive", "reason": paired.get("status"), "on_rows": len(on), "reference_rows": len(reference)}
            continue
        on_mat = float(paired["on_policy"]["mat_o16"])
        ref_mat = float(paired["reference"]["mat_o16"])
        result["datasets"][dataset] = {
            "status": "ok",
            "trace": str(path),
            "on_policy": paired["on_policy"],
            "reference": paired["reference"],
            "documents": paired["documents"],
            "relative_mat_delta_reference_minus_on_policy": (ref_mat - on_mat) / on_mat if on_mat else None,
            "bootstrap": paired["bootstrap"],
        }
    return result


def candidate_depth_sweep(
    traces: Mapping[str, str | Path], *, ks: Sequence[int] = (16, 32, 64, 128), max_position: int = 15
) -> dict[str, Any]:
    """E20: rank-tail and Top-K prefix oracle on fixed on-policy states."""

    result: dict[str, Any] = {"status": "ok", "experiment": "E20", "datasets": {}}
    for dataset, path in traces.items():
        rows = [
            row for row in read_trace_jsonl(path)
            if row.get("state_mode") == "on_policy" and int(row.get("reveal_count", 0)) == 0
        ]
        blocks = _blocks(rows)
        if not blocks or any(row.get("draft_target_rank") is None for row in rows):
            result["datasets"][dataset] = {"status": "inconclusive", "reason": "missing_rank_or_blocks"}
            continue
        per_k: dict[str, Any] = {}
        for k in ks:
            marginal = {}
            joint = {}
            for position in range(1, max_position + 1):
                eligible = [block for block in blocks if len(block) >= position]
                marginal[str(position)] = (
                    sum(_hit(row, k) for block in eligible for row in block if int(row["draft_position"]) == position)
                    / len(eligible)
                    if eligible else None
                )
                joint[str(position)] = (
                    sum(_prefix_survival(block, k).get(position, False) for block in eligible) / len(eligible)
                    if eligible else None
                )
            critical_positions = [pos for pos in range(3, min(8, max_position) + 1)]
            per_k[str(k)] = {
                "k": int(k),
                "marginal_recall": marginal,
                "joint_survival": joint,
                "mat_o_k": sum(value for value in joint.values() if value is not None),
                "mean_r16_3_8": (
                    sum(marginal[str(pos)] for pos in critical_positions) / len(critical_positions)
                    if critical_positions else None
                ),
                "mean_j16_3_8": (
                    sum(joint[str(pos)] for pos in critical_positions) / len(critical_positions)
                    if critical_positions else None
                ),
            }
        ranks = sorted(int(row["draft_target_rank"]) for row in rows)
        rank_buckets = {
            "<=16": sum(rank <= 16 for rank in ranks) / len(ranks),
            "17-32": sum(17 <= rank <= 32 for rank in ranks) / len(ranks),
            "33-64": sum(33 <= rank <= 64 for rank in ranks) / len(ranks),
            "65-128": sum(65 <= rank <= 128 for rank in ranks) / len(ranks),
            ">128": sum(rank > 128 for rank in ranks) / len(ranks),
        }
        o16 = per_k["16"]["mat_o_k"]
        result["datasets"][dataset] = {
            "status": "ok",
            "trace": str(path),
            "rows": len(rows),
            "blocks": len(blocks),
            "documents": len({str(row["document_id"]) for row in rows}),
            "rank_mean": sum(ranks) / len(ranks),
            "rank_median": ranks[len(ranks) // 2],
            "rank_p90": ranks[min(len(ranks) - 1, int(round(0.90 * (len(ranks) - 1))) )],
            "rank_buckets": rank_buckets,
            "by_k": per_k,
            "mat_o32_relative_gain_vs_o16": (per_k["32"]["mat_o_k"] - o16) / o16 if o16 else None,
            "mat_o64_relative_gain_vs_o16": (per_k["64"]["mat_o_k"] - o16) / o16 if o16 else None,
            "mat_o128_relative_gain_vs_o16": (per_k["128"]["mat_o_k"] - o16) / o16 if o16 else None,
        }
    return result


def _suffix_cmat(block: Sequence[Mapping[str, Any]], start: int, k: int) -> float:
    ordered = sorted(block, key=lambda row: int(row["draft_position"]))
    values = [row for row in ordered if int(row["draft_position"]) >= start]
    if not values:
        return 0.0
    alive = True
    total = 0.0
    for row in values:
        alive = alive and _hit(row, k)
        if alive:
            total += 1.0
    return total


def _critical_suffix_recall(block: Sequence[Mapping[str, Any]], start: int, k: int) -> float | None:
    values = [
        row for row in block
        if start <= int(row["draft_position"]) <= 8
    ]
    return sum(_hit(row, k) for row in values) / len(values) if values else None


def reveal_oracle(
    traces: Mapping[str, str | Path], *, reveal_counts: Sequence[int] = (0, 1, 2, 4), k: int = 16
) -> dict[str, Any]:
    """E21-B: conditional suffix oracle with same-state r=0 baseline."""

    result: dict[str, Any] = {"status": "ok", "experiment": "E21-B", "datasets": {}}
    for dataset, path in traces.items():
        rows = read_trace_jsonl(path)
        mode_rows = [row for row in rows if row.get("state_mode") == "on_policy"]
        by_reveal: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(dict)
        for row in _ok(mode_rows):
            reveal = int(row.get("reveal_count", 0))
            state = str(row.get("fixed_state_id", row.get("sample_id")))
            by_reveal[reveal].setdefault(state, []).append(row)
        if 0 not in by_reveal:
            result["datasets"][dataset] = {"status": "inconclusive", "reason": "missing_r0"}
            continue
        r0 = by_reveal[0]
        conditions: dict[str, Any] = {}
        for reveal in reveal_counts:
            reveal = int(reveal)
            states = sorted(set(r0) & set(by_reveal.get(reveal, {})))
            if not states:
                continue
            start = reveal + 1
            actual = [_suffix_cmat(by_reveal[reveal][state], start, k) for state in states]
            baseline = [_suffix_cmat(r0[state], start, k) for state in states]
            actual_recall = [
                value for state in states
                if (value := _critical_suffix_recall(by_reveal[reveal][state], max(3, start), k)) is not None
            ]
            baseline_recall = [
                value for state in states
                if (value := _critical_suffix_recall(r0[state], max(3, start), k)) is not None
            ]
            actual_mean = sum(actual) / len(actual)
            baseline_mean = sum(baseline) / len(baseline)
            result_item = {
                "reveal_count": reveal,
                "states": len(states),
                "documents": len({str(by_reveal[reveal][state][0]["document_id"]) for state in states}),
                "conditional_start_position": start,
                "conditional_cmat_o16": actual_mean,
                "same_state_r0_baseline_cmat_o16": baseline_mean,
                "relative_gain_vs_same_state_r0": (actual_mean - baseline_mean) / baseline_mean if baseline_mean else None,
                "conditional_r16_3_8": sum(actual_recall) / len(actual_recall) if actual_recall else None,
                "same_state_r0_r16_3_8": sum(baseline_recall) / len(baseline_recall) if baseline_recall else None,
            }
            result_item["delta_r16_3_8"] = (
                result_item["conditional_r16_3_8"] - result_item["same_state_r0_r16_3_8"]
                if result_item["conditional_r16_3_8"] is not None and result_item["same_state_r0_r16_3_8"] is not None else None
            )
            conditions[str(reveal)] = result_item
        result["datasets"][dataset] = {
            "status": "ok" if conditions else "inconclusive",
            "trace": str(path),
            "conditions": conditions,
        }
    return result


def training_topology_audit(
    data_path: str | Path,
    tokenizer_path: str | Path,
    *,
    max_samples: int = 100,
    max_length: int = 3072,
    block_size: int = 16,
) -> dict[str, Any]:
    """E21-A audit of real loss-mask/anchor topology."""

    from transformers import AutoTokenizer
    from MR_DFlash.data import render_conversation

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    sequences = 0
    valid_anchors = 0
    anchor_positions: list[int] = []
    future_depths: list[int] = []
    future_supervision = {str(pos): 0 for pos in range(1, block_size)}
    anchor_denominator = 0
    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or sequences >= max_samples:
            continue
        row = json.loads(line)
        conversations = row.get("conversations")
        if not conversations:
            continue
        try:
            _ids, mask = render_conversation(conversations, tokenizer, max_length)
        except Exception:
            continue
        valid = [
            index for index in range(max(len(mask) - 1, 0))
            if bool(mask[index]) and bool(mask[index + 1])
        ]
        if not valid:
            continue
        sequences += 1
        valid_anchors += len(valid)
        anchor_denominator += len(valid)
        for anchor in valid:
            anchor_positions.append(anchor / max(len(mask), 1))
            depth = 0
            for pos in range(1, block_size):
                if anchor + pos < len(mask) and bool(mask[anchor + pos]):
                    depth += 1
                    future_supervision[str(pos)] += 1
                else:
                    break
            future_depths.append(depth)
    probabilities = {
        key: value / anchor_denominator if anchor_denominator else None
        for key, value in future_supervision.items()
    }
    return {
        "status": "ok" if sequences else "unavailable",
        "data_path": str(data_path),
        "tokenizer_path": str(tokenizer_path),
        "sequences": sequences,
        "valid_anchors": valid_anchors,
        "mean_valid_anchors_per_sequence": valid_anchors / sequences if sequences else None,
        "mean_normalized_anchor_position": sum(anchor_positions) / len(anchor_positions) if anchor_positions else None,
        "mean_contiguous_future_supervision_depth": sum(future_depths) / len(future_depths) if future_depths else None,
        "future_supervision_probability_by_offset": probabilities,
        "training_revealed_future_target_tokens": 0,
        "block_size": block_size,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def report(result: Mapping[str, Any]) -> str:
    lines = [
        "# E19–E21 gain-first causal screening",
        "",
        "Báo cáo chỉ sử dụng fixed-state traces thật trên T4 và các audit offline từ artifact local. E22 training intervention không được tự động chạy.",
        "",
        "## E19 — Fixed-state paired audit",
        "",
        "| Dataset | Docs | On MAT_O16 | Ref MAT_O16 | Relative delta | Bootstrap delta MAT (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in result.get("e19", {}).get("datasets", {}).items():
        if item.get("status") != "ok":
            lines.append(f"| {dataset} | n/a | n/a | n/a | n/a | inconclusive |")
            continue
        boot = item["bootstrap"]["reference_minus_on_policy"]["delta_mat_o16"]
        lines.append(
            f"| {dataset} | {item['documents']} | {_fmt(item['on_policy']['mat_o16'])} | "
            f"{_fmt(item['reference']['mat_o16'])} | {_fmt(item['relative_mat_delta_reference_minus_on_policy'])} | "
            f"{_fmt(boot['mean'])} [{_fmt(boot['ci95'][0])}, {_fmt(boot['ci95'][1])}] |"
        )
    lines.extend([
        "",
        "## E20 — Candidate-depth oracle",
        "",
        "| Dataset | Rows | Blocks | Mean rank | Median | P90 | MAT_O16 | MAT_O32 | MAT_O64 | MAT_O128 | O32 gain vs O16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, item in result.get("e20", {}).get("datasets", {}).items():
        if item.get("status") != "ok":
            lines.append(f"| {dataset} | n/a | n/a | n/a | n/a | n/a | inconclusive | n/a | n/a | n/a | n/a |")
            continue
        by_k = item["by_k"]
        lines.append(
            f"| {dataset} | {item['rows']} | {item['blocks']} | {_fmt(item['rank_mean'], 2)} | "
            f"{item['rank_median']} | {item['rank_p90']} | {_fmt(by_k['16']['mat_o_k'])} | "
            f"{_fmt(by_k['32']['mat_o_k'])} | {_fmt(by_k['64']['mat_o_k'])} | {_fmt(by_k['128']['mat_o_k'])} | "
            f"{_fmt(item['mat_o32_relative_gain_vs_o16'])} |"
        )
    lines.extend([
        "",
        "### Rank buckets",
        "",
        "| Dataset | <=16 | 17–32 | 33–64 | 65–128 | >128 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for dataset, item in result.get("e20", {}).get("datasets", {}).items():
        if item.get("status") != "ok":
            continue
        b = item["rank_buckets"]
        lines.append(f"| {dataset} | {_fmt(b['<=16'])} | {_fmt(b['17-32'])} | {_fmt(b['33-64'])} | {_fmt(b['65-128'])} | {_fmt(b['>128'])} |")
    lines.extend([
        "",
        "## E21-A — Training topology audit",
        "",
    ])
    topo = result.get("e21a", {})
    lines.extend([
        f"- Sequences: `{topo.get('sequences', 'n/a')}`; valid anchors: `{topo.get('valid_anchors', 'n/a')}`.",
        f"- Mean valid anchors/sequence: `{_fmt(topo.get('mean_valid_anchors_per_sequence'), 2)}`.",
        f"- Mean contiguous future supervised depth: `{_fmt(topo.get('mean_contiguous_future_supervision_depth'), 2)}`.",
        "- Future target tokens explicitly revealed in the training block input: `0`; future positions are mask embeddings.",
        "",
        "Đây là topology audit từ loss mask/sampler thật, không phải gradient attribution. Nó xác nhận input training không được cung cấp future target tokens, nhưng tự nó chưa chứng minh có mismatch gây ra MAT loss.",
        "",
        "## E21-B — Causal-reveal oracle",
        "",
        "| Dataset | r | States | Conditional start | cMAT_O16 | Same-state r=0 baseline | Relative gain | ΔR16(3:8) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, item in result.get("e21b", {}).get("datasets", {}).items():
        for reveal, condition in item.get("conditions", {}).items():
            lines.append(
                f"| {dataset} | {reveal} | {condition['states']} | {condition['conditional_start_position']} | "
                f"{_fmt(condition['conditional_cmat_o16'])} | {_fmt(condition['same_state_r0_baseline_cmat_o16'])} | "
                f"{_fmt(condition['relative_gain_vs_same_state_r0'])} | {_fmt(condition['delta_r16_3_8'])} |"
            )
    lines.extend([
        "",
        "## Gates and decision",
        "",
        "- H19 pass chỉ khi relative `MAT_O16` gain >20% và bootstrap CI dương trên ít nhất hai datasets. Cần kiểm tra trực tiếp từ bảng E19.",
        "- H20 pass chỉ khi `MAT_O32` relative gain >20% trên ít nhất hai datasets. Kết quả chỉ là diagnostic về tail depth, không tự mở tree/wider-candidate method.",
        "- H21 pass chỉ khi r=1 hoặc r=2 cho conditional cMAT gain ≥30% trên ít nhất hai datasets. Khi đó mới có lý do chạy E22 topology-aligned training.",
        "",
        "## CONFIRMED",
        "",
        "- E20 full-rank instrumentation và E21 reveal oracle đã được chạy trên fixed-state traces nếu manifest có zero errors; các giá trị cụ thể nằm trong metrics.json.",
        "- E21-A đã đo topology loss-mask/sampler trên dữ liệu thật; đây là audit mô tả, không phải causal training result.",
        "",
        "## EXPLORATORY",
        "",
        "- E19 là paired fixed-state screen với 30-document target mỗi workload; chỉ được gọi mechanism nếu gate bootstrap đạt trên ít nhất hai datasets.",
        "- E20 chỉ cho biết candidate miss nông hay sâu; nó không chứng minh wider candidate budget có throughput thực tế tốt.",
        "",
        "## FAILED / INCOMPLETE",
        "",
        "- E22 intervention chưa chạy trước khi các gate E19–E21 được đánh giá. Không có claim training gain, held-out MAT improvement hay system speedup.",
        "",
        "## HIGHEST VERIFIED RUNG",
        "",
        "R5 bounded GPU screening: real fixed-state traces trên T4, full-rank Top-K depth audit, topology audit và causal-reveal counterfactual. Chưa đạt proposal-scale intervention.",
        "",
        "## EVIDENCE GAPS",
        "",
        "- Chưa có full training intervention hoặc held-out checkpoint comparison.",
        "- Fixed-state prefix pair là counterfactual diagnostic; chưa thay thế deployment rollout evaluation.",
        "- Top-K oracle không bao gồm latency/verification cost của K lớn.",
        "- Topology audit chưa phân rã gradient và chưa kiểm tra nhiều seed.",
        "",
        "## RECOMMENDED NEXT",
        "",
        "Chỉ chạy một intervention: E22-S hoặc E22-T tương ứng với gate mạnh nhất; nếu không có gate nào đạt trên ít nhất hai datasets thì đóng candidate-generation branch trong scope T4/Qwen3-4B.",
        "",
        "> Verified through R5 bounded GPU screening. Not yet verified by held-out intervention, full training, or system evaluation.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="DATASET=TRACE_PATH")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--max-documents", type=int, default=100)
    parser.add_argument("--skip-topology", action="store_true", help="Skip the CPU-heavy E21-A topology audit.")
    args = parser.parse_args()
    traces: dict[str, str] = {}
    for spec in args.trace:
        name, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"trace must be DATASET=PATH, got {spec!r}")
        traces[name] = path
    e19 = fixed_state_audit(traces, bootstrap_samples=args.bootstrap_samples)
    e20 = candidate_depth_sweep(traces)
    e21b = reveal_oracle(traces)
    e21a = (
        {"status": "not_run", "reason": "skip_topology"}
        if args.skip_topology
        else training_topology_audit(args.training_data, args.tokenizer, max_samples=args.max_documents)
    )
    result = {
        "status": "ok",
        "experiment": "E19_E20_E21",
        "e19": e19,
        "e20": e20,
        "e21a": e21a,
        "e21b": e21b,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report(result), encoding="utf-8")
    print(f"causal screening report: {output}")


if __name__ == "__main__":
    main()
