"""E22 rank-band repair oracle for fixed-budget DFlash traces.

The oracle changes only the selected token at one detached target-rank band.
It then recomputes the longest exact prefix on the original recorded block.
This keeps the experiment offline and makes the prefix-amplification effect
explicit instead of approximating it with marginal recall.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import read_trace_jsonl, write_metrics_bundle
from .metrics import _blocks, _observed_acceptance


# ``None`` is the open upper bound for the final band.
BAND_SPECS: dict[str, tuple[int, int | None]] = {
    "2-16": (2, 16),
    "17-32": (17, 32),
    "33-64": (33, 64),
    ">64": (65, None),
}


def _rank_in_band(rank: int | None, band: tuple[int, int | None]) -> bool:
    if rank is None:
        return False
    lower, upper = band
    return int(rank) >= lower and (upper is None or int(rank) <= upper)


def repaired_prefix_length(
    block: Sequence[Mapping[str, Any]],
    band: tuple[int, int | None] | None,
) -> int:
    """Return longest prefix after repairing exactly one rank band.

    ``band=None`` is the unchanged draft path.  For a row in ``band`` the
    oracle substitutes the verifier target for the original selected token;
    all later rows remain exactly as recorded.  This is intentionally a
    prefix calculation, not a row-count calculation.
    """

    accepted = 0
    for row in sorted(block, key=lambda item: int(item["draft_position"])):
        target = int(row["target_token_id"])
        selected = int(row["dflash_selected_token_id"])
        rank = row.get("draft_target_rank")
        if band is not None and _rank_in_band(int(rank) if rank is not None else None, band):
            selected = target
        if selected != target:
            break
        accepted += 1
    return accepted


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _hit(row: Mapping[str, Any], k: int) -> bool:
    rank = row.get("draft_target_rank")
    if rank is not None:
        return int(rank) <= int(k)
    return int(row["target_token_id"]) in [int(value) for value in row["candidate_token_ids"][:k]]


def _repaired_hit(row: Mapping[str, Any], band: tuple[int, int | None], k: int) -> bool:
    rank = row.get("draft_target_rank")
    return _rank_in_band(int(rank) if rank is not None else None, band) or _hit(row, k)


def _critical_recall(
    blocks: Sequence[Sequence[Mapping[str, Any]]],
    *,
    hit_fn: Any,
    k: int = 16,
) -> float | None:
    values: list[float] = []
    for position in range(3, 9):
        eligible = [block for block in blocks if len(block) >= position]
        if not eligible:
            continue
        hits = []
        for block in eligible:
            row = next(
                row for row in block if int(row["draft_position"]) == position
            )
            hits.append(float(hit_fn(row, k)))
        values.append(sum(hits) / len(hits))
    return _mean(values)


def _bootstrap_document_delta(
    blocks: Sequence[Sequence[Mapping[str, Any]]],
    band: tuple[int, int | None],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_document: dict[str, list[Sequence[Mapping[str, Any]]]] = defaultdict(list)
    for block in blocks:
        by_document[str(block[0]["document_id"])].append(block)
    documents = sorted(by_document)
    if len(documents) < 2:
        return {"status": "inconclusive", "documents": len(documents)}

    def document_means(document: str) -> tuple[float, float]:
        document_blocks = by_document[document]
        baseline = _mean([float(_observed_acceptance(block)) for block in document_blocks]) or 0.0
        repaired = _mean([float(repaired_prefix_length(block, band)) for block in document_blocks]) or 0.0
        return baseline, repaired

    values = {document: document_means(document) for document in documents}
    rng = random.Random(seed)
    deltas: list[float] = []
    relatives: list[float] = []
    for _ in range(max(1, int(bootstrap_samples))):
        sampled = [rng.choice(documents) for _ in documents]
        baseline = _mean([values[document][0] for document in sampled]) or 0.0
        repaired = _mean([values[document][1] for document in sampled]) or 0.0
        delta = repaired - baseline
        deltas.append(delta)
        relatives.append(delta / baseline if baseline else 0.0)

    def ci(items: Sequence[float]) -> list[float | None]:
        ordered = sorted(items)
        if not ordered:
            return [None, None]
        return [
            ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))],
        ]

    observed_baseline = _mean([values[document][0] for document in documents]) or 0.0
    observed_repaired = _mean([values[document][1] for document in documents]) or 0.0
    observed_delta = observed_repaired - observed_baseline
    return {
        "status": "ok",
        "documents": len(documents),
        "samples": max(1, int(bootstrap_samples)),
        "seed": int(seed),
        "delta_mat": {
            "observed": observed_delta,
            "mean": _mean(deltas),
            "ci95": ci(deltas),
        },
        "relative_gain": {
            "observed": observed_delta / observed_baseline if observed_baseline else None,
            "mean": _mean(relatives),
            "ci95": ci(relatives),
        },
    }


def _band_metrics(
    blocks: Sequence[Sequence[Mapping[str, Any]]],
    rows: Sequence[Mapping[str, Any]],
    band_name: str,
    band: tuple[int, int | None],
    *,
    baseline_mat: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    repaired_values = [float(repaired_prefix_length(block, band)) for block in blocks]
    repaired_mat = _mean(repaired_values)
    selected_rows = [
        row for row in rows if _rank_in_band(
            int(row["draft_target_rank"]), band
        )
    ]
    result: dict[str, Any] = {
        "band": band_name,
        "rank_lower": band[0],
        "rank_upper": band[1],
        "rows_in_band": len(selected_rows),
        "row_fraction": len(selected_rows) / len(rows) if rows else None,
        "mat_repaired": repaired_mat,
        "absolute_gain_vs_mat_d": repaired_mat - baseline_mat if repaired_mat is not None else None,
        "relative_gain_vs_mat_d": (
            (repaired_mat - baseline_mat) / baseline_mat
            if repaired_mat is not None and baseline_mat
            else None
        ),
        "r_at_1_repaired": _mean([
            float(_repaired_hit(row, band, 1)) for row in rows
        ]),
        "r_at_16_repaired": _mean([
            float(_repaired_hit(row, band, 16)) for row in rows
        ]),
        "r_at_16_3_8_repaired": _critical_recall(
            blocks,
            hit_fn=lambda row, k: _repaired_hit(row, band, k),
        ),
        "bootstrap": _bootstrap_document_delta(
            blocks,
            band,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
    }
    return result


def rank_band_repair_oracle(
    traces: Mapping[str, str | Path],
    *,
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Run E22 on fixed on-policy, no-reveal traces."""

    result: dict[str, Any] = {
        "status": "ok",
        "experiment": "E22",
        "bands": list(BAND_SPECS),
        "datasets": {},
    }
    pass_count = 0
    eligible_count = 0
    for dataset, path in traces.items():
        raw_rows = read_trace_jsonl(path)
        rows = [
            row for row in raw_rows
            if row.get("status", "ok") == "ok"
            and row.get("state_mode") == "on_policy"
            and int(row.get("reveal_count", 0)) == 0
        ]
        if not rows:
            result["datasets"][dataset] = {
                "status": "inconclusive",
                "reason": "no_fixed_on_policy_rows",
                "trace": str(path),
            }
            continue
        missing_rank = [row for row in rows if row.get("draft_target_rank") is None]
        if missing_rank:
            result["datasets"][dataset] = {
                "status": "inconclusive",
                "reason": "missing_full_vocabulary_rank",
                "rows": len(rows),
                "missing_rank_rows": len(missing_rank),
                "trace": str(path),
            }
            continue
        blocks = _blocks(rows)
        if not blocks:
            result["datasets"][dataset] = {
                "status": "inconclusive",
                "reason": "no_blocks",
                "trace": str(path),
            }
            continue
        eligible_count += 1
        baseline_values = [float(_observed_acceptance(block)) for block in blocks]
        baseline_mat = _mean(baseline_values) or 0.0
        baseline_r1 = _mean([float(_hit(row, 1)) for row in rows])
        baseline_r16 = _mean([float(_hit(row, 16)) for row in rows])
        baseline_r16_3_8 = _critical_recall(
            blocks,
            hit_fn=lambda row, k: _hit(row, k),
        )
        bands = {
            name: _band_metrics(
                blocks,
                rows,
                name,
                band,
                baseline_mat=baseline_mat,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            for name, band in BAND_SPECS.items()
        }
        shallow_gain = bands["17-32"]["relative_gain_vs_mat_d"]
        if shallow_gain is not None and shallow_gain > 0.20:
            pass_count += 1
        result["datasets"][dataset] = {
            "status": "ok",
            "trace": str(path),
            "rows": len(rows),
            "blocks": len(blocks),
            "documents": len({str(row["document_id"]) for row in rows}),
            "mat_d": baseline_mat,
            "r_at_1_d": baseline_r1,
            "r_at_16_d": baseline_r16,
            "r_at_16_3_8_d": baseline_r16_3_8,
            "bands": bands,
        }
    result["gate"] = {
        "name": "E22 shallow-band headroom",
        "threshold_relative_gain": 0.20,
        "required_datasets": 2,
        "eligible_datasets": eligible_count,
        "passing_datasets": pass_count,
        "decision": "PASS" if pass_count >= 2 else "FAIL",
    }
    return result


def render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# E22 — Rank-Band Repair Oracle",
        "",
        "Oracle offline trên fixed on-policy traces; mỗi band được sửa độc lập.",
        "",
        "| Dataset | Docs | Blocks | MAT_D | [2,16] | [17,32] | [33,64] | >64 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in result.get("datasets", {}).items():
        if item.get("status") != "ok":
            lines.append(f"| {dataset} | n/a | n/a | n/a | inconclusive | inconclusive | inconclusive | inconclusive |")
            continue
        values = []
        for name in BAND_SPECS:
            values.append(
                f"{item['bands'][name].get('relative_gain_vs_mat_d'):.4f}"
                if item["bands"][name].get("relative_gain_vs_mat_d") is not None
                else "n/a"
            )
        lines.append(
            f"| {dataset} | {item['documents']} | {item['blocks']} | {item['mat_d']:.4f} | "
            + " | ".join(values)
            + " |"
        )
    gate = result.get("gate", {})
    lines.extend([
        "",
        f"**Gate:** `{gate.get('decision', 'INCONCLUSIVE')}` — "
        f"{gate.get('passing_datasets', 0)}/{gate.get('eligible_datasets', 0)} dataset đạt "
        f">{gate.get('threshold_relative_gain', 0.20):.0%} relative MAT gain ở band [17,32].",
        "",
        "Các CI bootstrap theo document và toàn bộ metric chi tiết nằm trong `metrics.json`.",
        "",
    ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="DATASET=TRACE_PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    traces: dict[str, str] = {}
    for spec in args.trace:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"trace must be DATASET=TRACE_PATH, got {spec!r}")
        traces[name] = path
    result = rank_band_repair_oracle(
        traces,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output)
    write_metrics_bundle(output, result, report=render_report(result))
    print(json.dumps(result["gate"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
