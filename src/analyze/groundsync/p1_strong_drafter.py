"""EAGLE-3 strong-drafter replication and direct E2E benchmark.

The controlled P0 traces use Qwen3-0.6B as a deliberately small drafter.  This
module exercises the locally cached ``AngelSlim/Qwen3-4B_eagle3`` head paired
with the canonical Qwen3-4B target.  It records per-verification acceptance
rounds and, optionally, a matched greedy autoregressive run.  The latter is a
real model-level E2E measurement, but it is explicitly labelled direct EAGLE
inference rather than vLLM/API serving.

The analysis helpers are CPU-safe.  Only ``run_eagle`` loads CUDA models.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .p0_decision import across_round_persistence, summarize_within_block_burstiness


def load_dataset(path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def acceptance_rows_from_generations(
    generations: Sequence[Mapping[str, Any]],
    *,
    max_k: int = 4,
) -> list[dict[str, Any]]:
    """Expand EAGLE per-generation acceptance lists into P0-compatible rows."""

    result: list[dict[str, Any]] = []
    for generation in generations:
        document_id = str(generation.get("document_id"))
        acceptance = generation.get("acceptance_lengths") or []
        remaining = int(generation.get("new_tokens", 0) or 0)
        for round_index, raw_committed in enumerate(acceptance):
            committed = max(0, min(int(raw_committed), remaining))
            remaining -= committed
            # EAGLE's acceptance length includes one target fallback token.
            accepted_draft = max(0, committed - 1)
            result.append({
                "status": "ok",
                "document_id": document_id,
                "start_position": round_index,
                "max_k": max_k,
                "accepted_len": accepted_draft,
                "first_reject_rel": accepted_draft + 1 if accepted_draft < max_k else None,
                "proposal_token_ids": [0] * max_k,
                "draft_confidence": [0.0] * max_k,
                "eagle_committed_tokens": committed,
                "eagle_acceptance_length_raw": int(raw_committed),
            })
            if remaining <= 0:
                break
    return result


def analyze_acceptance_rows(
    generations: Sequence[Mapping[str, Any]],
    *,
    max_k: int = 4,
    persistence_deltas: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """Compute within-block and across-round burstiness for EAGLE output."""

    rows = acceptance_rows_from_generations(generations, max_k=max_k)
    within = summarize_within_block_burstiness(rows, max_k=max_k)
    persistence = across_round_persistence(rows, deltas=persistence_deltas)
    per_doc_rounds = Counter(str(row.get("document_id")) for row in rows)
    admission = [int(int(row.get("accepted_len", 0) or 0) > 0) for row in rows]
    return {
        "schema_version": "groundsync.p1.strong_drafter.analysis.v1",
        "status": "ok" if rows else "UNAVAILABLE",
        "coverage": {
            "generation_count": len(generations),
            "round_count": len(rows),
            "document_count": len(per_doc_rounds),
            "rounds_per_document": dict(sorted(per_doc_rounds.items())),
            "admission_rate": statistics.fmean(admission) if admission else None,
        },
        "within_block_burstiness": within,
        "across_round_persistence": persistence,
        "decision_gate": "within-block ratio > 1 and at least one persistence delta has lower CI > 0",
        "decision": (
            "PASS"
            if within.get("decision") == "PASS"
            and any(
                value.get("excess_ci")
                and float(value["excess_ci"]["low"]) > 0.0
                for value in persistence.get("by_delta", {}).values()
            )
            else "FAIL" if rows else "UNAVAILABLE"
        ),
    }


def _dataset_prompt(row: Mapping[str, Any]) -> str:
    document = row.get("document") or row.get("text") or row.get("prompt")
    if not document:
        raise ValueError("dataset row has no document/text/prompt")
    if "Summarize" in str(document)[:80]:
        return str(document)
    return (
        "Summarize the following document faithfully and concisely. "
        "Return only the summary.\n\nDocument:\n" + str(document)
    )


def _build_input_ids(tokenizer: Any, prompt: str, device: str) -> Any:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return encoded.to(device)


def run_eagle(
    *,
    base_model: Path,
    eagle_model: Path,
    dataset: Path,
    output_dir: Path,
    max_samples: int,
    max_new_tokens: int,
    max_input_tokens: int,
    total_token: int,
    depth: int,
    top_k: int,
    include_naive: bool,
) -> dict[str, Any]:
    """Run local EAGLE-3 on a dataset and write raw+summary artifacts."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("strong-drafter/E2E run requires visible CUDA")
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "externals" / "EAGLE"))
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values

    rows = load_dataset(dataset, max_samples=max_samples)
    model = EaModel.from_pretrained(
        base_model_path=str(base_model),
        ea_model_path=str(eagle_model),
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        use_eagle3=True,
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    inputs = []
    for row in rows:
        input_ids = _build_input_ids(tokenizer, _dataset_prompt(row), "cuda")
        if max_input_tokens > 0:
            input_ids = input_ids[:, :max_input_tokens]
        inputs.append(input_ids)
    max_prompt = max(int(value.shape[1]) for value in inputs)
    kv_max_length = max_prompt + max_new_tokens + total_token + 32
    past_kv, past_data, current_length = initialize_past_key_values(
        model.base_model, max_length=kv_max_length
    )
    model.past_key_values = past_kv
    model.past_key_values_data = past_data
    model.current_length_data = current_length

    warmup = _build_input_ids(tokenizer, "Summarize: warmup", "cuda")
    with torch.inference_mode():
        model.eagenerate(
            warmup, temperature=0.0, max_new_tokens=4,
            max_length=warmup.shape[1] + 4 + total_token + 32,
        )
        if include_naive:
            model.naivegenerate(
                warmup, temperature=0.0, max_new_tokens=4,
                max_length=warmup.shape[1] + 4 + total_token + 32,
            )

    generations: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "strong_drafter_generations.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for index, (row, input_ids) in enumerate(zip(rows, inputs)):
            max_length = int(input_ids.shape[1]) + max_new_tokens + total_token + 32
            with torch.inference_mode():
                torch.cuda.synchronize()
                start = time.perf_counter()
                eagle_ids, _, _, eagle_elapsed, acceptance = model.eagenerate(
                    input_ids,
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    log=True,
                    return_stats=True,
                )
                eagle_elapsed = float(eagle_elapsed)
                naive_elapsed = None
                naive_ids = None
                if include_naive:
                    naive_ids, _, _, naive_elapsed = model.naivegenerate(
                        input_ids,
                        temperature=0.0,
                        max_new_tokens=max_new_tokens,
                        max_length=max_length,
                        log=True,
                        return_stats=True,
                    )
                    naive_elapsed = float(naive_elapsed)
                torch.cuda.synchronize()
            new_tokens = int(eagle_ids.shape[1] - input_ids.shape[1])
            exact_match = None
            if naive_ids is not None:
                exact_match = bool(torch.equal(eagle_ids, naive_ids))
            generation = {
                "status": "ok",
                "document_id": str(row.get("id", index)),
                "dataset": row.get("dataset"),
                "new_tokens": new_tokens,
                "acceptance_lengths": [int(value) for value in acceptance],
                "eagle_time_s": eagle_elapsed,
                "eagle_tokens_per_s": new_tokens / eagle_elapsed if eagle_elapsed > 0 else None,
                "naive_time_s": naive_elapsed,
                "naive_tokens_per_s": (
                    new_tokens / naive_elapsed
                    if naive_elapsed is not None and naive_elapsed > 0 else None
                ),
                "speedup": (
                    naive_elapsed / eagle_elapsed
                    if naive_elapsed is not None and eagle_elapsed > 0 else None
                ),
                "exact_match_to_naive": exact_match,
                "input_tokens": int(input_ids.shape[1]),
            }
            generations.append(generation)
            handle.write(json.dumps(generation, ensure_ascii=False) + "\n")
            print(
                f"[{index + 1}/{len(rows)}] doc={generation['document_id']} "
                f"tokens={new_tokens} rounds={len(acceptance)} "
                f"eagle={eagle_elapsed:.3f}s"
            )
    analysis = analyze_acceptance_rows(generations)
    e2e = summarize_e2e(generations)
    result = {
        "schema_version": "groundsync.p1p2.eagle3.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
        },
        "models": {"base_model": str(base_model), "eagle_model": str(eagle_model)},
        "protocol": {
            "max_samples": max_samples,
            "max_new_tokens": max_new_tokens,
            "max_input_tokens": max_input_tokens,
            "total_token": total_token,
            "depth": depth,
            "top_k": top_k,
            "include_naive": include_naive,
            "timing_basis": "decode-only direct EAGLE model timing; prefill excluded by EaModel",
        },
        "coverage": {"requested": max_samples, "completed": len(generations), "raw_file": str(raw_path)},
        "strong_drafter_analysis": analysis,
        "e2e": e2e,
    }
    (output_dir / "strong_drafter_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "strong_drafter_report.md").write_text(render_report(result), encoding="utf-8")
    return result


def summarize_e2e(generations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in generations if row.get("status") == "ok"]
    with_naive = [row for row in valid if row.get("naive_time_s") is not None]
    exact_match_rate = (
        statistics.fmean(int(bool(row.get("exact_match_to_naive"))) for row in with_naive)
        if with_naive else None
    )
    aggregate_speedup = (
        sum(float(row["naive_time_s"]) for row in with_naive) / sum(float(row["eagle_time_s"]) for row in with_naive)
        if with_naive else None
    )
    return {
        "status": "ok" if with_naive else "UNAVAILABLE",
        "generation_count": len(valid),
        "paired_count": len(with_naive),
        "mean_eagle_tokens_per_s": statistics.fmean(row["eagle_tokens_per_s"] for row in with_naive) if with_naive else None,
        "aggregate_eagle_tokens_per_s": (
            sum(int(row["new_tokens"]) for row in with_naive) / sum(float(row["eagle_time_s"]) for row in with_naive)
            if with_naive else None
        ),
        "mean_naive_tokens_per_s": statistics.fmean(row["naive_tokens_per_s"] for row in with_naive) if with_naive else None,
        "aggregate_naive_tokens_per_s": (
            sum(int(row["new_tokens"]) for row in with_naive) / sum(float(row["naive_time_s"]) for row in with_naive)
            if with_naive else None
        ),
        "aggregate_speedup": aggregate_speedup,
        "exact_match_rate": exact_match_rate,
        "decision_gate": "paired timing complete, exact_match_rate == 1.0, and speedup > 1.0",
        "decision": (
            "PASS" if with_naive and exact_match_rate == 1.0 and aggregate_speedup is not None and aggregate_speedup > 1.0
            else "FAIL" if with_naive else "UNAVAILABLE"
        ),
    }


def render_report(result: Mapping[str, Any]) -> str:
    strong = result.get("strong_drafter_analysis", {})
    coverage = strong.get("coverage", {})
    within = strong.get("within_block_burstiness", {})
    persistence = strong.get("across_round_persistence", {})
    e2e = result.get("e2e", {})
    lines = [
        "# P1 strong-drafter / P2 direct E2E result",
        "",
        "Strong drafter là EAGLE-3 Qwen3-4B head cục bộ ghép với canonical Qwen3-4B.",
        "Acceptance length của EAGLE gồm token fallback của target; phân tích trừ",
        "một token fallback để lấy số draft token được chấp nhận.",
        "",
        "## Coverage và burstiness",
        "",
        f"- Generation: {coverage.get('generation_count', 0)}, rounds: {coverage.get('round_count', 0)}, documents: {coverage.get('document_count', 0)}.",
        f"- Admission rate: {coverage.get('admission_rate')!s}.",
        f"- Within-block h1/later ratio: {within.get('h1_to_later_hazard_ratio')!s}; decision: **{strong.get('decision', 'UNAVAILABLE')}**.",
        "- Persistence được bootstrap theo document; chỉ coi là có persistence nếu CI 95% thấp hơn 0.",
        "",
        "## P2 direct E2E",
        "",
        f"- Paired generations: {e2e.get('paired_count', 0)}; aggregate speedup: {e2e.get('aggregate_speedup')!s}x; decision: **{e2e.get('decision', 'UNAVAILABLE')}**.",
        f"- Exact match EAGLE/naive: {e2e.get('exact_match_rate')!s}.",
        "- Timing là decode-only direct model timing; đây chưa phải benchmark server/API vLLM.",
        "",
        "## Quyết định",
        "",
        "- Không dùng kết quả này để thay thế controlled P0: EAGLE acceptance round không có target attention transition tương ứng.",
        "- Strong-drafter replication chỉ đạt gate khi within-block asymmetry và persistence cùng có bằng chứng trên tập đã chạy.",
        "- P2 chỉ được gọi là bằng chứng E2E direct nếu paired run có output exact-match và có timing đầy đủ.",
    ]
    return "\n".join(lines) + "\n"


def refresh_existing_artifacts(output_dir: Path) -> dict[str, Any]:
    """Recompute analysis fields after an analysis-only code correction."""

    metrics_path = output_dir / "strong_drafter_metrics.json"
    raw_path = output_dir / "strong_drafter_generations.jsonl"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    generations = load_dataset(raw_path)
    result["strong_drafter_analysis"] = analyze_acceptance_rows(generations)
    result["e2e"] = summarize_e2e(generations)
    result["analysis_refreshed_at"] = datetime.now(timezone.utc).isoformat()
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "strong_drafter_report.md").write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--eagle-model", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--include-naive", action="store_true")
    parser.add_argument("--refresh-existing", type=Path, default=None,
                        help="refresh metrics/report from an existing run directory")
    args = parser.parse_args()
    if args.refresh_existing is not None:
        result = refresh_existing_artifacts(args.refresh_existing)
        print(json.dumps({"strong": result["strong_drafter_analysis"]["decision"], "e2e": result["e2e"]}, indent=2))
        return
    missing = [name for name in ("base_model", "eagle_model", "dataset", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    result = run_eagle(
        base_model=args.base_model,
        eagle_model=args.eagle_model,
        dataset=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        include_naive=args.include_naive,
    )
    print(json.dumps({"strong": result["strong_drafter_analysis"]["decision"], "e2e": result["e2e"]}, indent=2))


if __name__ == "__main__":
    main()
