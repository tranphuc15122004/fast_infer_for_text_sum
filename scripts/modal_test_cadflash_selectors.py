#!/usr/bin/env python3
"""Benchmark and compare proposed context selectors on Modal GPU (profile: phuclk30).

Evaluates:
1. Dense DFlash Baseline (Full Context)
2. M0 Baseline: Single Parent Query (Layer 18, Budget=2048)
3. M1 Proposed: Multi-Query Max-Pooling (Layer 18, Budget=2048)
4. M4 Proposed: Duo-Layer Ensemble (Layer 18 + Layer 35, Budget=2048, alpha=0.5)

Target: Qwen/Qwen3-4B (BF16)
Draft:  z-lab/Qwen3-4B-DFlash-b16 (BF16)
Dataset: Real GovReport document (~8k tokens context)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
MOUNT = Path("/mnt/fast-infer")
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "fast-infer-text-sum-cache")
GPU = os.environ.get("MODAL_GPU", "L40S")
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "numpy")
    .add_local_dir(ROOT / "src/TrainingFree", str(REMOTE_ROOT / "src/TrainingFree"), copy=True)
    .add_local_dir(ROOT / "externals/dflash", str(REMOTE_ROOT / "externals/dflash"), copy=True)
    .add_local_dir(ROOT / "data/longbench_100_14k", str(REMOTE_ROOT / "data/longbench_100_14k"), copy=True)
)

app = modal.App("fast-infer-cadflash-selection-test")


@app.function(
    image=IMAGE,
    gpu=GPU,
    volumes={str(MOUNT): VOLUME},
    timeout=1800,
    cpu=8,
    memory=32768,
)
def run_benchmark_on_modal(
    context_cap: int = 8192,
    max_new_tokens: int = 64,
    budget: int = 2048,
    sample_index: int = 0,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "externals/dflash"))

    from TrainingFree.context_adaptive.config import AdaptiveConfig
    from TrainingFree.context_adaptive.generation import generate_adaptive
    from TrainingFree.context_adaptive.prompt import prepare_prompt

    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[modal] GPU: {device_name} ({vram_gb:.1f} GB VRAM)", flush=True)

    # 1. Load Target and Draft models
    target_id = "Qwen/Qwen3-4B"
    draft_id = "z-lab/Qwen3-4B-DFlash-b16"
    print(f"[modal] Loading tokenizer from {target_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(target_id)

    print(f"[modal] Loading target model {target_id} in BF16...", flush=True)
    t0 = time.perf_counter()
    target = AutoModelForCausalLM.from_pretrained(
        target_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    print(f"[modal] Target loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    from dflash.model import DFlashDraftModel

    print(f"[modal] Loading draft model {draft_id} in BF16...", flush=True)
    t0 = time.perf_counter()
    draft = DFlashDraftModel.from_pretrained(
        draft_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    print(f"[modal] Draft loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Commit volume to keep model weights cached
    try:
        VOLUME.commit()
    except Exception as exc:
        print(f"[modal] Volume commit note: {exc}", flush=True)

    # 2. Load GovReport long-context sample
    data_path = REMOTE_ROOT / "data/longbench_100_14k/gov_report.jsonl"
    raw_sample = None
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == sample_index:
                raw_sample = json.loads(line)
                break
    if raw_sample is None:
        raise ValueError(f"sample_index không tồn tại: {sample_index}")

    context_text = raw_sample.get("context", "")
    sample = {
        "prompt": context_text,
        "raw": {
            "context": context_text,
            "dataset": "gov_report",
            "source_group_id": f"gov_report_{sample_index}",
        },
    }

    print(f"[modal] Preparing prompt (target context_cap={context_cap})...", flush=True)
    input_ids, layout = prepare_prompt(
        sample,
        tokenizer,
        chunk_size=128,
        max_tokens=context_cap,
    )
    actual_prompt_tokens = input_ids.shape[1]
    print(f"[modal] Prepared prompt tokens: {actual_prompt_tokens}, source chunks: {len(layout.source_chunks)}", flush=True)

    # 3. Warmup pass (8 tokens)
    print("[modal] Running warmup pass...", flush=True)
    warmup_cfg = AdaptiveConfig(budgets=(budget, "full"), gammas=(3, 7, 11, 15), selector="target_parent")
    _ = generate_adaptive(
        target,
        draft,
        input_ids,
        tokenizer,
        layout,
        warmup_cfg,
        max_new_tokens=8,
        fixed_budget="full",
        fixed_gamma=15,
        gamma_reference=15,
        variant="dflash_full_fixed",
    )
    torch.cuda.synchronize()
    print("[modal] Warmup completed.", flush=True)

    # 4. Define evaluation configurations
    eval_configs = [
        {
            "name": "DFlash Dense Baseline",
            "code": "Dense_Full",
            "selector": "target_parent",
            "fixed_budget": "full",
            "target_layer": 18,
            "duo_deep_layer": 35,
            "ensemble_alpha": 0.5,
            "query_pooling": "max",
            "variant": "dflash_full_fixed",
        },
        {
            "name": "M0 Baseline (Single Parent Q, L18)",
            "code": "M0_Single_L18",
            "selector": "target_parent",
            "fixed_budget": budget,
            "target_layer": 18,
            "duo_deep_layer": 35,
            "ensemble_alpha": 0.5,
            "query_pooling": "max",
            "variant": "best_fixed_pair",
        },
        {
            "name": "M1 Proposed (Multi-Query Max-Pooling, L18)",
            "code": "M1_MultiQ_L18",
            "selector": "multi_query_max_pooling",
            "fixed_budget": budget,
            "target_layer": 18,
            "duo_deep_layer": 35,
            "ensemble_alpha": 0.5,
            "query_pooling": "max",
            "variant": "best_fixed_pair",
        },
        {
            "name": "M2 Proposed (Multi-Query Mean-Pooling, L18)",
            "code": "M2_Mean_L18",
            "selector": "multi_query_mean_pooling",
            "fixed_budget": budget,
            "target_layer": 18,
            "duo_deep_layer": 35,
            "ensemble_alpha": 0.5,
            "query_pooling": "mean",
            "variant": "best_fixed_pair",
        },
        {
            "name": "M4 Proposed (Duo-Ensemble L18+L35, alpha=0.5)",
            "code": "M4_Duo_L18_L35",
            "selector": "duo_ensemble",
            "fixed_budget": budget,
            "target_layer": 18,
            "duo_deep_layer": 35,
            "ensemble_alpha": 0.5,
            "query_pooling": "max",
            "variant": "best_fixed_pair",
        },
    ]

    results = []
    dense_latency_ms = None
    dense_output_ids = None

    for item in eval_configs:
        print(f"\n[modal] >>> Running variant: {item['name']} (Budget={item['fixed_budget']})...", flush=True)
        cfg = AdaptiveConfig(
            budgets=(budget, "full"),
            gammas=(3, 7, 11, 15),
            selector=item["selector"],
            target_layer=item["target_layer"],
            duo_deep_layer=item["duo_deep_layer"],
            ensemble_alpha=item["ensemble_alpha"],
            query_pooling=item["query_pooling"],
        )

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        gen_result = generate_adaptive(
            target,
            draft,
            input_ids,
            tokenizer,
            layout,
            cfg,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            fixed_budget=item["fixed_budget"],
            fixed_gamma=15,
            gamma_reference=15,
            variant=item["variant"],
            run_id=f"modal_{item['code']}",
            sample_id=f"sample_{sample_index}",
            dataset="gov_report",
        )
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start_time) * 1000.0

        output_tokens = gen_result.output_tokens
        output_ids = gen_result.output_ids[:output_tokens]
        spec_rounds = sum(1 for r in gen_result.rounds if not r.get("boundary_round", False))
        boundary_rounds = sum(1 for r in gen_result.rounds if r.get("boundary_round", False))
        total_rounds = len(gen_result.rounds)

        accepted_total = int(gen_result.counters.get("draft_tokens_accepted", gen_result.counters.get("accepted_total", 0)))
        useful_accepted = int(gen_result.counters.get("useful_draft_tokens_accepted", gen_result.counters.get("useful_accepted_total", 0)))
        proposed_total = int(gen_result.counters.get("draft_tokens_proposed", 0))

        mean_alpha = float(gen_result.counters.get("mean_accept_length") or (accepted_total / spec_rounds if spec_rounds > 0 else 0.0))
        acceptance_rate = float(gen_result.counters.get("acceptance_rate") or (accepted_total / proposed_total if proposed_total > 0 else 0.0))
        tps = output_tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0.0

        print(
            f"[modal] Details for {item['code']}: spec_rounds={spec_rounds}, boundary_rounds={boundary_rounds}, "
            f"proposed={proposed_total}, accepted={accepted_total}, useful={useful_accepted}, mean_alpha={mean_alpha:.2f}",
            flush=True,
        )

        if dense_latency_ms is None:
            dense_latency_ms = wall_ms
            dense_output_ids = output_ids
            speedup = 1.00
            token_match_pct = 100.0
        else:
            speedup = dense_latency_ms / wall_ms if wall_ms > 0 else 1.0
            # Token match with dense baseline
            min_len = min(len(output_ids), len(dense_output_ids))
            matches = sum(1 for a, b in zip(output_ids[:min_len], dense_output_ids[:min_len]) if a == b)
            token_match_pct = (matches / min_len * 100.0) if min_len > 0 else 0.0

        decoded_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        preview = decoded_text.strip().replace("\n", " ")[:120]

        record = {
            "name": item["name"],
            "code": item["code"],
            "budget": str(item["fixed_budget"]),
            "wall_ms": round(wall_ms, 2),
            "output_tokens": output_tokens,
            "tps": round(tps, 2),
            "speedup_vs_dense": round(speedup, 3),
            "speculative_rounds": spec_rounds,
            "accepted_total": accepted_total,
            "useful_accepted": useful_accepted,
            "mean_alpha": round(mean_alpha, 3),
            "mean_alpha_definition": "mean processed commits per speculative round (1+L)",
            "mean_candidates_accepted": accepted_total / spec_rounds if spec_rounds else None,
            "acceptance_rate_pct": round(acceptance_rate * 100.0, 2),
            "token_match_pct": round(token_match_pct, 1),
            "token_match_reference": "dense_dflash_common_prefix",
            "exact_match_dense": output_ids == dense_output_ids,
            "text_preview": preview,
        }
        results.append(record)
        print(
            f"[modal] Completed {item['code']}: Latency={wall_ms:.1f}ms | TPS={tps:.1f} | "
            f"SpecRounds={spec_rounds} | BoundaryRounds={boundary_rounds} | Accepted={accepted_total} | "
            f"MeanAlpha={mean_alpha:.2f} | AcceptRate={acceptance_rate*100:.1f}% | Match={token_match_pct:.1f}%",
            flush=True,
        )

    # 5. Format Summary Report
    print("\n" + "=" * 100, flush=True)
    print("                 MODAL GPU INFERENCE EXPERIMENT RESULTS (Qwen3-4B + DFlash-b16)", flush=True)
    print(f" Prompt tokens: {actual_prompt_tokens} | Output tokens cap: {max_new_tokens} | Device: {device_name}", flush=True)
    print("=" * 100, flush=True)
    header = f"{'Method':<38} | {'Budget':<8} | {'Latency':<9} | {'TPS':<8} | {'Speedup':<8} | {'Rounds':<6} | {'Accepted':<8} | {'Mean Alpha':<10} | {'Acc Rate':<8} | {'Match %':<7}"
    print(header, flush=True)
    print("-" * 100, flush=True)
    for r in results:
        row = (
            f"{r['name']:<38} | {r['budget']:<8} | {r['wall_ms']:>6.1f} ms | {r['tps']:>6.1f} | "
            f"{r['speedup_vs_dense']:>6.2f}x | {r['speculative_rounds']:>6} | {r['accepted_total']:>8} | "
            f"{r['mean_alpha']:>10.2f} | {r['acceptance_rate_pct']:>6.1f}% | {r['token_match_pct']:>6.1f}%"
        )
        print(row, flush=True)
    print("=" * 100, flush=True)

    return {
        "status": "success",
        "gpu": device_name,
        "vram_gb": round(vram_gb, 1),
        "prompt_tokens": actual_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "budget": budget,
        "evaluation_phase": "A_only_pilot",
        "pipeline_ab": False,
        "results": results,
    }


@app.local_entrypoint()
def main(
    context_cap: int = 8192,
    max_new_tokens: int = 64,
    budget: int = 2048,
    sample_index: int = 0,
):
    if os.environ.get("MODAL_PROFILE") != "phuclk30":
        raise ValueError("Cần MODAL_PROFILE=phuclk30 và modal run --profile phuclk30")
    print(f"[local] Triggering Context-Adaptive DFlash benchmark on Modal GPU (profile: phuclk30)...")
    print(f"[local] Context Cap: {context_cap} | Max New Tokens: {max_new_tokens} | Context Budget: {budget}")
    res = run_benchmark_on_modal.remote(
        context_cap=context_cap,
        max_new_tokens=max_new_tokens,
        budget=budget,
        sample_index=sample_index,
    )
    out_file = ROOT / "outputs/modal_proposed_selectors_inference_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[local] Experiment successfully finished! Full results saved to: {out_file}")


if __name__ == "__main__":
    main()
