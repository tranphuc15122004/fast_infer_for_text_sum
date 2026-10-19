#!/usr/bin/env python3
"""Run DFlash Phase A vs Dense DFlash Pilot Benchmark on Modal NVIDIA B200.

Compares:
  1. vanilla (Autoregressive Reference, block_size=1)
  2. dflash_dense (Standard DFlash, block_size=16, dense KV)
  3. dflash_phase_a (DFlash + Phase A Context Focusing, B=2048, L18, M1, chunk=128)

Usage:
  MODAL_PROFILE=phuclk30 modal run --profile phuclk30 scripts/modal_phase_a_infer_pilot.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
VOLUME_MOUNT = Path("/mnt/fast-infer")
VOLUME_NAME = os.environ.get("MODAL_VOLUME", "fast-infer-text-sum-cache")
GPU = os.environ.get("MODAL_GPU", "B200")

volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.12.1",
        "accelerate==1.14.0",
        "numpy==2.2.6",
        "einops==0.8.2",
        "huggingface_hub==1.21.0",
        "rouge_score==0.1.2",
    )
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), ignore=["__pycache__", "*.pyc"])
    .add_local_dir(ROOT / "externals/dflash", str(REMOTE_ROOT / "externals/dflash"), ignore=["__pycache__", "*.pyc", ".git", "cache"])
    .add_local_dir(ROOT / "scripts/common", str(REMOTE_ROOT / "scripts/common"), ignore=["__pycache__", "*.pyc"])
    .add_local_file(ROOT / "data/longbench_200/gov_report.jsonl", "/bootstrap/gov_report.jsonl")
)

app = modal.App("dflash-phase-a-b200-pilot")


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(VOLUME_MOUNT): volume},
    timeout=3600,
    cpu=8,
    memory=65536,
)
def run_pilot_b200(
    num_samples: int = 2,
    max_new_tokens: int = 128,
    max_input_tokens: int = 10000,
    phase_a_budget: int = 2048,
    phase_a_chunk_size: int = 128,
    phase_a_layer: int = 18,
) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download

    # Register paths
    sys.path.insert(0, str(REMOTE_ROOT))
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    sys.path.insert(0, str(REMOTE_ROOT / "externals/dflash"))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))

    from dflash.model import DFlashDraftModel, dflash_generate
    from Benchmark.common.phase_a_selector import PhaseAConfig, PhaseAContextSelector

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"[modal-b200] Khởi động trên GPU: {gpu_name} ({total_mem:.1f} GiB VRAM)", flush=True)

    # 1. Tải Snapshot từ Volume cache
    os.environ["HF_HOME"] = str(VOLUME_MOUNT / "hf")
    os.environ["HF_HUB_CACHE"] = str(VOLUME_MOUNT / "hf/hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("[modal-b200] Nạp checkpoints...", flush=True)
    target_dir = snapshot_download("Qwen/Qwen3-4B", cache_dir=str(VOLUME_MOUNT / "hf/hub"), local_files_only=True)
    draft_dir = snapshot_download("z-lab/Qwen3-4B-DFlash-b16", cache_dir=str(VOLUME_MOUNT / "hf/hub"), local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(target_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(
        target_dir,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()

    draft = DFlashDraftModel.from_pretrained(
        draft_dir,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()

    torch.cuda.synchronize(device)
    print("[modal-b200] Models nạp thành công!", flush=True)

    # 2. Đọc dữ liệu mẫu từ GovReport
    samples = []
    with open("/bootstrap/gov_report.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_samples:
                break
            samples.append(json.loads(line))

    results = []

    for s_idx, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{s_idx}")
        raw_text = sample.get("context", "")
        # Format prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes long reports."},
            {"role": "user", "content": f"Summarize the following report:\n\n{raw_text}\n\nSummary:"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        if max_input_tokens > 0 and tokens.shape[1] > max_input_tokens:
            tokens = tokens[:, :max_input_tokens]
        input_len = tokens.shape[1]
        print(f"\n=======================================================", flush=True)
        print(f"[modal-b200] Mẫu {s_idx + 1}/{len(samples)}: id={sample_id}, input_tokens={input_len}", flush=True)
        print(f"=======================================================", flush=True)

        sample_runs = {}

        # Danh sách phương pháp chạy kiểm chứng
        methods_to_run = [
            ("vanilla", 1, False),
            ("dflash_dense", 16, False),
            ("dflash_phase_a", 16, True),
        ]

        for method_name, block_size, use_phase_a in methods_to_run:
            print(f"--> Đang chạy phương pháp: {method_name} (block_size={block_size}, phase_a={use_phase_a})...", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            selector = None
            if use_phase_a:
                cfg = PhaseAConfig(
                    target_layer=phase_a_layer,
                    budget=phase_a_budget,
                    chunk_size=phase_a_chunk_size,
                    sink_tokens=16,
                    recent_tokens=32,
                    query_pooling="max",
                )
                selector = PhaseAContextSelector(target, cfg)

            torch.cuda.synchronize(device)
            t_start = time.perf_counter()

            # Warmup run nhẹ 8 tokens để ổn định kernel CUDA
            if s_idx == 0:
                _ = dflash_generate(
                    draft,
                    target=target,
                    input_ids=tokens[:, :128],
                    max_new_tokens=8,
                    stop_token_ids=[-1],
                    temperature=0.0,
                    block_size=block_size,
                    return_stats=False,
                    phase_a_selector=selector,
                )
                torch.cuda.synchronize(device)

            t_start = time.perf_counter()
            eos_id = target.config.eos_token_id
            stop_ids = eos_id if isinstance(eos_id, list) else [eos_id]

            res = dflash_generate(
                draft,
                target=target,
                input_ids=tokens,
                max_new_tokens=max_new_tokens,
                stop_token_ids=[-1],  # Đo đủ max_new_tokens để tính TPOT công bằng
                temperature=0.0,
                block_size=block_size,
                return_stats=True,
                phase_a_selector=selector,
            )
            torch.cuda.synchronize(device)
            elapsed_sec = time.perf_counter() - t_start

            if selector is not None:
                selector.close()

            out_ids = res.output_ids[0, input_len:]
            gen_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            n_tokens = len(out_ids)

            decode_time_sec = res.time_per_output_token * n_tokens
            prefill_time_sec = res.time_to_first_token
            tpot_ms = res.time_per_output_token * 1000.0
            throughput = n_tokens / decode_time_sec if decode_time_sec > 0 else 0.0

            acceptance_rate = getattr(res, "acceptance_rate", None)
            avg_accept = getattr(res, "avg_accept_length", None)
            draft_lat = getattr(res, "draft_latency_ms", None)
            verify_lat = getattr(res, "verification_latency_ms", None)

            sample_runs[method_name] = {
                "method": method_name,
                "input_tokens": input_len,
                "output_tokens": n_tokens,
                "e2e_sec": round(elapsed_sec, 4),
                "prefill_ms": round(prefill_time_sec * 1000.0, 2),
                "decode_ms": round(decode_time_sec * 1000.0, 2),
                "tpot_ms": round(tpot_ms, 2),
                "decode_throughput_tok_s": round(throughput, 2),
                "acceptance_rate": acceptance_rate,
                "avg_accept_length": avg_accept,
                "draft_latency_ms": draft_lat,
                "verification_latency_ms": verify_lat,
                "generated_text_preview": gen_text[:120].replace("\n", " "),
                "token_ids": out_ids.tolist()[:16],
            }
            print(
                f"    [Hoàn tất] TPOT: {tpot_ms:.2f} ms/tok | Throughput: {throughput:.2f} tok/s | "
                f"Alpha: {acceptance_rate} | AvgAccept: {avg_accept} | E2E: {elapsed_sec:.2f}s",
                flush=True,
            )

        # Tính toán speedup ratios
        vanilla = sample_runs["vanilla"]
        dense = sample_runs["dflash_dense"]
        phase_a = sample_runs["dflash_phase_a"]

        vanilla_tpot = vanilla["tpot_ms"]
        dense_tpot = dense["tpot_ms"]
        phase_a_tpot = phase_a["tpot_ms"]

        dense["dsr_vs_vanilla"] = round(vanilla_tpot / dense_tpot, 3)
        dense["esr_vs_vanilla"] = round(vanilla["e2e_sec"] / dense["e2e_sec"], 3)

        phase_a["dsr_vs_vanilla"] = round(vanilla_tpot / phase_a_tpot, 3)
        phase_a["esr_vs_vanilla"] = round(vanilla["e2e_sec"] / phase_a["e2e_sec"], 3)
        phase_a["speedup_vs_dense"] = round(dense_tpot / phase_a_tpot, 3)
        phase_a["latency_reduction_pct"] = round((1.0 - phase_a_tpot / dense_tpot) * 100.0, 2)

        results.append({
            "sample_id": sample_id,
            "input_tokens": input_len,
            "runs": sample_runs,
        })

    return {
        "gpu_name": gpu_name,
        "total_memory_gb": round(total_mem, 1),
        "phase_a_config": {
            "budget": phase_a_budget,
            "chunk_size": phase_a_chunk_size,
            "layer": phase_a_layer,
        },
        "samples": results,
    }


@app.local_entrypoint()
def main(
    num_samples: int = 2,
    max_new_tokens: int = 128,
    max_input_tokens: int = 10000,
    phase_a_budget: int = 2048,
    output_path: str = "outputs/modal_pilot_b200/pilot_result.json",
):
    print(f"[local] Gửi job Pilot kiểm chứng DFlash Phase A lên Modal GPU (profile phuclk30, GPU={GPU})...")
    res = run_pilot_b200.remote(
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        phase_a_budget=phase_a_budget,
    )

    out_file = ROOT / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n[local] ================= KẾT QUẢ PILOT TRÊN {res['gpu_name']} =================")
    for sample in res["samples"]:
        s_id = sample["sample_id"]
        in_tok = sample["input_tokens"]
        runs = sample["runs"]
        v = runs["vanilla"]
        d = runs["dflash_dense"]
        pa = runs["dflash_phase_a"]

        print(f"\n--- Mẫu: {s_id} (Context: {in_tok} tokens, Sinh: {v['output_tokens']} tokens) ---")
        print(f"1. Vanilla HF:       TPOT = {v['tpot_ms']:6.2f} ms/tok | Throughput = {v['decode_throughput_tok_s']:5.1f} tok/s | E2E = {v['e2e_sec']:5.2f}s")
        print(f"2. DFlash Dense:     TPOT = {d['tpot_ms']:6.2f} ms/tok | Throughput = {d['decode_throughput_tok_s']:5.1f} tok/s | DSR = {d['dsr_vs_vanilla']:.2f}x | Alpha = {d['acceptance_rate']}")
        print(f"3. DFlash Phase A:   TPOT = {pa['tpot_ms']:6.2f} ms/tok | Throughput = {pa['decode_throughput_tok_s']:5.1f} tok/s | DSR = {pa['dsr_vs_vanilla']:.2f}x | Alpha = {pa['acceptance_rate']}")
        print(f"==> TỐC ĐỘ CẢI THIỆN: Phase A nhanh hơn Dense DFlash: {pa['speedup_vs_dense']:.2f}x (Giảm {pa['latency_reduction_pct']}% thời gian decode)")

    print(f"\n[local] Chi tiết đã được lưu tại: {out_file}")


if __name__ == "__main__":
    main()
