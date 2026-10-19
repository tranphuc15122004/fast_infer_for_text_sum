#!/usr/bin/env python3
"""
Khảo sát phân bố Attention của Qwen 3 4B trên tất cả các layer trên Modal GPU A100.
Đo attention của token đầu tiên sinh ra sau giai đoạn prefill (Decode step 1)
trên sample dài (8k và 16k tokens).
Vẽ histogram attention cho từng layer và báo cáo chuyên sâu 5 layer: 1, 9, 17, 25, 33.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone

import modal

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/qwen3_attention"

app = modal.App("qwen3-attention-survey")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.51.0",
        "accelerate",
        "safetensors",
        "matplotlib",
        "numpy",
        "scipy",
    )
)

@app.function(image=image, gpu="A100", timeout=1200)
def run_attention_survey(samples_data: list[dict]):
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = "cuda:0"
    model_name = "Qwen/Qwen3-4B"
    print(f"[modal] Bắt đầu khảo sát Attention model {model_name}...")
    print(f"[modal] Thiết bị GPU: {torch.cuda.get_device_name(0)}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    num_layers = len(model.model.layers)
    num_heads = model.config.num_attention_heads
    print(f"[modal] Model loaded: {num_layers} layers, {num_heads} heads.")
    
    results = {}
    plot_files = {}

    target_layers_1based = [1, 9, 17, 25, 33]
    target_layers_0based = [l - 1 for l in target_layers_1based]  # [0, 8, 16, 24, 32]
    
    for sample in samples_data:
        target_len = sample["target_tokens"]
        doc_text = sample["text"]
        sample_id = sample.get("id", f"sample_{target_len}")
        print(f"\n=======================================================")
        print(f"[modal] Xử lý sample: {sample_id} (mục tiêu ~{target_len} tokens)")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Document:\n{doc_text}\n\nPlease summarize the above document in detail."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        
        # Cắt ngắn hoặc căn chỉnh về đúng target_len tokens nếu cần
        if tokens.shape[1] > target_len:
            tokens = tokens[:, :target_len]
        actual_seq_len = tokens.shape[1]
        print(f"[modal] Actual prompt sequence length: {actual_seq_len} tokens")
        
        # 1. Giai đoạn PREFILL bằng SDPA (FlashAttention native - không OOM)
        model.config._attn_implementation = "sdpa"
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            prefill_out = model(tokens, use_cache=True, output_attentions=False)
        peak_prefill_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[modal] Prefill hoàn thành. Peak VRAM: {peak_prefill_gb:.2f} GB")
        
        # Xác định token đầu tiên sinh ra sau prefill
        next_token_logits = prefill_out.logits[:, -1, :]
        first_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        first_token_str = tokenizer.decode(first_token_id[0])
        print(f"[modal] Token đầu tiên sinh ra: ID={first_token_id.item()} | Repr={repr(first_token_str)}")
        
        # 2. Giai đoạn DECODE token đầu tiên bằng Eager Attention để trích xuất Attention matrix
        model.config._attn_implementation = "eager"
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            decode_out = model(
                first_token_id,
                past_key_values=prefill_out.past_key_values,
                use_cache=True,
                output_attentions=True,
            )
        peak_decode_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[modal] Decode step 1 hoàn thành. Peak VRAM: {peak_decode_gb:.2f} GB")
        
        attentions = decode_out.attentions  # Tuple of 36 tensors: shape (1, num_heads, 1, actual_seq_len + 1)
        
        sample_metrics = {
            "sample_id": sample_id,
            "target_len": target_len,
            "actual_seq_len": actual_seq_len,
            "first_token_id": int(first_token_id.item()),
            "first_token_str": first_token_str,
            "peak_prefill_gb": float(peak_prefill_gb),
            "peak_decode_gb": float(peak_decode_gb),
            "layers": {}
        }
        
        # Thu thập attention ma trận cho cả 36 layers
        all_layer_means = []  # shape: (36, actual_seq_len)
        per_layer_head_means = {}  # layer -> (num_heads, actual_seq_len)
        
        for l_idx in range(num_layers):
            # Attention của token đầu tiên chiếu lên tất cả prompt tokens (bỏ token tự sinh ở cuối)
            layer_attn = attentions[l_idx][0, :, 0, :actual_seq_len].float().cpu().numpy()  # (32, actual_seq_len)
            mean_heads = layer_attn.mean(axis=0)  # (actual_seq_len,)
            all_layer_means.append(mean_heads)
            
            if l_idx in target_layers_0based:
                per_layer_head_means[l_idx] = layer_attn
            
            # Tính toán các chỉ số phân bố & sparsity
            sink_4 = float(np.sum(mean_heads[:4]))
            sink_16 = float(np.sum(mean_heads[:16]))
            local_64 = float(np.sum(mean_heads[-64:]))
            local_256 = float(np.sum(mean_heads[-256:]))
            middle = float(np.sum(mean_heads[16:-64])) if actual_seq_len > 80 else 0.0
            
            sorted_attn = np.sort(mean_heads)[::-1]
            cum_attn = np.cumsum(sorted_attn)
            
            top_1pct_idx = max(1, int(0.01 * actual_seq_len))
            top_5pct_idx = max(1, int(0.05 * actual_seq_len))
            top_10pct_idx = max(1, int(0.10 * actual_seq_len))
            top_20pct_idx = max(1, int(0.20 * actual_seq_len))
            
            # Entropy & Perplexity
            eps = 1e-12
            probs = np.clip(mean_heads, eps, 1.0)
            entropy = float(-np.sum(probs * np.log(probs)))
            effective_tokens = float(np.exp(entropy))
            entropy_norm = float(entropy / np.log(actual_seq_len))
            
            # Gini Coefficient
            sorted_asc = np.sort(mean_heads)
            n = actual_seq_len
            index = np.arange(1, n + 1)
            gini = float((2 * np.sum(index * sorted_asc)) / (n * np.sum(sorted_asc)) - (n + 1) / n)
            
            sample_metrics["layers"][l_idx] = {
                "layer_1based": l_idx + 1,
                "layer_0based": l_idx,
                "sink_mass_4": sink_4,
                "sink_mass_16": sink_16,
                "local_mass_64": local_64,
                "local_mass_256": local_256,
                "middle_mass": middle,
                "top_1pct_mass": float(cum_attn[top_1pct_idx - 1]),
                "top_5pct_mass": float(cum_attn[top_5pct_idx - 1]),
                "top_10pct_mass": float(cum_attn[top_10pct_idx - 1]),
                "top_20pct_mass": float(cum_attn[top_20pct_idx - 1]),
                "effective_tokens": effective_tokens,
                "entropy_norm": entropy_norm,
                "gini": gini,
            }
            
        all_layer_means = np.array(all_layer_means)  # (36, actual_seq_len)
        results[sample_id] = sample_metrics
        
        # =========================================================================
        # VẼ ĐỒ THỊ
        # =========================================================================
        print(f"[modal] Đang tạo hình vẽ cho {sample_id}...")
        
        # 1. Heatmap toàn bộ 36 Layer vs Context Position (Downsampled into 100 bins)
        num_bins = 100
        bin_size = actual_seq_len // num_bins
        binned_matrix = np.zeros((num_layers, num_bins))
        for b in range(num_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < num_bins - 1 else actual_seq_len
            binned_matrix[:, b] = all_layer_means[:, start:end].sum(axis=1)
            
        fig, ax = plt.subplots(figsize=(12, 7))
        cax = ax.imshow(binned_matrix, aspect="auto", cmap="viridis", origin="lower",
                        extent=[0, 100, 1, num_layers])
        ax.set_title(f"Qwen 3 4B - Attention Distribution Across All 36 Layers ({target_len//1024}k tokens)", fontsize=14, pad=12)
        ax.set_xlabel("Context Position (%)", fontsize=12)
        ax.set_ylabel("Layer (1 to 36)", fontsize=12)
        cbar = fig.colorbar(cax, ax=ax, label="Attention Mass in 1% Context Window")
        
        # Đánh dấu các target layers
        for tl in target_layers_1based:
            ax.axhline(tl, color="red", linestyle="--", alpha=0.6, linewidth=1.2)
            ax.text(101, tl, f"L{tl}", color="red", verticalalignment="center", fontsize=9, fontweight="bold")
            
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        plt.close()
        plot_files[f"heatmap_all_layers_{target_len//1024}k.png"] = buf.getvalue()

        # 2. Báo cáo 5 Layer trọng điểm: 1, 9, 17, 25, 33
        # Đồ thị A: Histogram phân bố Positional Attention Mass (Binned theo vị trí context)
        fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        
        num_pos_bins = 80
        bin_edges = np.linspace(0, actual_seq_len, num_pos_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        for idx, (tl1, color) in enumerate(zip(target_layers_1based, colors)):
            tl0 = tl1 - 1
            attn_vec = all_layer_means[tl0]
            
            # Tính mass cho từng bin
            binned_mass = np.zeros(num_pos_bins)
            for b in range(num_pos_bins):
                s = int(bin_edges[b])
                e = int(bin_edges[b + 1])
                binned_mass[b] = np.sum(attn_vec[s:e])
                
            ax = axes[idx]
            ax.bar(bin_centers, binned_mass, width=(actual_seq_len / num_pos_bins) * 0.85,
                   color=color, alpha=0.85, edgecolor="none",
                   label=f"Layer {tl1} (idx {tl0})")
            
            sink_val = sample_metrics["layers"][tl0]["sink_mass_4"]
            local_val = sample_metrics["layers"][tl0]["local_mass_64"]
            mid_val = sample_metrics["layers"][tl0]["middle_mass"]
            gini_val = sample_metrics["layers"][tl0]["gini"]
            
            title_str = (f"Layer {tl1} (Index {tl0}) | "
                         f"Sink(0..3): {sink_val*100:.1f}% | "
                         f"Middle: {mid_val*100:.1f}% | "
                         f"Local(last 64): {local_val*100:.1f}% | "
                         f"Gini: {gini_val:.3f}")
            ax.set_title(title_str, fontsize=11, fontweight="bold", loc="left")
            ax.set_ylabel("Attention Mass", fontsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9)
            
        axes[-1].set_xlabel(f"Token Position in Context (0 to {actual_seq_len})", fontsize=12)
        fig.suptitle(f"Qwen 3 4B - Positional Attention Histogram for Target Layers (1, 9, 17, 25, 33) [{target_len//1024}k tokens]",
                     fontsize=14, y=0.995)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        plt.close()
        plot_files[f"positional_histogram_5layers_{target_len//1024}k.png"] = buf.getvalue()

        # Đồ thị B: Histogram phân bố độ lớn trọng số Attention (Log-Weight Distribution)
        # Thể hiện mức độ tập trung / thưa (sparsity) của attention
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=True)
        log_bins = np.linspace(-8, 0, 50)  # Từ 10^-8 đến 10^0 = 1.0
        
        for idx, (tl1, color) in enumerate(zip(target_layers_1based, colors)):
            tl0 = tl1 - 1
            attn_vec = all_layer_means[tl0]
            log_weights = np.log10(np.clip(attn_vec, 1e-10, 1.0))
            
            ax = axes[idx]
            ax.hist(log_weights, bins=log_bins, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
            ax.set_title(f"Layer {tl1} (idx {tl0})", fontsize=11, fontweight="bold")
            ax.set_xlabel("log10(Attention Weight)", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Number of Tokens", fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.6)
            
            # Thêm đường tham chiếu uniform attention = 1 / actual_seq_len
            uniform_log = np.log10(1.0 / actual_seq_len)
            ax.axvline(uniform_log, color="black", linestyle="--", label=f"Uniform ({uniform_log:.1f})")
            ax.legend(loc="upper right", fontsize=8)
            
        fig.suptitle(f"Qwen 3 4B - Attention Weight Value Distribution Histogram [{target_len//1024}k tokens]",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        plt.close()
        plot_files[f"weight_magnitude_histogram_5layers_{target_len//1024}k.png"] = buf.getvalue()

        # Đồ thị C: Biểu đồ xu hướng Sink vs Middle vs Local trên toàn bộ 36 layer
        fig, ax = plt.subplots(figsize=(12, 5.5))
        layer_indices = np.arange(1, num_layers + 1)
        sink_masses = [sample_metrics["layers"][l]["sink_mass_4"] for l in range(num_layers)]
        mid_masses = [sample_metrics["layers"][l]["middle_mass"] for l in range(num_layers)]
        local_masses = [sample_metrics["layers"][l]["local_mass_64"] for l in range(num_layers)]
        
        ax.plot(layer_indices, sink_masses, marker="o", linewidth=2, color="#d62728", label="Attention Sink (First 4 tokens)")
        ax.plot(layer_indices, mid_masses, marker="s", linewidth=2, color="#1f77b4", label="Middle Context (Tokens 16 to L-64)")
        ax.plot(layer_indices, local_masses, marker="^", linewidth=2, color="#2ca02c", label="Local Context (Last 64 tokens)")
        
        # Đánh dấu 5 layer được yêu cầu
        for tl in target_layers_1based:
            ax.axvline(tl, color="gray", linestyle=":", alpha=0.5)
            ax.text(tl, 0.95, f"L{tl}", ha="center", fontsize=9, fontweight="bold", color="darkred")
            
        ax.set_title(f"Qwen 3 4B - Context Allocation Trend Across 36 Layers ({target_len//1024}k tokens)", fontsize=13)
        ax.set_xlabel("Layer Index (1 to 36)", fontsize=11)
        ax.set_ylabel("Attention Mass Proportion (0.0 to 1.0)", fontsize=11)
        ax.set_xticks(range(1, num_layers + 1, 2))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="center right", framealpha=0.9)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        plt.close()
        plot_files[f"context_allocation_trend_36layers_{target_len//1024}k.png"] = buf.getvalue()

        # Đồ thị D: Phân tích 32 Head của 5 Layer mục tiêu (Head Specialization Heatmap)
        fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey=True)
        head_bins = 40
        h_edges = np.linspace(0, actual_seq_len, head_bins + 1)
        
        for idx, tl1 in enumerate(target_layers_1based):
            tl0 = tl1 - 1
            heads_mat = per_layer_head_means[tl0]  # (32, actual_seq_len)
            binned_heads = np.zeros((32, head_bins))
            for b in range(head_bins):
                s = int(h_edges[b])
                e = int(h_edges[b + 1])
                binned_heads[:, b] = heads_mat[:, s:e].sum(axis=1)
                
            ax = axes[idx]
            c = ax.imshow(binned_heads, aspect="auto", cmap="plasma", origin="lower",
                          extent=[0, 100, 1, 32])
            ax.set_title(f"Layer {tl1} (idx {tl0})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Context %", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Attention Head (1 to 32)", fontsize=11)
            fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            
        fig.suptitle(f"Qwen 3 4B - Per-Head Attention Distribution Across 32 Heads ({target_len//1024}k tokens)",
                     fontsize=13, y=0.98)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        plt.close()
        plot_files[f"head_specialization_5layers_{target_len//1024}k.png"] = buf.getvalue()

    # Đóng gói kết quả thành archive tar.gz
    archive_buf = io.BytesIO()
    with tarfile.open(fileobj=archive_buf, mode="w:gz") as tar:
        # JSON data
        json_bytes = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
        tarinfo = tarfile.TarInfo(name="attention_telemetry.json")
        tarinfo.size = len(json_bytes)
        tar.addfile(tarinfo, io.BytesIO(json_bytes))
        
        # Plot PNGs
        for fname, pdata in plot_files.items():
            tarinfo = tarfile.TarInfo(name=fname)
            tarinfo.size = len(pdata)
            tar.addfile(tarinfo, io.BytesIO(pdata))
            
    print("[modal] Toàn bộ thí nghiệm và tạo biểu đồ hoàn tất thành công.")
    return {
        "results": results,
        "archive": archive_buf.getvalue(),
        "filenames": list(plot_files.keys()) + ["attention_telemetry.json"]
    }


def main():
    print(f"Đọc dữ liệu mẫu từ repo cho 8k và 16k...")
    
    # 8k sample từ bin4_08k_12k.jsonl
    sample_8k_text = ""
    with open(ROOT / "data/length_bins/bin4_08k_12k.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("id") == "gov_report_9392a918215bb41b":
                sample_8k_text = d["context"]
                break
                
    # 16k sample từ bin5_12k_16k.jsonl
    sample_16k_text = ""
    with open(ROOT / "data/length_bins/bin5_12k_16k.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("dataset") == "gov_report" and d.get("input_tokens", 0) >= 14000:
                sample_16k_text = d["context"]
                break
    if not sample_16k_text:
        # Fallback to the longest in bin5
        with open(ROOT / "data/length_bins/bin5_12k_16k.jsonl", "r", encoding="utf-8") as f:
            sample_16k_text = json.loads(f.readline())["context"]

    samples_payload = [
        {"id": "gov_report_8k", "target_tokens": 8192, "text": sample_8k_text},
        {"id": "gov_report_16k", "target_tokens": 16384, "text": sample_16k_text},
    ]
    
    print(f"Chuẩn bị gọi Modal App...")
    with app.run():
        output = run_attention_survey.remote(samples_payload)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(output["archive"]), mode="r:gz") as tar:
        tar.extractall(OUTPUT_DIR)
        
    print(f"\n[local] Đã trích xuất {len(output['filenames'])} tệp vào: {OUTPUT_DIR}")
    for fn in output["filenames"]:
        print(f"  - {OUTPUT_DIR / fn}")


if __name__ == "__main__":
    main()
