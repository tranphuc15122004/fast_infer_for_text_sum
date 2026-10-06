#!/usr/bin/env python3
"""Đo attention DFlash và, tùy chọn, ghép với target trên cùng lượt verify."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "externals" / "dflash"))
from common.io_util import JsonlWriter


def summarize_vector(vector, prompt_length, prefix_length, *, bin_size=1024, top_k=1024):
    """Các span dùng chung denominator; coverage full tính trên mọi key."""
    vector = np.asarray(vector)
    if vector.ndim != 1 or not 0 < prompt_length <= prefix_length <= len(vector):
        raise ValueError("Span prompt/prefix không hợp lệ")
    if not np.isfinite(vector).all() or (vector < 0).any():
        raise RuntimeError("Attention không hữu hạn")
    prompt_vector = vector[:prompt_length]
    prompt_mass = float(prompt_vector.sum(dtype=np.float64))
    def bins(values, width):
        return [float(values[i:i + width].sum(dtype=np.float64))
                for i in range(0, len(values), width)]
    prompt_bins = bins(prompt_vector, bin_size)
    width = min(top_k, prompt_length)
    cumsum = np.concatenate(([0.], np.cumsum(prompt_vector, dtype=np.float64)))
    windows = cumsum[width:] - cumsum[:-width]
    best_start = int(windows.argmax())
    sorted_prompt = np.sort(prompt_vector)[::-1]
    sorted_all = np.sort(vector)[::-1]
    total_mass = float(vector.sum(dtype=np.float64))
    if abs(total_mass - 1.) > .005:
        raise RuntimeError(f"Attention tổng mass={total_mass}, không gần 1")
    cumulative_all = np.cumsum(sorted_all, dtype=np.float64)
    generated = vector[prompt_length:prefix_length]
    block = vector[prefix_length:]
    return {
        "prompt_mass": prompt_mass,
        "generated_mass": float(generated.sum(dtype=np.float64)),
        "draft_block_mass": float(block.sum(dtype=np.float64)),
        "draft_anchor_mass": float(block[0]) if len(block) else 0.,
        "draft_mask_mass": float(block[1:].sum(dtype=np.float64)),
        "mass_sum": total_mass,
        "key_tokens": len(vector),
        "prompt_bins_absolute": prompt_bins,
        "prompt_bins_normalized": [v / prompt_mass for v in prompt_bins] if prompt_mass else None,
        "generated_bins_absolute": bins(generated, 128),
        "generated_bin_size": 128,
        "all_bins_absolute": bins(vector, bin_size),
        "top_1k_all_coverage": float(sorted_all[:top_k].sum(dtype=np.float64) / total_mass),
        "tokens_for_90pct_all_mass": int(np.searchsorted(cumulative_all, .9 * total_mass)) + 1,
        "tokens_for_95pct_all_mass": int(np.searchsorted(cumulative_all, .95 * total_mass)) + 1,
        "top_1k_prompt_coverage": float(sorted_prompt[:width].sum(dtype=np.float64) / prompt_mass)
        if prompt_mass else None,
        "best_window_1k_prompt_coverage": float(windows[best_start] / prompt_mass)
        if prompt_mass else None,
        "best_window_start": best_start,
        "coverage_token_count": width,
    }


def summarize_attention(weights, prompt_length, prefix_length, *, bin_size=1024,
                        top_k=1024, full_vector=False):
    """Trung bình proposal queries/heads; bỏ anchor query, giữ mọi key."""
    if weights is None or weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[2] < 2:
        raise ValueError("Cần eager attention [1,H,anchor+proposal,K]")
    vector = weights[:, :, 1:, :].float().mean(dim=(0, 1, 2)).detach().cpu().numpy()
    record = summarize_vector(vector, prompt_length, prefix_length,
                              bin_size=bin_size, top_k=top_k)
    return record, vector if full_vector else vector[:prompt_length]


class AttentionCollector:
    def __init__(self, draft):
        self.prompt_length = 0
        self.prefix_length = 0
        self.layers = {}
        self.hooks = [layer.self_attn.register_forward_hook(self._hook(i))
                      for i, layer in enumerate(draft.layers)]

    def _hook(self, index):
        def collect(module, inputs, outputs):
            self.layers[index] = summarize_attention(
                outputs[1], self.prompt_length, self.prefix_length, full_vector=True)
        return collect

    def begin(self, prompt_length, prefix_length):
        self.prompt_length, self.prefix_length = prompt_length, prefix_length
        self.layers.clear()

    def finish(self):
        if len(self.layers) != len(self.hooks):
            raise RuntimeError("Thiếu attention của draft layer")
        ordered = [self.layers[i] for i in sorted(self.layers)]
        vectors = np.stack([v for _, v in ordered])
        # Coverage is computed after averaging layers, not by averaging their ratios.
        mean_vector = vectors.mean(axis=0)
        record = summarize_vector(mean_vector, self.prompt_length, self.prefix_length)
        record["layers"] = [{"layer": i, **r} for i, (r, _) in enumerate(ordered)]
        return record, vectors

    def close(self):
        for hook in self.hooks:
            hook.remove()


def prompt_for_cap(tokenizer, source_ids, cap):
    def encode(n):
        source = tokenizer.decode(source_ids[:n], skip_special_tokens=False)
        text = "Summarize the following report faithfully and concisely. Return only the summary.\n\nReport:\n" + source
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=False)
    overhead = encode(0).shape[1]
    if overhead >= cap:
        raise ValueError("Context cap nhỏ hơn instruction/template")
    n = min(len(source_ids), cap - overhead)
    ids = encode(n)
    # Tokenization at the source/template boundaries can change a few tokens.
    while ids.shape[1] > cap:
        n -= max(1, ids.shape[1] - cap)
        ids = encode(n)
    if cap - ids.shape[1] > 32:
        raise ValueError(f"Instance không đủ context {cap}: chỉ {ids.shape[1]}")
    return ids, n


@torch.inference_mode()
def collect_run(target, draft, tokenizer, ids, collector, *, cap, output_dir,
                writer, max_rounds, max_new_tokens, sample_id,
                comparison_collector=None, expected_rounds=None,
                save_vectors=True):
    from transformers import DynamicCache
    from dflash.model import extract_context_feature

    ids = ids.to(target.device)
    prompt_length = ids.shape[1]
    block_size = int(draft.block_size)
    mask_id = int(draft.mask_token_id)
    eos = tokenizer.eos_token_id
    cache_target, cache_draft = DynamicCache(), DynamicCache()
    output = target(ids, use_cache=True, past_key_values=cache_target,
                    output_hidden_states=True, logits_to_keep=1)
    features = extract_context_feature(output.hidden_states, draft.target_layer_ids)
    anchor = output.logits[:, -1:].argmax(-1)
    del output
    committed = []
    records = []
    stop_reason = "max_rounds"
    started = time.perf_counter()
    for round_index in range(max_rounds):
        start = prompt_length + len(committed)
        if len(committed) >= max_new_tokens:
            stop_reason = "max_new_tokens"
            break
        if eos is not None and int(anchor.item()) == eos:
            committed.append(eos)
            stop_reason = "eos"
            break
        block = torch.full((1, block_size), mask_id, dtype=torch.long, device=ids.device)
        block[:, :1] = anchor
        positions = torch.arange(cache_draft.get_seq_length(), start + block_size,
                                 device=ids.device).unsqueeze(0)
        collector.begin(prompt_length, start)
        hidden = draft(target_hidden=features, noise_embedding=target.model.embed_tokens(block),
                       position_ids=positions, past_key_values=cache_draft,
                       use_cache=True, is_causal=False)
        logits = target.lm_head(hidden[:, 1 - block_size:])
        if not torch.isfinite(logits).all():
            raise RuntimeError("Draft logits không hữu hạn")
        block[:, 1:] = logits.argmax(-1)
        cache_draft.crop(start)
        record, vectors = collector.finish()
        del hidden, logits
        parent_attention_comparison = None
        if comparison_collector is not None:
            parent_attention_comparison = comparison_collector.begin_round(
                prefix_length=start, prompt_length=prompt_length, draft_vectors=vectors,
                round_index=round_index, expected_rounds=expected_rounds,
                context_cap=cap, sample_id=sample_id)
        verified = target(block, position_ids=torch.arange(start, start + block_size,
                          device=ids.device).unsqueeze(0), past_key_values=cache_target,
                          use_cache=True, output_hidden_states=True)
        posterior = verified.logits.argmax(-1)
        accepted = int((block[:, 1:] == posterior[:, :-1]).cumprod(1).sum().item())
        if comparison_collector is not None:
            record["target_dflash_comparison"] = comparison_collector.finish_round(
                accepted_proposals=accepted)
            if parent_attention_comparison is not None:
                record["parent_attention_comparison"] = parent_attention_comparison
        to_commit = block[0, :accepted + 1].tolist()
        if eos in to_commit:
            to_commit = to_commit[:to_commit.index(eos) + 1]
        to_commit = to_commit[:max_new_tokens - len(committed)]
        committed.extend(to_commit)
        vector_file = f"context_{cap}_round_{round_index + 1:02d}.npz"
        if save_vectors:
            np.savez_compressed(output_dir / vector_file, per_layer_attention=vectors,
                                prompt_length=prompt_length, prefix_length=start)
        mean_vector = vectors.mean(0)
        top_positions = np.argsort(mean_vector[:prompt_length])[-8:][::-1]
        record.update({
            "type": "draft_attention", "method": "dflash_attention_probe",
            "dataset": "gov_report", "sample_id": sample_id,
            "context_cap": cap, "input_tokens": prompt_length,
            "round": round_index + 1, "output_offset": start - prompt_length,
            "prefix_tokens": start, "accepted_proposals": accepted,
            "output_tokens": len(committed),
            "committed_tokens": len(to_commit),
            "vectors_file": f"{output_dir.name}/{vector_file}" if save_vectors else None,
            "top_prompt_positions": [{"position": int(p), "token": tokenizer.decode([int(ids[0, p])]),
                                      "attention": float(mean_vector[p])} for p in top_positions],
        })
        writer.add(record)
        records.append(record)
        if round_index == 0 or (round_index + 1) % 16 == 0:
            print(f"[probe] sample={sample_id} cap={cap} round={round_index+1} "
                  f"output={len(committed)} prompt={record['prompt_mass']:.3f} "
                  f"generated={record['generated_mass']:.3f} block={record['draft_block_mass']:.3f} "
                  f"top1k_all={record['top_1k_all_coverage']:.3f}", flush=True)
        if not to_commit or eos in to_commit or len(committed) >= max_new_tokens:
            stop_reason = "eos" if eos in to_commit else "max_new_tokens"
            break
        cache_target.crop(prompt_length + len(committed))
        features = extract_context_feature(verified.hidden_states, draft.target_layer_ids)[:, :len(to_commit)]
        anchor = posterior[:, len(to_commit) - 1:len(to_commit)]
        del verified
    np.save(output_dir / f"context_{cap}_output_ids.npy", np.asarray(committed, dtype=np.int64))
    return {"sample_id": sample_id, "context_cap": cap, "input_tokens": prompt_length,
            "rounds": len(records), "stop_reason": stop_reason,
            "output_tokens": len(committed), "text": tokenizer.decode(committed, skip_special_tokens=True),
            "parent_attention_transitions": sum(
                1 for row in records if row.get("parent_attention_comparison") is not None),
            "parent_attention_bonus_transitions": sum(
                1 for row in records if row.get("parent_attention_comparison", {}).get(
                    "parent_token_role") == "bonus"),
            "elapsed_s": time.perf_counter() - started,
            "mean_prompt_mass": float(np.mean([r['prompt_mass'] for r in records])) if records else None,
            "mean_generated_mass": float(np.mean([r['generated_mass'] for r in records])) if records else None,
            "mean_draft_block_mass": float(np.mean([r['draft_block_mass'] for r in records])) if records else None,
            "mean_top_1k_all_coverage": float(np.mean([r['top_1k_all_coverage'] for r in records])) if records else None,
            "mean_top_1k_coverage": float(np.mean([r['top_1k_prompt_coverage'] for r in records])) if records else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--input", default=str(ROOT / "data/longbench_200/gov_report.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--caps", default="3072,5120,8192,16384")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Probe model thật cần GPU; local chỉ kiểm tra aggregation")
    if args.max_rounds < 1 or args.max_new_tokens < 1 or args.samples < 1:
        raise ValueError("max-rounds/max-new-tokens/samples phải > 0")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    from dflash.model import DFlashDraftModel
    from common.paths import snapshot_dir

    caps = sorted({int(c) for c in args.caps.split(",")})
    if not caps or min(caps) <= 0:
        raise ValueError("caps phải > 0")
    target_path = snapshot_dir(args.target) if not Path(args.target).is_dir() else Path(args.target)
    draft_path = snapshot_dir(args.draft) if not Path(args.draft).is_dir() else Path(args.draft)
    if target_path is None or draft_path is None:
        raise FileNotFoundError("Target/draft snapshot chưa có trong cache Modal")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(target_path), local_files_only=True)
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    order = np.random.default_rng(args.sample_seed).permutation(len(rows))
    selected, seen = [], set()
    for index in order:
        row = rows[int(index)]
        if row["id"] in seen:
            continue
        source = row["context"]
        source_ids = tokenizer.encode(source, add_special_tokens=False)
        if len(source_ids) < max(caps):
            continue
        prompts = {cap: prompt_for_cap(tokenizer, source_ids, cap) for cap in caps}
        selected.append((row, source_ids, prompts))
        seen.add(row["id"])
        if len(selected) == args.samples:
            break
    if len(selected) != args.samples:
        raise ValueError(f"Chỉ có {len(selected)} document đủ {max(caps)} token")
    print(f"[probe] selected {len(selected)} documents: {[r['id'] for r, _, _ in selected]}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(str(target_path), dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
    draft = DFlashDraftModel.from_pretrained(str(draft_path), dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True).to("cuda").eval()
    torch.manual_seed(42)
    manifest = {"status": "running", "target": str(target_path), "draft": str(draft_path),
        "samples": [{"sample_id": row["id"], "source_tokens": len(source_ids),
                     "source_sha256": hashlib.sha256(row['context'].encode()).hexdigest()}
                    for row, source_ids, _ in selected],
        "sample_count": len(selected), "dataset": "gov_report",
        "sample_seed": args.sample_seed,
        "sample_selection": "seeded permutation of dataset, unique documents with enough source tokens",
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "caps": caps, "max_rounds": args.max_rounds, "max_new_tokens": args.max_new_tokens,
        "block_size": draft.block_size, "feature_layer_ids": draft.target_layer_ids,
        "draft_layers": len(draft.layers), "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "transformers": transformers.__version__, "dtype": "bfloat16",
        "target_attention": "sdpa", "draft_attention": "eager", "seed": 42,
        "query_aggregation": "mean layers/heads/proposals excluding anchor",
        "prompt_scope": "full initial prompt including report and template/instruction",
        "key_scope": "all keys: initial prompt + committed generated prefix + anchor/masks in current draft block",
        "schema_version": 2, "prompt_bin_size": 1024, "generated_bin_size": 128}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    writer = JsonlWriter(output_dir / "attention.jsonl")
    collector = AttentionCollector(draft)
    summaries = []
    try:
        for sample_index, (row, _, prompts) in enumerate(selected):
            sample_dir = output_dir / f"sample_{sample_index:02d}"
            sample_dir.mkdir()
            for cap, (ids, source_count) in prompts.items():
                print(f"[probe] start sample={sample_index+1}/{len(selected)} id={row['id']} "
                      f"cap={cap} actual_input={ids.shape[1]} source={source_count}", flush=True)
                np.save(sample_dir / f"context_{cap}_input_ids.npy", ids.numpy())
                summaries.append(collect_run(target, draft, tokenizer, ids, collector, cap=cap,
                    output_dir=sample_dir, writer=writer, max_rounds=args.max_rounds,
                    max_new_tokens=args.max_new_tokens, sample_id=row["id"]))
                print(f"[probe] finished sample={sample_index+1} cap={cap} "
                      f"rounds={summaries[-1]['rounds']} output={summaries[-1]['output_tokens']} "
                      f"stop={summaries[-1]['stop_reason']}", flush=True)
                torch.cuda.empty_cache()
        summary = {"type": "summary", "status": "success", "sample_count": len(selected),
                   "run_count": len(summaries), "draft_rounds": sum(r['rounds'] for r in summaries),
                   "runs": summaries}
        writer.finalize(summary)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        manifest["status"] = "success"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        collector.close()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
