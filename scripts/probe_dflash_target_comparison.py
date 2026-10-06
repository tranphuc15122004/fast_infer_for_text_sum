#!/usr/bin/env python3
"""So phân phối attention DFlash/target theo vị trí key trên cùng lượt draft."""

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
from probe_dflash_attention import AttentionCollector, collect_run

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

SCORE_TOPK = [1024, 4096]
POSITION_BINS = 10


def js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    """Jensen-Shannon divergence theo log2, trong khoảng [0, 1]."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / first.sum()
    second = second / second.sum()
    middle = (first + second) / 2
    first_mask, second_mask = first > 0, second > 0
    left = np.sum(first[first_mask] * np.log2(first[first_mask] / middle[first_mask]))
    right = np.sum(second[second_mask] * np.log2(second[second_mask] / middle[second_mask]))
    return float((left + right) / 2)


def top_indices(vector: np.ndarray, width: int) -> np.ndarray:
    width = min(width, vector.size)
    return np.argpartition(vector, -width)[-width:]


def compare_parent_to_next_draft(parent: np.ndarray, draft: np.ndarray,
                                 prefix_length: int,
                                 target_layer_ids: list[int]) -> list[dict]:
    """Compare previous target anchor-query attention with next DFlash block."""
    parent = np.asarray(parent, dtype=np.float64)
    draft = np.asarray(draft, dtype=np.float64)
    if draft.ndim != 2 or parent.ndim != 2 or draft.shape[1] != prefix_length + 16:
        raise ValueError("Parent/next-draft vectors have incompatible key scope")
    if parent.shape[1] != prefix_length:
        raise ValueError(f"Parent cache length {parent.shape[1]} != next prefix {prefix_length}")

    result = []
    for target_index, parent_vector in enumerate(parent):
        parent_vector = parent_vector / parent_vector.sum()
        padded_parent = np.zeros(prefix_length + 16, dtype=np.float64)
        padded_parent[:prefix_length] = parent_vector
        for draft_index, draft_vector in enumerate(draft):
            draft_vector = draft_vector / draft_vector.sum()
            draft_cache = draft_vector[:prefix_length]
            draft_cache_mass = float(draft_cache.sum())
            if draft_cache_mass <= 0:
                raise ValueError("DFlash có zero attention mass trên shared cache")
            draft_cache_conditional = draft_cache / draft_cache_mass
            metrics = {
                "js_divergence_full_context": js_divergence(padded_parent, draft_vector),
                "js_divergence_cache_conditional": js_divergence(
                    parent_vector, draft_cache_conditional),
                "parent_cache_mass": 1.0,
                "dflash_cache_mass_full_denominator": draft_cache_mass,
                "dflash_current_block_mass_full_denominator": float(
                    draft_vector[prefix_length:].sum()),
            }
            for budget in SCORE_TOPK:
                width = min(budget, prefix_length)
                parent_positions = top_indices(parent_vector, width)
                draft_positions = top_indices(draft_cache, width)
                overlap = len(np.intersect1d(parent_positions, draft_positions,
                                             assume_unique=True))
                suffix = str(budget)
                metrics.update({
                    f"top_{budget}_cache_overlap_fraction": float(overlap / width),
                    f"parent_attention_mass_on_dflash_top_{suffix}_cache": float(
                        parent_vector[draft_positions].sum()),
                    f"dflash_full_mass_on_parent_top_{suffix}_cache": float(
                        draft_vector[parent_positions].sum()),
                    f"dflash_cache_conditional_mass_on_parent_top_{suffix}_cache": float(
                        draft_cache_conditional[parent_positions].sum()),
                    f"parent_top_{suffix}_own_mass": float(parent_vector[parent_positions].sum()),
                })
            result.append({"target_layer": target_layer_ids[target_index],
                           "draft_layer": draft_index, "metrics": metrics})
    return result


def model_summary(vector: np.ndarray, prompt_length: int, prefix_length: int) -> dict:
    total = float(vector.sum(dtype=np.float64))
    cache_mass = float(vector[:prefix_length].sum(dtype=np.float64))
    prompt_mass = float(vector[:prompt_length].sum(dtype=np.float64))
    output_mass = float(vector[prompt_length:prefix_length].sum(dtype=np.float64))
    boundaries = np.linspace(0, len(vector), POSITION_BINS + 1, dtype=int)
    position_bins = [float(vector[boundaries[i]:boundaries[i + 1]].sum(dtype=np.float64))
                     for i in range(POSITION_BINS)]
    top_cache = {}
    for budget in SCORE_TOPK:
        selected = top_indices(vector[:prefix_length], budget)
        top_cache[str(budget)] = {
            "keys": int(min(budget, prefix_length)),
            "coverage_total_mass": float(vector[selected].sum(dtype=np.float64)),
            "coverage_cache_mass": float(vector[selected].sum(dtype=np.float64) / cache_mass),
            "positions": selected,
        }
    return {
        "total_mass": total,
        "prompt_mass": prompt_mass,
        "committed_output_mass": output_mass,
        "context_cache_mass": cache_mass,
        "current_block_mass": float(vector[prefix_length:].sum(dtype=np.float64)),
        "top_cache": top_cache,
        "position_bins_full_context": position_bins,
    }


class TargetAttentionCollector:
    """Tính attention target cho 3 layer khi target đang verify block DFlash."""

    def __init__(self, target, target_layer_ids: list[int], *,
                 capture_parent_attention: bool = False):
        self.target = target
        self.target_layer_ids = target_layer_ids
        self.capture_parent_attention = capture_parent_attention
        self.active = False
        self.prefix_length = 0
        self.prompt_length = 0
        self.draft_vectors = None
        self.target_vectors = {}
        self.target_query_probabilities = {}
        self.future_block_mass = {}
        self.snapshots = {}
        self.parent_snapshots = {}
        self.pending_parent = None
        self.expected_rounds = {}
        self.original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def capture_then_sdpa(module, query, key, value, attention_mask,
                              dropout=0.0, scaling=None, is_causal=None, **kwargs):
            layer = getattr(module, "layer_idx", None)
            if self.active and layer in self.target_layer_ids:
                q_len, key_len = query.shape[-2], key.shape[-2]
                if key_len != self.prefix_length + q_len:
                    raise RuntimeError(
                        f"Target K/V shape mismatch: {key_len=} != "
                        f"{self.prefix_length=} + {q_len=}"
                    )
                kv_groups = getattr(module, "num_key_value_groups", 1)
                if key.shape[1] != query.shape[1]:
                    key_for_scores = key.repeat_interleave(kv_groups, dim=1)
                else:
                    key_for_scores = key
                scale = scaling if scaling is not None else getattr(module, "scaling", None)
                if scale is None:
                    scale = query.shape[-1] ** -0.5
                scores = torch.matmul(
                    query.float(), key_for_scores[..., :self.prefix_length + q_len, :]
                    .float().transpose(-1, -2)
                ) * scale
                if attention_mask is not None:
                    mask = attention_mask[..., :q_len, :self.prefix_length + q_len]
                    if mask.dtype == torch.bool:
                        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                    else:
                        scores = scores + mask.float()
                elif is_causal is not False:
                    key_positions = torch.arange(
                        self.prefix_length + q_len, device=scores.device)
                    query_positions = self.prefix_length + torch.arange(
                        q_len, device=scores.device)
                    allowed = key_positions[None, :] <= query_positions[:, None]
                    window = getattr(module, "sliding_window", None)
                    if window:
                        allowed &= key_positions[None, :] > query_positions[:, None] - window
                    scores = scores.masked_fill(~allowed[None, None, :, :],
                                                torch.finfo(scores.dtype).min)
                probabilities = torch.softmax(scores, dim=-1)
                aligned_queries = min(15, q_len - 1)
                block_probabilities = probabilities[:, :, :aligned_queries,
                                                      self.prefix_length:self.prefix_length + q_len]
                future_keys = (torch.arange(q_len, device=scores.device)[None, None, None, :]
                               > torch.arange(aligned_queries, device=scores.device)[None, None, :, None])
                future_mass = block_probabilities.masked_fill(~future_keys, 0).sum(-1).amax()
                if future_mass.item() > 1e-6:
                    raise RuntimeError(
                        f"Target attention leaked to future verifier keys: {future_mass.item():.3g}"
                    )
                self.future_block_mass[layer] = float(future_mass.item())
                # Target q[0:15] predicts the same token as DFlash's mask q[1:16].
                pooled = probabilities[:, :, :aligned_queries, :].mean(dim=(0, 1, 2))
                self.target_vectors[layer] = pooled
                if self.capture_parent_attention:
                    # [batch, heads, query, key] -> [query, key], retaining
                    # each target query while averaging only its heads.
                    self.target_query_probabilities[layer] = probabilities[0].mean(dim=0)
            return self.original_sdpa(
                module, query, key, value, attention_mask,
                dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
            )

        self.sdpa_wrapper = capture_then_sdpa
        # Keep the target config/backend label as "sdpa" so Transformers builds
        # exactly the original causal mask. Replace only the implementation
        # function used at dispatch; DFlash uses eager and is unaffected.
        ALL_ATTENTION_FUNCTIONS.register("sdpa", self.sdpa_wrapper)

    def begin_round(self, *, prefix_length, prompt_length, draft_vectors, round_index,
                    expected_rounds, context_cap, sample_id):
        self.active = False
        round_index = int(round_index)
        context_cap = int(context_cap)
        prefix_length = int(prefix_length)
        parent_comparison = None
        if self.capture_parent_attention:
            if round_index == 0:
                self.pending_parent = None
            else:
                parent = self.pending_parent
                if parent is None:
                    raise RuntimeError("Thiếu parent query attention từ lượt trước")
                if (parent["sample_id"] != sample_id or parent["context_cap"] != context_cap or
                        parent["round_index"] != round_index - 1):
                    raise RuntimeError("Parent attention không thuộc trajectory/lượt ngay trước")
                if parent["prefix_length"] != prefix_length:
                    raise RuntimeError(
                        f"Parent visible prefix {parent['prefix_length']} != DFlash prefix {prefix_length}"
                    )
                parent_vectors = np.stack([parent["vectors"][layer]
                                           for layer in self.target_layer_ids])
                draft_vectors = np.asarray(draft_vectors, dtype=np.float64)
                parent_comparison = {
                    "parent_round": parent["round_index"] + 1,
                    "draft_round": round_index + 1,
                    "parent_query_index": parent["query_index"],
                    "accepted_proposals": parent["accepted_proposals"],
                    "parent_token_role": parent["token_role"],
                    "shared_cache_length": prefix_length,
                    "parent_target_layers": self.target_layer_ids,
                    "draft_layers": list(range(draft_vectors.shape[0])),
                    "pairs": compare_parent_to_next_draft(
                        parent_vectors, draft_vectors, prefix_length, self.target_layer_ids),
                }
                expected_rounds = int(expected_rounds or 0)
                parent_phases = {1: "first_transition",
                                 expected_rounds // 2: "middle",
                                 expected_rounds - 1: "last"}
                phase = parent_phases.get(round_index)
                if phase is not None:
                    self.parent_snapshots.setdefault((sample_id, context_cap), {})[phase] = {
                        "draft": draft_vectors.astype(np.float16),
                        "parent_target": parent_vectors.astype(np.float16),
                        "prefix_length": prefix_length,
                        "prompt_length": self.prompt_length or int(prompt_length),
                        "round": round_index + 1,
                        "parent_round": parent["round_index"] + 1,
                        "parent_query_index": parent["query_index"],
                        "accepted_proposals": parent["accepted_proposals"],
                        "parent_token_role": parent["token_role"],
                    }
        self.active = True
        self.prefix_length = prefix_length
        self.prompt_length = int(prompt_length)
        self.draft_vectors = np.asarray(draft_vectors, dtype=np.float32)
        self.target_vectors.clear()
        self.target_query_probabilities.clear()
        self.future_block_mass.clear()
        self.round_index = round_index
        self.context_cap = context_cap
        self.sample_id = sample_id
        self.expected_rounds = int(expected_rounds or 0)
        return parent_comparison

    def finish_round(self, *, accepted_proposals: int | None = None) -> dict:
        self.active = False
        if sorted(self.target_vectors) != self.target_layer_ids:
            raise RuntimeError(
                f"Thiếu target layer: nhận {sorted(self.target_vectors)}, "
                f"cần {self.target_layer_ids}"
            )
        target_vectors = np.stack([
            self.target_vectors[layer].detach().float().cpu().numpy()
            for layer in self.target_layer_ids
        ]).astype(np.float64)
        draft_vectors = self.draft_vectors.astype(np.float64)
        if draft_vectors.shape[1] != self.prefix_length + 16:
            raise RuntimeError(f"DFlash vector có shape lạ: {draft_vectors.shape}")
        if target_vectors.shape[1] != draft_vectors.shape[1]:
            raise RuntimeError(f"Target/DFlash key scope lệch: {target_vectors.shape} vs {draft_vectors.shape}")

        target_summaries = [model_summary(v, self.prompt_length, self.prefix_length)
                            for v in target_vectors]
        draft_summaries = [model_summary(v, self.prompt_length, self.prefix_length)
                           for v in draft_vectors]
        pair_metrics = []
        for target_index, target_summary in enumerate(target_summaries):
            target_vector = target_vectors[target_index]
            target_cache = target_vector[:self.prefix_length]
            target_cache_conditional = target_cache / target_cache.sum()
            for draft_index, draft_summary in enumerate(draft_summaries):
                draft_vector = draft_vectors[draft_index]
                draft_cache = draft_vector[:self.prefix_length]
                draft_cache_conditional = draft_cache / draft_cache.sum()
                d_top = draft_summary["top_cache"]
                t_top = target_summary["top_cache"]
                measures = {
                    "js_divergence_full_context": js_divergence(target_vector, draft_vector),
                    "js_divergence_cache_conditional": js_divergence(
                        target_cache_conditional, draft_cache_conditional),
                }
                for budget in SCORE_TOPK:
                    name = str(budget)
                    d_positions = d_top[name]["positions"]
                    t_positions = t_top[name]["positions"]
                    overlap = len(np.intersect1d(d_positions, t_positions, assume_unique=True))
                    measures.update({
                        f"top_{budget}_cache_overlap_fraction": float(overlap / min(budget, self.prefix_length)),
                        f"target_total_mass_on_dflash_top_{budget}_cache": float(target_vector[d_positions].sum()),
                        f"dflash_total_mass_on_target_top_{budget}_cache": float(draft_vector[t_positions].sum()),
                    })
                pair_metrics.append({
                    "target_layer": self.target_layer_ids[target_index],
                    "draft_layer": draft_index,
                    "metrics": measures,
                })

        if not self.capture_parent_attention:
            for name, expected in [("first", 0), ("middle", self.expected_rounds // 2),
                                   ("last", self.expected_rounds - 1)]:
                if self.expected_rounds and self.round_index == expected:
                    self.snapshots.setdefault((self.sample_id, self.context_cap), {})[name] = {
                        "draft": draft_vectors.astype(np.float16),
                        "target": target_vectors.astype(np.float16),
                        "prefix_length": self.prefix_length,
                        "prompt_length": self.prompt_length,
                        "round": self.round_index + 1,
                    }

        if self.capture_parent_attention:
            if accepted_proposals is None:
                raise ValueError("Cần accepted_proposals để chọn query sinh anchor kế tiếp")
            query_index = int(accepted_proposals)
            if not 0 <= query_index <= 15:
                raise ValueError(f"Query parent không thuộc block 16: {query_index}")
            next_prefix_length = self.prefix_length + query_index + 1
            parent_vectors = {}
            for layer in self.target_layer_ids:
                query_probs = self.target_query_probabilities[layer]
                if query_index >= query_probs.shape[0] or query_probs.shape[-1] < next_prefix_length:
                    raise RuntimeError("Target parent query không phủ prefix kế tiếp")
                vector = query_probs[query_index, :next_prefix_length].detach().float().cpu().numpy()
                total = float(vector.sum(dtype=np.float64))
                if abs(total - 1.0) > 2e-3:
                    raise RuntimeError(f"Parent query attention không cộng 100%: {total}")
                parent_vectors[layer] = vector.astype(np.float64) / total
            self.pending_parent = {
                "sample_id": self.sample_id, "context_cap": self.context_cap,
                "round_index": self.round_index,
                "query_index": query_index,
                "accepted_proposals": query_index,
                "token_role": "bonus" if query_index == 15 else "correction",
                "prefix_length": next_prefix_length,
                "vectors": parent_vectors,
            }
        self.target_query_probabilities.clear()

        def compact(summary):
            return {key: value for key, value in summary.items() if key != "top_cache"}

        return {
            "query_alignment": "target verifier queries 0..14 paired with DFlash mask queries 1..15",
            "key_scope": "prompt + committed output prefix + current block; target causal future block keys have zero probability",
            "prefix_length": self.prefix_length,
            "target_layers": [
                {"layer": self.target_layer_ids[i], **compact(v)}
                for i, v in enumerate(target_summaries)
            ],
            "draft_layers": [
                {"layer": i, **compact(v)} for i, v in enumerate(draft_summaries)
            ],
            "pairwise": pair_metrics,
            "target_max_future_block_mass": max(self.future_block_mass.values()),
            "metric_note": "Full-context JS uses original total probability mass. Cache-conditional JS renormalizes only prompt+committed output. Top-K cache overlap excludes current 16 block slots.",
        }

    def close(self):
        self.active = False
        ALL_ATTENTION_FUNCTIONS.register("sdpa", self.original_sdpa)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path,
                        help="Artifact gốc có manifest, summary và input IDs trên Modal volume")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--caps", default="3072,5120,8192,16384")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--capture-parent-attention", action="store_true",
                        help="Ghép query target sinh anchor với DFlash ở lượt kế tiếp")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phép so sánh model thật cần GPU")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel

    source_manifest = json.loads((args.source_run / "manifest.json").read_text())
    source_summary = json.loads((args.source_run / "summary.json").read_text())
    if source_manifest["status"] != "success" or source_manifest["block_size"] != 16:
        raise ValueError("Run nguồn không thành công hoặc block size không phải 16")
    caps = [int(x) for x in args.caps.split(",")]
    if any(cap not in source_manifest["caps"] for cap in caps):
        raise ValueError("Caps phải tồn tại trong run nguồn để ghép đúng tokenized prompts")
    selected_samples = source_manifest["samples"][:args.samples]
    if len(selected_samples) != args.samples:
        raise ValueError("Số instance yêu cầu lớn hơn run nguồn")

    target_path = Path(source_manifest["target"])
    draft_path = Path(source_manifest["draft"])
    if not target_path.exists() or not draft_path.exists():
        raise FileNotFoundError("Model snapshot của run gốc không tồn tại trên Modal volume")
    tokenizer = AutoTokenizer.from_pretrained(str(target_path), local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        str(target_path), dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to("cuda").eval()
    draft = DFlashDraftModel.from_pretrained(
        str(draft_path), dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).to("cuda").eval()
    layer_count = len(target.model.layers)
    target_layers = sorted({0, layer_count // 2, layer_count - 1})
    if len(target_layers) != 3:
        raise ValueError(f"Target cần >= 3 layer, nhận {layer_count}")
    comparison = TargetAttentionCollector(
        target, target_layers,
        capture_parent_attention=args.capture_parent_attention)
    dflash_collector = AttentionCollector(draft)

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "source_run": source_manifest.get("source_run", "dflash-attention-fullmass-govreport10-l40s-20261006"),
        "source_run_manifest_sha256": hashlib.sha256((args.source_run / "manifest.json").read_bytes()).hexdigest(),
        "source_run_summary_sha256": hashlib.sha256((args.source_run / "summary.json").read_bytes()).hexdigest(),
        "target": source_manifest["target"], "draft": source_manifest["draft"],
        "target_layers": target_layers, "target_layer_count": layer_count,
        "draft_layers": len(draft.layers), "caps": caps,
        "samples": [s["sample_id"] for s in selected_samples], "sample_count": len(selected_samples),
        "sample_seed": args.sample_seed, "block_size": 16,
        "query_alignment": "DFlash mask positions 1..15 vs target verifier positions 0..14; both predict the same proposal token index.",
        "primary_metric": (
            "Jensen-Shannon divergence (base 2) between the previous target next-anchor query padded on the full next-round key scope and the next DFlash block distribution."
            if args.capture_parent_attention else
            "Jensen-Shannon divergence (base 2) between mean-over-heads-and-aligned-queries distributions on the full shared key scope."
        ),
        "secondary_metrics": ["cache-conditional JS", "Top-1K/4K cache-position overlap", "cross-model total-mass coverage of selected cache keys", "prompt/output/block mass"],
        "decision_scope": "Exploratory distribution comparison; no quality or acceptance conclusion without a later causal cache ablation.",
        "model_attention": "Target uses original SDPA for model outputs; registered wrapper computes selected-layer probabilities from the same Q/K and causal/sliding mask without materializing prefill attention matrices.",
        "dtype": "bfloat16 weights; target comparison probabilities accumulated in float32",
        "capture_parent_attention": args.capture_parent_attention,
        "parent_alignment": (
            "Target verifier query q=accepted_proposals predicts the anchor passed to the next DFlash round; q=15 is a literal bonus query only when all 15 proposals are accepted. Compare that target row to the next-round DFlash attention on its shared cache, with the next 16-key block retained in full-scope metrics."
            if args.capture_parent_attention else None),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    writer = JsonlWriter(args.output / "attention_comparison.jsonl")
    reference_runs = {(r["sample_id"], r["context_cap"]): r for r in source_summary["runs"]}
    run_summaries = []
    started = time.perf_counter()
    try:
        for sample_index, sample in enumerate(selected_samples):
            sample_dir = args.output / f"sample_{sample_index:02d}"
            sample_dir.mkdir()
            for cap in caps:
                input_path = args.source_run / f"sample_{sample_index:02d}/context_{cap}_input_ids.npy"
                if not input_path.exists():
                    raise FileNotFoundError(input_path)
                ids = torch.from_numpy(np.load(input_path).astype(np.int64))
                ref = reference_runs[sample["sample_id"], cap]
                result = collect_run(
                    target, draft, tokenizer, ids, dflash_collector, cap=cap,
                    output_dir=sample_dir, writer=writer, max_rounds=args.max_rounds,
                    max_new_tokens=args.max_new_tokens, sample_id=sample["sample_id"],
                    comparison_collector=comparison, expected_rounds=ref["rounds"],
                    save_vectors=False)
                generated = np.load(sample_dir / f"context_{cap}_output_ids.npy")
                reference = np.load(args.source_run / f"sample_{sample_index:02d}/context_{cap}_output_ids.npy")
                replay_match = bool(np.array_equal(generated, reference)) if args.max_rounds >= ref["rounds"] else None
                replay_rounds_match = result["rounds"] == ref["rounds"] if args.max_rounds >= ref["rounds"] else None
                run_summaries.append({
                    "sample_id": sample["sample_id"], "context_cap": cap,
                    "draft_rounds": result["rounds"], "reference_draft_rounds": ref["rounds"],
                    "output_tokens": result["output_tokens"], "reference_output_tokens": ref["output_tokens"],
                    "greedy_output_exact_match": replay_match,
                    "draft_round_count_exact_match": replay_rounds_match,
                    "parent_attention_transitions": result["parent_attention_transitions"],
                    "parent_attention_bonus_transitions": result["parent_attention_bonus_transitions"],
                })
                print(f"[compare] sample={sample_index+1}/{len(selected_samples)} cap={cap} "
                      f"rounds={result['rounds']} replay_match={replay_match} "
                      f"round_count_match={replay_rounds_match}", flush=True)
                if replay_match is False or replay_rounds_match is False:
                    raise RuntimeError(
                        f"Greedy replay diverged for {sample['sample_id']} cap={cap}; "
                        "comparison halted to keep trajectories paired"
                    )
                torch.cuda.empty_cache()

        saved_snapshots = 0
        if not args.capture_parent_attention:
            vector_dir = args.output / "vector_snapshots"
            vector_dir.mkdir()
            for (sample_id, cap), phases in comparison.snapshots.items():
                sample_index = next(i for i, s in enumerate(selected_samples) if s["sample_id"] == sample_id)
                for phase, snapshot in phases.items():
                    name = f"sample_{sample_index:02d}_context_{cap}_{phase}.npz"
                    np.savez_compressed(vector_dir / name, **snapshot)
                    saved_snapshots += 1
        parent_snapshot_dir = args.output / "parent_transition_snapshots"
        if args.capture_parent_attention:
            parent_snapshot_dir.mkdir()
            for (sample_id, cap), phases in comparison.parent_snapshots.items():
                sample_index = next(i for i, s in enumerate(selected_samples)
                                    if s["sample_id"] == sample_id)
                for phase, snapshot in phases.items():
                    name = f"sample_{sample_index:02d}_context_{cap}_{phase}.npz"
                    np.savez_compressed(parent_snapshot_dir / name, **snapshot)
        summary = {
            "type": "summary", "status": "success", "run_count": len(run_summaries),
            "draft_rounds": sum(r["draft_rounds"] for r in run_summaries),
            "exact_greedy_replay_runs": sum(r["greedy_output_exact_match"] is True for r in run_summaries),
            "exact_round_count_runs": sum(r["draft_round_count_exact_match"] is True for r in run_summaries),
            "runs": run_summaries, "vector_snapshots": saved_snapshots,
            "parent_attention_transitions": sum(
                r["parent_attention_transitions"] for r in run_summaries),
            "parent_attention_bonus_transitions": sum(
                r["parent_attention_bonus_transitions"] for r in run_summaries),
            "elapsed_s": time.perf_counter() - started,
        }
        writer.finalize(summary)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        manifest.update(status="success", elapsed_s=summary["elapsed_s"],
                        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        raise
    finally:
        comparison.close()
        dflash_collector.close()
    print(f"[compare] completed rounds={summary['draft_rounds']} elapsed={summary['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
