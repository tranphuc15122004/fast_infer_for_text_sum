#!/usr/bin/env python3
"""CPU experiment for DFlash draft-tree verification.

The tree builder follows the best-first prefix objective from DDTree. Target
verification remains greedy and exact; the script is a local CPU feasibility
benchmark, not a GPU speedup claim.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "scripts"
for path in (str(SCRIPT_ROOT), str(ROOT / "externals" / "dflash")):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.benchmark_data import render_prompt
from common.qwen3_paired import prepare_input_ids
from dflash.model import DFlashDraftModel, extract_context_feature
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclass(frozen=True)
class TreeNode:
    node_id: int
    token_id: int
    depth: int
    parent_id: int | None
    rank_path: tuple[int, ...]
    log_probability: float


@dataclass
class DraftTree:
    nodes: list[TreeNode]
    children: list[list[int]]


def build_draft_tree(
    top_token_ids: torch.Tensor | Iterable[Iterable[int]],
    top_log_probs: torch.Tensor | Iterable[Iterable[float]],
    budget: int,
) -> DraftTree:
    """Return the top-B prefixes of the factorized DFlash draft distribution."""

    token_rows = (
        top_token_ids.detach().cpu().tolist()
        if isinstance(top_token_ids, torch.Tensor)
        else [list(row) for row in top_token_ids]
    )
    log_rows = (
        top_log_probs.detach().cpu().tolist()
        if isinstance(top_log_probs, torch.Tensor)
        else [list(row) for row in top_log_probs]
    )
    if not token_rows or len(token_rows) != len(log_rows):
        raise ValueError("token and log-probability rows must have equal nonzero length")
    if budget <= 0:
        raise ValueError("budget must be positive")
    widths = {len(row) for row in token_rows} | {len(row) for row in log_rows}
    if len(widths) != 1 or 0 in widths:
        raise ValueError("each draft position must have the same nonempty top-k width")
    if any(len(ids) != len(logps) for ids, logps in zip(token_rows, log_rows)):
        raise ValueError("token and log-probability widths differ")

    width = len(token_rows[0])
    depth_limit = len(token_rows)
    budget = min(int(budget), width * depth_limit)
    heap: list[tuple[float, tuple[int, ...]]] = [(-float(log_rows[0][0]), (0,))]
    enqueued: set[tuple[int, ...]] = {(0,)}
    nodes: list[TreeNode] = []
    children: list[list[int]] = []
    tuple_to_id: dict[tuple[int, ...], int] = {}

    while heap and len(nodes) < budget:
        negative_score, ranks = heapq.heappop(heap)
        score = -negative_score
        depth = len(ranks)
        parent_path = ranks[:-1]
        parent_id = tuple_to_id.get(parent_path) if parent_path else None
        if parent_path and parent_id is None:
            raise RuntimeError("best-first tree lost prefix closure")

        node_id = len(nodes)
        node = TreeNode(
            node_id=node_id,
            token_id=int(token_rows[depth - 1][ranks[-1]]),
            depth=depth,
            parent_id=parent_id,
            rank_path=ranks,
            log_probability=score,
        )
        nodes.append(node)
        children.append([])
        tuple_to_id[ranks] = node_id
        if parent_id is not None:
            children[parent_id].append(node_id)

        last_rank = ranks[-1]
        sibling = ranks[:-1] + (last_rank + 1,)
        if last_rank + 1 < width and sibling not in enqueued:
            sibling_score = (
                score
                - float(log_rows[depth - 1][last_rank])
                + float(log_rows[depth - 1][last_rank + 1])
            )
            heapq.heappush(heap, (-sibling_score, sibling))
            enqueued.add(sibling)

        child = ranks + (0,)
        if depth < depth_limit and child not in enqueued:
            child_score = score + float(log_rows[depth][0])
            heapq.heappush(heap, (-child_score, child))
            enqueued.add(child)

    return DraftTree(nodes=nodes, children=children)


def build_tree_attention_mask(
    tree: DraftTree,
    *,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build additive ancestor-only attention mask and RoPE positions.

    Query row zero is the pending bonus token. Other rows correspond to the
    flattened tree nodes. Cached context and the bonus token are visible to all
    nodes; a draft node sees only its ancestors and itself among tree nodes.
    """

    query_length = 1 + len(tree.nodes)
    key_length = past_length + query_length
    mask = torch.full(
        (1, 1, query_length, key_length),
        -torch.inf,
        dtype=dtype,
        device=device,
    )
    mask[..., : past_length + 1] = 0
    positions = [past_length]
    for node in tree.nodes:
        positions.append(past_length + node.depth)

    for node in tree.nodes:
        query_row = 1 + node.node_id
        mask[0, 0, query_row, : past_length + 1] = 0
        ancestor_id = node.node_id
        while ancestor_id is not None:
            mask[0, 0, query_row, past_length + 1 + ancestor_id] = 0
            ancestor_id = tree.nodes[ancestor_id].parent_id

    return mask, torch.tensor([positions], dtype=torch.long, device=device)


def walk_verified_tree(
    tree: DraftTree,
    verifier_logits: torch.Tensor,
) -> tuple[list[int], int]:
    """Follow target argmax through matching children; return path and bonus."""

    if verifier_logits.ndim != 2 or verifier_logits.shape[0] != len(tree.nodes) + 1:
        raise ValueError("verifier logits must have one row for bonus and each tree node")
    accepted: list[int] = []
    parent_id: int | None = None
    query_row = 0
    while True:
        target_token = int(verifier_logits[query_row].argmax().item())
        children = (
            [node.node_id for node in tree.nodes if node.parent_id is None]
            if parent_id is None
            else tree.children[parent_id]
        )
        match = next(
            (node_id for node_id in children if tree.nodes[node_id].token_id == target_token),
            None,
        )
        if match is None:
            return accepted, target_token
        accepted.append(match)
        parent_id = match
        query_row = 1 + match


def _cache_seq_length(cache: DynamicCache) -> int:
    return int(cache.get_seq_length())


def _compact_cache(cache: DynamicCache, keep_indices: list[int]) -> DynamicCache:
    index = torch.tensor(keep_indices, dtype=torch.long, device=cache.layers[0].keys.device)
    layer_data = []
    for layer in cache.layers:
        layer_data.append(
            (
                layer.keys.index_select(-2, index),
                layer.values.index_select(-2, index),
            )
        )
    return DynamicCache(ddp_cache_data=layer_data)


def dflash_tree_generate(
    draft: DFlashDraftModel,
    target: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    node_budget: int,
    top_k: int,
    stop_token_ids: list[int],
) -> dict[str, Any]:
    """Generate greedily using DFlash proposals and exact tree verification."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("CPU tree runner currently requires input shape [1, sequence]")
    if node_budget <= 0 or top_k <= 0:
        raise ValueError("node_budget and top_k must be positive")

    input_length = input_ids.shape[1]
    max_length = input_length + max_new_tokens
    target_cache = DynamicCache()
    draft_cache = DynamicCache()
    prefill = target(
        input_ids,
        past_key_values=target_cache,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    target_cache = prefill.past_key_values
    target_hidden = extract_context_feature(prefill.hidden_states, draft.target_layer_ids)
    generated = [int(prefill.logits[0, -1].argmax().item())]
    rounds = 0
    accepted_lengths: list[int] = []
    tree_build_seconds = 0.0
    verification_seconds = 0.0
    cache_seconds = 0.0
    device = input_ids.device
    dtype = next(target.parameters()).dtype
    stop_set = {int(token) for token in stop_token_ids}

    if generated[0] in stop_set or max_new_tokens == 1:
        return {
            "output_ids": torch.tensor([generated[:max_new_tokens]], device=device),
            "rounds": rounds,
            "accepted_lengths": accepted_lengths,
            "tree_build_ms": 0.0,
            "verify_ms": 0.0,
            "cache_ms": 0.0,
        }

    while len(generated) < max_new_tokens:
        start = _cache_seq_length(target_cache)
        bonus = generated[-1]
        block_size = int(draft.block_size)
        noise_ids = torch.full(
            (1, block_size),
            int(draft.mask_token_id),
            dtype=torch.long,
            device=device,
        )
        noise_ids[0, 0] = bonus
        noise_embedding = target.model.embed_tokens(noise_ids)
        draft_position_ids = torch.arange(
            draft_cache.get_seq_length(), start + block_size, dtype=torch.long, device=device
        ).unsqueeze(0)
        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_position_ids,
            past_key_values=draft_cache,
            use_cache=True,
            is_causal=False,
        )
        draft_logits = target.lm_head(draft_hidden[:, 1 - block_size :, :])
        draft_cache.crop(start)

        build_start = time.perf_counter()
        proposal_length = draft_logits.shape[1]
        effective_top_k = min(top_k, int(draft_logits.shape[-1]))
        top_values, top_ids = torch.topk(
            torch.log_softmax(draft_logits[0].float(), dim=-1),
            k=effective_top_k,
            dim=-1,
        )
        tree = build_draft_tree(top_ids, top_values, node_budget)
        mask, position_ids = build_tree_attention_mask(
            tree,
            past_length=start,
            dtype=dtype,
            device=device,
        )
        tree_build_seconds += time.perf_counter() - build_start

        tree_token_ids = torch.tensor(
            [[node.token_id for node in tree.nodes]], dtype=torch.long, device=device
        )
        verify_ids = torch.cat(
            [torch.tensor([[bonus]], dtype=torch.long, device=device), tree_token_ids], dim=1
        )
        verify_start = time.perf_counter()
        verified = target(
            verify_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        target_cache = verified.past_key_values
        accepted_node_ids, next_bonus = walk_verified_tree(tree, verified.logits[0])
        verification_seconds += time.perf_counter() - verify_start

        cache_start = time.perf_counter()
        kept_tree_nodes = [1 + node_id for node_id in accepted_node_ids]
        keep_cache_indices = list(range(start)) + [start] + [start + item for item in kept_tree_nodes]
        target_cache = _compact_cache(target_cache, keep_cache_indices)
        hidden_rows = [0] + [1 + node_id for node_id in accepted_node_ids]
        selected_hidden = tuple(hidden[0, hidden_rows].unsqueeze(0) for hidden in verified.hidden_states)
        target_hidden = extract_context_feature(selected_hidden, draft.target_layer_ids)
        cache_seconds += time.perf_counter() - cache_start

        accepted_tokens = [tree.nodes[node_id].token_id for node_id in accepted_node_ids]
        generated.extend(accepted_tokens)
        if len(generated) < max_new_tokens:
            generated.append(int(next_bonus))
        accepted_lengths.append(len(accepted_tokens))
        rounds += 1
        if next_bonus in stop_set or any(token in stop_set for token in accepted_tokens):
            break

    output = torch.tensor([generated[:max_new_tokens]], dtype=torch.long, device=device)
    return {
        "output_ids": output,
        "rounds": rounds,
        "accepted_lengths": accepted_lengths,
        "tree_build_ms": tree_build_seconds * 1000.0,
        "verify_ms": verification_seconds * 1000.0,
        "cache_ms": cache_seconds * 1000.0,
    }


@torch.inference_mode()
def target_greedy_generate(
    target: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    stop_token_ids: list[int],
) -> torch.Tensor:
    cache = DynamicCache()
    output = target(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    token = int(output.logits[0, -1].argmax().item())
    generated = [token]
    stop_set = set(stop_token_ids)
    while len(generated) < max_new_tokens and token not in stop_set:
        next_output = target(
            torch.tensor([[token]], dtype=torch.long, device=input_ids.device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = next_output.past_key_values
        token = int(next_output.logits[0, -1].argmax().item())
        generated.append(token)
    return torch.tensor([generated], dtype=torch.long, device=input_ids.device)


def _load_models(args: argparse.Namespace):
    target_config = AutoConfig.from_pretrained(args.target_model, local_files_only=True)
    draft_config = AutoConfig.from_pretrained(args.draft_model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    for config in (target_config, draft_config):
        eos = getattr(config, "eos_token_id", None)
        if isinstance(eos, int) and not 0 <= eos < vocab_size:
            config.eos_token_id = tokenizer.eos_token_id
        bos = getattr(config, "bos_token_id", None)
        if isinstance(bos, int) and not 0 <= bos < vocab_size:
            config.bos_token_id = tokenizer.bos_token_id
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        config=target_config,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cpu").eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        config=draft_config,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cpu").eval()
    return tokenizer, target, draft


def _read_samples(dataset: str, count: int) -> list[dict[str, Any]]:
    path = ROOT / "data" / "longbench_100_14k" / f"{dataset}.jsonl"
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
        if len(records) >= count:
            break
    if len(records) < count:
        raise ValueError(f"requested {count} rows, found {len(records)} in {path}")
    return records


def _rss_gb() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer, target, draft = _load_models(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    eos = tokenizer.eos_token_id
    stop_token_ids = [] if eos is None else [int(eos)]

    conditions = ["target_ar", "dflash_chain_4", "dflash_chain_16", "dflash_tree_16"]
    for dataset in args.datasets.split(","):
        rows = _read_samples(dataset.strip(), args.samples_per_dataset)
        for row_index, sample in enumerate(rows):
            prompt = render_prompt(sample)
            encoded = tokenizer(prompt, return_tensors="pt")
            prepared = prepare_input_ids(encoded.input_ids, args.max_input_tokens)
            input_ids = prepared.input_ids.to("cpu").contiguous()

            # Warm all execution paths once on a short deterministic slice.
            if args.warmup and dataset == args.datasets.split(",")[0].strip() and row_index == 0:
                warm_ids = input_ids[:, -min(input_ids.shape[1], 64) :]
                target_greedy_generate(
                    target, warm_ids, max_new_tokens=2, stop_token_ids=stop_token_ids
                )
                for block_size in (4, 16):
                    from dflash.model import dflash_generate

                    dflash_generate(
                        draft,
                        target,
                        warm_ids,
                        max_new_tokens=2,
                        stop_token_ids=stop_token_ids,
                        temperature=0.0,
                        block_size=block_size,
                    )
                dflash_tree_generate(
                    draft,
                    target,
                    warm_ids,
                    max_new_tokens=2,
                    node_budget=args.node_budget,
                    top_k=args.top_k,
                    stop_token_ids=stop_token_ids,
                )

            randomized_conditions = conditions.copy()
            random.shuffle(randomized_conditions)
            sample_outputs: dict[str, list[int]] = {}
            for condition in randomized_conditions:
                started = time.perf_counter()
                stats: dict[str, Any] = {}
                if condition == "target_ar":
                    output_ids = target_greedy_generate(
                        target,
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        stop_token_ids=stop_token_ids,
                    )
                elif condition == "dflash_tree_16":
                    stats = dflash_tree_generate(
                        draft,
                        target,
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        node_budget=args.node_budget,
                        top_k=args.top_k,
                        stop_token_ids=stop_token_ids,
                    )
                    output_ids = stats["output_ids"]
                else:
                    from dflash.model import dflash_generate

                    block_size = 4 if condition.endswith("_4") else 16
                    result = dflash_generate(
                        draft,
                        target,
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        stop_token_ids=stop_token_ids,
                        temperature=0.0,
                        block_size=block_size,
                    )
                    output_ids = result[:, input_ids.shape[1] :]
                    stats = {
                        "rounds": None,
                        "accepted_lengths": [],
                        "tree_build_ms": 0.0,
                        "verify_ms": 0.0,
                        "cache_ms": 0.0,
                    }
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                ids = [int(token) for token in output_ids[0].tolist()]
                sample_outputs[condition] = ids
                all_rows.append(
                    {
                        "record_type": "sample",
                        "dataset": dataset.strip(),
                        "sample_id": sample.get("id", row_index),
                        "condition": condition,
                        "input_tokens": int(input_ids.shape[1]),
                        "output_tokens": len(ids),
                        "elapsed_ms": round(elapsed_ms, 3),
                        "output_ids": ids,
                        "rounds": stats.get("rounds"),
                        "accepted_lengths": stats.get("accepted_lengths", []),
                        "tree_build_ms": round(float(stats.get("tree_build_ms", 0.0)), 3),
                        "verify_ms": round(float(stats.get("verify_ms", 0.0)), 3),
                        "cache_ms": round(float(stats.get("cache_ms", 0.0)), 3),
                    }
                )

            target_ids = sample_outputs["target_ar"]
            for record in all_rows[-len(conditions) :]:
                if record["dataset"] == dataset.strip() and record["sample_id"] == sample.get("id", row_index):
                    record["exact_to_target"] = record["output_ids"] == target_ids

    summary_rows = [row for row in all_rows if row["record_type"] == "sample"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((row["dataset"], row["condition"]), []).append(row)
    summary = []
    for (dataset, condition), rows in sorted(grouped.items()):
        latencies = [float(row["elapsed_ms"]) for row in rows]
        summary.append(
            {
                "record_type": "summary",
                "dataset": dataset,
                "condition": condition,
                "n": len(rows),
                "mean_e2e_ms": round(sum(latencies) / len(latencies), 3),
                "median_e2e_ms": round(float(torch.tensor(latencies).median().item()), 3),
                "exact_rate": round(sum(bool(row["exact_to_target"]) for row in rows) / len(rows), 4),
                "mean_output_tokens": round(sum(int(row["output_tokens"]) for row in rows) / len(rows), 2),
            }
        )
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows + summary),
        encoding="utf-8",
    )
    manifest = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "device": "cpu",
        "dtype": "float32",
        "attention_backend": "sdpa",
        "datasets": args.datasets.split(","),
        "samples_per_dataset": args.samples_per_dataset,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "node_budget": args.node_budget,
        "top_k": args.top_k,
        "threads": args.threads,
        "seed": args.seed,
        "peak_rss_gb": round(_rss_gb(), 3),
        "results": summary,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-model",
        default="/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    )
    parser.add_argument(
        "--draft-model",
        default="/home/tuantb/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/61ab4992e5b5ec5913c7f8a9618367b4309533a3",
    )
    parser.add_argument("--datasets", default="gov_report,multi_news,qmsum")
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--node-budget", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "dflash_ddtree_cpu" / "screen.jsonl"),
    )
    args = parser.parse_args()
    manifest = _run(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
