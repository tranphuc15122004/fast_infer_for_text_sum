"""Rebuild full-prefix states and evaluate immutable single-round actions."""

from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from TrainingFree.context_adaptive.cache import DraftContextBank
from TrainingFree.context_adaptive.generation import (
    _clone_dynamic_cache,
    _draft_and_verify_action,
    _feature_rows,
)
from TrainingFree.context_adaptive.types import Action, PromptLayout, Selection
from TrainingFree.context_interventions.attention import (
    MultiLayerTargetQueryCapture,
    target_full_key_scores,
)
from TrainingFree.context_interventions.masks import make_selection


TARGET_SCORING_LAYERS = (1, 9, 17, 25, 33)
TARGET_PARENT_LAYER = 18


@dataclass(frozen=True)
class ReferencePrefix:
    round_index: int
    processed_output_tokens: int
    pending_anchor: int
    remaining_output_tokens: int
    prefix_hash: str
    previous_dense_round_index: int | None


@dataclass
class ReplayState:
    cache: Any
    bank: DraftContextBank
    target_queries: dict[int, torch.Tensor]
    target_scores: dict[int, dict[int, float]]
    output_ids: tuple[int, ...]
    pending_anchor: int
    processed_output_tokens: int
    prefix_hash: str
    bank_build_ms: float
    parent_entropy: float | None = None
    parent_normalized_entropy: float | None = None
    parent_confidence: float | None = None
    parent_probability_margin: float | None = None
    parent_logit_margin: float | None = None


def choose_reference_prefixes(
    rounds: Sequence[Mapping[str, Any]],
    output_ids: Sequence[int],
    *,
    prompt_length: int,
    max_new_tokens: int,
    eos_ids: set[int],
    max_prefixes: int = 6,
) -> list[ReferencePrefix]:
    """Select prefill, second and progress-quantile full-DFlash states."""
    generated = [int(token) for token in output_ids[:max_new_tokens]]
    natural_end = next((index for index, token in enumerate(generated) if token in eos_ids), len(generated))
    candidates: list[tuple[int, int, int | None]] = []
    for row in rounds:
        if row.get("type") != "round" or int(row.get("gamma_executed", 0)) < 15:
            continue
        round_index = int(row["round_index"])
        processed = int(row.get("processed_output_before", int(row["logical_length_before"]) - prompt_length))
        if processed < 0 or processed >= natural_end or processed >= len(generated):
            continue
        if natural_end - processed < 16:
            continue
        # Capped runs require the same remaining horizon as a natural-EOS run.
        if max_new_tokens - processed < 16:
            continue
        previous = [
            int(prior["round_index"])
            for prior in rounds
            if prior.get("type") == "round"
            and int(prior.get("gamma_executed", 0)) >= 15
            and int(prior.get("round_index", -1)) < round_index
        ]
        candidates.append((round_index, processed, max(previous) if previous else None))
    if not candidates:
        return []
    candidates.sort(key=lambda row: row[0])
    chosen: list[tuple[int, int, int | None]] = [candidates[0]]
    if len(candidates) > 1:
        chosen.append(candidates[1])
    for fraction in (0.25, 0.50, 0.75, 0.90):
        target = int(round(fraction * max(natural_end - 1, 0)))
        nearest = min(candidates, key=lambda row: (abs(row[1] - target), row[1], row[0]))
        if nearest not in chosen:
            chosen.append(nearest)
        if len(chosen) >= max_prefixes:
            break
    chosen = sorted(chosen[:max_prefixes], key=lambda row: row[0])
    result = []
    for round_index, processed, previous_round in chosen:
        anchor = generated[processed]
        prefix_hash = hashlib.sha256(
            (",".join(map(str, generated[:processed])) + f"|{anchor}|{round_index}").encode("ascii")
        ).hexdigest()
        result.append(ReferencePrefix(
            round_index=round_index,
            processed_output_tokens=processed,
            pending_anchor=anchor,
            remaining_output_tokens=max_new_tokens - processed,
            prefix_hash=prefix_hash,
            previous_dense_round_index=previous_round,
        ))
    return result


def rebuild_replay_state(
    runtime: Mapping[str, Any],
    input_ids: torch.Tensor,
    reference_output_ids: Sequence[int],
    prefix: ReferencePrefix,
    layout: PromptLayout,
    *,
    max_new_tokens: int,
    target_layers: Sequence[int] = (*TARGET_SCORING_LAYERS, TARGET_PARENT_LAYER),
) -> ReplayState:
    """Rebuild dense target KV and DFlash bank from a causal token prefix."""
    from transformers import DynamicCache

    target = runtime["target"]
    draft = runtime["draft"]
    processed = prefix.processed_output_tokens
    output_prefix = torch.tensor(
        list(reference_output_ids[:processed]), dtype=input_ids.dtype, device=input_ids.device
    ).reshape(1, -1)
    all_ids = torch.cat((input_ids.to(target.device), output_prefix.to(target.device)), dim=1)
    if all_ids.shape[1] < 1:
        raise ValueError("reference prefix must contain at least one target token")
    cache = DynamicCache()
    capture = MultiLayerTargetQueryCapture(target, list(target_layers))
    try:
        capture.enable(last_only=True)
        with torch.inference_mode():
            output = target(
                all_ids,
                position_ids=torch.arange(all_ids.shape[1], dtype=torch.long, device=all_ids.device).reshape(1, -1),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                output_hidden_states=True,
            )
        queries = capture.disable()
        bank = DraftContextBank(draft, int(all_ids.shape[1]) + int(max_new_tokens) + int(draft.block_size))
        with torch.inference_mode():
            features = _feature_rows(output.hidden_states, draft.target_layer_ids)
            bank_start = time.perf_counter()
            bank.append(features, torch.arange(all_ids.shape[1], dtype=torch.long, device=all_ids.device))
            if torch.cuda.is_available():
                torch.cuda.synchronize(all_ids.device)
            bank_ms = (time.perf_counter() - bank_start) * 1000.0
        next_token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
        from TrainingFree.speculation_experiments.signals import summarize_draft_logits

        parent_signal = summarize_draft_logits(output.logits[:, -1:, :])
        if next_token != prefix.pending_anchor:
            raise RuntimeError(
                "reference prefix anchor differs from full target greedy output: "
                f"round={prefix.round_index}, expected={prefix.pending_anchor}, actual={next_token}"
            )
        target_scores: dict[int, dict[int, float]] = {}
        for layer_index in target_layers:
            query = queries.get(int(layer_index))
            if query is None:
                raise RuntimeError(f"target query capture missing configured layer {layer_index}")
            attention = target.model.layers[int(layer_index)].self_attn
            target_scores[int(layer_index)] = target_full_key_scores(
                query,
                cache,
                int(layer_index),
                0,
                all_ids.shape[1] - 1,
                attention,
            )
        prompt_hash = hashlib.sha256(all_ids[0].detach().cpu().numpy().tobytes()).hexdigest()
        del output, features
        return ReplayState(
            cache=cache,
            bank=bank,
            target_queries=queries,
            target_scores=target_scores,
            output_ids=tuple(int(token) for token in reference_output_ids[:processed]),
            pending_anchor=prefix.pending_anchor,
            processed_output_tokens=processed,
            prefix_hash=prompt_hash,
            bank_build_ms=bank_ms,
            parent_entropy=parent_signal["raw_entropy"][0],
            parent_normalized_entropy=parent_signal["normalized_entropy"][0],
            parent_confidence=parent_signal["confidence"][0],
            parent_probability_margin=parent_signal["probability_margin"][0],
            parent_logit_margin=parent_signal["logit_margin"][0],
        )
    finally:
        capture.close()


def mean_target_scoring_vectors(target_scores: Mapping[int, Mapping[int, float]]) -> dict[str, Any]:
    missing = set(TARGET_SCORING_LAYERS) - set(target_scores)
    if missing:
        raise ValueError(f"target attention layers missing: {sorted(missing)}")
    vectors = {f"target_{layer}": dict(target_scores[layer]) for layer in TARGET_SCORING_LAYERS}
    positions = set.intersection(*(set(target_scores[layer]) for layer in TARGET_SCORING_LAYERS))
    if len(positions) != len(target_scores[TARGET_SCORING_LAYERS[0]]):
        raise ValueError("five target scoring vectors do not cover the same causal key positions")
    vectors["target_mean5"] = {
        position: sum(float(target_scores[layer][position]) for layer in TARGET_SCORING_LAYERS) / len(TARGET_SCORING_LAYERS)
        for position in positions
    }
    vectors["target_corresponding"] = [dict(target_scores[layer]) for layer in TARGET_SCORING_LAYERS]
    return vectors


def dense_draft_scores(
    runtime: Mapping[str, Any],
    state: ReplayState,
    layout: PromptLayout,
    *,
    gamma: int = 15,
) -> dict[int, dict[int, float]]:
    """Collect current full-context DFlash scores at one immutable prefix."""
    draft = runtime["draft"]
    device = state.bank.device
    bank_length = state.bank.length
    selection = make_selection(
        [tuple(range(bank_length)) for _ in draft.layers],
        requested_budget="full",
        selector_id="dense_dflash",
    )
    mask_token_id = getattr(draft, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(draft.config, "dflash_config", {}).get("mask_token_id")
    if mask_token_id is None:
        raise ValueError("DFlash mask_token_id is missing")
    block = torch.full(
        (1, gamma + 1), int(mask_token_id), dtype=torch.long, device=device
    )
    block[0, 0] = int(state.pending_anchor)
    logical_length = bank_length
    positions = torch.arange(logical_length, logical_length + gamma + 1, device=device)
    with torch.inference_mode():
        proposal = __import__("TrainingFree.context_adaptive.attention", fromlist=["draft_block"]).draft_block(
            draft,
            runtime["target"],
            state.bank,
            selection,
            block,
            positions,
            Action("full", gamma),
            layout=layout,
            collect_full_context_scores=True,
        )
    return proposal.attention_scores_by_layer


def evaluate_replay_action(
    runtime: Mapping[str, Any],
    input_ids: torch.Tensor,
    state: ReplayState,
    layout: PromptLayout,
    positions_by_layer: Sequence[Sequence[int]],
    *,
    requested_budget: int | str,
    selector_id: str,
    remaining_output_tokens: int,
    gamma: int = 15,
    ranking_origin_round: int | None = None,
    collect_attention_scores: bool = False,
    collect_uncertainty: bool = False,
    cleanup_after_action: bool = True,
) -> dict[str, Any]:
    """Evaluate a single action with a fresh target-cache fork."""
    if remaining_output_tokens < gamma + 1:
        return {
            "status": "infeasible",
            "reason": "fewer_than_gamma_plus_anchor_tokens_remain",
            "accepted_candidates": None,
            "actual_context_tokens_by_layer": [len(row) for row in positions_by_layer],
        }
    draft = runtime["draft"]
    selection = make_selection(
        positions_by_layer,
        requested_budget=requested_budget,
        selector_id=selector_id,
        ranking_origin_round=ranking_origin_round,
    )
    for layer_positions in selection.positions_by_layer:
        if not layer_positions or layer_positions[-1] >= state.bank.length:
            raise ValueError("action mask references an invalid/unprocessed draft-bank position")
    mask_token_id = getattr(draft, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(draft.config, "dflash_config", {}).get("mask_token_id")
    if mask_token_id is None:
        raise ValueError("DFlash mask_token_id is missing")
    action = Action(requested_budget, gamma)
    device = input_ids.device
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    replay_setup_started = time.perf_counter()
    forked_cache = _clone_dynamic_cache(state.cache)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    replay_setup_ms = (time.perf_counter() - replay_setup_started) * 1000.0
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    action_started = time.perf_counter()
    with torch.inference_mode():
        outcome = _draft_and_verify_action(
            target=runtime["target"],
            draft=draft,
            bank=state.bank,
            cache=forked_cache,
            layout=layout,
            input_ids=input_ids.to(device),
            anchor=state.pending_anchor,
            logical_length=state.bank.length,
            action=action,
            selection=selection,
            temperature=0.0,
            mask_token_id=int(mask_token_id),
            collect_refresh_scores=collect_attention_scores,
            collect_full_context_scores=collect_attention_scores,
            collect_uncertainty=collect_uncertainty,
            query_capture=None,
        )
        (
            block,
            target_output,
            accepted,
            _posterior,
            scores,
            draft_ms,
            gather_host_ms,
            gathered_bytes,
            _target_query,
            draft_events,
            verify_event,
            gather_events,
            uncertainty,
            _verification_gamma,
            _verification_decision,
            _verification_policy_ms,
        ) = outcome
        bank_update_started = time.perf_counter()
        committed = 1 + int(accepted)
        retained = _feature_rows(target_output.hidden_states, list(draft.target_layer_ids))[:, :committed, :]
        original_bank_length = state.bank.length
        forked_cache.crop(original_bank_length + committed)
        cache_length_after_crop = int(forked_cache.get_seq_length())
        if cache_length_after_crop != original_bank_length + committed:
            raise RuntimeError(
                f"replay target cache crop mismatch: {cache_length_after_crop} != {original_bank_length + committed}"
            )
        state.bank.append(
            retained,
            torch.arange(original_bank_length, original_bank_length + committed, device=device),
        )
        # This is a diagnostic fork. Reuse the immutable state while keeping the
        # appended storage beyond the active prefix unreachable to later actions.
        state.bank.length = original_bank_length
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    bank_update_ms = (time.perf_counter() - bank_update_started) * 1000.0
    if state.bank.length != original_bank_length:
        raise RuntimeError("replay draft-bank logical length changed after diagnostic action")
    action_wall_ms = (time.perf_counter() - action_started) * 1000.0
    gather_ms = (
        float(gather_events[0].elapsed_time(gather_events[1]))
        if gather_events is not None and torch.cuda.is_available()
        else float(gather_host_ms)
    )
    draft_component_ms = (
        float(draft_events[0].elapsed_time(draft_events[1]))
        if draft_events and draft_events[0] is not None and torch.cuda.is_available()
        else float(draft_ms)
    )
    verify_component_ms = (
        float(verify_event[0].elapsed_time(verify_event[1]))
        if verify_event and verify_event[0] is not None and torch.cuda.is_available()
        else float(verify_event[2])
    )
    row = {
        "status": "ok",
        "accepted_candidates": int(accepted),
        "accepted_prefix": int(accepted),
        "proposed_candidates": gamma,
        "delta_accepted": None,
        "actual_context_tokens_by_layer": [len(value) for value in selection.positions_by_layer],
        "positions_sha256_by_layer": [
            hashlib.sha256(",".join(map(str, value)).encode("ascii")).hexdigest()
            for value in selection.positions_by_layer
        ],
        "ranking_origin_round": ranking_origin_round,
        "draft_ms": draft_component_ms,
        "gather_ms": gather_ms,
        "verify_ms": verify_component_ms,
        "bank_update_ms": bank_update_ms,
        "cache_length_after_crop": cache_length_after_crop,
        "action_wall_ms": action_wall_ms,
        "replay_setup_ms": replay_setup_ms,
        "gather_bytes": int(gathered_bytes),
        "proposed_token_ids": [int(token) for token in block[0, 1:].detach().cpu().tolist()],
        "attention_scores_by_layer": scores,
        "draft_uncertainty": (
            {key: list(value) for key, value in uncertainty.items() if key != "signal_extraction_ms"}
            if uncertainty is not None else None
        ),
        "signal_extraction_ms": (
            float(uncertainty["signal_extraction_ms"]) if uncertainty is not None else None
        ),
        "parent_signal": {
            "entropy": state.parent_entropy,
            "normalized_entropy": state.parent_normalized_entropy,
            "confidence": state.parent_confidence,
            "probability_margin": state.parent_probability_margin,
            "logit_margin": state.parent_logit_margin,
        },
    }
    del retained, target_output, block, outcome
    if cleanup_after_action:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row
