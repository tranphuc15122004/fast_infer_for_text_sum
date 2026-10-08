"""Exact target verification with adaptive DFlash context and block length."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from transformers import DynamicCache

from .attention import draft_block
from .cache import DraftContextBank
from .config import AdaptiveConfig
from .controller import AdaptiveController
from .selection import rank_chunks, select_context
from .signals import TargetQueryCapture, normalized_entropy, refresh_concentration, target_parent_scores
from .statistics import AdaptiveStatistics, state_bucket
from .types import Action, GenerationResult, PromptLayout, RoundState, Selection


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


def _feature_rows(hidden_states: Sequence[torch.Tensor], layer_ids: Sequence[int]) -> torch.Tensor:
    return torch.cat([hidden_states[int(index) + 1] for index in layer_ids], dim=-1)


def _eos_ids(target: Any) -> set[int]:
    value = getattr(target.config, "eos_token_id", None)
    if value is None:
        value = getattr(target.generation_config, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def _capture_parent(
    query: torch.Tensor | None,
    cache: Any,
    layer_index: int,
    query_index: int,
    query_position: int,
    layout: PromptLayout,
    attention: Any,
) -> tuple[dict[int, float] | None, float | None]:
    if query is None or not layout.source_chunks:
        return None, None
    scores, concentration = target_parent_scores(
        query,
        cache,
        layer_index,
        query_index,
        query_position,
        layout.source_chunks,
        head_groups=int(attention.num_key_value_groups),
        scaling=float(attention.scaling),
        sliding_window=getattr(attention, "sliding_window", None),
    )
    return scores, concentration


def _action_grid(config: AdaptiveConfig, bank_length: int, max_gamma: int) -> tuple[Action, ...]:
    actions: dict[str, Action] = {}
    for budget in config.budgets:
        canonical_budget: int | str = "full" if budget == "full" or int(budget) >= bank_length else int(budget)
        for gamma in config.gammas:
            if gamma <= max_gamma:
                action = Action(canonical_budget, gamma, config.length_mode)
                actions[action.key()] = action
    return tuple(actions.values())


def _state_hash(state: RoundState, action: Action, prompt_hash: str = "") -> str:
    text = "|".join(
        (
            str(state.logical_length),
            str(state.parent_entropy),
            str(state.source_concentration),
            str(state.history_acceptance),
            str(state.refresh_required),
            action.key(),
            prompt_hash,
        )
    )
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _complete_round_schema(
    rounds: list[dict[str, Any]],
    *,
    run_id: str,
    sample_id: str,
    dataset: str,
    split: str,
    variant: str,
    repetition: int,
) -> None:
    """Fill invariant identity/accounting fields for speculative and AR rows."""
    for index, row in enumerate(rounds):
        gamma = int(row.get("gamma_executed", 0))
        before = int(row.get("logical_length_before", 0))
        commits = int(row.get("processed_commits", 1))
        defaults = {
            "schema_version": "cadflash.round.v1",
            "type": "round",
            "status": "ok",
            "run_id": run_id,
            "sample_id": sample_id,
            "dataset": dataset,
            "split": split,
            "variant": variant,
            "repetition": repetition,
            "round_index": index,
            "logical_length_before": before,
            "processed_output_before": 0,
            "pending_anchor_position": before,
            "parent_query_origin_round": None,
            "parent_query_index": 0,
            "parent_entropy": None,
            "entropy_signal_temperature": 1.0,
            "source_concentration": None,
            "history_acceptance": None,
            "ranking_origin_round": None,
            "ranking_age": None,
            "requested_budget": "full",
            "selected_context_tokens_by_layer": [],
            "physical_context_tokens_by_layer": [],
            "gamma_requested": gamma,
            "gamma_executed": gamma,
            "block_size": gamma + 1,
            "length_mode": "autoregressive" if gamma == 0 else "draft_shape",
            "selector_id": "none",
            "refresh_required": False,
            "action_reason": "target_only_step" if gamma == 0 else "draft_action",
            "accepted_candidates": 0,
            "processed_commits": commits,
            "logical_length_after": before + commits,
            "all_candidates_accepted": False,
            "prefix_right_censored": False,
            "eos_offset": None,
            "boundary_round": gamma == 0,
            "trimmed_commits": 0,
            "signal_ms": 0.0,
            "controller_ms": 0.0,
            "selection_ms": 0.0,
            "bank_update_ms": 0.0,
            "draft_ms": 0.0,
            "verify_ms": 0.0,
            "round_gpu_span_ms": None,
            "round_host_ms": 0.0,
            "gather_bytes": 0,
            "cost_observation_ready": False,
            "predicted_cost_per_commit": None,
            "fallback_reason": None,
            "wasted_work_ms": 0.0,
        }
        for key, value in defaults.items():
            row.setdefault(key, value)


def _verify_one(
    target: Any,
    cache: Any,
    token_ids: torch.Tensor,
    positions: torch.Tensor,
) -> Any:
    return target(
        token_ids,
        position_ids=positions.reshape(1, -1),
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
    )


def _draft_and_verify_action(
    *,
    target: Any,
    draft: Any,
    bank: DraftContextBank,
    cache: Any,
    layout: PromptLayout,
    input_ids: torch.Tensor,
    anchor: int,
    logical_length: int,
    action: Action,
    selection: Selection,
    temperature: float,
    mask_token_id: int,
    collect_refresh_scores: bool,
    query_capture: TargetQueryCapture | None,
) -> tuple[torch.Tensor, Any, int, torch.Tensor, dict[int, dict[int, float]], float, float, torch.Tensor | None, Any, Any]:
    device = input_ids.device
    block = torch.full((1, action.block_size), mask_token_id, dtype=input_ids.dtype, device=device)
    block[0, 0] = int(anchor)
    position_ids = torch.arange(logical_length, logical_length + action.block_size, dtype=torch.long, device=device)
    draft_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    draft_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if draft_start is not None:
        draft_start.record()
    proposal = draft_block(
        draft,
        target,
        bank,
        selection,
        block,
        position_ids,
        action,
        layout=layout,
        collect_refresh_scores=collect_refresh_scores,
    )
    block[0, 1:] = proposal.token_ids[0]
    if draft_end is not None:
        draft_end.record()
    if query_capture is not None:
        query_capture.enable(last_only=False)
    verify_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    verify_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if verify_start is not None:
        verify_start.record()
    verify_started = time.perf_counter()
    target_output = _verify_one(target, cache, block, position_ids)
    verify_wall_ms = (time.perf_counter() - verify_started) * 1000.0
    if verify_end is not None:
        verify_end.record()
    target_query = query_capture.disable() if query_capture is not None else None
    posterior = _sample(target_output.logits, temperature)
    matches = (block[:, 1:] == posterior[:, :-1]).to(torch.int32)
    accepted = int(torch.cumprod(matches, dim=1).sum().item())
    return (
        block,
        target_output,
        accepted,
        posterior,
        proposal.attention_scores_by_layer,
        proposal.draft_latency_ms,
        proposal.gather_bytes,
        target_query,
        (draft_start, draft_end),
        (verify_start, verify_end, verify_wall_ms),
    )


def _target_prefill(
    target: Any,
    input_ids: torch.Tensor,
    query_capture: TargetQueryCapture | None,
) -> tuple[Any, torch.Tensor | None, float]:
    cache = DynamicCache()
    if query_capture is not None:
        query_capture.enable(last_only=True)
    started = time.perf_counter()
    output = target(
        input_ids,
        position_ids=torch.arange(input_ids.shape[1], device=input_ids.device).reshape(1, -1),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        # The DFlash context bank is required for every selector, not only
        # when the selected target layer is instrumented.
        output_hidden_states=True,
    )
    query = query_capture.disable() if query_capture is not None else None
    return (cache, output), query, (time.perf_counter() - started) * 1000.0


def _clone_dynamic_cache(cache: Any) -> Any:
    """Fork the current target prefix for an isolated action replay."""
    from transformers import DynamicCache

    rows = []
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise TypeError("calibration replay requires initialized dense DynamicCache layers")
        rows.append((keys.clone(), values.clone()))
    return DynamicCache(ddp_cache_data=rows)


def _replay_calibration_actions(
    *,
    rows: list[dict[str, Any]],
    actions: Sequence[Action],
    state: RoundState,
    target: Any,
    draft: Any,
    bank: DraftContextBank,
    cache: Any,
    layout: PromptLayout,
    input_ids: torch.Tensor,
    anchor: int,
    logical_length: int,
    remaining: int,
    rankings: Sequence[Sequence[int]],
    config: AdaptiveConfig,
    tokenizer: Any,
    generated: Sequence[int],
    eos_ids: set[int],
    temperature: float,
    mask_token_id: int,
    ranking_origin_round: int | None,
    query_capture: TargetQueryCapture | None,
    target_layer: int,
    target_attention: Any,
) -> None:
    """Replay feasible actions against one immutable causal prefix.

    This is an offline calibration-only path. Each action receives a cloned
    target cache; the live trajectory and full draft bank remain unchanged.
    """
    for action in actions:
        if action.gamma > remaining - 1:
            continue
        selection_started = time.perf_counter()
        selection = select_context(
            state,
            layout,
            rankings,
            action.budget,
            processed_output_start=layout.prompt_tokens,
            bank_length=bank.length,
            source_anchors=config.source_anchors,
            recent_output=config.recent_output,
            selector_id=config.selector,
            ranking_origin_round=ranking_origin_round,
        )
        selection_ms = (time.perf_counter() - selection_started) * 1000.0
        if selection is None:
            continue
        setup_started = time.perf_counter()
        cloned_cache = _clone_dynamic_cache(cache)
        replay_setup_ms = (time.perf_counter() - setup_started) * 1000.0
        if config.timing_mode == "diagnostic" and torch.cuda.is_available():
            torch.cuda.synchronize(input_ids.device)
        started = time.perf_counter()
        block, output, accepted, posterior, draft_scores, _, gather_bytes, query, _, _ = _draft_and_verify_action(
            target=target,
            draft=draft,
            bank=bank,
            cache=cloned_cache,
            layout=layout,
            input_ids=input_ids,
            anchor=anchor,
            logical_length=logical_length,
            action=action,
            selection=selection,
            temperature=temperature,
            mask_token_id=mask_token_id,
            collect_refresh_scores=state.refresh_required and config.selector == "draft_refresh",
            query_capture=query_capture,
        )
        eos_offset = next(
            (index for index, token in enumerate(block[0, 1 : accepted + 1].tolist()) if int(token) in eos_ids),
            None,
        )
        committed = accepted if eos_offset is None else eos_offset + 1
        retained = _feature_rows(output.hidden_states, list(draft.target_layer_ids))[:, : 1 + committed, :]
        bank_length = bank.length
        bank.append(retained, torch.arange(logical_length, logical_length + 1 + committed, device=input_ids.device))
        bank.length = bank_length
        if query is not None and eos_offset is None:
            _capture_parent(
                query,
                cloned_cache,
                target_layer,
                accepted,
                logical_length + accepted,
                layout,
                target_attention,
            )
        if state.refresh_required and draft_scores:
            refresh_concentration(draft_scores, layout.source_chunks)
        if config.timing_mode == "diagnostic" and torch.cuda.is_available():
            torch.cuda.synchronize(input_ids.device)
        elapsed_ms = selection_ms + (time.perf_counter() - started) * 1000.0
        row = {
            "type": "calibration_action",
            "status": "ok",
            "round_index": state.round_index,
            "state_hash": _state_hash(state, action, layout.prompt_hash),
            "logical_length": state.logical_length,
            "calibration_checkpoint": state.processed_output_tokens,
            "prompt_length": layout.prompt_tokens,
            "remaining_output_tokens": remaining,
            "parent_entropy": state.parent_entropy,
            "source_concentration": state.source_concentration,
            "history_acceptance": state.history_acceptance,
            "refresh_required": state.refresh_required,
            "ranking_age": state.ranking_age,
            "source_origin_round": state.source_origin_round,
            "requested_budget": action.budget,
            "gamma_executed": action.gamma,
            "length_mode": action.length_mode,
            "accepted_candidates": committed,
            "verifier_accepted_candidates": accepted,
            "all_candidates_accepted": accepted == action.gamma,
            "prefix_right_censored": accepted == action.gamma and eos_offset is None,
            "eos_offset": eos_offset,
            "selected_context_tokens_by_layer": [len(item) for item in selection.positions_by_layer],
            "gather_bytes": gather_bytes,
            "action_cost_ms": elapsed_ms,
            "round_total_ms": elapsed_ms,
            "selection_ms": selection_ms,
            "replay_setup_ms": replay_setup_ms,
            "timing_mode": config.timing_mode,
            "selector_id": config.selector,
            "target_cache_fork_bytes": sum(
                int(layer.keys.numel() + layer.values.numel()) * layer.keys.element_size()
                for layer in cache.layers
            ),
        }
        rows.append(row)
        del output, retained, cloned_cache, query, block, posterior, draft_scores


@torch.inference_mode()
def generate_adaptive(
    target: Any,
    draft: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    layout: PromptLayout,
    config: AdaptiveConfig,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    statistics: AdaptiveStatistics | None = None,
    variant: str = "dflash_full_fixed",
    fixed_budget: int | str = "full",
    fixed_gamma: int = 15,
    gamma_reference: int = 15,
    run_id: str = "local",
    sample_id: str = "sample",
    dataset: str = "unknown",
    split: str = "dev",
    repetition: int = 0,
    stop_token_ids: Sequence[int] | None = None,
    calibration_action_rows: list[dict[str, Any]] | None = None,
    calibration_checkpoints: Sequence[int] = (0, 512, 1024, 1536),
) -> GenerationResult:
    """Generate one sequence; target sees full context on every verification."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Context-Adaptive DFlash V1 supports batch size one")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    input_ids = input_ids.to(target.device)
    prompt_length = input_ids.shape[1]
    max_length = prompt_length + max_new_tokens
    layer_ids = list(draft.target_layer_ids)
    if not layer_ids or min(layer_ids) < 0 or max(layer_ids) >= int(target.config.num_hidden_layers):
        raise ValueError("DFlash target_layer_ids are incompatible with target checkpoint")
    if max(config.gammas) > int(draft.block_size) - 1:
        raise ValueError(
            f"configured gamma {max(config.gammas)} exceeds checkpoint capability {int(draft.block_size) - 1}"
        )
    if fixed_gamma not in config.gammas or gamma_reference not in config.gammas:
        raise ValueError("fixed_gamma and gamma_reference must belong to the configured gamma grid")
    if fixed_budget != "full" and fixed_budget not in config.budgets:
        raise ValueError("fixed_budget must belong to the configured context-budget grid")
    target_layer = target.config.num_hidden_layers // 2 if config.target_layer == "middle" else int(config.target_layer)
    if target_layer >= len(target.model.layers):
        raise ValueError("target attention layer is outside target model")
    target_attention = target.model.layers[target_layer].self_attn
    if fixed_gamma > int(draft.block_size) - 1 or gamma_reference > int(draft.block_size) - 1:
        raise ValueError("fixed_gamma/gamma_reference exceeds DFlash checkpoint capability")
    bank = DraftContextBank(draft, max_length)
    query_capture = TargetQueryCapture(target, target_layer) if config.selector == "target_parent" else None
    capture_cleanup = query_capture.close if query_capture is not None else (lambda: None)
    controller = AdaptiveController(
        config,
        statistics,
        variant=variant,
        fixed_budget=fixed_budget,
        fixed_gamma=fixed_gamma,
        gamma_reference=gamma_reference,
    )
    eos_ids = _eos_ids(target) if stop_token_ids is None else {int(token) for token in stop_token_ids}
    mask_token_id = getattr(draft, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(draft.config, "dflash_config", {}).get("mask_token_id")
    if mask_token_id is None or not 0 <= int(mask_token_id) < int(target.config.vocab_size):
        capture_cleanup()
        raise ValueError("DFlash mask_token_id is missing or outside the target vocabulary")

    rounds: list[dict[str, Any]] = []
    draft_events: list[tuple[Any, Any]] = []
    verify_events: list[tuple[Any, Any]] = []
    gpu_round_events: list[tuple[Any, Any]] = []
    generated: list[int] = []
    processed_output_tokens = 0
    processed_commits_total = 0
    trimmed_commits_total = 0
    proposed_total = accepted_total = useful_accepted_total = speculative_rounds = 0
    signal_ms_total = selection_ms_total = bank_update_ms_total = 0.0
    draft_ms_wall = verify_ms_wall = 0.0
    fallback_rounds = dense_refresh_rounds = 0
    draft_context_total = gamma_total = 0
    prefill_ms = 0.0
    ttft_ms = 0.0
    draft_prefill_ms = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
        torch.cuda.reset_peak_memory_stats(input_ids.device)
    start_time = time.perf_counter()

    ranking_by_layer: tuple[tuple[int, ...], ...] = ()
    target_scores: dict[int, float] | None = None
    source_concentration: float | None = None
    source_origin_round: int | None = None
    ranking_origin_round: int | None = None
    ranking_scores_by_layer: dict[int, dict[int, float]] = {}
    calibration_points = tuple(sorted({int(value) for value in calibration_checkpoints if int(value) >= 0}))
    calibration_checkpoint_index = 0
    parent_entropy: float | None = None
    parent_entropy_origin_round: int | None = None
    parent_query_index_for_state: int | None = None
    initial_query: torch.Tensor | None = None
    try:
        prefill, initial_query, prefill_ms = _target_prefill(target, input_ids, query_capture)
        target_cache, prefill_output = prefill
        prefill_logits = prefill_output.logits[:, -1, :]
        anchor_tensor = _sample(prefill_logits, temperature).reshape(1, 1)
        anchor = int(anchor_tensor.item())
        ttft_ms = (time.perf_counter() - start_time) * 1000.0
        prefill_ms = ttft_ms
        parent_entropy = normalized_entropy(prefill_logits, config.entropy_signal_temperature)
        parent_entropy_origin_round = -1
        parent_query_index_for_state = prompt_length - 1
        prefill_features = _feature_rows(prefill_output.hidden_states, layer_ids)
        bank_started = time.perf_counter()
        bank.append(prefill_features, torch.arange(prompt_length, device=input_ids.device))
        draft_prefill_ms = (time.perf_counter() - bank_started) * 1000.0
        del prefill_output, prefill_features
        generated = [anchor]
        if query_capture is not None:
            target_scores, source_concentration = _capture_parent(
                initial_query, target_cache, target_layer, 0, prompt_length - 1, layout, target_attention
            )
            if target_scores:
                ranking_by_layer = rank_chunks(
                    layout,
                    selector="target_parent",
                    scores_by_layer=[target_scores],
                    num_draft_layers=len(draft.layers),
                )
                source_origin_round = -1

        round_index = 0
        stopped_by = "max_new_tokens"
        while processed_output_tokens < max_new_tokens:
            if generated and generated[processed_output_tokens] in eos_ids:
                stopped_by = "eos"
                break
            remaining = max_new_tokens - processed_output_tokens
            if remaining <= 1:
                break
            logical_length = prompt_length + processed_output_tokens
            refresh_required = config.selector == "draft_refresh" and (
                round_index == 0 or round_index % config.refresh_period == 0
            )
            max_gamma = min(int(draft.block_size) - 1, remaining - 1)
            actions = _action_grid(config, bank.length, max_gamma)
            if refresh_required:
                actions = tuple(action for action in actions if action.budget == "full")
            if not actions:
                # AR boundary step: process the pending anchor with the full
                # target and let it produce the next target-selected token.
                boundary_parent_entropy = parent_entropy
                boundary_parent_origin = parent_entropy_origin_round
                boundary_parent_query_index = parent_query_index_for_state
                token = torch.tensor([[generated[processed_output_tokens]]], dtype=input_ids.dtype, device=input_ids.device)
                position = torch.tensor([logical_length], dtype=torch.long, device=input_ids.device)
                if query_capture is not None:
                    query_capture.enable(last_only=False)
                ar_started = time.perf_counter()
                ar_output = target(
                    token,
                    position_ids=position.reshape(1, 1),
                    past_key_values=target_cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
                ar_query = query_capture.disable() if query_capture is not None else None
                logits = ar_output.logits[:, -1, :]
                next_anchor = int(_sample(logits, temperature).reshape(-1)[0].item())
                features = _feature_rows(ar_output.hidden_states, layer_ids)
                bank.append(features, position)
                ar_elapsed = (time.perf_counter() - ar_started) * 1000.0
                processed_output_tokens += 1
                processed_commits_total += 1
                if processed_output_tokens < max_new_tokens:
                    generated.append(next_anchor)
                if query_capture is not None and ar_query is not None:
                    target_scores, source_concentration = _capture_parent(
                        ar_query, target_cache, target_layer, 0, logical_length, layout, target_attention
                    )
                    if target_scores:
                        ranking_by_layer = rank_chunks(layout, selector="target_parent", scores_by_layer=[target_scores], num_draft_layers=len(draft.layers))
                        source_origin_round = round_index
                    else:
                        ranking_by_layer = ()
                        source_origin_round = None
                parent_entropy = normalized_entropy(logits, config.entropy_signal_temperature)
                parent_entropy_origin_round = round_index
                parent_query_index_for_state = 0
                rounds.append({
                    "type": "round", "status": "ok", "variant": variant,
                    "sample_id": sample_id, "dataset": dataset, "split": split,
                    "repetition": repetition, "round_index": round_index,
                    "logical_length_before": logical_length,
                    "processed_output_before": processed_output_tokens - 1,
                    "pending_anchor_position": logical_length,
                    "parent_query_origin_round": boundary_parent_origin,
                    "parent_query_index": boundary_parent_query_index,
                    "next_parent_entropy": parent_entropy,
                    "requested_budget": "full", "selected_context_tokens_by_layer": [bank.length] * len(draft.layers),
                    "physical_context_tokens_by_layer": [bank.length] * len(draft.layers),
                    "gamma_requested": 0, "gamma_executed": 0, "block_size": 1,
                    "length_mode": "autoregressive", "selector_id": config.selector,
                    "refresh_required": False, "action_reason": "boundary_ar",
                    "accepted_candidates": 0, "processed_commits": 1,
                    "logical_length_after": logical_length + 1, "all_candidates_accepted": False,
                    "prefix_right_censored": False, "eos_offset": None, "boundary_round": True,
                    "trimmed_commits": 0, "signal_ms": 0.0, "controller_ms": 0.0,
                    "selection_ms": 0.0, "bank_update_ms": 0.0, "draft_ms": 0.0,
                    "verify_ms": ar_elapsed, "round_gpu_span_ms": None,
                    "round_host_ms": ar_elapsed, "gather_bytes": 0,
                    "cost_observation_ready": False, "predicted_cost_per_commit": None,
                    "fallback_reason": "no_supported_boundary_shape", "wasted_work_ms": 0.0,
                    "parent_entropy": boundary_parent_entropy,
                    "source_concentration": source_concentration,
                    "history_acceptance": controller.history_acceptance,
                })
                round_index += 1
                if next_anchor in eos_ids:
                    stopped_by = "eos"
                    break
                continue

            round_started = time.perf_counter()
            state = RoundState(
                round_index=round_index,
                logical_length=logical_length,
                processed_output_tokens=processed_output_tokens,
                remaining_output_tokens=remaining,
                parent_entropy=parent_entropy,
                source_concentration=source_concentration,
                history_acceptance=controller.history_acceptance,
                ranking_age=(round_index - ranking_origin_round) if ranking_origin_round is not None else 10**9,
                refresh_required=refresh_required,
                source_origin_round=source_origin_round,
            )
            state_parent_entropy_origin = parent_entropy_origin_round
            state_parent_query_index = parent_query_index_for_state
            action, action_reason, controller_ms = controller.choose(state, actions)
            if refresh_required and action.budget != "full":
                action = Action("full", action.gamma, action.length_mode)
                action_reason += ":refresh_full"

            if config.selector == "target_parent":
                rankings = ranking_by_layer
                source_scores_for_refresh = None
            elif config.selector == "draft_refresh":
                rankings = rank_chunks(
                    layout,
                    selector="draft_refresh",
                    scores_by_layer=[ranking_scores_by_layer[index] for index in sorted(ranking_scores_by_layer)] if ranking_scores_by_layer else None,
                    num_draft_layers=len(draft.layers),
                ) if ranking_scores_by_layer else ()
                source_scores_for_refresh = ranking_scores_by_layer
            else:
                rankings = rank_chunks(
                    layout,
                    selector=config.selector,
                    features=bank.features,
                    output_ids=generated,
                    last_processed_position=logical_length - 1 if logical_length > prompt_length else None,
                    num_draft_layers=len(draft.layers),
                    seed_key=f"{config.seed}:{layout.prompt_hash}:{round_index}",
                    source_token_ids=input_ids[0].tolist(),
                    tokenizer=tokenizer,
                )
                source_scores_for_refresh = None
            if not rankings:
                rankings = tuple(tuple(range(len(layout.source_chunks))) for _ in range(len(draft.layers)))
            if config.selector == "target_parent" and not ranking_by_layer:
                action = Action("full", action.gamma, action.length_mode)
                action_reason += ":missing_target_signal_full"
            selection_started = time.perf_counter()
            selection = select_context(
                state,
                layout,
                rankings,
                action.budget,
                processed_output_start=prompt_length,
                bank_length=bank.length,
                source_anchors=config.source_anchors,
                recent_output=config.recent_output,
                selector_id=config.selector,
                ranking_origin_round=(ranking_origin_round if config.selector == "draft_refresh" else source_origin_round),
            )
            if selection is None:
                fallback = Action("full", action.gamma, action.length_mode)
                selection = select_context(
                    state,
                    layout,
                    rankings,
                    "full",
                    processed_output_start=prompt_length,
                    bank_length=bank.length,
                    source_anchors=config.source_anchors,
                    recent_output=config.recent_output,
                    selector_id=config.selector,
                    ranking_origin_round=ranking_origin_round,
                )
                action = fallback
                fallback_rounds += 1
                action_reason += ":protection_or_budget_fallback"
            selection_ms = (time.perf_counter() - selection_started) * 1000.0
            if refresh_required:
                dense_refresh_rounds += 1

            collect_target_signal = query_capture is not None and round_index % config.signal_update_period == 0
            calibration_due = False
            while (
                calibration_checkpoint_index < len(calibration_points)
                and processed_output_tokens >= calibration_points[calibration_checkpoint_index]
            ):
                calibration_due = True
                calibration_checkpoint_index += 1
            if calibration_action_rows is not None and calibration_due:
                _replay_calibration_actions(
                    rows=calibration_action_rows,
                    actions=actions,
                    state=state,
                    target=target,
                    draft=draft,
                    bank=bank,
                    cache=target_cache,
                    layout=layout,
                    input_ids=input_ids,
                    anchor=generated[processed_output_tokens],
                    logical_length=logical_length,
                    remaining=remaining,
                    rankings=rankings,
                    config=config,
                    tokenizer=tokenizer,
                    generated=generated,
                    eos_ids=eos_ids,
                    temperature=temperature,
                    mask_token_id=int(mask_token_id),
                    ranking_origin_round=(ranking_origin_round if config.selector == "draft_refresh" else source_origin_round),
                    query_capture=query_capture if collect_target_signal else None,
                    target_layer=target_layer,
                    target_attention=target_attention,
                )

            if config.timing_mode == "diagnostic" and torch.cuda.is_available():
                torch.cuda.synchronize(input_ids.device)
            gpu_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            gpu_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if gpu_start is not None:
                gpu_start.record()
            (
                block,
                target_output,
                accepted,
                posterior,
                draft_scores,
                draft_ms,
                gathered_bytes,
                target_query,
                draft_event_pair,
                verify_event_data,
            ) = _draft_and_verify_action(
                target=target,
                draft=draft,
                bank=bank,
                cache=target_cache,
                layout=layout,
                input_ids=input_ids,
                anchor=generated[processed_output_tokens],
                logical_length=logical_length,
                action=action,
                selection=selection,
                temperature=temperature,
                mask_token_id=int(mask_token_id),
                collect_refresh_scores=refresh_required and config.selector == "draft_refresh",
                query_capture=query_capture if collect_target_signal else None,
            )
            proposed_total += action.gamma
            speculative_rounds += 1
            gamma_total += action.gamma
            draft_context_total += sum(len(positions) for positions in selection.positions_by_layer) / len(selection.positions_by_layer)

            eos_candidate_index = next(
                (index for index, token in enumerate(block[0, 1 : accepted + 1].tolist()) if int(token) in eos_ids),
                None,
            )
            accepted_for_commit = accepted if eos_candidate_index is None else eos_candidate_index + 1
            accepted_total += accepted
            useful_accepted_total += accepted_for_commit
            commits_for_output = 1 + accepted_for_commit
            original_commits = 1 + accepted
            trimmed = original_commits - commits_for_output
            trimmed_commits_total += trimmed
            candidate_ids = [int(token) for token in block[0, 1 : accepted_for_commit + 1].tolist()]
            generated.extend(candidate_ids)
            processed_output_tokens += commits_for_output
            processed_commits_total += original_commits
            logical_after = logical_length + commits_for_output

            pending_anchor = int(posterior[0, accepted].item())
            eos_in_accepted = eos_candidate_index is not None
            if eos_in_accepted:
                pending_anchor = -1
            signal_started = time.perf_counter()
            parent_logits = target_output.logits[:, accepted, :]
            parent_entropy = normalized_entropy(parent_logits, config.entropy_signal_temperature)
            parent_entropy_origin_round = round_index
            parent_query_index_for_state = accepted
            if target_query is not None and not eos_in_accepted:
                target_scores, source_concentration = _capture_parent(
                    target_query,
                    target_cache,
                    target_layer,
                    accepted,
                    logical_length + accepted,
                    layout,
                    target_attention,
                )
                if target_scores:
                    ranking_by_layer = rank_chunks(
                        layout,
                        selector="target_parent",
                        scores_by_layer=[target_scores],
                        num_draft_layers=len(draft.layers),
                    )
                    source_origin_round = round_index
                else:
                    # A query with no source attention is missing evidence;
                    # do not keep an older ranking and present it as current.
                    ranking_by_layer = ()
                    source_origin_round = None
            signal_ms = (time.perf_counter() - signal_started) * 1000.0
            bank_started = time.perf_counter()
            retained = _feature_rows(target_output.hidden_states, layer_ids)[:, :commits_for_output, :]
            target_cache.crop(logical_after)
            bank.append(
                retained,
                torch.arange(logical_length, logical_after, device=input_ids.device),
            )
            bank_update_ms = (time.perf_counter() - bank_started) * 1000.0
            del retained
            if gpu_end is not None:
                gpu_end.record()

            if not eos_in_accepted and processed_output_tokens < max_new_tokens:
                generated.append(pending_anchor)
            pending_eos = (
                not eos_in_accepted
                and processed_output_tokens < max_new_tokens
                and pending_anchor in eos_ids
            )
            all_accepted = accepted == action.gamma
            state_hash = _state_hash(state, action, layout.prompt_hash)
            action_cost_ms = None
            round_total_ms = None
            if config.timing_mode == "diagnostic" and torch.cuda.is_available():
                torch.cuda.synchronize(input_ids.device)
                round_total_ms = (time.perf_counter() - round_started) * 1000.0
                action_cost_ms = max(0.0, round_total_ms - controller_ms)
            controller.observe(
                state,
                action,
                accepted_for_commit,
                action_cost_ms if config.cost_update_mode == "async_event" else None,
            )

            if refresh_required and draft_scores:
                ranking_scores_by_layer = draft_scores
                ranking_origin_round = round_index
                source_origin_round = round_index
                source_concentration = refresh_concentration(draft_scores, layout.source_chunks)
                ranking_by_layer = rank_chunks(
                    layout,
                    selector="draft_refresh",
                    scores_by_layer=[draft_scores[index] for index in sorted(draft_scores)],
                    num_draft_layers=len(draft.layers),
                )

            selection_ms_total += selection_ms
            signal_ms_total += signal_ms
            bank_update_ms_total += bank_update_ms
            draft_ms_wall += draft_ms
            if gpu_start is not None and gpu_end is not None:
                gpu_round_events.append((gpu_start, gpu_end))
            if draft_event_pair[0] is not None:
                draft_events.append(draft_event_pair)
            if verify_event_data[0] is not None:
                verify_events.append((verify_event_data[0], verify_event_data[1]))
            host_elapsed = (time.perf_counter() - round_started) * 1000.0
            rounds.append({
                "type": "round", "schema_version": "cadflash.round.v1", "status": "ok",
                "run_id": run_id, "sample_id": sample_id, "dataset": dataset, "split": split,
                "variant": variant, "repetition": repetition, "round_index": round_index,
                "logical_length_before": logical_length,
                "processed_output_before": logical_length - prompt_length,
                "pending_anchor_position": logical_length,
                "parent_query_origin_round": state_parent_entropy_origin,
                "parent_query_index": state_parent_query_index,
                "parent_entropy": state.parent_entropy,
                "next_parent_entropy": parent_entropy,
                "entropy_signal_temperature": config.entropy_signal_temperature,
                "source_concentration": state.source_concentration,
                "history_acceptance": state.history_acceptance,
                "ranking_origin_round": ranking_origin_round,
                "ranking_age": state.ranking_age,
                "requested_budget": selection.requested_budget,
                "selected_context_tokens_by_layer": [len(x) for x in selection.positions_by_layer],
                "physical_context_tokens_by_layer": [len(x) for x in selection.positions_by_layer],
                "gamma_requested": action.gamma, "gamma_executed": action.gamma,
                "block_size": action.block_size, "length_mode": action.length_mode,
                "selector_id": config.selector, "refresh_required": refresh_required,
                "action_reason": action_reason,
                "accepted_candidates": accepted,
                "emitted_accepted_candidates": accepted_for_commit,
                "processed_commits": original_commits,
                "logical_length_after": logical_after, "all_candidates_accepted": all_accepted,
                "prefix_right_censored": all_accepted and eos_candidate_index is None,
                "eos_offset": eos_candidate_index,
                "boundary_round": remaining <= max(action.gamma + 1, 1),
                "trimmed_commits": trimmed, "signal_ms": signal_ms,
                "controller_ms": controller_ms, "selection_ms": selection_ms,
                "bank_update_ms": bank_update_ms, "draft_ms": draft_ms,
                "verify_ms": verify_event_data[2], "round_gpu_span_ms": None,
                "round_host_ms": host_elapsed,
                "gather_bytes": gathered_bytes, "cost_observation_ready": action_cost_ms is not None,
                "round_total_ms": round_total_ms,
                "action_cost_ms": action_cost_ms,
                "verify_event_ms": verify_event_data[2],
                "draft_event_ms": None,
                "predicted_cost_per_commit": None, "fallback_reason": None,
                "wasted_work_ms": 0.0,
                "state_entropy_bin": (state_bucket(state, statistics.cutpoints)["entropy"] if statistics else "unknown"),
                "state_concentration_bin": (state_bucket(state, statistics.cutpoints)["concentration"] if statistics else "unknown"),
                "state_history_bin": (state_bucket(state, statistics.cutpoints)["history"] if statistics else "unknown"),
                "state_context_bucket": str(math.ceil(logical_length / 2048)),
                "state_refresh": "1" if refresh_required else "0",
                "state_hash": state_hash,
            })
            if gpu_start is not None and gpu_end is not None:
                rounds[-1]["_gpu_span_events"] = (gpu_start, gpu_end)
            if draft_event_pair[0] is not None:
                rounds[-1]["_draft_events"] = draft_event_pair
            if verify_event_data[0] is not None:
                rounds[-1]["_verify_events"] = (verify_event_data[0], verify_event_data[1])
            del target_output, parent_logits, target_query, draft_scores
            del block, posterior
            round_index += 1
            if eos_in_accepted:
                stopped_by = "eos"
                break
            if pending_eos:
                stopped_by = "eos"
                break
            if processed_output_tokens >= max_new_tokens:
                break
    finally:
        capture_cleanup()

    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    e2e_ms = (time.perf_counter() - start_time) * 1000.0
    for row in rounds:
        for field, event_key in (("draft_ms", "_draft_events"), ("verify_ms", "_verify_events")):
            events = row.pop(event_key, None)
            if events is not None:
                row[field] = float(events[0].elapsed_time(events[1]))
        span = row.pop("_gpu_span_events", None)
        if span is not None:
            row["round_gpu_span_ms"] = float(span[0].elapsed_time(span[1]))
    output_tokens = min(len(generated), max_new_tokens)
    generated = generated[:output_tokens]
    if not generated:
        raise RuntimeError("generation produced no target-selected token")
    _complete_round_schema(
        rounds,
        run_id=run_id,
        sample_id=sample_id,
        dataset=dataset,
        split=split,
        variant=variant,
        repetition=repetition,
    )
    draft_ms = sum(float(record.get("draft_ms") or 0.0) for record in rounds)
    verify_ms = sum(float(record.get("verify_ms") or 0.0) for record in rounds)
    accepted_lengths = [1 + int(row["accepted_candidates"]) for row in rounds if int(row.get("gamma_executed", 0)) > 0]
    useful_accepted_lengths = [
        1 + int(row.get("emitted_accepted_candidates", row["accepted_candidates"]))
        for row in rounds if int(row.get("gamma_executed", 0)) > 0
    ]
    acceptance_rate = accepted_total / proposed_total if proposed_total else None
    final_pending = max(0, output_tokens - (processed_commits_total - trimmed_commits_total))
    decode_ms = max(e2e_ms - ttft_ms, 0.0)
    timings = {
        "prefill_ms": prefill_ms,
        "draft_prefill_ms": draft_prefill_ms,
        "decode_ms": decode_ms,
        "e2e_ms": e2e_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": decode_ms / (output_tokens - 1) if output_tokens > 1 else None,
        "throughput_tok_s": output_tokens * 1000.0 / e2e_ms if e2e_ms > 0 else None,
        "qps": 1000.0 / e2e_ms if e2e_ms > 0 else None,
        "draft_latency_ms": draft_ms,
        "verification_latency_ms": verify_ms,
        "signal_latency_ms": signal_ms_total,
        "selection_latency_ms": selection_ms_total,
        "bank_update_latency_ms": bank_update_ms_total,
        "controller_latency_ms": controller.choose_ms_total,
    }
    counters = {
        "processed_commits": processed_commits_total,
        "final_pending_emitted_count": final_pending,
        "trimmed_commits": trimmed_commits_total,
        "draft_tokens_proposed": proposed_total,
        "draft_tokens_accepted": accepted_total,
        "useful_draft_tokens_accepted": useful_accepted_total,
        "mean_accept_length": sum(accepted_lengths) / len(accepted_lengths) if accepted_lengths else None,
        "mean_effective_accept_length": sum(useful_accepted_lengths) / len(useful_accepted_lengths) if useful_accepted_lengths else None,
        "acceptance_rate": acceptance_rate,
        "effective_acceptance_rate": useful_accepted_total / proposed_total if proposed_total else None,
        "mean_gamma": gamma_total / speculative_rounds if speculative_rounds else None,
        "mean_draft_context_tokens": draft_context_total / speculative_rounds if speculative_rounds else None,
        "fallback_rounds": fallback_rounds,
        "dense_refresh_rounds": dense_refresh_rounds,
        "dense_bank_bytes": bank.allocated_bytes,
        "gather_bytes": sum(int(record.get("gather_bytes", 0)) for record in rounds),
        "round_count": len(rounds),
        "stopped_by": stopped_by,
        "output_accounting_valid": processed_commits_total + final_pending - trimmed_commits_total == output_tokens,
        "peak_memory_gib": torch.cuda.max_memory_allocated(input_ids.device) / (2**30) if torch.cuda.is_available() else None,
    }
    _complete_round_schema(
        rounds,
        run_id="target-only",
        sample_id="sample",
        dataset="unknown",
        split="unknown",
        variant="ar",
        repetition=0,
    )
    correctness = "not_compared" if variant != "ar" else "target_reference"
    return GenerationResult(
        output_ids=torch.tensor(generated, dtype=torch.long, device=input_ids.device),
        output_tokens=output_tokens,
        rounds=rounds,
        counters=counters,
        timings=timings,
        correctness_status=correctness,
        status="ok" if counters["output_accounting_valid"] else "error",
        error=None if counters["output_accounting_valid"] else "output token accounting invariant failed",
    )


@torch.inference_mode()
def generate_target_only(
    target: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    stop_token_ids: Sequence[int] | None = None,
) -> GenerationResult:
    """Paired autoregressive reference with the same sampling/stopping rule."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or max_new_tokens < 1:
        raise ValueError("target-only reference requires batch one and positive output cap")
    input_ids = input_ids.to(target.device)
    prompt_length = input_ids.shape[1]
    stop = _eos_ids(target) if stop_token_ids is None else set(int(item) for item in stop_token_ids)
    cache = DynamicCache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
        torch.cuda.reset_peak_memory_stats(input_ids.device)
    started = time.perf_counter()
    prefill = target(
        input_ids,
        position_ids=torch.arange(prompt_length, device=input_ids.device).reshape(1, -1),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    next_token = int(_sample(prefill.logits[:, -1, :], temperature).reshape(-1)[0].item())
    generated = [next_token]
    ttft_ms = (time.perf_counter() - started) * 1000.0
    rounds: list[dict[str, Any]] = []
    processed = 0
    stopped_by = "max_new_tokens"
    while len(generated) < max_new_tokens:
        if generated[-1] in stop:
            stopped_by = "eos"
            break
        position = prompt_length + processed
        token = torch.tensor([[generated[-1]]], dtype=input_ids.dtype, device=input_ids.device)
        row_started = time.perf_counter()
        output = target(
            token,
            position_ids=torch.tensor([[position]], dtype=torch.long, device=input_ids.device),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
        )
        selected = int(_sample(output.logits[:, -1, :], temperature).reshape(-1)[0].item())
        row_ms = (time.perf_counter() - row_started) * 1000.0
        rounds.append({
            "type": "round", "schema_version": "cadflash.round.v1", "status": "ok",
            "variant": "ar", "round_index": len(rounds),
            "logical_length_before": position, "processed_output_before": processed,
            "pending_anchor_position": position, "parent_query_index": 0,
            "requested_budget": "full", "gamma_requested": 0, "gamma_executed": 0,
            "block_size": 1, "length_mode": "autoregressive", "selector_id": "none",
            "refresh_required": False, "action_reason": "target_only_step",
            "accepted_candidates": 0, "processed_commits": 1,
            "logical_length_after": position + 1, "all_candidates_accepted": False,
            "prefix_right_censored": False, "boundary_round": True, "trimmed_commits": 0,
            "signal_ms": 0.0, "controller_ms": 0.0, "selection_ms": 0.0,
            "bank_update_ms": 0.0, "draft_ms": 0.0, "verify_ms": row_ms,
            "round_gpu_span_ms": None, "round_host_ms": row_ms, "gather_bytes": 0,
            "cost_observation_ready": False, "fallback_reason": None,
        })
        generated.append(selected)
        processed += 1
    if generated[-1] in stop:
        stopped_by = "eos"
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    e2e_ms = (time.perf_counter() - started) * 1000.0
    output_count = min(len(generated), max_new_tokens)
    generated = generated[:output_count]
    final_pending = max(0, output_count - processed)
    decode_ms = max(0.0, e2e_ms - ttft_ms)
    timings = {
        "prefill_ms": ttft_ms, "decode_ms": decode_ms, "e2e_ms": e2e_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": decode_ms / (output_count - 1) if output_count > 1 else None,
        "throughput_tok_s": output_count * 1000.0 / e2e_ms if e2e_ms else None,
        "qps": 1000.0 / e2e_ms if e2e_ms else None,
        "draft_latency_ms": 0.0, "verification_latency_ms": decode_ms,
        "signal_latency_ms": 0.0, "selection_latency_ms": 0.0,
        "bank_update_latency_ms": 0.0, "controller_latency_ms": 0.0,
    }
    counters = {
        "processed_commits": processed, "final_pending_emitted_count": final_pending,
        "trimmed_commits": 0, "draft_tokens_proposed": 0, "draft_tokens_accepted": 0,
        "mean_accept_length": None, "acceptance_rate": None, "mean_gamma": 0.0,
        "mean_draft_context_tokens": None, "fallback_rounds": 0,
        "dense_refresh_rounds": 0, "dense_bank_bytes": 0, "gather_bytes": 0,
        "round_count": len(rounds), "stopped_by": stopped_by,
        "output_accounting_valid": processed + final_pending == output_count,
        "peak_memory_gib": torch.cuda.max_memory_allocated(input_ids.device) / (2**30) if torch.cuda.is_available() else None,
    }
    return GenerationResult(
        output_ids=torch.tensor(generated, dtype=torch.long, device=input_ids.device),
        output_tokens=output_count, rounds=rounds, counters=counters, timings=timings,
        correctness_status="target_reference", status="ok" if counters["output_accounting_valid"] else "error",
    )
