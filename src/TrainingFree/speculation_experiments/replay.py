"""Same-prefix draft-shape and candidate-signal collection for campaign B."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from TrainingFree.context_adaptive.attention import draft_block
from TrainingFree.context_adaptive.generation import _clone_dynamic_cache, _feature_rows, _sample, _verify_one
from TrainingFree.context_adaptive.types import Action
from TrainingFree.context_interventions.masks import make_selection
from TrainingFree.context_interventions.replay import ReplayState, evaluate_replay_action
from TrainingFree.speculation_experiments.protocol import CampaignConfig, state_identity


def expected_prefix_acceptance(full_accepted: int, verify_prefix: int) -> int:
    if not 0 <= int(full_accepted) <= 15:
        raise ValueError("full_accepted must be within [0, 15]")
    if int(verify_prefix) not in CampaignConfig().verify_prefixes:
        raise ValueError("verify_prefix must be one of the preregistered values")
    return min(int(full_accepted), int(verify_prefix))


def _draft_full_block(runtime: Mapping[str, Any], state: ReplayState, layout: Any):
    draft = runtime["draft"]
    device = state.bank.device
    selection = make_selection(
        [tuple(range(state.bank.length)) for _ in draft.layers],
        requested_budget="full",
        selector_id="B_full_context",
    )
    mask_token_id = getattr(draft, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(draft.config, "dflash_config", {}).get("mask_token_id")
    if mask_token_id is None:
        raise ValueError("DFlash mask_token_id is missing")
    gamma = 15
    block = torch.full((1, gamma + 1), int(mask_token_id), dtype=torch.long, device=device)
    block[0, 0] = int(state.pending_anchor)
    positions = torch.arange(state.bank.length, state.bank.length + gamma + 1, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        draft_wall_started = time.perf_counter()
        draft_start = torch.cuda.Event(enable_timing=True)
        draft_end = torch.cuda.Event(enable_timing=True)
        draft_start.record()
    else:
        draft_wall_started = time.perf_counter()
        draft_start = draft_end = None
    with torch.inference_mode():
        proposal = draft_block(
            draft,
            runtime["target"],
            state.bank,
            selection,
            block,
            positions,
            Action("full", gamma),
            layout=layout,
        )
    block[0, 1:] = proposal.token_ids[0]
    if draft_end is not None:
        draft_end.record()
        torch.cuda.synchronize(device)
        draft_gpu_ms = float(draft_start.elapsed_time(draft_end))
        draft_ms = (time.perf_counter() - draft_wall_started) * 1000.0
    else:
        draft_gpu_ms = (time.perf_counter() - draft_wall_started) * 1000.0
        draft_ms = draft_gpu_ms
    return block, positions, selection, draft_ms, draft_gpu_ms, proposal


def _verify_prefix_once(
    runtime: Mapping[str, Any],
    input_ids: torch.Tensor,
    state: ReplayState,
    block: torch.Tensor,
    k: int,
) -> dict[str, Any]:
    target = runtime["target"]
    draft = runtime["draft"]
    device = input_ids.device
    logical_length = state.bank.length
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    cache_setup_started = time.perf_counter()
    cache = _clone_dynamic_cache(state.cache)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    cache_setup_ms = (time.perf_counter() - cache_setup_started) * 1000.0
    prefix_block = block[:, : k + 1].contiguous()
    positions = torch.arange(logical_length, logical_length + k + 1, dtype=torch.long, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        verify_wall_started = time.perf_counter()
        verify_start = torch.cuda.Event(enable_timing=True)
        verify_end = torch.cuda.Event(enable_timing=True)
        verify_start.record()
    else:
        verify_wall_started = time.perf_counter()
        verify_start = verify_end = None
    with torch.inference_mode():
        output = _verify_one(target, cache, prefix_block, positions)
        posterior = _sample(output.logits, 0.0)
        matches = (prefix_block[:, 1:] == posterior[:, :-1]).to(torch.int32)
        accepted = int(torch.cumprod(matches, dim=1).sum().item())
        pending_anchor = int(posterior[0, accepted].item())
    if verify_end is not None:
        verify_end.record()
        torch.cuda.synchronize(device)
        verify_ms = float(verify_start.elapsed_time(verify_end))
        verify_wall_ms = (time.perf_counter() - verify_wall_started) * 1000.0
    else:
        verify_ms = (time.perf_counter() - verify_wall_started) * 1000.0
        verify_wall_ms = verify_ms

    with torch.inference_mode():
        bank_update_started = time.perf_counter()
        committed = 1 + accepted
        retained = _feature_rows(output.hidden_states, list(draft.target_layer_ids))[:, :committed, :]
        cache.crop(logical_length + committed)
        cache_length_after_crop = int(cache.get_seq_length())
        if cache_length_after_crop != logical_length + committed:
            raise RuntimeError(
                f"verify-prefix target cache crop mismatch: {cache_length_after_crop} != {logical_length + committed}"
            )
        original_length = state.bank.length
        state.bank.append(
            retained,
            torch.arange(logical_length, logical_length + committed, device=device),
        )
        state.bank.length = original_length
        if state.bank.length != logical_length:
            raise RuntimeError("verify-prefix draft-bank logical length changed after replay")
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    bank_update_ms = (time.perf_counter() - bank_update_started) * 1000.0
    del output, retained, cache, prefix_block
    return {
        "accepted_candidates": accepted,
        "pending_anchor": pending_anchor,
        "verify_ms": verify_ms,
        "verify_wall_ms": verify_wall_ms,
        "bank_update_ms": bank_update_ms,
        "cache_setup_ms": cache_setup_ms,
        "cache_length_after_crop": cache_length_after_crop,
        "bank_active_length": int(state.bank.length),
    }


def run_verification_prefix_sweep(
    runtime: Mapping[str, Any],
    input_ids: torch.Tensor,
    state: ReplayState,
    layout: Any,
    *,
    dataset: str,
    source_group_id: str,
    sample_id: str,
    prompt_token_ids: Sequence[int],
    reference_output_ids: Sequence[int],
    max_new_tokens: int,
    round_index: int,
    full_accepted: int,
    verify_prefixes: Sequence[int] = (3, 7, 11, 15),
    repetitions: int = 6,
    selection_repetitions: int = 3,
    warmups: int = 3,
    seed: int = 42,
    expected_proposal_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Draft 15 candidates once, then replay causal verifier prefixes."""

    k_values = tuple(int(value) for value in verify_prefixes)
    if not k_values or tuple(sorted(set(k_values))) != k_values:
        raise ValueError("verify_prefixes must be unique and sorted")
    if any(value not in CampaignConfig().verify_prefixes for value in k_values):
        raise ValueError("unsupported verification prefix")
    if 15 not in k_values:
        raise ValueError("verify prefix 15 is required as the full verifier reference")
    if repetitions < 2 or not 1 <= selection_repetitions < repetitions:
        raise ValueError("repetitions must exceed selection_repetitions >= 1")
    if max_new_tokens - len(state.output_ids) < 16:
        raise ValueError("B1-V requires at least 16 output tokens of configured horizon")
    identity = state_identity(
        dataset=dataset,
        source_group_id=source_group_id,
        prompt_token_ids=prompt_token_ids,
        output_prefix_ids=state.output_ids,
        pending_anchor=int(state.pending_anchor),
        anchor_position=len(prompt_token_ids) + len(state.output_ids),
        round_index=round_index,
    )
    rows: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    prior_proposal: tuple[int, ...] | None = None
    for repetition in range(-warmups, repetitions):
        block, _, _, draft_ms, draft_gpu_ms, _ = _draft_full_block(runtime, state, layout)
        proposal_ids = tuple(int(token) for token in block[0, 1:].detach().cpu().tolist())
        if expected_proposal_ids is not None and proposal_ids != tuple(int(value) for value in expected_proposal_ids):
            raise RuntimeError("B1-V gamma15 proposal differs from the paired B1 candidate trace")
        if prior_proposal is not None and proposal_ids != prior_proposal:
            raise RuntimeError("gamma15 proposals changed across B1-V timing repetitions")
        prior_proposal = proposal_ids
        if repetition < 0:
            for k in k_values:
                warmup_result = _verify_prefix_once(runtime, input_ids, state, block, k)
                if int(warmup_result["accepted_candidates"]) != expected_prefix_acceptance(full_accepted, k):
                    raise RuntimeError(f"verify-prefix warmup acceptance identity failed for k={k}")
            del block
            continue
        order = list(k_values)
        random.Random(seed + repetition).shuffle(order)
        split = "selection" if repetition < selection_repetitions else "evaluation"
        schedule.append({"repetition": repetition, "split": split, "k_order": order})
        for k in order:
            measured = _verify_prefix_once(runtime, input_ids, state, block, k)
            expected_l = expected_prefix_acceptance(full_accepted, k)
            if int(measured["accepted_candidates"]) != expected_l:
                raise RuntimeError(
                    f"verify-prefix acceptance identity failed for k={k}: "
                    f"observed={measured['accepted_candidates']} expected={expected_l}"
                )
            reference_position = len(state.output_ids) + 1 + expected_l
            has_reference_anchor = reference_position < len(reference_output_ids)
            pending_matches = (
                int(measured["pending_anchor"]) == int(reference_output_ids[reference_position])
                if has_reference_anchor else None
            )
            if pending_matches is False:
                raise RuntimeError(f"verify-prefix pending anchor differs from reference for k={k}")
            total_ms = float(draft_ms) + float(measured["verify_wall_ms"]) + float(measured["bank_update_ms"])
            rows.append({
                "schema_version": "dflash-speculation-b.verify-prefix.v1",
                "type": "verify_prefix_action",
                "state_id": identity,
                "dataset": dataset,
                "source_group_id": source_group_id,
                "sample_id": sample_id,
                "round_index": int(round_index),
                "processed_output_tokens": len(state.output_ids),
                "prompt_tokens": len(prompt_token_ids),
                "context_tokens": int(state.bank.length),
                "max_new_tokens": int(max_new_tokens),
                "draft_gamma": 15,
                "k": k,
                "repetition": repetition,
                "repetition_split": split,
                "accepted_candidates": int(measured["accepted_candidates"]),
                "expected_accepted_candidates": expected_l,
                "acceptance_identity_valid": True,
                "pending_anchor": int(measured["pending_anchor"]),
                "pending_anchor_matches_reference": pending_matches,
                "committed_tokens": 1 + int(measured["accepted_candidates"]),
                "draft_ms": float(draft_ms),
                "draft_gpu_ms": float(draft_gpu_ms),
                "verify_ms": float(measured["verify_ms"]),
                "verify_wall_ms": float(measured["verify_wall_ms"]),
                "bank_update_ms": float(measured["bank_update_ms"]),
                "cache_setup_ms": float(measured["cache_setup_ms"]),
                "cache_length_after_crop": int(measured["cache_length_after_crop"]),
                "bank_active_length": int(measured["bank_active_length"]),
                "total_round_ms": total_ms,
                "cost_per_commit_ms": total_ms / (1 + int(measured["accepted_candidates"])),
                "proposed_token_ids": list(proposal_ids),
                "selection_schedule_seed": seed,
            })
        del block
    return {"state_id": identity, "actions": rows, "schedule": schedule, "proposal_token_ids": list(prior_proposal or ())}


def run_same_prefix_sweep(
    runtime: Mapping[str, Any],
    input_ids: torch.Tensor,
    state: ReplayState,
    layout: Any,
    *,
    dataset: str,
    source_group_id: str,
    sample_id: str,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    round_index: int,
    gammas: Sequence[int] = (3, 7, 11, 15),
    repetitions: int = 6,
    selection_repetitions: int = 3,
    warmups: int = 3,
    seed: int = 42,
    collect_uncertainty: bool = True,
    history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay each gamma's own draft forward from an isolated identical state.

    Cache copying is timed separately by ``evaluate_replay_action``. Timing
    repetitions are paired by state and use a seeded, saved action order.
    Selection repetitions choose the diagnostic oracle; later repetitions are
    reserved to estimate its selected-cost performance.
    """

    gamma_values = tuple(int(value) for value in gammas)
    if not gamma_values or tuple(sorted(set(gamma_values))) != gamma_values:
        raise ValueError("gammas must be unique and sorted")
    if any(value not in CampaignConfig().gammas for value in gamma_values):
        raise ValueError("unsupported gamma in same-prefix sweep")
    if repetitions < 2 or not 1 <= selection_repetitions < repetitions:
        raise ValueError("repetitions must exceed selection_repetitions >= 1")
    if warmups < 0:
        raise ValueError("warmups must be nonnegative")
    if max_new_tokens < 1 or round_index < 0:
        raise ValueError("max_new_tokens must be positive and round_index nonnegative")

    output_prefix = tuple(int(token) for token in state.output_ids)
    identity = state_identity(
        dataset=dataset,
        source_group_id=source_group_id,
        prompt_token_ids=prompt_token_ids,
        output_prefix_ids=output_prefix,
        pending_anchor=int(state.pending_anchor),
        anchor_position=len(prompt_token_ids) + len(output_prefix),
        round_index=round_index,
    )
    full_positions = tuple(range(state.bank.length))
    masks = [full_positions for _ in runtime["draft"].layers]
    by_gamma: dict[int, dict[str, Any]] = {}
    warmup_rows: list[dict[str, Any]] = []

    for gamma in gamma_values:
        for warmup_index in range(warmups):
            warmup = evaluate_replay_action(
                runtime,
                input_ids,
                state,
                layout,
                masks,
                requested_budget="full",
                selector_id="B_full_context",
                remaining_output_tokens=max_new_tokens - len(output_prefix),
                gamma=gamma,
                cleanup_after_action=False,
            )
            if warmup.get("status") != "ok":
                raise RuntimeError(f"warmup failed for gamma={gamma}: {warmup.get('reason')}")
            warmup_rows.append({"gamma": gamma, "warmup_index": warmup_index, "L": warmup["accepted_candidates"]})
        by_gamma[gamma] = {"rows": [], "proposal": None, "accepted": None}

    schedule: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        order = list(gamma_values)
        random.Random(seed + repetition).shuffle(order)
        split = "selection" if repetition < selection_repetitions else "evaluation"
        schedule.append({"repetition": repetition, "split": split, "gamma_order": order})
        for gamma in order:
            outcome = evaluate_replay_action(
                runtime,
                input_ids,
                state,
                layout,
                masks,
                requested_budget="full",
                selector_id="B_full_context",
                remaining_output_tokens=max_new_tokens - len(output_prefix),
                gamma=gamma,
                cleanup_after_action=False,
            )
            if outcome.get("status") != "ok":
                raise RuntimeError(f"action failed for gamma={gamma}: {outcome.get('reason')}")
            proposal = tuple(int(token) for token in outcome["proposed_token_ids"])
            accepted = int(outcome["accepted_candidates"])
            prior_proposal = by_gamma[gamma]["proposal"]
            prior_accepted = by_gamma[gamma]["accepted"]
            if prior_proposal is not None and proposal != prior_proposal:
                raise RuntimeError(f"greedy proposals changed across timing repeats for gamma={gamma}")
            if prior_accepted is not None and accepted != prior_accepted:
                raise RuntimeError(f"accepted prefix changed across timing repeats for gamma={gamma}")
            by_gamma[gamma]["proposal"] = proposal
            by_gamma[gamma]["accepted"] = accepted
            by_gamma[gamma]["rows"].append({
                "schema_version": "dflash-speculation-b.action.v1",
                "type": "draft_shape_action",
                "state_id": identity,
                "dataset": dataset,
                "source_group_id": source_group_id,
                "sample_id": sample_id,
                "round_index": int(round_index),
                "processed_output_tokens": len(output_prefix),
                "prompt_tokens": len(prompt_token_ids),
                "context_tokens": int(state.bank.length),
                "max_new_tokens": int(max_new_tokens),
                "action": f"gamma{gamma}",
                "gamma": gamma,
                "repetition": repetition,
                "repetition_split": split,
                "accepted_candidates": accepted,
                "committed_tokens": 1 + accepted,
                "proposed_candidates": gamma,
                "proposed_token_ids": list(proposal),
                "draft_ms": outcome["draft_ms"],
                "gather_ms": outcome["gather_ms"],
                "verify_ms": outcome["verify_ms"],
                "bank_update_ms": outcome["bank_update_ms"],
                "cache_length_after_crop": outcome["cache_length_after_crop"],
                "round_ms": outcome["action_wall_ms"],
                "replay_setup_ms": outcome["replay_setup_ms"],
                "signal_extraction_ms": None,
                "selection_schedule_seed": seed,
            })

    signal_row = None
    if collect_uncertainty and 15 in gamma_values:
        signal = evaluate_replay_action(
            runtime,
            input_ids,
            state,
            layout,
            masks,
            requested_budget="full",
            selector_id="B_full_context_signal",
            remaining_output_tokens=max_new_tokens - len(output_prefix),
            gamma=15,
            collect_uncertainty=True,
            cleanup_after_action=False,
        )
        if signal.get("status") != "ok":
            raise RuntimeError(f"gamma15 signal collection failed: {signal.get('reason')}")
        if tuple(signal["proposed_token_ids"]) != by_gamma[15]["proposal"]:
            raise RuntimeError("uncertainty collection changed gamma15 candidate tokens")
        if int(signal["accepted_candidates"]) != by_gamma[15]["accepted"]:
            raise RuntimeError("uncertainty collection changed gamma15 accepted prefix")
        signal_row = {
            "schema_version": "dflash-speculation-b.signals.v1",
            "type": "candidate_signals",
            "state_id": identity,
            "dataset": dataset,
            "source_group_id": source_group_id,
            "sample_id": sample_id,
            "round_index": int(round_index),
            "processed_output_tokens": len(output_prefix),
            "prompt_tokens": len(prompt_token_ids),
            "context_tokens": int(state.bank.length),
            "max_new_tokens": int(max_new_tokens),
            "gamma": 15,
            "accepted_candidates": int(signal["accepted_candidates"]),
            "proposed_token_ids": list(signal["proposed_token_ids"]),
            "draft_uncertainty": signal["draft_uncertainty"],
            "parent_signal": signal["parent_signal"],
            "history": dict(history or {}),
            "history_ema": (history or {}).get("ema_acceptance"),
            "previous_accepted_candidates": (history or {}).get("previous_accepted_candidates"),
            "previous_all_accepted": (history or {}).get("previous_all_accepted"),
            "history_missing": bool((history or {}).get("missing", True)),
            "signal_extraction_ms": signal["signal_extraction_ms"],
            "instrumented_round_ms": signal["action_wall_ms"],
            "history_origin": "reference_full_context_gamma15_prior_rounds",
        }

    return {
        "state_id": identity,
        "actions": [row for gamma in gamma_values for row in by_gamma[gamma]["rows"]],
        "signal": signal_row,
        "schedule": schedule,
        "warmups": warmup_rows,
        "proposal_by_gamma": {str(gamma): list(values["proposal"]) for gamma, values in by_gamma.items()},
        "accepted_by_gamma": {str(gamma): values["accepted"] for gamma, values in by_gamma.items()},
    }
