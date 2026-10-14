"""B200 model execution for parity and fixed-prefix A1/A2 replay."""

from __future__ import annotations

import gc
import hashlib
import random
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from TrainingFree.context_adaptive.config import AdaptiveConfig
from TrainingFree.context_adaptive.generation import generate_adaptive, generate_target_only
from TrainingFree.context_adaptive.prompt import prepare_prompt
from TrainingFree.context_interventions.attention import target_full_key_scores
from TrainingFree.context_interventions.masks import (
    build_region_positions,
    exact_budget_positions,
    global_protected_positions,
    make_selection,
    matched_deletion_positions,
    remove_attention_positions,
)
from TrainingFree.context_interventions.matrix import a1_matrix, a2_matrix
from TrainingFree.context_interventions.masks import build_online_selection_policy
from TrainingFree.context_interventions.replay import (
    TARGET_PARENT_LAYER,
    ReferencePrefix,
    ReplayState,
    choose_reference_prefixes,
    dense_draft_scores,
    evaluate_replay_action,
    mean_target_scoring_vectors,
    rebuild_replay_state,
)
from TrainingFree.context_interventions.resilience import CampaignIntegrityError, is_fatal_compute_error


def _eos_ids(target: Any) -> set[int]:
    value = getattr(target.config, "eos_token_id", None)
    if value is None:
        value = getattr(target.generation_config, "eos_token_id", None)
    if value is None:
        return set()
    return {int(value)} if isinstance(value, int) else {int(item) for item in value}


def _native_full_dflash(
    runtime: Mapping[str, Any], input_ids: torch.Tensor, max_new_tokens: int, *, gamma: int = 15
):
    from dflash.model import dflash_generate

    eos_ids = sorted(_eos_ids(runtime["target"])) or [-1]
    with torch.inference_mode():
        return dflash_generate(
            runtime["draft"],
            target=runtime["target"],
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            stop_token_ids=eos_ids,
            temperature=0.0,
            block_size=int(gamma) + 1,
            return_stats=True,
        )


def _ids(result: Any, prompt_tokens: int, max_new_tokens: int) -> list[int]:
    values = result.output_ids
    if values.ndim == 2:
        values = values[0]
    return [int(value) for value in values[prompt_tokens : prompt_tokens + max_new_tokens].detach().cpu().tolist()]


def _first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def run_reference_pair(
    runtime: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    max_new_tokens: int,
    source_anchors: int = 64,
    recent_output: int = 256,
    gamma: int = 15,
    reference_ar_ids: Sequence[int] | None = None,
    require_ar_match: bool = True,
    collect_uncertainty: bool = False,
) -> dict[str, Any]:
    """Run AR/native DFlash/adapter Full on the exact same tokenized prompt."""
    ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=128)
    ids = ids.to(runtime["target"].device).contiguous()
    if layout.source_group_id is None:
        object.__setattr__(layout, "source_group_id", str(sample.get("_cad_source_group_id", "")))
    if not layout.source_available:
        raise RuntimeError(f"source span mapping is unavailable: {layout.source_reason}")
    if reference_ar_ids is None:
        torch.cuda.synchronize(ids.device)
        ar_started = time.perf_counter()
        with torch.inference_mode():
            ar = generate_target_only(runtime["target"], ids, max_new_tokens=max_new_tokens, temperature=0.0)
        if ar.status != "ok":
            raise RuntimeError(f"AR reference generation failed: {ar.error}")
        torch.cuda.synchronize(ids.device)
        ar_ms: float | None = (time.perf_counter() - ar_started) * 1000.0
        ar_ids = [int(value) for value in ar.output_ids.detach().cpu().tolist()]
    else:
        ar_ms = None
        ar_ids = [int(value) for value in reference_ar_ids[:max_new_tokens]]

    torch.cuda.synchronize(ids.device)
    native_started = time.perf_counter()
    native = _native_full_dflash(runtime, ids, max_new_tokens, gamma=gamma)
    torch.cuda.synchronize(ids.device)
    native_ms = (time.perf_counter() - native_started) * 1000.0
    native_ids = _ids(native, ids.shape[1], max_new_tokens)

    config = AdaptiveConfig(
        budgets=("full",),
        gammas=(int(gamma),),
        selector="target_parent",
        target_layer=TARGET_PARENT_LAYER,
        source_anchors=source_anchors,
        recent_output=recent_output,
        timing_mode="wall",
        output_root="outputs/dflash_context_interventions",
    )
    torch.cuda.synchronize(ids.device)
    adapter_started = time.perf_counter()
    with torch.inference_mode():
        adapter = generate_adaptive(
            runtime["target"],
            runtime["draft"],
            ids,
            runtime["tokenizer"],
            layout,
            config,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            variant="dflash_full_fixed",
            fixed_budget="full",
            fixed_gamma=int(gamma),
            gamma_reference=int(gamma),
            sample_id=str(sample["id"]),
            dataset=str(sample["dataset"]),
            split=str(sample.get("_cad_split", "unknown")),
            collect_uncertainty=collect_uncertainty,
        )
    torch.cuda.synchronize(ids.device)
    adapter_ms = (time.perf_counter() - adapter_started) * 1000.0
    if adapter.status != "ok":
        raise RuntimeError(f"full-adapter reference generation failed: {adapter.error}")
    adapter_ids = [int(value) for value in adapter.output_ids.detach().cpu().tolist()]

    native_matches_ar = native_ids == ar_ids
    adapter_matches_ar = adapter_ids == ar_ids
    native_adapter_match = native_ids == adapter_ids
    return {
        "sample": dict(sample),
        "input_ids": ids,
        "layout": layout,
        "ar_ids": ar_ids,
        "native_ids": native_ids,
        "adapter_ids": adapter_ids,
        "adapter_rounds": adapter.rounds,
        "eos_ids": _eos_ids(runtime["target"]),
        "parity": {
            "gamma": int(gamma),
            "status": "pass" if native_adapter_match and (
                not require_ar_match or (native_matches_ar and adapter_matches_ar)
            ) else "fail",
            "gate_basis": "ar_and_native_adapter" if require_ar_match else "native_equals_adapter",
            "ar_exact_match_required": bool(require_ar_match),
            "native_equals_ar": native_matches_ar,
            "adapter_equals_ar": adapter_matches_ar,
            "native_equals_adapter": native_adapter_match,
            "native_first_mismatch": _first_mismatch(native_ids, ar_ids),
            "adapter_first_mismatch": _first_mismatch(adapter_ids, ar_ids),
            "ar_output_tokens": len(ar_ids),
            "native_output_tokens": len(native_ids),
            "adapter_output_tokens": len(adapter_ids),
            "ar_e2e_ms": ar_ms,
            "native_full_e2e_ms": native_ms,
            "adapter_full_e2e_ms": adapter_ms,
            "native_stats": str(getattr(native, "stats", None)),
        },
    }


def _action_row(
    *,
    experiment: str,
    phase: str,
    sample: Mapping[str, Any],
    prefix: ReferencePrefix,
    label: str,
    baseline_accepted: int,
    outcome: Mapping[str, Any],
    nominal_budget: int | str,
    attention_scoring: str | None = None,
    deletion_side: str | None = None,
    deletion_count: int | None = None,
    random_seed: int | None = None,
    ranking_origin_round: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cell_id = "|".join((
        experiment,
        str(sample["dataset"]),
        str(sample["_cad_source_group_id"]),
        str(prefix.round_index),
        label,
    ))
    row = {
        "schema_version": "dflash-context-interventions.replay.v1",
        "type": "replay",
        "experiment": experiment,
        "phase": phase,
        "sample_id": str(sample["id"]),
        "dataset": str(sample["dataset"]),
        "source_group_id": str(sample["_cad_source_group_id"]),
        "split": str(sample.get("_cad_split", "unknown")),
        "prefix_round_index": prefix.round_index,
        "processed_output_tokens": prefix.processed_output_tokens,
        "pending_anchor": prefix.pending_anchor,
        "prefix_hash": prefix.prefix_hash,
        "previous_dense_round_index": prefix.previous_dense_round_index,
        "selector_id": label,
        "nominal_budget": nominal_budget,
        "attention_scoring": attention_scoring,
        "deletion_side": deletion_side,
        "deletion_count": deletion_count,
        "random_seed": random_seed,
        "L_full": baseline_accepted,
        "L_action": outcome.get("accepted_candidates"),
        "delta_L": (
            int(outcome["accepted_candidates"]) - baseline_accepted
            if outcome.get("accepted_candidates") is not None else None
        ),
        "status": outcome.get("status", "error"),
        "reason": outcome.get("reason"),
        "actual_context_tokens_by_layer": outcome.get("actual_context_tokens_by_layer"),
        "positions_sha256_by_layer": outcome.get("positions_sha256_by_layer"),
        "ranking_origin_round": (
            ranking_origin_round
            if ranking_origin_round is not None
            else outcome.get("ranking_origin_round")
        ),
        "draft_ms": outcome.get("draft_ms"),
        "verify_ms": outcome.get("verify_ms"),
        "gather_bytes": outcome.get("gather_bytes"),
        "run_id": str(sample.get("_campaign_run_id", "")),
    }
    if extra:
        row.update(dict(extra))
    row["cell_id"] = cell_id
    return row


def _previous_dflash_evidence(
    outcome: Mapping[str, Any] | None,
    *,
    reference_accepted: int | None,
    layer_count: int,
    round_index: int | None,
) -> tuple[dict[int, dict[int, float]] | None, dict[str, Any]]:
    """Validate the optional previous-round ranking signal without aborting a campaign.

    A previous-round replay supplies a selector feature, not the full-DFlash
    baseline for the current experimental cell. Its attention map can remain
    usable when verifier acceptance differs; that discrepancy is retained as
    row-level evidence. Missing replay output or incomplete attention maps make
    only the previous_dflash selector ineligible.
    """
    diagnostic: dict[str, Any] = {
        "round_index": round_index,
        "status": "unavailable",
        "reference_accepted_candidates": (
            int(reference_accepted) if reference_accepted is not None else None
        ),
        "replay_accepted_candidates": None,
        "acceptance_matches": None,
        "missing_attention_layers": list(range(max(int(layer_count), 0))),
    }
    if outcome is None:
        diagnostic["reason"] = "previous_dense_round_missing"
        return None, diagnostic

    diagnostic["replay_status"] = outcome.get("status")
    if outcome.get("status") != "ok":
        diagnostic["status"] = "replay_failed"
        diagnostic["reason"] = outcome.get("reason") or "previous_dense_replay_not_ok"
        return None, diagnostic

    accepted = outcome.get("accepted_candidates")
    if accepted is not None:
        diagnostic["replay_accepted_candidates"] = int(accepted)
    if reference_accepted is not None and accepted is not None:
        diagnostic["acceptance_matches"] = int(accepted) == int(reference_accepted)

    raw_scores = outcome.get("attention_scores_by_layer")
    scores: dict[int, dict[int, float]] = {}
    missing_layers: list[int] = []
    if not isinstance(raw_scores, Mapping):
        missing_layers = list(range(max(int(layer_count), 0)))
    else:
        for layer_index in range(max(int(layer_count), 0)):
            layer_scores = raw_scores.get(layer_index, raw_scores.get(str(layer_index)))
            if not isinstance(layer_scores, Mapping) or not layer_scores:
                missing_layers.append(layer_index)
                continue
            try:
                scores[layer_index] = {
                    int(position): float(score) for position, score in layer_scores.items()
                }
            except (TypeError, ValueError, OverflowError):
                missing_layers.append(layer_index)
    diagnostic["missing_attention_layers"] = missing_layers
    if missing_layers:
        diagnostic["status"] = "attention_unavailable"
        diagnostic["reason"] = "previous_dense_attention_map_missing_layers"
        return None, diagnostic

    if reference_accepted is None or accepted is None:
        diagnostic["status"] = "acceptance_unavailable"
    elif diagnostic["acceptance_matches"]:
        diagnostic["status"] = "ok"
    else:
        diagnostic["status"] = "acceptance_mismatch"
    return scores, diagnostic


def _replay_previous_dflash_round(
    runtime: Mapping[str, Any],
    input_ids: torch.Tensor,
    reference_output: Sequence[int],
    prefix: ReferencePrefix,
    layout,
    *,
    max_new_tokens: int,
) -> Mapping[str, Any]:
    """Rebuild the optional prior-round signal and turn local failures into evidence."""
    state = None
    try:
        state = rebuild_replay_state(
            runtime, input_ids, reference_output, prefix, layout,
            max_new_tokens=max_new_tokens,
        )
        return evaluate_replay_action(
            runtime, input_ids, state, layout,
            [tuple(range(state.bank.length)) for _ in runtime["draft"].layers],
            requested_budget="full", selector_id="previous_dense_reference",
            remaining_output_tokens=prefix.remaining_output_tokens,
            collect_attention_scores=True,
        )
    except Exception as exc:
        if is_fatal_compute_error(exc):
            raise CampaignIntegrityError(f"previous DFlash replay: {exc}") from exc
        state = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}:{str(exc)[:500]}",
            "accepted_candidates": None,
            "attention_scores_by_layer": None,
        }
    finally:
        state = None


def _evaluate_mask(
    runtime: Mapping[str, Any],
    input_ids: torch.Tensor,
    state,
    layout,
    sample: Mapping[str, Any],
    prefix: ReferencePrefix,
    *,
    phase: str,
    experiment: str,
    label: str,
    nominal_budget: int | str,
    masks: Sequence[Sequence[int]],
    baseline_accepted: int,
    attention_scoring: str | None = None,
    deletion_side: str | None = None,
    deletion_count: int | None = None,
    random_seed: int | None = None,
    ranking_origin_round: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        outcome = evaluate_replay_action(
            runtime,
            input_ids,
            state,
            layout,
            masks,
            requested_budget=nominal_budget,
            selector_id=label,
            remaining_output_tokens=prefix.remaining_output_tokens,
            ranking_origin_round=ranking_origin_round,
        )
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        outcome = {"status": "error", "reason": f"out_of_memory:{str(exc)[:300]}", "accepted_candidates": None}
    except Exception as exc:
        if is_fatal_compute_error(exc):
            raise CampaignIntegrityError(f"replay action {label}: {exc}") from exc
        outcome = {"status": "error", "reason": f"{type(exc).__name__}:{str(exc)[:500]}", "accepted_candidates": None}
    return _action_row(
        experiment=experiment,
        phase=phase,
        sample=sample,
        prefix=prefix,
        label=label,
        baseline_accepted=baseline_accepted,
        outcome=outcome,
        nominal_budget=nominal_budget,
        attention_scoring=attention_scoring,
        deletion_side=deletion_side,
        deletion_count=deletion_count,
        random_seed=random_seed,
        extra=extra,
    )


def run_a1_prefix(
    runtime: Mapping[str, Any],
    sample: Mapping[str, Any],
    input_ids: torch.Tensor,
    layout,
    prefix: ReferencePrefix,
    reference_output: Sequence[int],
    *,
    phase: str,
    max_new_tokens: int,
    state: ReplayState | None = None,
    full_outcome: Mapping[str, Any] | None = None,
    reference_accepted: int | None = None,
    source_anchors: int = 64,
    recent_output: int = 256,
) -> list[dict[str, Any]]:
    if state is None:
        state = rebuild_replay_state(
            runtime, input_ids, reference_output, prefix, layout,
            max_new_tokens=max_new_tokens,
        )
    bank_length = state.bank.length
    baseline_mask = tuple(range(bank_length))
    baseline = full_outcome or evaluate_replay_action(
        runtime, input_ids, state, layout,
        [baseline_mask for _ in runtime["draft"].layers],
        requested_budget="full", selector_id="full", remaining_output_tokens=prefix.remaining_output_tokens,
        collect_attention_scores=False,
    )
    if baseline.get("status") != "ok":
        raise RuntimeError(f"full DFlash replay failed at prefix {prefix.round_index}: {baseline.get('reason')}")
    base_l = int(baseline["accepted_candidates"])
    if reference_accepted is not None and base_l != int(reference_accepted):
        raise RuntimeError(
            f"full DFlash replay acceptance differs from reference trace at round {prefix.round_index}: "
            f"trace={reference_accepted}, replay={base_l}"
        )
    g0 = global_protected_positions(layout, bank_length=bank_length, source_anchors=source_anchors)
    matrix = a1_matrix(phase)
    rows: list[dict[str, Any]] = []
    for control in matrix["controls"]:
        positions = build_region_positions(
            control,
            layout=layout,
            bank_length=bank_length,
            processed_output_start=layout.prompt_tokens,
            source_anchors=source_anchors,
            recent_output=recent_output,
        )
        masks = [positions for _ in runtime["draft"].layers]
        rows.append(_evaluate_mask(
            runtime, input_ids, state, layout, sample, prefix,
            phase=phase, experiment="A1_region", label=control,
            nominal_budget="full" if control == "full" else len(positions), masks=masks,
            baseline_accepted=base_l,
        ))

    source = set(layout.source_positions)
    recent_start = max(layout.prompt_tokens, bank_length - recent_output)
    recent = set(range(recent_start, bank_length))
    old_output = set(range(layout.prompt_tokens, recent_start))
    region_positions = {"source": source, "recent": recent, "old_output": old_output}
    for deletion in matrix["matched_deletions"]:
        for region_name in matrix["matched_regions"]:
            try:
                keep = matched_deletion_positions(
                    bank_length=bank_length,
                    region_positions=tuple(sorted(region_positions[region_name])),
                    protected_positions=g0,
                    count=int(deletion),
                    seed=42,
                )
                masks = [keep for _ in runtime["draft"].layers]
                rows.append(_evaluate_mask(
                    runtime, input_ids, state, layout, sample, prefix,
                    phase=phase, experiment="A1_matched_deletion",
                    label=f"delete_{region_name}_D{deletion}",
                    nominal_budget=len(keep), masks=masks, baseline_accepted=base_l,
                    deletion_count=int(deletion),
                    extra={"eligible_region_keys": len(region_positions[region_name] - set(g0))},
                ))
            except ValueError as exc:
                rows.append(_action_row(
                    experiment="A1_matched_deletion", phase=phase, sample=sample, prefix=prefix,
                    label=f"delete_{region_name}_D{deletion}", baseline_accepted=base_l,
                    outcome={"status": "infeasible", "reason": str(exc), "accepted_candidates": None},
                    nominal_budget=bank_length - int(deletion), deletion_count=int(deletion),
                    extra={"eligible_region_keys": len(region_positions[region_name] - set(g0))},
                ))

    scoring_vectors = mean_target_scoring_vectors(state.target_scores)
    for scoring_id in matrix["attention_scoring"]:
        for deletion in matrix["attention_deletions"]:
            for side in ("low", "high"):
                try:
                    if scoring_id == "target_corresponding":
                        vectors = scoring_vectors[scoring_id]
                        masks = [
                            remove_attention_positions(
                                bank_length=bank_length,
                                protected_positions=g0,
                                scores=vectors[index],
                                count=int(deletion),
                                side=side,
                            )
                            for index in range(len(runtime["draft"].layers))
                        ]
                    else:
                        scores = scoring_vectors[scoring_id]
                        keep = remove_attention_positions(
                            bank_length=bank_length,
                            protected_positions=g0,
                            scores=scores,
                            count=int(deletion),
                            side=side,
                        )
                        masks = [keep for _ in runtime["draft"].layers]
                    rows.append(_evaluate_mask(
                        runtime, input_ids, state, layout, sample, prefix,
                        phase=phase, experiment="A1_attention",
                        label=f"{scoring_id}_{side}_D{deletion}",
                        nominal_budget=bank_length - int(deletion), masks=masks,
                        baseline_accepted=base_l, attention_scoring=scoring_id,
                        deletion_side=side, deletion_count=int(deletion),
                    ))
                except ValueError as exc:
                    rows.append(_action_row(
                        experiment="A1_attention", phase=phase, sample=sample, prefix=prefix,
                        label=f"{scoring_id}_{side}_D{deletion}", baseline_accepted=base_l,
                        outcome={"status": "infeasible", "reason": str(exc), "accepted_candidates": None},
                        nominal_budget=bank_length - int(deletion), attention_scoring=scoring_id,
                        deletion_side=side, deletion_count=int(deletion),
                    ))
    del state, baseline
    gc.collect()
    return rows


def _prefix_from_trace(
    round_row: Mapping[str, Any],
    *,
    output_ids: Sequence[int],
    max_new_tokens: int,
    previous_round_index: int | None,
) -> ReferencePrefix:
    processed = int(round_row.get("processed_output_before", 0))
    pending = int(output_ids[processed])
    prefix_hash = hashlib.sha256(
        (",".join(map(str, output_ids[:processed])) + f"|{pending}|{round_row['round_index']}").encode("ascii")
    ).hexdigest()
    return ReferencePrefix(
        round_index=int(round_row["round_index"]),
        processed_output_tokens=processed,
        pending_anchor=pending,
        remaining_output_tokens=max_new_tokens - processed,
        prefix_hash=prefix_hash,
        previous_dense_round_index=previous_round_index,
    )


def run_a2_prefix(
    runtime: Mapping[str, Any],
    sample: Mapping[str, Any],
    input_ids: torch.Tensor,
    layout,
    prefix: ReferencePrefix,
    reference_output: Sequence[int],
    *,
    phase: str,
    max_new_tokens: int,
    state: ReplayState | None = None,
    full_outcome: Mapping[str, Any] | None = None,
    replay_cache: dict[int, Mapping[str, Any]] | None = None,
    previous_prefix: ReferencePrefix | None = None,
    reference_accepted: int | None = None,
    previous_reference_accepted: int | None = None,
    source_anchors: int = 64,
    recent_output: int = 256,
) -> list[dict[str, Any]]:
    if state is None:
        state = rebuild_replay_state(runtime, input_ids, reference_output, prefix, layout, max_new_tokens=max_new_tokens)
    bank_length = state.bank.length
    full_mask = tuple(range(bank_length))
    full = full_outcome or evaluate_replay_action(
        runtime, input_ids, state, layout,
        [full_mask for _ in runtime["draft"].layers],
        requested_budget="full", selector_id="full", remaining_output_tokens=prefix.remaining_output_tokens,
        collect_attention_scores=True,
    )
    if full.get("status") != "ok":
        raise RuntimeError(f"full DFlash replay failed at prefix {prefix.round_index}: {full.get('reason')}")
    baseline_l = int(full["accepted_candidates"])
    if reference_accepted is not None and baseline_l != int(reference_accepted):
        raise RuntimeError(
            f"full DFlash replay acceptance differs from reference trace at round {prefix.round_index}: "
            f"trace={reference_accepted}, replay={baseline_l}"
        )
    current_scores = full.get("attention_scores_by_layer") or dense_draft_scores(runtime, state, layout)
    previous_scores: dict[int, dict[int, float]] | None = None
    previous_replay_check: dict[str, Any] | None = None
    if prefix.previous_dense_round_index is not None:
        previous_replay_check = {
            "round_index": prefix.previous_dense_round_index,
            "status": "unavailable",
            "reference_accepted_candidates": previous_reference_accepted,
            "replay_accepted_candidates": None,
            "acceptance_matches": None,
            "missing_attention_layers": list(range(len(runtime["draft"].layers))),
            "reason": "previous_dense_round_not_replayed",
        }
    if prefix.previous_dense_round_index is not None and previous_prefix is not None:
        previous_full = (replay_cache or {}).get(previous_prefix.round_index)
        if previous_full is None:
            previous_full = _replay_previous_dflash_round(
                runtime, input_ids, reference_output, previous_prefix, layout,
                max_new_tokens=max_new_tokens,
            )
            if replay_cache is not None:
                replay_cache[previous_prefix.round_index] = previous_full
        previous_scores, previous_replay_check = _previous_dflash_evidence(
            previous_full,
            reference_accepted=previous_reference_accepted,
            layer_count=len(runtime["draft"].layers),
            round_index=previous_prefix.round_index,
        )
        del previous_full
    protected = global_protected_positions(
        layout,
        bank_length=bank_length,
        source_anchors=source_anchors,
        processed_output_start=layout.prompt_tokens,
        recent_output=recent_output,
        protect_recent=True,
    )
    matrix = a2_matrix(phase)
    rows: list[dict[str, Any]] = []
    for budget in matrix["budgets"]:
        canonical_full = budget == "full" or int(budget) >= bank_length
        effective_budget: int | str = "full" if canonical_full else int(budget)
        for selector in matrix["selectors"]:
            seeds = matrix["random_seeds"] if selector == "random" else (42,)
            for seed in seeds:
                label = f"{selector}_B{budget}" + (f"_seed{seed}" if selector == "random" else "")
                if canonical_full:
                    rows.append(_action_row(
                        experiment="A2_selector", phase=phase, sample=sample, prefix=prefix,
                        label=label, baseline_accepted=baseline_l,
                        outcome={
                            "status": "canonicalized_alias",
                            "reason": "requested budget is at least the full context length; Full is already represented by L_full",
                            "accepted_candidates": None,
                            "actual_context_tokens_by_layer": [bank_length] * len(runtime["draft"].layers),
                        },
                        nominal_budget=budget,
                        random_seed=int(seed) if selector == "random" else None,
                        extra={"canonicalized_full": True,
                               "oracle_diagnostic_only": selector == "current_dflash_oracle",
                               "previous_replay_check": previous_replay_check},
                    ))
                    continue
                try:
                    if selector == "random":
                        keep = exact_budget_positions(
                            bank_length=bank_length, budget=int(budget), protected_positions=protected,
                            strategy="random", seed=int(seed),
                        )
                        masks = [keep for _ in runtime["draft"].layers]
                    elif selector == "recency":
                        keep = exact_budget_positions(
                            bank_length=bank_length, budget=int(budget), protected_positions=protected,
                            strategy="recency",
                        )
                        masks = [keep for _ in runtime["draft"].layers]
                    elif selector == "target_parent":
                        scores = state.target_scores[TARGET_PARENT_LAYER]
                        keep = exact_budget_positions(
                            bank_length=bank_length, budget=int(budget), protected_positions=protected,
                            scores=scores, strategy="score_high",
                        )
                        masks = [keep for _ in runtime["draft"].layers]
                    elif selector == "previous_dflash":
                        if previous_scores is None:
                            detail = (previous_replay_check or {}).get("reason") or (
                                previous_replay_check or {}
                            ).get("status", "no_previous_dense_round")
                            raise ValueError(f"previous_dense_round_unavailable:{detail}")
                        masks = []
                        for layer_index in range(len(runtime["draft"].layers)):
                            prior = previous_scores.get(layer_index)
                            if prior is None:
                                raise ValueError(f"previous_dflash_layer_{layer_index}_missing")
                            carried = {position: float(prior.get(position, 0.0)) for position in range(bank_length)}
                            masks.append(exact_budget_positions(
                                bank_length=bank_length, budget=int(budget), protected_positions=protected,
                                scores=carried, strategy="score_high",
                            ))
                    elif selector == "current_dflash_oracle":
                        masks = []
                        for layer_index in range(len(runtime["draft"].layers)):
                            scores = current_scores.get(layer_index)
                            if scores is None:
                                raise ValueError(f"current_dflash_layer_{layer_index}_missing")
                            masks.append(exact_budget_positions(
                                bank_length=bank_length, budget=int(budget), protected_positions=protected,
                                scores=scores, strategy="score_high",
                            ))
                    else:
                        raise ValueError(f"unsupported selector {selector}")
                    outcome = evaluate_replay_action(
                        runtime, input_ids, state, layout, masks,
                        requested_budget=effective_budget, selector_id=selector,
                        remaining_output_tokens=prefix.remaining_output_tokens,
                        ranking_origin_round=prefix.previous_dense_round_index if selector == "previous_dflash" else None,
                    )
                    if outcome.get("status") == "ok":
                        outcome["ranking_coverage"] = (
                            sum(len(previous_scores.get(i, {})) for i in range(len(runtime["draft"].layers)))
                            / max(len(runtime["draft"].layers) * bank_length, 1)
                            if selector == "previous_dflash" and previous_scores is not None else None
                        )
                        outcome["ranking_age"] = (
                            prefix.round_index - prefix.previous_dense_round_index
                            if selector == "previous_dflash" and prefix.previous_dense_round_index is not None else None
                        )
                    rows.append(_action_row(
                        experiment="A2_selector", phase=phase, sample=sample, prefix=prefix,
                        label=label, baseline_accepted=baseline_l, outcome=outcome,
                        nominal_budget=budget, random_seed=int(seed) if selector == "random" else None,
                        ranking_origin_round=prefix.previous_dense_round_index if selector == "previous_dflash" else None,
                        extra={"ranking_coverage": outcome.get("ranking_coverage"), "ranking_age": outcome.get("ranking_age"),
                               "oracle_diagnostic_only": selector == "current_dflash_oracle",
                               "canonicalized_full": canonical_full,
                               "previous_replay_check": previous_replay_check},
                    ))
                except ValueError as exc:
                    rows.append(_action_row(
                        experiment="A2_selector", phase=phase, sample=sample, prefix=prefix,
                        label=label, baseline_accepted=baseline_l,
                        outcome={"status": "infeasible" if "budget" in str(exc) or "protected" in str(exc) else "not_eligible",
                                 "reason": str(exc), "accepted_candidates": None},
                        nominal_budget=budget,
                        random_seed=int(seed) if selector == "random" else None,
                        ranking_origin_round=prefix.previous_dense_round_index if selector == "previous_dflash" else None,
                        extra={"oracle_diagnostic_only": selector == "current_dflash_oracle",
                               "canonicalized_full": canonical_full,
                               "previous_replay_check": previous_replay_check},
                    ))
    del state, full
    gc.collect()
    return rows


def run_a3_rollout(
    runtime: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    max_new_tokens: int,
    selector: str,
    budget: int,
    refresh_period: int,
    repetition: int = 0,
    source_anchors: int = 64,
    recent_output: int = 256,
) -> dict[str, Any]:
    """Run one online fixed-policy rollout and retain all overhead counters."""
    if selector not in {"recency", "target_parent", "previous_dflash"}:
        raise ValueError("A3 rollout supports recency, target_parent, or previous_dflash")
    ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=128)
    ids = ids.to(runtime["target"].device).contiguous()
    if not layout.source_available:
        raise RuntimeError(f"source span mapping is unavailable: {layout.source_reason}")
    use_refresh = selector == "previous_dflash"
    core_selector = "draft_refresh" if use_refresh else ("target_parent" if selector == "target_parent" else "recent_only")
    config = AdaptiveConfig(
        budgets=(int(budget), "full"),
        gammas=(15,),
        selector=core_selector,
        target_layer=TARGET_PARENT_LAYER,
        refresh_period=int(refresh_period),
        source_anchors=source_anchors,
        recent_output=recent_output,
        timing_mode="wall",
        output_root="outputs/dflash_context_interventions",
        seed=42,
    )
    policy = build_online_selection_policy(
        selector,
        budget=int(budget),
        source_anchors=source_anchors,
        recent_output=recent_output,
        random_seed=42,
    )
    torch.cuda.synchronize(ids.device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = generate_adaptive(
            runtime["target"], runtime["draft"], ids, runtime["tokenizer"], layout, config,
            max_new_tokens=max_new_tokens, temperature=0.0,
            variant="fixed", fixed_budget=int(budget), fixed_gamma=15, gamma_reference=15,
            sample_id=str(sample["id"]), dataset=str(sample["dataset"]),
            split=str(sample.get("_cad_split", "unknown")), repetition=repetition,
            selection_policy=policy,
            collect_full_target_scores=selector == "target_parent",
            collect_full_draft_scores=selector == "previous_dflash",
        )
    torch.cuda.synchronize(ids.device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema_version": "dflash-context-interventions.rollout.v1",
        "type": "rollout",
        "sample_id": str(sample["id"]),
        "dataset": str(sample["dataset"]),
        "source_group_id": str(sample.get("_cad_source_group_id", "")),
        "split": str(sample.get("_cad_split", "unknown")),
        "variant": selector,
        "budget": int(budget),
        "refresh_period": int(refresh_period),
        "repetition": int(repetition),
        "status": result.status,
        "error": result.error,
        "output_ids": [int(value) for value in result.output_ids.detach().cpu().tolist()],
        "output_tokens": int(result.output_tokens),
        "wall_e2e_ms": wall_ms,
        "reported_e2e_ms": result.timings.get("e2e_ms"),
        "timings": result.timings,
        "counters": result.counters,
        "rounds": result.rounds,
        "correctness_status": result.correctness_status,
    }


def run_a3_comparison(
    runtime: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    max_new_tokens: int,
    selector: str,
    budget: int,
    refresh_period: int,
    repetition: int = 0,
    warmups: int = 1,
    source_anchors: int = 64,
    recent_output: int = 256,
) -> list[dict[str, Any]]:
    """Paired AR/native Full/adapter Full/selected end-to-end timing run."""
    ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=128)
    ids = ids.to(runtime["target"].device).contiguous()
    if not layout.source_available:
        raise RuntimeError(f"source span mapping is unavailable: {layout.source_reason}")
    use_refresh = selector == "previous_dflash"
    core_selector = "draft_refresh" if use_refresh else ("target_parent" if selector == "target_parent" else "recent_only")
    full_config = AdaptiveConfig(
        budgets=("full",), gammas=(15,), selector="recent_only", target_layer=TARGET_PARENT_LAYER,
        source_anchors=source_anchors, recent_output=recent_output, timing_mode="wall",
    )
    reduced_config = AdaptiveConfig(
        budgets=(int(budget), "full"), gammas=(15,), selector=core_selector,
        target_layer=TARGET_PARENT_LAYER, refresh_period=int(refresh_period),
        source_anchors=source_anchors, recent_output=recent_output, timing_mode="wall",
    )
    policy = build_online_selection_policy(
        selector, budget=int(budget), source_anchors=source_anchors,
        recent_output=recent_output, random_seed=42,
    )

    def run_one(name: str) -> tuple[dict[str, Any], list[int]]:
        torch.cuda.synchronize(ids.device)
        started = time.perf_counter()
        with torch.inference_mode():
            if name == "ar":
                result = generate_target_only(runtime["target"], ids, max_new_tokens=max_new_tokens, temperature=0.0)
                output = [int(value) for value in result.output_ids.detach().cpu().tolist()]
                timings = result.timings
                counters = result.counters
                rounds = []
                status = result.status
                error = result.error
            elif name == "native_full_dflash":
                result = _native_full_dflash(runtime, ids, max_new_tokens)
                output = _ids(result, ids.shape[1], max_new_tokens)
                timings = {
                    "e2e_ms": None,
                    "ttft_ms": float(getattr(result, "time_to_first_token", 0.0)) * 1000.0,
                    "draft_ms": getattr(result, "draft_latency_ms", None),
                    "verify_ms": getattr(result, "verification_latency_ms", None),
                }
                acceptance_lengths = [int(length) for length in getattr(result, "acceptance_lengths", [])]
                counters = {
                    "acceptance_rate": getattr(result, "acceptance_rate", None),
                    "avg_accept_length": getattr(result, "avg_accept_length", None),
                    "mean_accept_length": (
                        sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else None
                    ),
                    "mean_accepted_candidates": (
                        sum(max(length - 1, 0) for length in acceptance_lengths) / len(acceptance_lengths)
                        if acceptance_lengths else None
                    ),
                }
                rounds = []
                status = "ok"
                error = None
            elif name == "adapter_full":
                result = generate_adaptive(
                    runtime["target"], runtime["draft"], ids, runtime["tokenizer"], layout, full_config,
                    max_new_tokens=max_new_tokens, temperature=0.0, variant="dflash_full_fixed",
                    fixed_budget="full", fixed_gamma=15, gamma_reference=15,
                    sample_id=str(sample["id"]), dataset=str(sample["dataset"]), repetition=repetition,
                )
                output = [int(value) for value in result.output_ids.detach().cpu().tolist()]
                timings, counters, rounds, status, error = result.timings, result.counters, result.rounds, result.status, result.error
            else:
                result = generate_adaptive(
                    runtime["target"], runtime["draft"], ids, runtime["tokenizer"], layout, reduced_config,
                    max_new_tokens=max_new_tokens, temperature=0.0, variant="fixed",
                    fixed_budget=int(budget), fixed_gamma=15, gamma_reference=15,
                    sample_id=str(sample["id"]), dataset=str(sample["dataset"]),
                    split=str(sample.get("_cad_split", "unknown")), repetition=repetition,
                    selection_policy=policy,
                    collect_full_target_scores=selector == "target_parent",
                    collect_full_draft_scores=selector == "previous_dflash",
                )
                output = [int(value) for value in result.output_ids.detach().cpu().tolist()]
                timings, counters, rounds, status, error = result.timings, result.counters, result.rounds, result.status, result.error
        if name in {"adapter_full", "selected"} and counters.get("mean_accept_length") is not None:
            counters = dict(counters)
            # Adaptive counters count the committed anchor plus accepted draft
            # candidates; native DFlash stores the same convention in
            # acceptance_lengths. Report the candidate count separately for
            # the preregistered ΔL non-inferiority comparison.
            counters["mean_accepted_candidates"] = max(float(counters["mean_accept_length"]) - 1.0, 0.0)
        torch.cuda.synchronize(ids.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        row = {
            "variant": name if name != "selected" else selector,
            "status": status,
            "error": error,
            "output_tokens": len(output),
            "wall_e2e_ms": wall_ms,
            "reported_timings": timings,
            "counters": counters,
            "rounds": rounds,
            "correctness_status": getattr(result, "correctness_status", "target_reference" if name == "ar" else None),
            "output_ids": output,
        }
        del result
        gc.collect()
        return row, output

    order = ["ar", "native_full_dflash", "adapter_full", "selected"]
    seed_text = f"{sample['id']}:{selector}:{budget}:{refresh_period}:{repetition}"
    random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)).shuffle(order)
    for warmup in range(max(0, int(warmups))):
        warmup_order = list(order)
        random.Random(f"{seed_text}:warmup:{warmup}").shuffle(warmup_order)
        for variant in warmup_order:
            run_one(variant)
        print(f"[A3 warmup] {sample['dataset']}:{sample['id']} {warmup + 1}/{warmups}", flush=True)
    measured: dict[str, dict[str, Any]] = {}
    outputs: dict[str, list[int]] = {}
    for variant in order:
        row, output = run_one(variant)
        measured[variant] = row
        outputs[variant] = output
    ar_output = outputs["ar"]
    records = []
    reference_text = str(sample.get("reference") or sample.get("reference_output") or
                         (sample.get("raw") or {}).get("reference_output") or
                         (((sample.get("raw") or {}).get("answers") or [""])[0]))
    for variant in order:
        row = measured[variant]
        generated_text = runtime["tokenizer"].decode(outputs[variant], skip_special_tokens=True)
        row.update({
            "schema_version": "dflash-context-interventions.rollout.v1",
            "type": "rollout",
            "sample_id": str(sample["id"]),
            "dataset": str(sample["dataset"]),
            "source_group_id": str(sample.get("_cad_source_group_id", "")),
            "split": str(sample.get("_cad_split", "unknown")),
            "selector": selector,
            "budget": int(budget),
            "refresh_period": int(refresh_period),
            "repetition": int(repetition),
            "matches_ar": outputs[variant] == ar_output,
            "first_mismatch": _first_mismatch(outputs[variant], ar_output),
            "matches_native_full_dflash": outputs[variant] == outputs["native_full_dflash"],
            "first_mismatch_native_full_dflash": _first_mismatch(
                outputs[variant], outputs["native_full_dflash"]
            ),
            "stopped_by": "eos" if any(token in _eos_ids(runtime["target"]) for token in outputs[variant]) else "max_new_tokens",
            "generated_text": generated_text,
            "reference_text": reference_text or None,
        })
        try:
            from common.rouge import add_rouge

            add_rouge(row, generated_text, reference_text)
        except Exception as exc:
            row["rouge_status"] = f"error:{type(exc).__name__}:{str(exc)[:200]}"
        records.append(row)
    return records


def run_a3_prefix(
    runtime: Mapping[str, Any],
    sample: Mapping[str, Any],
    input_ids: torch.Tensor,
    layout,
    prefix: ReferencePrefix,
    reference_output: Sequence[int],
    *,
    phase: str,
    selector: str,
    max_new_tokens: int,
    state: ReplayState | None = None,
    full_outcome: Mapping[str, Any] | None = None,
    replay_cache: dict[int, Mapping[str, Any]] | None = None,
    previous_prefix: ReferencePrefix | None = None,
    ranking_age: int | None = None,
    reference_accepted: int | None = None,
    previous_reference_accepted: int | None = None,
    source_anchors: int = 64,
    recent_output: int = 256,
) -> list[dict[str, Any]]:
    """Replay one deployable selector over the preregistered A3 budget curve."""
    if selector not in {"recency", "target_parent", "previous_dflash"}:
        raise ValueError("A3 budget sweep needs a deployable selector")
    if state is None:
        state = rebuild_replay_state(runtime, input_ids, reference_output, prefix, layout, max_new_tokens=max_new_tokens)
    bank_length = state.bank.length
    full_mask = tuple(range(bank_length))
    full = full_outcome or evaluate_replay_action(
        runtime, input_ids, state, layout,
        [full_mask for _ in runtime["draft"].layers],
        requested_budget="full", selector_id="full", remaining_output_tokens=prefix.remaining_output_tokens,
        collect_attention_scores=False,
    )
    if full.get("status") != "ok":
        raise RuntimeError(f"full DFlash replay failed at prefix {prefix.round_index}: {full.get('reason')}")
    baseline_l = int(full["accepted_candidates"])
    if reference_accepted is not None and baseline_l != int(reference_accepted):
        raise RuntimeError(
            f"full DFlash replay acceptance differs from reference trace at round {prefix.round_index}: "
            f"trace={reference_accepted}, replay={baseline_l}"
        )
    previous_scores: dict[int, dict[int, float]] | None = None
    previous_replay_check: dict[str, Any] | None = None
    if selector == "previous_dflash" and previous_prefix is not None:
        old_full = (replay_cache or {}).get(previous_prefix.round_index)
        if old_full is None:
            old_full = _replay_previous_dflash_round(
                runtime, input_ids, reference_output, previous_prefix, layout,
                max_new_tokens=max_new_tokens,
            )
            if replay_cache is not None:
                replay_cache[previous_prefix.round_index] = old_full
        previous_scores, previous_replay_check = _previous_dflash_evidence(
            old_full,
            reference_accepted=previous_reference_accepted,
            layer_count=len(runtime["draft"].layers),
            round_index=previous_prefix.round_index,
        )
        del old_full
    elif selector == "previous_dflash":
        previous_scores, previous_replay_check = _previous_dflash_evidence(
            None,
            reference_accepted=previous_reference_accepted,
            layer_count=len(runtime["draft"].layers),
            round_index=None,
        )

    ranking_origin = previous_prefix.round_index if previous_prefix is not None else None
    actual_ranking_age = prefix.round_index - ranking_origin if ranking_origin is not None else ranking_age

    protected = global_protected_positions(
        layout, bank_length=bank_length, source_anchors=source_anchors,
        processed_output_start=layout.prompt_tokens, recent_output=recent_output, protect_recent=True,
    )
    budgets: Sequence[int] = (512, 1024, 2048, 4096, 8192) if phase == "dev" else ((1024, 2048) if phase == "pilot" else (2048,))
    rows: list[dict[str, Any]] = []
    if selector != "previous_dflash" or actual_ranking_age in (None, 1):
        rows.append(_action_row(
            experiment="A3_budget_replay", phase=phase, sample=sample, prefix=prefix,
            label="full_Bfull", baseline_accepted=baseline_l,
            outcome={
                "status": "ok", "accepted_candidates": baseline_l,
                "actual_context_tokens_by_layer": [bank_length] * len(runtime["draft"].layers),
                "positions_sha256_by_layer": [hashlib.sha256(",".join(map(str, full_mask)).encode("ascii")).hexdigest()]
                    * len(runtime["draft"].layers),
                "draft_ms": full.get("draft_ms"), "verify_ms": full.get("verify_ms"),
            },
            nominal_budget="full", extra={
                "canonicalized_full": True,
                "selection_and_rebuild_included": False,
                "previous_replay_check": previous_replay_check,
            },
        ))
    for budget in budgets:
        full_budget = int(budget) >= bank_length
        requested: int | str = "full" if full_budget else int(budget)
        label = (
            f"{selector}_age{actual_ranking_age}_B{budget}"
            if selector == "previous_dflash" else f"{selector}_B{budget}"
        )
        if full_budget:
            rows.append(_action_row(
                experiment="A3_budget_replay", phase=phase, sample=sample, prefix=prefix,
                label=label, baseline_accepted=baseline_l,
                outcome={
                    "status": "canonicalized_alias",
                    "reason": "requested budget is at least the full context length; see full_Bfull",
                    "accepted_candidates": None,
                    "actual_context_tokens_by_layer": [bank_length] * len(runtime["draft"].layers),
                },
                nominal_budget=budget, ranking_origin_round=ranking_origin,
                extra={"canonicalized_full": True, "alias_of": "full_Bfull", "ranking_age": actual_ranking_age,
                       "previous_replay_check": previous_replay_check},
            ))
            continue
        try:
            if selector == "recency":
                selected = exact_budget_positions(
                    bank_length=bank_length, budget=int(budget), protected_positions=protected, strategy="recency"
                )
                masks = [selected for _ in runtime["draft"].layers]
            elif selector == "target_parent":
                selected = exact_budget_positions(
                    bank_length=bank_length, budget=int(budget), protected_positions=protected,
                    scores=state.target_scores[TARGET_PARENT_LAYER], strategy="score_high",
                )
                masks = [selected for _ in runtime["draft"].layers]
            else:
                if previous_scores is None:
                    detail = (previous_replay_check or {}).get("reason") or (
                        previous_replay_check or {}
                    ).get("status", "no_previous_dense_round")
                    raise ValueError(f"previous_dense_round_unavailable:{detail}")
                masks = []
                for layer_index in range(len(runtime["draft"].layers)):
                    score_map = previous_scores.get(layer_index)
                    if score_map is None:
                        raise ValueError(f"previous_dflash_layer_{layer_index}_missing")
                    carried = {position: float(score_map.get(position, 0.0)) for position in range(bank_length)}
                    masks.append(exact_budget_positions(
                        bank_length=bank_length, budget=int(budget), protected_positions=protected,
                        scores=carried, strategy="score_high",
                    ))
            outcome = evaluate_replay_action(
                runtime, input_ids, state, layout, masks,
                requested_budget=requested, selector_id=selector,
                remaining_output_tokens=prefix.remaining_output_tokens,
                ranking_origin_round=ranking_origin if selector == "previous_dflash" else None,
            )
            extra = {
                "canonicalized_full": False,
                "ranking_age": actual_ranking_age if selector == "previous_dflash" else None,
                "ranking_origin_round": ranking_origin if selector == "previous_dflash" else None,
                "ranking_coverage": (
                    sum(len(previous_scores.get(i, {})) for i in range(len(runtime["draft"].layers)))
                    / max(len(runtime["draft"].layers) * bank_length, 1)
                    if selector == "previous_dflash" and previous_scores is not None else None
                ),
                "component_draft_plus_verify_ms": (
                    float(outcome.get("draft_ms", 0.0)) + float(outcome.get("verify_ms", 0.0))
                    if outcome.get("status") == "ok" else None
                ),
                "selection_and_rebuild_included": False,
                "previous_replay_check": previous_replay_check,
            }
        except ValueError as exc:
            masks = [full_mask for _ in runtime["draft"].layers]
            outcome = {"status": "not_eligible" if "previous_dense" in str(exc) else "infeasible",
                       "reason": str(exc), "accepted_candidates": None,
                       "actual_context_tokens_by_layer": [len(mask) for mask in masks]}
            extra = {"canonicalized_full": False, "selection_and_rebuild_included": False,
                     "previous_replay_check": previous_replay_check}
        rows.append(_action_row(
            experiment="A3_budget_replay", phase=phase, sample=sample, prefix=prefix,
            label=label, baseline_accepted=baseline_l, outcome=outcome,
            nominal_budget=budget, ranking_origin_round=ranking_origin if selector == "previous_dflash" else None,
            extra=extra,
        ))
    del state, full
    gc.collect()
    return rows
