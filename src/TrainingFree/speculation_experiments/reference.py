"""Reference-rollout prefix sampling and strictly causal history features."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from TrainingFree.context_interventions.replay import ReferencePrefix


def select_reference_prefixes(
    rounds: Sequence[Mapping[str, Any]],
    output_ids: Sequence[int],
    *,
    prompt_length: int,
    max_new_tokens: int,
    eos_ids: set[int],
    max_prefixes: int = 15,
) -> list[ReferencePrefix]:
    """Choose up to five regular reference states in each output third.

    States need at least a full gamma-15 block before natural EOS and the
    configured cap. The selection rule may inspect the completed reference
    output; these are diagnostic states and not deployable controller inputs.
    """

    if prompt_length < 1 or max_new_tokens < 1 or max_prefixes < 1:
        raise ValueError("prompt_length, max_new_tokens, and max_prefixes must be positive")
    generated = [int(token) for token in output_ids[:max_new_tokens]]
    natural_end = next((index for index, token in enumerate(generated) if token in eos_ids), len(generated))
    if natural_end < 1:
        return []
    candidates: list[tuple[int, int]] = []
    for row in rounds:
        if row.get("type") != "round" or int(row.get("gamma_executed", 0)) != 15:
            continue
        round_index = int(row.get("round_index", -1))
        processed = int(row.get("processed_output_before", int(row.get("logical_length_before", prompt_length)) - prompt_length))
        if round_index < 0 or processed < 0 or processed >= natural_end:
            continue
        if natural_end - processed < 16 or max_new_tokens - processed < 16:
            continue
        candidates.append((round_index, processed))
    if not candidates:
        return []
    selected: set[tuple[int, int]] = set()
    per_stratum = max(1, min(5, max_prefixes // 3))
    for stratum in range(3):
        left = natural_end * stratum / 3.0
        right = natural_end * (stratum + 1) / 3.0
        local = [item for item in candidates if left <= item[1] < right]
        if not local:
            continue
        count = min(per_stratum, len(local))
        targets = [left + (index + 0.5) * (right - left) / count for index in range(count)]
        for target in targets:
            nearest = min(local, key=lambda item: (abs(item[1] - target), item[1], item[0]))
            selected.add(nearest)
    # Small/short traces can leave one stratum empty. Fill remaining slots with
    # states farthest from already selected output positions, with early ties.
    remaining = [item for item in candidates if item not in selected]
    while remaining and len(selected) < max_prefixes:
        if selected:
            next_row = max(
                remaining,
                key=lambda item: (
                    min(abs(item[1] - chosen[1]) for chosen in selected),
                    -item[1],
                    -item[0],
                ),
            )
        else:
            next_row = remaining[0]
        selected.add(next_row)
        remaining.remove(next_row)
    selected_rows = sorted(selected, key=lambda item: item[0])[:max_prefixes]
    result: list[ReferencePrefix] = []
    for round_index, processed in selected_rows:
        anchor = generated[processed]
        prefix_hash = hashlib.sha256(
            (",".join(map(str, generated[:processed])) + f"|{anchor}|{round_index}").encode("ascii")
        ).hexdigest()
        previous = [
            (int(row.get("round_index", -1)), int(row.get("accepted_candidates", 0)))
            for row in rounds
            if row.get("type") == "round"
            and int(row.get("gamma_executed", 0)) == 15
            and int(row.get("round_index", -1)) < round_index
        ]
        previous_round = max((index for index, _ in previous), default=None)
        result.append(ReferencePrefix(
            round_index=round_index,
            processed_output_tokens=processed,
            pending_anchor=anchor,
            remaining_output_tokens=max_new_tokens - processed,
            prefix_hash=prefix_hash,
            previous_dense_round_index=previous_round,
        ))
    return result


def reference_history(
    rounds: Sequence[Mapping[str, Any]], *, current_round: int, gamma: int = 15
) -> dict[str, Any]:
    """Summarize only full-reference rounds with index less than current_round."""

    prior = sorted(
        (
            (int(row["round_index"]), int(row["accepted_candidates"]))
            for row in rounds
            if row.get("type") == "round"
            and int(row.get("gamma_executed", 0)) == gamma
            and int(row.get("round_index", -1)) < current_round
            and row.get("accepted_candidates") is not None
        ),
        key=lambda value: value[0],
    )
    if not prior:
        return {
            "previous_accepted_candidates": None,
            "mean4_acceptance": None,
            "ema_acceptance": None,
            "previous_all_accepted": None,
            "missing": True,
            "origin_rounds": [],
        }
    normalized = [(index, accepted / float(gamma)) for index, accepted in prior]
    ema = normalized[0][1]
    for _, value in normalized[1:]:
        ema = 0.1 * value + 0.9 * ema
    last_index, last_value = normalized[-1]
    recent = normalized[-4:]
    last_accepted = int(prior[-1][1])
    return {
        "previous_accepted_candidates": last_accepted,
        "mean4_acceptance": sum(value for _, value in recent) / len(recent),
        "ema_acceptance": ema,
        "previous_all_accepted": bool(last_accepted == gamma),
        "missing": False,
        "origin_rounds": [index for index, _ in prior],
        "latest_origin_round": last_index,
        "latest_acceptance_fraction": last_value,
    }
