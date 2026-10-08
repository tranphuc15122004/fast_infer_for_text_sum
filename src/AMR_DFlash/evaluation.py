"""Fixed-state, same-prefix acceptance evaluation for AMR-DFlash."""

from __future__ import annotations

from typing import Any

import torch

from .core import greedy_acceptance, resolve_greedy_commit
from .memory import AMRMemory
from .model import AMRDFlashDraft


@torch.inference_mode()
def evaluate_one_fixed_state(
    target: Any,
    draft: AMRDFlashDraft,
    memory: AMRMemory,
    *,
    context_ids: torch.Tensor,
    projected_context: torch.Tensor,
    anchor_id: int,
    remaining_output_budget: int,
    mode: str,
    eos_token_ids: set[int],
) -> dict[str, Any]:
    """Draft and verify one frozen state without mutating its saved context."""
    if context_ids.ndim != 2 or context_ids.shape[0] != 1 or context_ids.shape[1] < 1:
        raise ValueError("fixed-state evaluation requires context_ids [1,N], N>=1")
    if projected_context.ndim != 3 or projected_context.shape[:2] != context_ids.shape:
        raise ValueError("projected_context must align with context_ids [1,N]")
    if remaining_output_budget < 1:
        raise ValueError("remaining_output_budget must be positive")
    if mode not in {"dense", "selection", "compressor", "amr"}:
        raise ValueError("mode must be dense, selection, compressor, or amr")

    device = context_ids.device
    context_length = int(context_ids.shape[1])
    prefix_positions = torch.arange(context_length, device=device).unsqueeze(0)
    prefix = target(
        input_ids=context_ids,
        position_ids=prefix_positions,
        use_cache=True,
        output_hidden_states=False,
        logits_to_keep=1,
        return_dict=True,
    )
    past = prefix.past_key_values
    anchor = torch.tensor([[anchor_id]], device=device, dtype=torch.long)
    anchor_embedding = target.get_input_embeddings()(anchor)[:, 0, :]
    live_positions = torch.arange(
        context_length,
        context_length + draft.block_size,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    draft_memory = memory.build(
        projected_context,
        anchor_embedding,
        live_positions,
        mode=mode,
        use_cost_gate=False,
    )
    block_ids = torch.full(
        (1, draft.block_size), draft.mask_token_id, device=device, dtype=torch.long
    )
    block_ids[0, 0] = int(anchor_id)
    noise_embedding = target.get_input_embeddings()(block_ids)
    position_ids = torch.cat((draft_memory.positions, live_positions), dim=1)
    draft_hidden = draft.forward_projected(
        projected_context=draft_memory.features,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        attention_mask=draft_memory.attention_mask,
    )
    proposal_logits = target.get_output_embeddings()(draft_hidden)[:, 1:, :]
    proposals = proposal_logits.argmax(dim=-1)
    candidate_ids = torch.cat((anchor, proposals), dim=1)
    verification = target(
        input_ids=candidate_ids,
        position_ids=live_positions,
        past_key_values=past,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )
    target_choices = verification.logits[:, :-1, :].argmax(dim=-1)
    acceptance = greedy_acceptance(proposals, target_choices)
    correction = int(verification.logits[0, acceptance.accepted].argmax().item())
    proposal_ids = [int(token) for token in proposals[0].tolist()]
    accepted_prefix = proposal_ids[: acceptance.accepted]
    commit = resolve_greedy_commit(
        accepted_prefix,
        accepted=acceptance.accepted,
        correction=correction,
        remaining=remaining_output_budget,
        eos_token_ids=eos_token_ids,
    )
    censored = bool(
        any(token in eos_token_ids for token in accepted_prefix)
        or correction in eos_token_ids
        or remaining_output_budget <= draft.block_size - 1
    )
    return {
        "record_type": "fixed_state",
        "mode": mode,
        "context_length": context_length,
        "anchor_id": int(anchor_id),
        "remaining_output_budget": int(remaining_output_budget),
        "accepted_proposals_raw": acceptance.accepted,
        "accepted_proposals_committed": len(commit.committed_proposals),
        "committed_tokens": len(commit.emitted_tokens),
        "survival": [
            int(index < acceptance.accepted) for index in range(draft.block_size - 1)
        ],
        "first_mismatch": acceptance.first_mismatch,
        "correction_token_id": correction,
        "proposal_ids": proposal_ids,
        "target_choices": [int(token) for token in target_choices[0].tolist()],
        "censored": censored,
        "stopped_on_eos": commit.stopped_on_eos,
        "raw_positions": [int(value) for value in draft_memory.raw_positions[0].tolist()],
        "raw_tokens": int(draft_memory.raw_positions.shape[1]),
        "slot_positions": [int(value) for value in draft_memory.slot_positions[0].tolist()],
        "slot_tokens": int(draft_memory.slot_valid_mask.sum().item()),
        "selection_was_bypassed": draft_memory.bypassed,
        "status": "success",
    }
