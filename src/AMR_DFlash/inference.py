"""Greedy AMR-DFlash generation with an exact cached target verifier."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .core import MemoryConfig, greedy_acceptance
from .memory import AMRMemory
from .model import AMRDFlashDraft, extract_target_features


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class GenerationResult:
    output_ids: torch.Tensor
    acceptance_lengths: list[int]
    accepted_proposals: list[int]
    proposed_tokens: int
    accepted_tokens: int
    ttft_ms: float
    prefill_ms: float
    draft_ms: float
    selector_ms: float
    verification_ms: float
    e2e_ms: float
    peak_memory_gb: float
    projected_context: torch.Tensor | None = None
    states: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


class AMRDFlashEngine:
    """Batch-1 greedy draft/verify engine; target always sees its full prefix."""

    def __init__(
        self,
        target: Any,
        tokenizer: Any,
        draft: AMRDFlashDraft,
        memory: AMRMemory,
        *,
        device: torch.device | None = None,
        mode: str = "amr",
        use_cost_gate: bool = True,
    ) -> None:
        self.target = target.eval()
        self.tokenizer = tokenizer
        self.draft = draft.eval()
        self.memory = memory.eval()
        self.device = device or next(target.parameters()).device
        self.mode = mode
        self.use_cost_gate = use_cost_gate
        if mode not in {"dense", "selection", "amr"}:
            raise ValueError("mode must be dense, selection, or amr")
        if target.get_input_embeddings() is None or target.get_output_embeddings() is None:
            raise ValueError("target model must expose input embeddings and output head")
        if self.draft.block_size != 16:
            raise ValueError("AMR-DFlash V0 requires block_size=16")

    @property
    def embed_tokens(self):
        return self.target.get_input_embeddings()

    @property
    def lm_head(self):
        return self.target.get_output_embeddings()

    def _target_forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        past_key_values: Any = None,
        output_hidden_states: bool,
    ) -> Any:
        return self.target(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: list[int] | tuple[int, ...] | int | None = None,
        capture_states: bool = False,
    ) -> GenerationResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("AMR-DFlash V0 requires input_ids with shape [1,N], N>=1")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.mode != "dense" and self.memory.selector is None:
            raise ValueError("selector is required for selection or AMR mode")
        ids = input_ids.to(device=self.device, dtype=torch.long).contiguous()
        if eos_token_ids is None:
            eos_value = getattr(self.tokenizer, "eos_token_id", None)
        else:
            eos_value = eos_token_ids
        if eos_value is None:
            eos_ids: set[int] = set()
        elif isinstance(eos_value, int):
            eos_ids = {int(eos_value)}
        else:
            eos_ids = {int(value) for value in eos_value}

        _sync(self.device)
        e2e_start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        position_ids = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        prefill_start = time.perf_counter()
        prefill = self._target_forward(
            ids,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        _sync(self.device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        past = prefill.past_key_values
        first_token = int(prefill.logits[:, -1, :].argmax(dim=-1)[0].item())
        raw_features = extract_target_features(prefill, self.draft.target_layer_ids)
        projected = self.draft.project_context(raw_features)
        del raw_features, prefill

        generated = [first_token]
        output_ids = torch.cat((ids, torch.tensor([[first_token]], device=self.device)), dim=1)
        ttft_ms = prefill_ms
        draft_ms = selector_ms = verification_ms = 0.0
        acceptance_lengths: list[int] = []
        accepted_proposals: list[int] = []
        proposed_tokens = accepted_tokens = 0
        states: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        if first_token in eos_ids:
            _sync(self.device)
            total_ms = (time.perf_counter() - e2e_start) * 1000.0
            return GenerationResult(
                output_ids=output_ids,
                acceptance_lengths=acceptance_lengths,
                accepted_proposals=accepted_proposals,
                proposed_tokens=0,
                accepted_tokens=0,
                ttft_ms=ttft_ms,
                prefill_ms=prefill_ms,
                draft_ms=0.0,
                selector_ms=0.0,
                verification_ms=0.0,
                e2e_ms=total_ms,
                peak_memory_gb=self._peak_memory_gb(),
                projected_context=projected.detach() if capture_states else None,
                states=states,
                trace=trace,
            )

        while len(generated) < max_new_tokens:
            context_length = int(past.get_seq_length())
            anchor_id = int(output_ids[0, context_length].item())
            cursor = context_length
            live_positions = torch.arange(
                cursor,
                cursor + self.draft.block_size,
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(0)
            anchor_embedding = self.embed_tokens(
                torch.tensor([[anchor_id]], device=self.device, dtype=torch.long)
            )[:, 0, :]
            selector_start = time.perf_counter()
            draft_memory = self.memory.build(
                projected,
                anchor_embedding,
                live_positions,
                mode=self.mode,
                use_cost_gate=self.use_cost_gate,
            )
            _sync(self.device)
            selector_ms += (time.perf_counter() - selector_start) * 1000.0
            if capture_states:
                states.append(
                    {
                        "context_length": context_length,
                        "anchor_id": anchor_id,
                        "anchor_embedding": anchor_embedding[0].detach().cpu(),
                        "remaining_output_budget": max_new_tokens - len(generated),
                        "bypassed": draft_memory.bypassed,
                        "raw_positions": draft_memory.raw_positions[0].detach().cpu(),
                        "slot_positions": draft_memory.slot_positions[0].detach().cpu(),
                    }
                )
            block_ids = torch.full(
                (1, self.draft.block_size),
                self.draft.mask_token_id,
                device=self.device,
                dtype=torch.long,
            )
            block_ids[0, 0] = anchor_id
            noise_embedding = self.embed_tokens(block_ids)
            position_ids = torch.cat(
                (draft_memory.positions, live_positions), dim=1
            )
            draft_start = time.perf_counter()
            draft_hidden = self.draft.forward_projected(
                projected_context=draft_memory.features,
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                attention_mask=draft_memory.attention_mask,
            )
            proposal_logits = self.lm_head(draft_hidden)[:, 1:, :]
            proposal_ids = proposal_logits.argmax(dim=-1)
            _sync(self.device)
            block_draft_ms = (time.perf_counter() - draft_start) * 1000.0
            draft_ms += block_draft_ms

            candidate_ids = torch.cat(
                (torch.tensor([[anchor_id]], device=self.device), proposal_ids), dim=1
            )
            verify_start = time.perf_counter()
            verification = self._target_forward(
                candidate_ids,
                position_ids=live_positions,
                past_key_values=past,
                output_hidden_states=True,
            )
            target_choices = verification.logits[:, :-1, :].argmax(dim=-1)
            acceptance = greedy_acceptance(proposal_ids, target_choices)
            accepted = acceptance.accepted
            proposed_tokens += int(proposal_ids.shape[1])
            accepted_proposals.append(accepted)
            accepted_tokens += accepted
            acceptance_lengths.append(accepted + 1)
            _sync(self.device)
            verification_ms += (time.perf_counter() - verify_start) * 1000.0

            correction = int(verification.logits[0, accepted].argmax().item())
            proposal_list = [int(value) for value in proposal_ids[0, :accepted].tolist()]
            remaining = max_new_tokens - len(generated)
            eos_in_acceptance = next(
                (index for index, token in enumerate(proposal_list) if token in eos_ids),
                None,
            )
            stop_on_eos = eos_in_acceptance is not None
            if stop_on_eos:
                committed_proposals = proposal_list[: eos_in_acceptance + 1]
                emitted = committed_proposals
                processed_count = 1 + len(committed_proposals)
            elif accepted >= remaining:
                committed_proposals = proposal_list[:remaining]
                emitted = committed_proposals
                processed_count = 1 + len(committed_proposals)
            else:
                committed_proposals = proposal_list
                emitted = committed_proposals + [correction]
                processed_count = 1 + len(committed_proposals)

            emitted = emitted[:remaining]
            if emitted:
                generated.extend(emitted)
                output_ids = torch.cat(
                    (
                        output_ids,
                        torch.tensor([emitted], device=self.device, dtype=torch.long),
                    ),
                    dim=1,
                )
            # The verifier cache contains the full speculative suffix. Keep
            # only anchor + target-accepted proposals; the correction remains
            # a pending anchor and is processed at the next block.
            new_cache_length = cursor + processed_count
            past.crop(new_cache_length)
            accepted_features = extract_target_features(
                verification, self.draft.target_layer_ids
            )[:, :processed_count, :]
            projected_addition = self.draft.project_context(accepted_features)
            projected = torch.cat((projected, projected_addition), dim=1)
            del accepted_features, projected_addition
            trace.append(
                {
                    "accepted_proposals": accepted,
                    "first_mismatch": acceptance.first_mismatch,
                    "committed_proposals": len(committed_proposals),
                    "emitted_tokens": len(emitted),
                    "bypassed": draft_memory.bypassed,
                    "raw_tokens": int(draft_memory.raw_positions.shape[1]),
                    "slot_tokens": int(draft_memory.slot_positions.shape[1]),
                    "cursor": cursor,
                }
            )
            del verification, draft_hidden, proposal_logits, target_choices
            if stop_on_eos or any(token in eos_ids for token in emitted):
                break

        _sync(self.device)
        total_ms = (time.perf_counter() - e2e_start) * 1000.0
        return GenerationResult(
            output_ids=output_ids,
            acceptance_lengths=acceptance_lengths,
            accepted_proposals=accepted_proposals,
            proposed_tokens=proposed_tokens,
            accepted_tokens=accepted_tokens,
            ttft_ms=ttft_ms,
            prefill_ms=prefill_ms,
            draft_ms=draft_ms,
            selector_ms=selector_ms,
            verification_ms=verification_ms,
            e2e_ms=total_ms,
            peak_memory_gb=self._peak_memory_gb(),
            projected_context=projected.detach() if capture_states else None,
            states=states,
            trace=trace,
        )

    def _peak_memory_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device) / (1024**3))
