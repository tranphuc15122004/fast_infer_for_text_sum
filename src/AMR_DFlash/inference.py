"""Greedy AMR-DFlash generation with an exact cached target verifier."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .core import (
    CompressedMemory,
    CompressorState,
    MemoryConfig,
    greedy_acceptance,
    resolve_greedy_commit,
)
from .memory import AMRMemory
from .model import AMRDFlashDraft, extract_target_features


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _StageClock:
    """CPU wall intervals or CUDA events, resolved after the sample boundary."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.intervals: dict[str, list[Any]] = {}
        self.active: dict[str, Any] = {}

    def start(self, name: str) -> None:
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
        else:
            start = time.perf_counter()
        self.active[name] = start

    def stop(self, name: str) -> None:
        start = self.active.pop(name)
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
            interval = (start, end)
        else:
            interval = (time.perf_counter() - start) * 1000.0
        self.intervals.setdefault(name, []).append(interval)

    def total(self, name: str) -> float:
        intervals = self.intervals.get(name, [])
        if self.device.type == "cuda":
            return sum(start.elapsed_time(end) for start, end in intervals)
        return sum(intervals)


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
    decode_ms: float = 0.0
    memory_update_ms: float = 0.0
    feature_projection_ms: float = 0.0


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
        if mode not in {"dense", "selection", "compressor", "amr"}:
            raise ValueError("mode must be dense, selection, compressor, or amr")
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
        logits_to_keep: int = 0,
    ) -> Any:
        return self.target(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=output_hidden_states,
            logits_to_keep=logits_to_keep,
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
        clock = _StageClock(self.device)
        position_ids = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        clock.start("prefill")
        prefill = self._target_forward(
            ids,
            position_ids=position_ids,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        clock.stop("prefill")
        past = prefill.past_key_values
        first_token = int(prefill.logits[:, -1, :].argmax(dim=-1)[0].item())
        clock.start("projection")
        raw_features = extract_target_features(prefill, self.draft.target_layer_ids)
        projected = self.draft.project_context(raw_features)
        clock.stop("projection")
        del raw_features, prefill
        compressor_state: CompressorState | None = None
        selector_keys: torch.Tensor | None = None

        generated = [first_token]
        output_ids = torch.cat((ids, torch.tensor([[first_token]], device=self.device)), dim=1)
        _sync(self.device)
        ttft_ms = (time.perf_counter() - e2e_start) * 1000.0
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
                prefill_ms=clock.total("prefill"),
                draft_ms=0.0,
                selector_ms=0.0,
                verification_ms=0.0,
                e2e_ms=total_ms,
                peak_memory_gb=self._peak_memory_gb(),
                projected_context=projected.detach() if capture_states else None,
                states=states,
                trace=trace,
                decode_ms=max(0.0, total_ms - ttft_ms),
                feature_projection_ms=clock.total("projection"),
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
            clock.start("build")
            selector_active = (
                self.mode in {"selection", "amr"}
                and (
                    not self.use_cost_gate
                    or context_length >= self.memory.config.min_context_tokens
                )
            )
            if selector_active and selector_keys is None:
                selector_keys = self.memory.selector.encode_keys(projected)
            compressed_override: CompressedMemory | None = None
            slots_enabled = (
                self.mode in {"amr", "compressor"}
                and self.memory.config.num_slots > 0
                and (
                    not self.use_cost_gate
                    or context_length >= self.memory.config.min_context_tokens
                )
            )
            if slots_enabled:
                compressor_dtype = self.memory.compressor.value_projection.weight.dtype
                if compressor_state is None:
                    compressor_state = self.memory.compressor.empty_state(
                        batch_size=1,
                        device=self.device,
                        dtype=compressor_dtype,
                    )
                    self.memory.compressor.update(
                        compressor_state,
                        projected.to(compressor_dtype),
                        torch.arange(context_length, device=self.device).unsqueeze(0),
                    )
                slot_values, slot_positions = self.memory.compressor.finalize(compressor_state)
                compressed_override = CompressedMemory(
                    values=slot_values,
                    positions=slot_positions,
                    valid_mask=compressor_state.weight_sum > 0,
                )
            draft_memory = self.memory.build(
                projected,
                anchor_embedding,
                live_positions,
                mode=self.mode,
                use_cost_gate=self.use_cost_gate,
                compressed_override=compressed_override,
                selector_keys=selector_keys,
            )
            clock.stop("build")
            if capture_states:
                states.append(
                    {
                        "context_length": context_length,
                        "anchor_id": anchor_id,
                        "anchor_embedding": anchor_embedding[0].detach().cpu(),
                        "remaining_output_budget": max_new_tokens - len(generated),
                        "bypassed": draft_memory.bypassed,
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
            clock.start("draft")
            draft_hidden = self.draft.forward_projected(
                projected_context=draft_memory.features,
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                attention_mask=draft_memory.attention_mask,
            )
            proposal_logits = self.lm_head(draft_hidden)[:, 1:, :]
            proposal_ids = proposal_logits.argmax(dim=-1)
            clock.stop("draft")

            candidate_ids = torch.cat(
                (torch.tensor([[anchor_id]], device=self.device), proposal_ids), dim=1
            )
            clock.start("verification")
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
            acceptance_lengths.append(accepted + 1)
            clock.stop("verification")

            correction = int(verification.logits[0, accepted].argmax().item())
            proposal_list = [int(value) for value in proposal_ids[0, :accepted].tolist()]
            remaining = max_new_tokens - len(generated)
            commit = resolve_greedy_commit(
                proposal_list,
                accepted=accepted,
                correction=correction,
                remaining=remaining,
                eos_token_ids=eos_ids,
            )
            committed_proposals = commit.committed_proposals
            emitted = commit.emitted_tokens
            processed_count = commit.processed_tokens
            stop_on_eos = commit.stopped_on_eos

            if emitted:
                generated.extend(emitted)
                output_ids = torch.cat(
                    (
                        output_ids,
                        torch.tensor([emitted], device=self.device, dtype=torch.long),
                    ),
                    dim=1,
                )
            accepted_tokens += len(committed_proposals)
            # The verifier cache contains the full speculative suffix. Keep
            # only anchor + target-accepted proposals; the correction remains
            # a pending anchor and is processed at the next block.
            new_cache_length = cursor + processed_count
            past.crop(new_cache_length)
            clock.start("projection")
            accepted_features = extract_target_features(
                verification, self.draft.target_layer_ids
            )[:, :processed_count, :]
            projected_addition = self.draft.project_context(accepted_features)
            clock.stop("projection")
            clock.start("update")
            if selector_keys is not None:
                selector_keys = torch.cat(
                    (
                        selector_keys,
                        self.memory.selector.encode_keys(projected_addition),
                    ),
                    dim=1,
                )
            if compressor_state is not None:
                self.memory.compressor.update(
                    compressor_state,
                    projected_addition.to(
                        self.memory.compressor.value_projection.weight.dtype
                    ),
                    torch.arange(
                        cursor,
                        cursor + processed_count,
                        device=self.device,
                    ).unsqueeze(0),
                )
            projected = torch.cat((projected, projected_addition), dim=1)
            clock.stop("update")
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
            prefill_ms=clock.total("prefill"),
            draft_ms=clock.total("draft"),
            selector_ms=clock.total("build") + clock.total("update"),
            verification_ms=clock.total("verification"),
            e2e_ms=total_ms,
            peak_memory_gb=self._peak_memory_gb(),
            projected_context=projected.detach() if capture_states else None,
            states=states,
            trace=trace,
            decode_ms=max(0.0, total_ms - ttft_ms),
            memory_update_ms=clock.total("update"),
            feature_projection_ms=clock.total("projection"),
        )

    def _peak_memory_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device) / (1024**3))
