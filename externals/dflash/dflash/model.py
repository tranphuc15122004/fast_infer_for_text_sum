import time
import torch
from types import SimpleNamespace
from typing import Any, Callable, Optional
from typing_extensions import Unpack
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    model: "DFlashDraftModel",
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    return_stats: bool = False,
    verification_prefix_controller: Optional[Any] = None,
    calibration_collector: Optional[list[Any]] = None,
    phase_a_selector: Optional[Any] = None,
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id
    if phase_a_selector is not None:
        phase_a_selector.reset()
    phase_a_controller = getattr(phase_a_selector, "controller", None)
    phase_a_adaptive = phase_a_controller is not None
    phase_a_round_trace = []
    phase_a_budget_trajectory = []
    phase_a_pending_started = None
    phase_a_pending_budget = None
    phase_a_pending_context = None
    phase_a_pending_fallback_reason = None
    phase_a_controller_time_ms = 0.0

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    if phase_a_selector is not None and block_size > 1:
        phase_a_selector.begin_verification_round(prefill=True)
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
        if phase_a_selector is not None:
            try:
                focused_indices = phase_a_selector.end_verification_round(
                    past_key_values_target,
                    accepted_length=1,
                    total_length=num_input_tokens,
                    prefill=True,
                )
                model.config._phase_a_focused_indices = focused_indices
                phase_a_pending_fallback_reason = phase_a_selector.last_fallback_reason
            except Exception as exc:
                model.config._phase_a_focused_indices = None
                phase_a_selector.selected_indices = None
                phase_a_selector.last_fallback_reason = f"prefill_selector_error:{type(exc).__name__}"
                phase_a_pending_fallback_reason = phase_a_selector.last_fallback_reason
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    proposed_lengths = []
    draft_events = []
    verification_events = []
    start = num_input_tokens
    draft_prefill = True
    last_parent_entropy = None
    last_parent_confidence = None

    while start < max_length:
        round_start_position = start
        round_budget_used = (
            int(phase_a_selector.current_budget)
            if phase_a_selector is not None else None
        )
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        draft_round_events = None
        verify_round_events = None
        if block_size > 1:
            draft_start_event = (
                torch.cuda.Event(enable_timing=True)
                if (return_stats and torch.cuda.is_available())
                else None
            )
            if draft_start_event is not None:
                draft_start_event.record()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, 1 - block_size :, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_start_event is not None:
                draft_end_event = torch.cuda.Event(enable_timing=True)
                draft_end_event.record()
                draft_events.append((draft_start_event, draft_end_event))
                draft_round_events = (draft_start_event, draft_end_event)
            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _cuda_time()

        # Phase B: adaptive verification prefix length selection
        verify_k = block_size - 1 if block_size > 1 else 0
        if verification_prefix_controller is not None and block_size > 1:
            try:
                if getattr(verification_prefix_controller, "should_bypass_signals", None) and verification_prefix_controller.should_bypass_signals(start):
                    verify_k = getattr(verification_prefix_controller, "fallback_prefix", block_size - 1)
                else:
                    from TrainingFree.speculation_experiments.signals import summarize_draft_logits

                    req_entropy = getattr(verification_prefix_controller, "requires_entropy", False)
                    draft_unc = summarize_draft_logits(draft_logits, include_entropy=req_entropy)
                    allowed_p = getattr(verification_prefix_controller, "allowed_prefixes", (3, 7, 11, 15))
                    avail_p = [p for p in allowed_p if p <= min(block_size - 1, max_length - start)]
                if not avail_p:
                    avail_p = [min(block_size - 1, max_length - start)]
                curr_state = SimpleNamespace(
                    logical_length=start,
                    processed_output_tokens=start - num_input_tokens,
                    parent_entropy=last_parent_entropy,
                    parent_confidence=last_parent_confidence,
                    history_acceptance=(
                        (sum(acceptance_lengths) / (len(acceptance_lengths) * (block_size - 1)))
                        if acceptance_lengths else None
                    ),
                )
                decision = verification_prefix_controller(
                    state=curr_state,
                    draft_uncertainty=draft_unc,
                    draft_gamma=block_size - 1,
                    available_prefixes=avail_p,
                )
                chosen_k = int(decision.get("prefix", block_size - 1))
                verify_k = max(1, min(chosen_k, block_size - 1, max_length - start))
            except Exception:
                verify_k = min(block_size - 1, max_length - start)

        verify_len = verify_k + 1
        curr_verify_ids = block_output_ids[:, :verify_len]
        curr_verify_pos = block_position_ids[:, :verify_len]

        verification_start_event = (
            torch.cuda.Event(enable_timing=True)
            if (return_stats and torch.cuda.is_available())
            else None
        )
        if verification_start_event is not None:
            verification_start_event.record()
        if phase_a_selector is not None:
            phase_a_selector.begin_verification_round()
        output = target(
            curr_verify_ids,
            position_ids=curr_verify_pos,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (curr_verify_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        if verification_start_event is not None:
            verification_end_event = torch.cuda.Event(enable_timing=True)
            verification_end_event.record()
            verification_events.append(
                (verification_start_event, verification_end_event)
            )
            verify_round_events = (verification_start_event, verification_end_event)
        output_ids[:, start : start + acceptance_length + 1] = curr_verify_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        acceptance_lengths.append(acceptance_length + 1)
        proposed_lengths.append(verify_k)

        committed_tokens = min(acceptance_length + 1, max_length - round_start_position)
        measured_round_ms = None
        measured_budget = phase_a_pending_budget
        observation_used = False
        observation_skip_reason = None
        draft_round_ms = None
        verify_round_ms = None
        if phase_a_adaptive:
            # The acceptance .item() above synchronizes verification. Timing
            # starts before the previous controller/selection and therefore
            # accounts for their cost with the action they prepared.
            if phase_a_pending_started is not None:
                measured_round_ms = max(0.0, (time.perf_counter() - phase_a_pending_started) * 1000.0)
                measured_budget = phase_a_pending_budget
            if draft_round_events is not None and draft_round_events[1].query():
                draft_round_ms = float(draft_round_events[0].elapsed_time(draft_round_events[1]))
            if verify_round_events is not None and verify_round_events[1].query():
                verify_round_ms = float(verify_round_events[0].elapsed_time(verify_round_events[1]))

        if phase_a_selector is not None:
            next_action_started = time.perf_counter() if phase_a_adaptive else None
            decision_started = time.perf_counter() if phase_a_adaptive else None
            next_budget = int(phase_a_selector.current_budget)
            decision_reason = "fixed_budget"
            if phase_a_adaptive:
                if measured_round_ms is not None and measured_budget is not None:
                    if phase_a_pending_fallback_reason is None:
                        phase_a_controller.observe(
                            budget=measured_budget,
                            round_ms=measured_round_ms,
                            committed_tokens=committed_tokens,
                            context_length=int(phase_a_pending_context or round_start_position),
                            draft_ms=draft_round_ms,
                            verification_ms=verify_round_ms,
                        )
                        observation_used = True
                    else:
                        observation_skip_reason = phase_a_pending_fallback_reason
                if start >= max_length:
                    next_budget = None
                    decision_reason = "generation_complete"
                else:
                    next_budget, decision_reason = phase_a_controller.choose(
                        context_length=start,
                        allow_exploration=(max_length - start > block_size),
                    )
            decision_ms = (
                (time.perf_counter() - decision_started) * 1000.0
                if decision_started is not None else 0.0
            )
            selection_launch_ms = 0.0
            selected_tokens = None
            if next_budget is not None:
                selector_started = time.perf_counter() if phase_a_adaptive else None
                try:
                    focused_indices = phase_a_selector.end_verification_round(
                        past_key_values_target,
                        accepted_length=acceptance_length + 1,
                        total_length=start,
                        budget_override=next_budget if phase_a_adaptive else None,
                    )
                    model.config._phase_a_focused_indices = focused_indices
                except Exception as exc:
                    model.config._phase_a_focused_indices = None
                    phase_a_selector.selected_indices = None
                    phase_a_selector.last_fallback_reason = f"selector_error:{type(exc).__name__}"
                if selector_started is not None:
                    selection_launch_ms = (time.perf_counter() - selector_started) * 1000.0
                selected_tokens = (
                    int(phase_a_selector.selected_indices.shape[0])
                    if phase_a_selector.selected_indices is not None else None
                )
            if phase_a_adaptive:
                phase_a_controller_time_ms += decision_ms
                if next_budget is not None:
                    phase_a_budget_trajectory.append(int(next_budget))
                phase_a_round_trace.append({
                    "round": len(acceptance_lengths),
                    "budget_used": int(measured_budget if measured_budget is not None else round_budget_used),
                    "next_budget": int(next_budget) if next_budget is not None else None,
                    "selected_context_tokens_for_next_round": selected_tokens,
                    "committed_tokens": int(committed_tokens),
                    "round_ms": round(measured_round_ms, 4) if measured_round_ms is not None else None,
                    "draft_cuda_ms": round(draft_round_ms, 4) if draft_round_ms is not None else None,
                    "verification_cuda_ms": round(verify_round_ms, 4) if verify_round_ms is not None else None,
                    "controller_decision_ms": round(decision_ms, 4),
                    "selection_launch_ms": round(selection_launch_ms, 4),
                    "decision": decision_reason,
                    "measurement_valid": measured_round_ms is not None,
                    "controller_observation": observation_used,
                    "observation_skip_reason": observation_skip_reason,
                    "action_fallback_reason": phase_a_pending_fallback_reason,
                    "next_selection_fallback_reason": phase_a_selector.last_fallback_reason,
                })
                if next_budget is not None:
                    phase_a_pending_started = next_action_started
                    phase_a_pending_budget = int(next_budget)
                    phase_a_pending_context = int(start)
                    phase_a_pending_fallback_reason = phase_a_selector.last_fallback_reason
                else:
                    phase_a_pending_started = None
                    phase_a_pending_budget = None
                    phase_a_pending_context = None
                    phase_a_pending_fallback_reason = None

        if calibration_collector is not None and block_size > 1:
            try:
                cand_k = block_size - 1
                top_logits, _ = torch.topk(draft_logits[0, :cand_k].float().detach(), k=2, dim=-1)
                margins = (top_logits[:, 0] - top_logits[:, 1]).cpu().tolist()
                confidences = torch.softmax(draft_logits[0, :cand_k].float().detach(), dim=-1).max(dim=-1).values.cpu().tolist()
                calibration_collector.append({
                    "accepted_candidates": int(acceptance_length),
                    "draft_uncertainty": {
                        "confidence": confidences,
                        "probability_margin": [max(0.0, 2.0 * float(c) - 1.0) for c in confidences],
                        "logit_margin": margins,
                    },
                    "source_group_id": "warmup",
                })
            except Exception:
                pass

        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    if hasattr(getattr(model, "config", None), "_phase_a_focused_indices"):
        model.config._phase_a_focused_indices = None

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    def _event_total_ms(events):
        return round(sum(start.elapsed_time(end) for start, end in events), 3) if events else 0.0

    proposed = sum(proposed_lengths) if proposed_lengths else sum(max(int(block_size) - 1, 0) for _ in acceptance_lengths)
    accepted = sum(max(int(length) - 1, 0) for length in acceptance_lengths)
    acceptance_rate = accepted / proposed if proposed else None
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
        draft_latency_ms=_event_total_ms(draft_events),
        verification_latency_ms=_event_total_ms(verification_events),
        draft_tokens_proposed=proposed,
        draft_tokens_accepted=accepted,
        acceptance_rate=round(acceptance_rate, 4)
        if acceptance_rate is not None else None,
        rejected_draft_ratio=round(1.0 - acceptance_rate, 4)
        if acceptance_rate is not None else None,
        avg_accept_length=(
            round(sum(acceptance_lengths) / len(acceptance_lengths), 4)
            if acceptance_lengths else None
        ),
        proposed_lengths=proposed_lengths,
        phase_a_budget=phase_a_selector.config.budget if phase_a_selector is not None else None,
        phase_a_selected_tokens=(
            len(phase_a_selector.selected_indices)
            if (phase_a_selector is not None and phase_a_selector.selected_indices is not None)
            else None
        ),
        phase_a_budget_mode=(phase_a_selector.config.budget_mode if phase_a_selector is not None else None),
        phase_a_budget_min=(phase_a_selector.config.min_budget if phase_a_selector is not None else None),
        phase_a_budget_max=(phase_a_selector.config.max_budget if phase_a_selector is not None else None),
        phase_a_budget_current=(phase_a_selector.current_budget if phase_a_selector is not None else None),
        phase_a_budget_trajectory=phase_a_budget_trajectory,
        phase_a_round_trace=phase_a_round_trace,
        phase_a_controller=(phase_a_controller.snapshot() if phase_a_adaptive else None),
        phase_a_controller_time_ms=round(phase_a_controller_time_ms, 4),
    )


# ---------------------------------------------------------------------------
# DFlash model
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        focused_indices = getattr(self.config, "_phase_a_focused_indices", None)
        if focused_indices is not None:
            past_len = k.shape[-2] - q_len
            if past_len > 0 and len(focused_indices) < past_len:
                k = torch.cat([k[:, :, focused_indices, :], k[:, :, past_len:, :]], dim=-2)
                v = torch.cat([v[:, :, focused_indices, :], v[:, :, past_len:, :]], dim=-2)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get(
            "target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        verification_prefix_controller: Optional[Any] = None,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            verification_prefix_controller=verification_prefix_controller,
        )
