"""Adapter worker opt-in cho vLLM 0.30, giữ target forward/verification native.

Đường A3 dùng Qwen3 projections/RMSNorm/RoPE gốc và gather paged KV theo vị
trí logic. Full native không đi qua attention thay thế. Các can thiệp chỉ
hoạt động trong ContextVar của đúng proposer call, không ảnh hưởng target.
"""

from __future__ import annotations

import atexit
from contextvars import ContextVar
from dataclasses import dataclass, field
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Callable

import torch

from .context import ContextState, gather_flash_kv, prompt_key, selected_attention, select_positions


_RUNNER: ContextVar[Any] = ContextVar('research_runner', default=None)
_CALL: ContextVar[Any] = ContextVar('research_draft_call', default=None)
_INSTALLED = False
_SNAPSHOT_CALLBACK: Callable | None = None


def register_snapshot_callback(callback: Callable | None) -> None:
    """Worker-local hook trước native propose; callback chịu trách nhiệm fork/rollback."""
    global _SNAPSHOT_CALLBACK
    _SNAPSHOT_CALLBACK = callback


def truncate_proposal(proposal: torch.Tensor, k: int) -> torch.Tensor:
    if proposal.ndim != 2 or proposal.shape[0] != 1:
        raise ValueError('variable verifier prefix requires max_num_seqs=1')
    if not 1 <= int(k) <= proposal.shape[1]:
        raise ValueError('verifier prefix must be within the actual drafted block')
    return proposal[:, :int(k)].contiguous()


@dataclass
class DraftCall:
    runner: Any
    config: dict
    request_ids: list[str]
    states: list[ContextState]
    rows: list[dict] = field(default_factory=list)
    layer_indices: dict[str, int] = field(default_factory=dict)
    uncertainty: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    drafted_gamma: int = 0
    gather_bytes: list[int] = field(default_factory=list)
    read_tokens: list[list[int]] = field(default_factory=list)


class TraceWriter:
    def __init__(self, directory: str):
        from Benchmark.common.io_util import JsonlWriter
        self.writer = JsonlWriter(Path(directory) / f'worker-{os.getpid()}.jsonl')
        self.count = 0
        atexit.register(self.close)

    def add(self, record):
        self.writer.add(record)
        # JsonlWriter normally retains samples for summary; token traces may be huge.
        self.writer.records.clear()
        self.count += 1

    def close(self):
        if self.writer is not None:
            self.writer.finalize({'type': 'summary', 'records': self.count,
                                  'adapter': 'vllm-0.30-qwen3-dflash-v1'})
            self.writer = None


def _configuration(runner):
    if not hasattr(runner, '_fast_infer_research_config'):
        path = os.environ.get('FAST_INFER_VLLM_RESEARCH_CONFIG')
        if not path:
            return None
        config = json.loads(Path(path).read_text())
        layout_file = config.get('parameters', {}).get('prompt_layouts_file')
        if layout_file:
            config['parameters']['prompt_layouts'] = json.loads(Path(layout_file).read_text())
        if runner.speculative_config is None or runner.speculative_config.method != 'dflash':
            raise RuntimeError('research worker adapter only supports native method=dflash')
        if runner.vllm_config.parallel_config.tensor_parallel_size != 1:
            raise RuntimeError('research worker requires tensor_parallel_size=1')
        if not runner.vllm_config.model_config.enforce_eager:
            raise RuntimeError('research worker requires enforce_eager=True')
        if runner.speculative_config.use_local_argmax_reduction:
            raise RuntimeError('research signal collector requires use_local_argmax_reduction=False')
        if runner.use_async_scheduling:
            raise RuntimeError('research adapter requires async_scheduling=False')
        if config['variant'] in {'b3_fixed_k', 'b3', 'b4'}:
            if runner.vllm_config.scheduler_config.max_num_seqs != 1:
                raise RuntimeError('B3/B4 variable lengths require max_num_seqs=1')
            if runner.speculative_config.num_speculative_tokens != 15:
                raise RuntimeError('B3/B4 engine must allocate maximum gamma=15')
        runner._fast_infer_research_config = config
        runner._fast_infer_context_states = {}
        runner._fast_infer_research_trace = TraceWriter(config['trace_dir']) if config.get('trace_dir') else None
    return runner._fast_infer_research_config


def _states_for_requests(runner, config, request_ids):
    store = runner._fast_infer_context_states
    for stale in set(store) - set(runner.requests):
        del store[stale]
    states = []
    layouts = config['parameters'].get('prompt_layouts', {})
    for request_id in request_ids:
        if request_id not in store:
            request = runner.requests[request_id]
            token_ids = request.prompt_token_ids
            if token_ids is None:
                raise RuntimeError('research adapter requires token-ID text prompts')
            fingerprint = prompt_key(token_ids)
            layout = dict(layouts.get(fingerprint, {}))
            if config['variant'] == 'a3' and not layout:
                raise RuntimeError(f'A3 missing exact source/global prompt layout: {fingerprint}')
            if int(layout.get('prompt_tokens', len(token_ids))) != len(token_ids):
                raise RuntimeError('layout prompt length does not match native vLLM prompt')
            store[request_id] = ContextState(
                prompt_tokens=len(token_ids),
                source_positions=tuple(layout.get('source_positions', ())),
                global_positions=tuple(layout.get('global_positions', layout.get('protected_positions', ()))),
                metadata={**layout, 'prompt_hash': fingerprint},
            )
        states.append(store[request_id])
    return states


def _observe_verification(call, sampled_token_ids, spec_metadata):
    if spec_metadata is None:
        return
    if isinstance(sampled_token_ids, torch.Tensor):
        rows = sampled_token_ids.tolist()
        rows = [[token for token in row if token >= 0] for row in rows]
    else:
        rows = sampled_token_ids
    for state, tokens in zip(call.states, rows):
        if not tokens or state.pending_trace is None:
            continue
        previous = state.pending_trace
        accepted = max(0, len(tokens) - 1)
        if accepted > previous['verified_candidates']:
            raise RuntimeError('vLLM sampled more accepted candidates than proposed')
        rate = accepted / max(previous['verified_candidates'], 1)
        state.history_acceptance = rate if state.history_acceptance is None else .8 * state.history_acceptance + .2 * rate
        if call.runner._fast_infer_research_trace is not None:
            call.runner._fast_infer_research_trace.add({
                **previous, 'type': 'verification', 'accepted_candidates': accepted,
                'committed_tokens': len(tokens), 'committed_token_ids': list(tokens),
                'history_ema_after': state.history_acceptance,
                'outcome_observed_at_round': state.round_index,
            })
        state.pending_trace = None


def _round_state(state: ContextState, logical_length: int):
    return SimpleNamespace(round_index=state.round_index, logical_length=logical_length,
                           processed_output_tokens=max(0, logical_length - state.prompt_tokens),
                           history_acceptance=state.history_acceptance,
                           parent_entropy=None, source_concentration=None)


def _b4_decision(call, gamma: int, bank_length: int):
    from TrainingFree.speculation_experiments.lagged import LaggedDraftLengthController
    state = call.states[0]
    parameters = call.config['parameters']
    controller = LaggedDraftLengthController(parameters['model'])
    feasible = tuple(value for value in (3, 7, 11, 15) if value <= gamma)
    decision = controller(state=_round_state(state, bank_length),
                          previous_draft_uncertainty=state.previous_uncertainty,
                          previous_draft_gamma=state.previous_gamma,
                          feasible_gammas=feasible, fallback_gamma=max(feasible))
    chosen = int(decision['gamma'])
    if chosen not in feasible:
        raise RuntimeError('B4 policy returned an infeasible actual draft gamma')
    call.decisions = [decision]
    return chosen


def _b3_decision(call, proposal):
    parameters = call.config['parameters']
    if call.config['variant'] == 'b3_fixed_k':
        return int(parameters['k']), {'reason': 'fixed_verifier_prefix'}
    from TrainingFree.speculation_experiments.b_controller import VerificationPrefixController
    state = call.states[0]
    dataset = state.metadata.get('dataset', parameters.get('dataset', 'unknown'))
    allowed = tuple(parameters.get('allowed_prefixes', (7, 11, 15)))
    controller = VerificationPrefixController(
        model=parameters['model'],
        cost_profiles=parameters['cost_profiles'],
        dataset=dataset,
        overhead_ms=float(parameters.get('overhead_ms', 0)),
        opportunity_threshold_eta=float(parameters.get('opportunity_threshold_eta', 0.10)),
    )
    decision = controller(
        state=_round_state(state, call.rows[0]['bank_length']),
        draft_uncertainty=call.uncertainty[0],
        draft_gamma=proposal.shape[1],
        available_prefixes=tuple(k for k in allowed if k <= proposal.shape[1]),
    )
    return int(decision['prefix']), decision


def install() -> bool:
    """Patch class idempotent; mỗi runner nạp config của chính engine ấy."""
    global _INSTALLED
    if _INSTALLED:
        return False
    if importlib.metadata.version('vllm') != '0.30.0':
        raise RuntimeError('research worker adapter supports exactly vLLM==0.30.0')
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Attention, DFlashQwen3ForCausalLM
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    native_runner = GPUModelRunner.propose_draft_token_ids
    native_propose = DFlashProposer.propose
    native_inputs = DFlashProposer.set_inputs_first_pass
    native_attention = DFlashQwen3Attention.forward
    native_logits = DFlashQwen3ForCausalLM.compute_logits

    def runner_propose(self, scheduler_output, sampled_token_ids, sampling_metadata,
                       hidden_states, sample_hidden_states, aux_hidden_states,
                       spec_decode_metadata, common_attn_metadata, slot_mappings):
        config = _configuration(self)
        if config is None:
            return native_runner(self, scheduler_output, sampled_token_ids, sampling_metadata,
                hidden_states, sample_hidden_states, aux_hidden_states, spec_decode_metadata,
                common_attn_metadata, slot_mappings)
        request_ids = self.input_batch.req_ids.copy()
        call = DraftCall(self, config, request_ids, _states_for_requests(self, config, request_ids))
        _observe_verification(call, sampled_token_ids, spec_decode_metadata)
        token = _RUNNER.set(call)
        try:
            return native_runner(self, scheduler_output, sampled_token_ids, sampling_metadata,
                hidden_states, sample_hidden_states, aux_hidden_states, spec_decode_metadata,
                common_attn_metadata, slot_mappings)
        finally:
            _RUNNER.reset(token)

    def propose(self, *args, **kwargs):
        call = _RUNNER.get()
        if call is None:
            return native_propose(self, *args, **kwargs)
        if args:
            raise RuntimeError('vLLM DFlash propose ABI changed: expected named arguments')
        if _SNAPSHOT_CALLBACK is not None:
            _SNAPSHOT_CALLBACK(proposer=self, runner=call.runner, call=call, inputs=kwargs)
        parameters = call.config['parameters']
        gamma = int(kwargs['num_speculative_tokens'])
        if gamma < 1:
            raise RuntimeError('research adapter does not support scheduler gamma=0')
        if call.config['variant'] == 'b4':
            # Use logical last target position before native input expansion; padded
            # rejected tails are removed below by native metadata preparation.
            positions = kwargs['target_positions']
            rejected = kwargs.get('num_rejected_tokens_gpu')
            bank = int(positions[-1].item()) + 1 - (int(rejected[0].item()) if rejected is not None else 0)
            gamma = _b4_decision(call, gamma, bank)
            kwargs = {**kwargs, 'num_speculative_tokens': gamma}
        call.drafted_gamma = gamma
        call.uncertainty = [{} for _ in call.states]
        call.gather_bytes = [0 for _ in call.states]
        call.read_tokens = [[] for _ in call.states]
        active = _CALL.set(call)
        started = time.perf_counter()
        configured_gamma = self.num_speculative_tokens
        try:
            if call.config['variant'] == 'b4':
                # DFlashProposer.set_inputs_first_pass reads this field to size
                # its query block. Pass the selected gamma through both the
                # propose argument and the proposer state, then restore the
                # engine's maximum allocation for the next request/round.
                self.num_speculative_tokens = gamma
            try:
                proposal = native_propose(self, **kwargs)
            finally:
                if call.config['variant'] == 'b4':
                    self.num_speculative_tokens = configured_gamma
            if call.config['variant'] in {'b3', 'b3_fixed_k'}:
                k, decision = _b3_decision(call, proposal)
                proposal = truncate_proposal(proposal, k)
                if self._last_draft_probs is not None:
                    self._last_draft_probs = self._last_draft_probs[:, :k].contiguous()
                call.decisions = [decision]
            elapsed_ms = (time.perf_counter() - started) * 1000
            traced = call.runner._fast_infer_research_trace
            for row, (request_id, state) in enumerate(zip(call.request_ids, call.states)):
                if not call.rows[row]['valid']:
                    continue  # Chunked-prefill placeholder is discarded by native scheduler.
                bank = call.rows[row]['bank_length']
                dense = call.rows[row]['dense']
                if dense and parameters.get('selector') == 'previous_dflash':
                    state.ranking_length = bank
                    state.ranking_round = state.round_index
                trace_identity = {name: state.metadata[name] for name in
                                  ('sample_id', 'dataset', 'source_group_id', 'prompt_hash', 'prompt_tokens')
                                  if name in state.metadata}
                trace = {
                    'request_id': request_id, **trace_identity,
                    'round_index': state.round_index, 'variant': call.config['variant'],
                    'context_tokens': bank, 'logical_length': bank,
                    'processed_output_tokens': max(0, bank - state.prompt_tokens),
                    'drafted_candidates': gamma, 'verified_candidates': proposal.shape[1],
                    'gamma': gamma, 'k': proposal.shape[1], 'dense_refresh': dense,
                    'ranking_origin_round': state.ranking_round,
                    'draft_uncertainty': call.uncertainty[row],
                    'history_ema': state.history_acceptance,
                    'proposal_token_ids': proposal[row].tolist() if traced else [],
                    'anchor': call.rows[row]['anchor'],
                    'context_tokens_by_layer': call.read_tokens[row],
                    'kv_read_bytes': call.gather_bytes[row],
                    'controller': call.decisions[row] if call.decisions else None,
                    'draft_and_controller_wall_ms': elapsed_ms,
                    'timing_scope': 'host_wall_includes_sync_for_signals; not standalone_cuda_latency',
                    'accepted_candidates': None, 'committed_tokens': None,
                }
                state.pending_trace = trace
                if traced:
                    traced.add({**trace, 'type': 'proposal'})
                state.previous_uncertainty = call.uncertainty[row]
                state.previous_gamma = gamma
                state.round_index += 1
            return proposal
        finally:
            _CALL.reset(active)

    def set_inputs(self, *args, **kwargs):
        result = native_inputs(self, *args, **kwargs)
        call = _CALL.get()
        if call is None:
            return result
        if not isinstance(self.model, DFlashQwen3ForCausalLM):
            raise RuntimeError('research adapter requires unwrapped Qwen3 DFlash model in eager mode')
        _, _, cad = result
        gamma = self.num_speculative_tokens
        seq_lens = cad.seq_lens.tolist()
        query_positions = self.positions[:len(call.states) * (gamma + 1)]
        query_ids = self.input_ids[:len(call.states) * (gamma + 1)]
        call.layer_indices = {layer.self_attn.layer_name: idx
                             for idx, layer in enumerate(self.model.model.layers)}
        period = int(call.config['parameters'].get('refresh_period', 1))
        full_budget = call.config['parameters'].get('budget') == 'full'
        call.rows = []
        for row, state in enumerate(call.states):
            bank = int(seq_lens[row]) - gamma - 1
            valid = bank >= state.prompt_tokens
            dense = full_budget or not valid or call.config['variant'] != 'a3' or state.round_index % period == 0
            if call.config['variant'] == 'a3' and not state.rankings and call.config['parameters']['selector'] == 'previous_dflash':
                dense = True
            call.rows.append({'bank_length': bank, 'dense': dense, 'valid': valid,
                              'block_table': cad.block_table_tensor[row],
                              'query_positions': query_positions[row * (gamma + 1):(row + 1) * (gamma + 1)],
                              'anchor': int(query_ids[row * (gamma + 1)].item())})
        return result

    def attention(self, positions, hidden_states):
        call = _CALL.get()
        if call is None or call.config['variant'] not in {'a3', 'full_adapter'}:
            return native_attention(self, positions, hidden_states)
        if self.attention_sink_bias is not None:
            raise RuntimeError('research sparse attention does not support attention_sink_bias')
        if self.sliding_window is not None:
            raise RuntimeError('research paged-KV gather requires full-attention cache; sliding-window page eviction is unsupported')
        if self.attn.impl.__class__.__module__ != 'vllm.v1.attention.backends.flash_attn':
            raise RuntimeError('research context adapter requires FLASH_ATTN backend')
        if str(self.attn.kv_cache_dtype).startswith('fp8'):
            raise RuntimeError('research context adapter requires unquantized KV')
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.q_size)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        output = torch.zeros_like(q)
        layer = call.layer_indices[self.layer_name]
        parameters = call.config['parameters']
        gamma = call.drafted_gamma
        for row, state in enumerate(call.states):
            metadata = call.rows[row]
            bank = metadata['bank_length']
            budget = parameters.get('budget', bank)
            if metadata['dense'] or budget == 'full' or bank <= int(budget):
                selected = torch.arange(bank, device=q.device)
            else:
                selected = select_positions(state, layer=layer, bank_length=bank,
                    budget=int(parameters['budget']), selector=parameters['selector'],
                    source_anchors=int(parameters.get('source_anchors', 64)),
                    recent_output=int(parameters.get('recent_output', 256))).to(q.device)
            context_k, context_v = gather_flash_kv(self.attn.kv_cache, metadata['block_table'],
                selected, call.runner.drafter.block_size, self.head_dim)
            sl = slice(row * (gamma + 1), (row + 1) * (gamma + 1))
            collect = metadata['valid'] and metadata['dense'] and parameters.get('selector') == 'previous_dflash'
            values, scores = selected_attention(q[sl], context_k, context_v, k[sl], v[sl],
                selected, metadata['query_positions'], causal=self.causal,
                scaling=self.scaling, collect_scores=collect, sliding_window=self.sliding_window)
            output[sl] = values
            if scores is not None:
                state.rankings[layer] = scores.detach()
            call.read_tokens[row].append(selected.numel())
            call.gather_bytes[row] += (context_k.numel() + context_v.numel()) * context_k.element_size()
        projected, _ = self.o_proj(output.reshape(-1, self.q_size))
        return projected

    def compute_logits(self, hidden_states):
        logits = native_logits(self, hidden_states)
        call = _CALL.get()
        if call is None or logits is None:
            return logits
        collect = call.config['variant'] in {'b3', 'b4', 'trace'} or call.config['parameters'].get('collect_uncertainty', False)
        if collect:
            start = time.perf_counter()
            top_vals, top_inds = torch.topk(logits.float(), 2, dim=-1)
            logit_margin = top_vals[..., 0] - top_vals[..., 1]
            model_variant = call.config.get('parameters', {}).get('model', {}).get('variant', '')
            req_entropy = model_variant in ('R-Hazard-Suffix-Entropy', 'R-DH', 'R-HA', 'R-PreAll', 'R-DAll') or call.config.get('parameters', {}).get('require_entropy', False)
            if req_entropy:
                log_probs = logits.float().log_softmax(dim=-1)
                probabilities = log_probs.exp()
                terms = torch.where(probabilities > 0, probabilities * log_probs, 0.)
                entropy = -terms.sum(dim=-1) / max(math.log(logits.shape[-1]), 1e-12)
                top_probs = log_probs.gather(-1, top_inds).exp()
                confidence = top_probs[..., 0]
                prob_margin = top_probs[..., 0] - top_probs[..., 1]
            else:
                top_probs = torch.softmax(top_vals, dim=-1)
                confidence = top_probs[..., 0]
                prob_margin = top_probs[..., 0] - top_probs[..., 1]
                entropy = torch.zeros_like(confidence)

            entropy = entropy.reshape(len(call.states), call.drafted_gamma).tolist()
            confidence = confidence.reshape(len(call.states), call.drafted_gamma).tolist()
            prob_margin = prob_margin.reshape(len(call.states), call.drafted_gamma).tolist()
            logit_margin = logit_margin.reshape(len(call.states), call.drafted_gamma).tolist()
            duration = (time.perf_counter() - start) * 1000
            call.uncertainty = [{'normalized_entropy': e, 'confidence': c,
                                 'probability_margin': pm, 'logit_margin': lm,
                                 'signal_extraction_ms': duration}
                                for e, c, pm, lm in zip(entropy, confidence, prob_margin, logit_margin)]
        return logits

    GPUModelRunner.propose_draft_token_ids = runner_propose
    DFlashProposer.propose = propose
    DFlashProposer.set_inputs_first_pass = set_inputs
    DFlashQwen3Attention.forward = attention
    DFlashQwen3ForCausalLM.compute_logits = compute_logits
    _INSTALLED = True
    return True
