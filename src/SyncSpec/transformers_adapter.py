"""Offline Transformers adapters with exact target cache transactions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .model import SyncSpecDrafter, build_masked_block, top_m_candidates
from .synthetic import DraftOutput
from .verifier import VerificationResult, greedy_verify, stochastic_verify


@dataclass
class TransformersTargetState:
    source_ids: torch.Tensor
    input_ids: torch.Tensor
    past_key_values: Any
    next_logits: torch.Tensor
    anchor_hidden: torch.Tensor
    recent_hidden: torch.Tensor
    generated: list[int]
    source_hidden: torch.Tensor | None = None


def _field(output: Any, name: str, index: int | None = None) -> Any:
    value = getattr(output, name, None)
    if value is not None:
        return value
    if index is not None and isinstance(output, (tuple, list)):
        return output[index]
    raise AttributeError(f"model output has no {name}")


class TransformersTargetAdapter:
    """A target adapter that never downloads and preserves full-context KV."""

    def __init__(self, model, tokenizer=None, device: str | torch.device = "cpu", eos_token_id: int | None = None):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.eos_token_id = eos_token_id if eos_token_id is not None else getattr(getattr(model, "config", None), "eos_token_id", None)

    def _clone_cache(self, cache: Any) -> Any:
        if cache is None:
            return None
        # Transformers 5 DynamicCache contains DynamicLayer objects and
        # non-leaf tensors. deepcopy() is rejected by PyTorch for those tensors;
        # reconstruct the cache from detached key/value leaves instead.
        if hasattr(cache, "layers"):
            try:
                from transformers import DynamicCache

                layers = []
                for layer in cache.layers:
                    if not getattr(layer, "is_initialized", False):
                        layers.append((None, None))
                    else:
                        layers.append((layer.keys.detach().clone(), layer.values.detach().clone()))
                return DynamicCache(ddp_cache_data=layers, config=getattr(cache, "config", None))
            except (ImportError, TypeError, AttributeError):
                pass
        if isinstance(cache, tuple):
            return tuple(
                tuple(item.detach().clone() if torch.is_tensor(item) else deepcopy(item) for item in pair)
                if isinstance(pair, tuple) else pair
                for pair in cache
            )
        return deepcopy(cache)

    def _stack_caches(self, caches: list[Any]) -> Any:
        """Build a transient batch cache from equal-length request caches."""
        if not caches:
            return None
        first = caches[0]
        if torch.is_tensor(first):
            return torch.cat(caches, dim=0)
        if hasattr(first, "layers"):
            from transformers import DynamicCache

            layers = []
            for layer_index in range(len(first.layers)):
                layer_values = [cache.layers[layer_index] for cache in caches]
                if not all(getattr(layer, "is_initialized", False) for layer in layer_values):
                    layers.append((None, None))
                    continue
                layers.append((
                    torch.cat([layer.keys for layer in layer_values], dim=0),
                    torch.cat([layer.values for layer in layer_values], dim=0),
                ))
            return DynamicCache(ddp_cache_data=layers, config=getattr(first, "config", None))
        if isinstance(first, tuple):
            return tuple(
                tuple(torch.cat([cache[index][part] for cache in caches], dim=0)
                      for part in range(len(first[index])))
                for index in range(len(first))
            )
        raise TypeError(f"unsupported cache type for batch verification: {type(first)!r}")

    def _slice_cache(self, cache: Any, row: int) -> Any:
        """Detach one request cache from a transient batched target cache."""
        if cache is None:
            return None
        if torch.is_tensor(cache):
            return cache[row:row + 1].detach().clone()
        if hasattr(cache, "layers"):
            from transformers import DynamicCache

            layers = []
            for layer in cache.layers:
                if not getattr(layer, "is_initialized", False):
                    layers.append((None, None))
                else:
                    layers.append((
                        layer.keys[row:row + 1].detach().clone(),
                        layer.values[row:row + 1].detach().clone(),
                    ))
            return DynamicCache(ddp_cache_data=layers, config=getattr(cache, "config", None))
        if isinstance(cache, tuple):
            return tuple(
                tuple(value[row:row + 1].detach().clone() if torch.is_tensor(value) else value
                      for value in pair)
                if isinstance(pair, tuple) else pair
                for pair in cache
            )
        raise TypeError(f"unsupported cache type for batch verification: {type(cache)!r}")

    def _run(self, input_ids: torch.Tensor, past_key_values: Any = None, output_hidden_states: bool = False):
        kwargs = {
            "input_ids": input_ids.to(self.device),
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": output_hidden_states,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        with torch.no_grad():
            return self.model(**kwargs)

    def _final_hidden_module(self):
        """Find the final normalization module used by common decoder models.

        Capturing this module avoids asking Transformers to retain every layer's
        hidden state for a long-context prefill.  The result is only a small
        compatibility helper; models without a recognizable final norm use the
        older ``output_hidden_states`` fallback below.
        """
        candidates = (
            (getattr(getattr(self.model, "model", None), "norm", None)),
            (getattr(getattr(self.model, "model", None), "final_layernorm", None)),
            (getattr(getattr(self.model, "model", None), "final_layer_norm", None)),
            (getattr(getattr(self.model, "transformer", None), "ln_f", None)),
        )
        return next((module for module in candidates if module is not None), None)

    def _run_with_final_hidden(
        self, input_ids: torch.Tensor, past_key_values: Any = None,
    ) -> tuple[Any, torch.Tensor | None]:
        """Run a target forward while capturing only the final hidden tensor."""
        captured: list[torch.Tensor] = []
        module = self._final_hidden_module()
        handle = None
        if module is not None and hasattr(module, "register_forward_hook"):
            def capture(_module, _inputs, output):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if torch.is_tensor(value):
                    captured.append(value)

            handle = module.register_forward_hook(capture)
        try:
            output = self._run(
                input_ids, past_key_values=past_key_values,
                output_hidden_states=False,
            )
        finally:
            if handle is not None:
                handle.remove()
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and captured:
            hidden = captured[-1]
        return output, hidden

    def prefill(self, source_ids: torch.Tensor) -> TransformersTargetState:
        source = source_ids.to(self.device).flatten().to(torch.long)
        if source.numel() == 0:
            raise ValueError("target prefill requires at least one source/prompt token")
        context_limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(context_limit, int) and context_limit > 0 and source.numel() > context_limit:
            raise ValueError(
                f"full-context target input ({source.numel()}) exceeds model limit ({context_limit})"
            )
        output, hidden = self._run_with_final_hidden(source.unsqueeze(0))
        logits = _field(output, "logits", 0)[0, -1].detach()
        if hidden is None:
            # Compatibility fallback for a custom model without a final norm
            # hook or direct last_hidden_state output.  Common Llama/Qwen
            # models take the hook path above and never materialize all layers.
            output = self._run(source.unsqueeze(0), output_hidden_states=True)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(output, "hidden_states", None)
                hidden = hidden_states[-1] if hidden_states else None
        if hidden is None:
            # A custom local model may not expose hidden states; target anchor
            # still remains well-defined as a zero vector for the drafter.
            width = int(getattr(getattr(self.model, "config", None), "hidden_size", 1))
            hidden = torch.zeros((1, source.numel(), width), device=self.device)
        hidden = hidden.detach()
        return TransformersTargetState(
            source_ids=source.clone(),
            input_ids=source.clone(),
            past_key_values=_field(output, "past_key_values", 1),
            next_logits=logits,
            anchor_hidden=hidden[0, -1].clone(),
            recent_hidden=hidden[0, -128:].clone(),
            generated=[],
            source_hidden=hidden[0].clone(),
        )

    def next_logits(self, state: TransformersTargetState) -> torch.Tensor:
        return state.next_logits.clone()

    def remaining_context_tokens(self, state: TransformersTargetState) -> int | None:
        """Expose target context headroom so the engine can cap generation."""
        context_limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if not isinstance(context_limit, int) or context_limit <= 0:
            return None
        return max(0, int(context_limit) - int(state.input_ids.numel()))

    def _verification_pass(
        self, state: TransformersTargetState, proposals: torch.Tensor,
    ) -> tuple[torch.Tensor, Any, torch.Tensor | None, torch.Tensor | None]:
        proposals = proposals.to(self.device).flatten().to(torch.long)
        if proposals.numel() == 0:
            return torch.empty((0, state.next_logits.numel()), device=self.device), None, None, None
        vocab_size = int(self.model.get_input_embeddings().num_embeddings)
        invalid = (proposals < 0) | (proposals >= vocab_size)
        if invalid.any():
            # Invalid IDs cannot be embedded.  They are necessarily a target
            # mismatch, so compute only up to the first invalid position and
            # pad the unused diagnostic rows for the pure verifier contract.
            first_invalid = int(invalid.nonzero(as_tuple=False)[0].item())
            valid_prefix = proposals[:first_invalid]
            if valid_prefix.numel():
                output = self._run(
                    valid_prefix.unsqueeze(0),
                    past_key_values=self._clone_cache(state.past_key_values),
                    output_hidden_states=False,
                )
                prefix_logits = _field(output, "logits", 0)[0]
                logits = torch.cat([state.next_logits.unsqueeze(0), prefix_logits], dim=0)
            else:
                logits = state.next_logits.unsqueeze(0)
            if logits.shape[0] < proposals.numel():
                logits = torch.cat([
                    logits,
                    logits[-1:].expand(proposals.numel() - logits.shape[0], -1),
                ], dim=0)
            return logits, None, None, None
        # Evaluate the complete proposal block once.  The logits for the
        # proposal positions are the current state's next logits followed by
        # rows emitted after each preceding proposal.  The resulting cache is
        # retained as a transaction payload for the all-accepted fast commit.
        output, hidden = self._run_with_final_hidden(
            proposals.unsqueeze(0),
            past_key_values=self._clone_cache(state.past_key_values),
        )
        block_logits = _field(output, "logits", 0)[0]
        logits = torch.cat([
            state.next_logits.unsqueeze(0), block_logits[:-1],
        ], dim=0)[: proposals.numel()]
        if hidden is None:
            # Compatibility fallback for a custom model without a final norm
            # hook or direct last_hidden_state output.  This path is not taken
            # by the standard Llama/Qwen model families.
            output = self._run(
                proposals.unsqueeze(0),
                past_key_values=self._clone_cache(state.past_key_values),
                output_hidden_states=True,
            )
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(output, "hidden_states", None)
                hidden = hidden_states[-1] if hidden_states else None
        hidden_rows = hidden[0].detach() if hidden is not None else None
        return (
            logits, _field(output, "past_key_values", 1),
            block_logits[-1].detach(), hidden_rows,
        )

    def _verification_logits(self, state: TransformersTargetState, proposals: torch.Tensor) -> torch.Tensor:
        return self._verification_pass(state, proposals)[0]

    def verify(self, state: TransformersTargetState, proposals: torch.Tensor, **kwargs) -> VerificationResult:
        proposals = proposals.to(self.device).flatten().to(torch.long)
        logits, transaction_cache, next_logits, hidden_states = self._verification_pass(state, proposals)
        if kwargs.get("stochastic", False):
            result = stochastic_verify(
                proposals,
                logits.softmax(dim=-1),
                kwargs["proposal_probs"].to(logits.device),
                kwargs.get("generator"),
            )
        else:
            result = greedy_verify(proposals, logits)
        if result.rejected_position is None and proposals.numel():
            result.transaction_cache = transaction_cache
            result.transaction_next_logits = next_logits
            result.transaction_last_hidden = hidden_states[-1] if hidden_states is not None else None
            result.transaction_hidden_states = hidden_states
            result.transaction_proposal_length = int(proposals.numel())
            result.transaction_base_length = int(state.input_ids.numel())
        return result

    def verify_batch(
        self, states: list[TransformersTargetState], proposals: torch.Tensor,
        stochastic: bool = False, proposal_probs: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> list[VerificationResult]:
        """Verify equal-length proposal blocks with one batched target forward."""
        if not states:
            return []
        if proposals.ndim == 1 and len(states) == 1:
            proposals = proposals.unsqueeze(0)
        if proposals.ndim != 2 or proposals.shape[0] != len(states):
            raise ValueError("batched proposals must be [batch, K]")
        lengths = {int(state.input_ids.numel()) for state in states}
        if len(lengths) != 1:
            raise ValueError("batch verification requires equal current context lengths")
        proposals = proposals.to(self.device).to(torch.long)
        vocab_size = int(self.model.get_input_embeddings().num_embeddings)
        if ((proposals < 0) | (proposals >= vocab_size)).any():
            # Preserve the scalar invalid-ID compatibility path.  Real
            # drafter candidates are always in-vocabulary and use the batched
            # fast path below.
            return [self.verify(states[row], proposals[row], stochastic=stochastic,
                                proposal_probs=(proposal_probs[row] if proposal_probs is not None else None),
                                generator=generator)
                    for row in range(len(states))]
        cache = self._stack_caches([state.past_key_values for state in states])
        output, hidden = self._run_with_final_hidden(
            proposals, past_key_values=self._clone_cache(cache),
        )
        block_logits = _field(output, "logits", 0)
        logits = torch.cat([
            torch.stack([state.next_logits for state in states], dim=0).unsqueeze(1),
            block_logits[:, :-1],
        ], dim=1)
        if hidden is None:
            output = self._run(
                proposals, past_key_values=self._clone_cache(cache),
                output_hidden_states=True,
            )
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(output, "hidden_states", None)
                hidden = hidden_states[-1] if hidden_states else None
        if stochastic:
            if proposal_probs is None or proposal_probs.ndim != 3:
                raise ValueError("batched stochastic verification requires [batch, K, V] proposal_probs")
            results = [
                stochastic_verify(
                    proposals[row], logits[row].softmax(dim=-1),
                    proposal_probs[row].to(logits.device), generator,
                )
                for row in range(len(states))
            ]
        else:
            results = [greedy_verify(proposals[row], logits[row]) for row in range(len(states))]
        for row, result in enumerate(results):
            if result.rejected_position is None and proposals.shape[1]:
                result.transaction_cache = self._slice_cache(
                    _field(output, "past_key_values", 1), row,
                )
                result.transaction_next_logits = block_logits[row, -1].detach()
                result.transaction_last_hidden = hidden[row, -1].detach() if hidden is not None else None
                result.transaction_hidden_states = hidden[row].detach() if hidden is not None else None
                result.transaction_proposal_length = int(proposals.shape[1])
                result.transaction_base_length = int(states[row].input_ids.numel())
        return results

    def commit(self, state: TransformersTargetState, result: VerificationResult) -> None:
        committed = result.committed_ids.to(self.device).flatten().to(torch.long)
        if committed.numel() == 0:
            return
        context_limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if (
            isinstance(context_limit, int) and context_limit > 0
            and state.input_ids.numel() + committed.numel() > context_limit
        ):
            raise ValueError(
                f"committed generation exceeds model context limit ({context_limit})"
            )
        can_reuse_transaction = (
            getattr(result, "transaction_cache", None) is not None
            and getattr(result, "rejected_position", None) is None
            and getattr(result, "transaction_proposal_length", None) == int(committed.numel())
            and getattr(result, "transaction_base_length", None) == int(state.input_ids.numel())
            and getattr(result, "transaction_next_logits", None) is not None
        )
        if can_reuse_transaction:
            state.past_key_values = result.transaction_cache
            state.next_logits = result.transaction_next_logits.detach().clone()
            state.input_ids = torch.cat([state.input_ids, committed.detach()])
            state.generated.extend(int(x) for x in committed.tolist())
            hidden_states = getattr(result, "transaction_hidden_states", None)
            if hidden_states is not None and hidden_states.numel():
                state.anchor_hidden = hidden_states[-1].detach().clone()
                state.recent_hidden = torch.cat([state.recent_hidden, hidden_states.detach()], dim=0)[-128:]
            return
        output, hidden = self._run_with_final_hidden(
            committed.unsqueeze(0),
            past_key_values=state.past_key_values,
        )
        state.past_key_values = _field(output, "past_key_values", 1)
        state.next_logits = _field(output, "logits", 0)[0, -1].detach()
        state.input_ids = torch.cat([state.input_ids, committed.detach()])
        state.generated.extend(int(x) for x in committed.tolist())
        if hidden is None:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(output, "hidden_states", None)
                hidden = hidden_states[-1] if hidden_states else None
        if hidden is not None:
            state.anchor_hidden = hidden[0, -1].detach().clone()
            state.recent_hidden = torch.cat([state.recent_hidden, hidden[0].detach()], dim=0)[-128:]

    def source_embeddings(self, source_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.model.get_input_embeddings()
        return embedding(source_ids.to(self.device).flatten()).detach()

    def generate_greedy(self, source_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Exact vanilla-AR reference using the same adapter/cache state."""
        state = self.prefill(source_ids)
        generated: list[int] = []
        limit = max(0, int(max_new_tokens))
        context_remaining = self.remaining_context_tokens(state)
        if context_remaining is not None:
            limit = min(limit, context_remaining)
        for _ in range(limit):
            token = self.next_logits(state).argmax().reshape(1)
            result = VerificationResult(token, 0)
            self.commit(state, result)
            generated.append(int(token.item()))
            eos = self.eos_token_id
            if eos is not None and (int(token.item()) in eos if isinstance(eos, (list, tuple, set)) else int(token.item()) == eos):
                break
        return torch.tensor(generated, dtype=torch.long, device=self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
        local_files_only: bool = True,
    ) -> "TransformersTargetAdapter":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(model_path)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
        model_kwargs = {"dtype": torch_dtype, "local_files_only": local_files_only}
        try:
            # Transformers 5 renamed the loading keyword to ``dtype``.  Keep
            # the compatibility fallback narrowly scoped so a real model
            # loading error is not accidentally retried with different args.
            model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except TypeError as exc:
            message = str(exc).lower()
            if "unexpected keyword argument" not in message or "dtype" not in message:
                raise
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        eos = getattr(tokenizer, "eos_token_id", None)
        return cls(model, tokenizer, device=device, eos_token_id=eos)


class NativeDrafterAdapter:
    """Adapter from a saved native SyncSpec drafter to the engine contract."""

    def __init__(self, model: SyncSpecDrafter, target: TransformersTargetAdapter, mask_token_id: int | None = None):
        self.model = model.eval()
        self.target = target
        self.mask_token_id = mask_token_id if mask_token_id is not None else model.config.mask_token_id
        if self.mask_token_id is None:
            self.mask_token_id = model.config.vocab_size - 1

    def draft(self, state: TransformersTargetState, kd: int, source_memory=None, **_) -> DraftOutput:
        anchor_token = state.input_ids[-1].reshape(1)
        ids = build_masked_block(anchor_token, kd, self.mask_token_id)
        if source_memory is None:
            source = None
        else:
            retrieved = source_memory.retrieve(state.anchor_hidden, top_r=source_memory.top_r)
            source = retrieved.descriptors.unsqueeze(0)
        with torch.inference_mode():
            output = self.model(
                ids,
                target_anchor=state.anchor_hidden.unsqueeze(0),
                recent_hidden=state.recent_hidden.unsqueeze(0),
                source_memory=source,
                position_offset=int(state.input_ids.numel()),
            )
        candidate_ids, candidate_logits = top_m_candidates(output.logits[0], self.model.config.top_m)
        return DraftOutput(candidate_ids, candidate_logits, output.hidden[0])

    def draft_batch(
        self, states: list[TransformersTargetState], source_memories: list[Any] | None = None,
        kd: int = 0,
    ) -> list[DraftOutput]:
        """Run one shallow drafter forward for equal-length request states."""
        if not states:
            return []
        if int(kd) <= 0:
            raise ValueError("batched drafting requires K_d > 0")
        context_lengths = {int(state.input_ids.numel()) for state in states}
        if len(context_lengths) != 1:
            raise ValueError("batched drafting requires equal current context lengths")
        if source_memories is None:
            source_memories = [None] * len(states)
        if len(source_memories) != len(states):
            raise ValueError("one source memory is required per draft state")
        anchor_ids = torch.stack([state.input_ids[-1] for state in states])
        recent_lengths = {int(state.recent_hidden.shape[0]) for state in states}
        if len(recent_lengths) != 1:
            raise ValueError("batched drafting requires equal recent-window lengths")
        target_anchor = torch.stack([state.anchor_hidden for state in states])
        recent_hidden = torch.stack([state.recent_hidden for state in states])
        retrieved: list[torch.Tensor] = []
        for state, memory in zip(states, source_memories):
            if memory is None:
                continue
            result = memory.retrieve(state.anchor_hidden, top_r=memory.top_r)
            retrieved.append(result.descriptors)
        source_memory = None
        if retrieved:
            if len(retrieved) != len(states) or len({tuple(value.shape) for value in retrieved}) != 1:
                raise ValueError("batched drafting requires equal source-memory descriptor shapes")
            source_memory = torch.stack(retrieved)
        ids = build_masked_block(anchor_ids, int(kd), self.mask_token_id)
        with torch.inference_mode():
            output = self.model(
                ids, target_anchor=target_anchor, recent_hidden=recent_hidden,
                source_memory=source_memory,
                position_offset=next(iter(context_lengths)),
            )
        return [
            DraftOutput(
                *top_m_candidates(output.logits[row], self.model.config.top_m),
                output.hidden[row],
            )
            for row in range(len(states))
        ]
