"""End-to-end synchronized speculative generation."""

from __future__ import annotations

import time
import json
from collections import defaultdict
from pathlib import Path

import torch

from .config import SyncSpecConfig
from .controller import PostDraftController, PreDraftGate, RuntimeFeedback, context_bin
from .evidence import SourceMemoryBank, SourceNgramIndex
from .schema import InferenceResult
from .selector import SourceCoherentSelector
from .survival import SurvivalHead
from .verifier import VerificationResult


def _is_eos(token: int, eos_token_id) -> bool:
    if eos_token_id is None:
        return False
    return (
        token in eos_token_id
        if isinstance(eos_token_id, (list, tuple, set))
        else token == eos_token_id
    )


def _truncate_result_at_eos(result: VerificationResult, eos_token_id) -> VerificationResult:
    """Drop proposal/cache state after EOS before committing a verified block."""
    committed = result.committed_ids.flatten()
    if committed.numel() == 0 or eos_token_id is None:
        return result
    eos_positions = [
        index for index, token in enumerate(committed.tolist())
        if _is_eos(int(token), eos_token_id)
    ]
    if not eos_positions:
        return result
    keep = eos_positions[0] + 1
    if keep == committed.numel():
        return result
    accepted = (
        keep if result.rejected_position is None
        else min(int(result.accepted_length), keep - 1)
    )
    # Omit the opaque transaction payload: the adapter recomputes only this
    # committed prefix, preventing cache entries after EOS from leaking.
    return VerificationResult(
        committed_ids=committed[:keep].clone(),
        accepted_length=accepted,
        correction_token_id=result.correction_token_id,
        residual_probs=result.residual_probs,
        rejected_position=result.rejected_position,
    )


class SyncSpecEngine:
    def __init__(self, target, drafter, config: SyncSpecConfig, selector=None, survival_head=None):
        self.target = target
        self.drafter = drafter
        self.config = config
        hidden = config.hidden_size or int(
            getattr(getattr(drafter, "model", drafter), "config", None).hidden_size
            if getattr(getattr(drafter, "model", drafter), "config", None) is not None
            and hasattr(getattr(getattr(drafter, "model", drafter), "config", None), "hidden_size")
            else 16
        )
        serving_device = torch.device(getattr(target, "device", "cpu"))
        self.selector = (selector or SourceCoherentSelector(
            hidden, rank=min(config.selector_rank, hidden),
            temperature=config.selector_temperature,
            vocab_size=config.vocab_size or None,
        )).to(serving_device)
        self.survival_head = (survival_head or SurvivalHead(
            8, hidden_size=min(64, hidden * 2)
        )).to(serving_device)
        self.pre_gate = PreDraftGate(
            config.gate_epsilon, config.budget_profiles,
            default_gain=config.predicted_spec_gain, gain_table=config.gate_table,
        )
        self.controller = PostDraftController()

    def _remaining_context_tokens(self, state) -> int | None:
        """Return target context headroom when the adapter exposes a limit."""
        resolver = getattr(self.target, "remaining_context_tokens", None)
        if resolver is None:
            return None
        remaining = resolver(state)
        if remaining is None:
            return None
        remaining = int(remaining)
        if remaining < 0:
            raise ValueError("target context headroom must be non-negative")
        return remaining

    def _synchronize(self) -> None:
        """Make component wall-clock measurements valid for asynchronous CUDA."""
        device = getattr(self.target, "device", None)
        if device is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))

    def _profile_costs(
        self, kd: int, kv_max: int, batch_size: int = 1,
        context_length: int | None = None,
    ) -> dict[int, float]:
        if self.config.runtime_profile:
            path = Path(self.config.runtime_profile)
            if path.is_file():
                from .profile import round_costs_from_profile
                matching = self._matching_profile_records(
                    path, kd, batch_size, context_length,
                )
                measured = round_costs_from_profile(matching)
                if measured:
                    # Profile components are one shared kernel/scheduler
                    # interval for a homogeneous microbatch, while the
                    # post-draft controller scores one request at a time.
                    # Normalize only measured batch costs; the local fallback
                    # below is already expressed per request.
                    divisor = max(1, int(batch_size))
                    return {
                        kv: cost / divisor for kv, cost in measured.items()
                    }
                if self.config.require_measured_profile:
                    return {}
        if self.config.require_measured_profile:
            return {}
        # Replace with profile JSON on real hardware. This monotonic fallback is
        # only for the synthetic/local reference path and is never presented as
        # a hardware measurement.
        return {k: 1.0 + 0.12 * k + 0.02 * kd for k in range(1, kv_max + 1)}

    def _matching_profile_records(
        self, path: Path, kd: int, batch_size: int,
        context_length: int | None,
    ) -> list[dict]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        return [
            record for record in records
            if isinstance(record, dict)
            and self._profile_matches(record, kd, batch_size, context_length)
        ]

    def _ar_cost(
        self, feedback: RuntimeFeedback, kd: int, batch_size: int,
        context_length: int,
    ) -> float | None:
        """Resolve a per-token AR opportunity cost for post-draft gating."""
        if self.config.runtime_profile:
            path = Path(self.config.runtime_profile)
            if path.is_file():
                from .profile import ar_cost_from_profile
                matching = self._matching_profile_records(
                    path, kd, batch_size, context_length,
                )
                measured = ar_cost_from_profile(matching)
                if measured is not None:
                    return measured
        if feedback.ar_rounds > 0 and feedback.ar_latency_ema_ms > 0.0:
            return feedback.ar_latency_ema_ms
        return None

    def _can_speculate(
        self, kd: int, batch_size: int, context_length: int,
        max_kv: int | None = None,
    ) -> bool:
        if not self.config.require_measured_profile:
            return True
        costs = self._profile_costs(kd, kd, batch_size, context_length)
        if not costs:
            return False
        candidates = {
            profile.kv for profile in self.config.budget_profiles
            if profile.kd == kd and profile.kv > 0
            and (max_kv is None or profile.kv <= int(max_kv))
        }
        return bool(candidates.intersection(costs))

    def _profile_matches(
        self, record: dict, kd: int, batch_size: int = 1,
        context_length: int | None = None,
    ) -> bool:
        """Accept measured costs only for the active serving configuration."""
        key = record.get("key", {})
        if not isinstance(key, dict) or "kv" not in key:
            return False
        if self.config.require_measured_profile and record.get("source") != "measured":
            return False
        for field, expected in (
            ("model", self.config.target_model),
            ("checkpoint", self.config.drafter_checkpoint),
            ("precision", self.config.dtype),
            ("selector_checkpoint", self.config.selector_checkpoint),
            ("survival_checkpoint", self.config.survival_checkpoint),
        ):
            if expected is not None:
                # New production profiles carry the trained component paths.
                # A missing axis is accepted only for legacy/local profiles
                # when measured-profile gating is disabled.
                if field not in key:
                    if self.config.require_measured_profile:
                        return False
                elif str(key[field]) != str(expected):
                    return False
        if "kd" in key and int(key["kd"]) != int(kd):
            return False
        if str(key.get("batch_bin", "")) != f"batch{int(batch_size)}":
            return False
        if context_length is not None:
            if str(key.get("context_bin", "")) != context_bin(int(context_length)):
                return False
        profile_gpu = str(key.get("gpu", "")).strip().lower()
        device = torch.device(getattr(self.target, "device", "cpu"))
        if profile_gpu:
            if device.type == "cpu" and profile_gpu not in {"cpu", "unknown"}:
                return False
            if device.type == "cuda":
                current_gpu = torch.cuda.get_device_name(device).lower()
                if profile_gpu not in current_gpu and current_gpu not in profile_gpu:
                    return False
        return True

    def _feedback_gain(
        self, feedback: RuntimeFeedback, context_length: int, batch_size: int,
        kd: int | None = None,
    ) -> float:
        prior = self.pre_gate.estimated_gain(
            context_length, batch_size, kd=kd,
        )
        return feedback.adjusted_gain(prior)

    def _feedback_gains(
        self, feedback: RuntimeFeedback, context_length: int, batch_size: int,
    ) -> dict[int, float]:
        """Return request-local priors for each finite pre-draft K_d."""
        return {
            kd: self._feedback_gain(feedback, context_length, batch_size, kd=kd)
            for kd in sorted({
                profile.kd for profile in self.config.budget_profiles if profile.kd > 0
            })
        }

    def _choose_pre_gate(
        self,
        context_length: int,
        batch_size: int,
        predicted_gain,
        max_kv: int | None = None,
    ):
        """Choose a prior-optimal K_d that also has a usable cost profile."""
        choice = self.pre_gate.choose(
            context_length=context_length,
            batch_size=batch_size,
            predicted_gain=predicted_gain,
        )
        if choice.kd == 0 or self._can_speculate(
            choice.kd, batch_size, context_length, max_kv=max_kv,
        ):
            return choice
        available = {
            profile.kd for profile in self.config.budget_profiles
            if profile.kd > 0 and self._can_speculate(
                profile.kd, batch_size, context_length, max_kv=max_kv,
            )
        }
        return self.pre_gate.choose(
            context_length=context_length,
            batch_size=batch_size,
            predicted_gain=predicted_gain,
            allowed_kds=available,
        )

    def _survival_features(self, draft, selection) -> torch.Tensor:
        k = selection.token_ids.numel()
        if k == 0:
            return torch.empty((0, 8), dtype=selection.q.dtype, device=selection.q.device)
        positions = torch.arange(1, k + 1, dtype=selection.q.dtype, device=selection.q.device)
        positions = positions / max(1, k)
        entropy = -(selection.q * selection.q.clamp_min(1e-9).log()).sum(dim=-1)
        return torch.stack([
            positions,
            selection.q.max(dim=-1).values,
            entropy,
            selection.gates,
            selection.ngram_features[:, :, 0].mean(dim=-1),
            selection.ngram_features[:, :, 1].mean(dim=-1),
            torch.full((k,), float(k), dtype=selection.q.dtype, device=selection.q.device),
            torch.ones(k, dtype=selection.q.dtype, device=selection.q.device),
        ], dim=-1)

    def _commit_bonus_token(
        self, state, result, stochastic: bool, generator: torch.Generator,
    ) -> torch.Tensor | None:
        """Commit the target correction/bonus token after an all-accepted block."""
        committed = getattr(result, "committed_ids", None)
        if committed is None or committed.numel() == 0:
            return None
        if getattr(result, "rejected_position", None) is not None:
            return None
        logits = getattr(result, "transaction_next_logits", None)
        if logits is None:
            logits = self.target.next_logits(state)
        if stochastic:
            token = torch.multinomial(logits.softmax(-1), 1, generator=generator)
        else:
            token = logits.argmax().reshape(1)
        bonus_result = type(
            "BonusResult", (), {"committed_ids": token, "accepted_length": 0}
        )()
        self.target.commit(state, bonus_result)
        return token.reshape(-1)

    def generate(
        self,
        source_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        seed: int = 0,
        stochastic: bool = False,
        batch_size: int = 1,
        force_kv: int | None = None,
        max_rounds: int | None = None,
    ) -> InferenceResult:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if force_kv is not None and int(force_kv) <= 0:
            raise ValueError("force_kv must be positive or None")
        if max_rounds is not None and int(max_rounds) <= 0:
            raise ValueError("max_rounds must be positive or None")
        batch_size = int(batch_size)
        torch.manual_seed(seed)
        source_ids = source_ids.flatten()
        self._synchronize()
        started = time.perf_counter()
        prefill_started = time.perf_counter()
        state = self.target.prefill(source_ids)
        self._synchronize()
        prefill_ms = (time.perf_counter() - prefill_started) * 1e3
        ngram = SourceNgramIndex(
            source_ids.tolist(), self.config.source_ngram_min, self.config.source_ngram_max
        )
        embeddings = getattr(state, "source_hidden", None)
        if embeddings is None and hasattr(self.target, "source_embeddings"):
            embeddings = self.target.source_embeddings(source_ids)
        memory = SourceMemoryBank.from_source(
            source_ids, embeddings=embeddings, chunk_size=self.config.source_chunk_size,
            top_r=self.config.source_top_r,
        )
        if hasattr(state, "source_hidden"):
            state.source_hidden = None
        output: list[torch.Tensor] = []
        accepted: list[int] = []
        budgets: list[dict[str, int]] = []
        timings = {"prefill": prefill_ms, "draft": 0.0, "selector": 0.0, "survival": 0.0, "verify": 0.0, "scheduler": 0.0}
        feedback = RuntimeFeedback(alpha=self.config.feedback_alpha)
        rollout_features: list[list[float]] = []
        rollout_labels: list[float] = []
        rounds = 0
        fallback = 0
        limit = self.config.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if limit < 0:
            raise ValueError("max_new_tokens must be non-negative")
        context_remaining = self._remaining_context_tokens(state)
        if context_remaining is not None:
            limit = min(limit, context_remaining)
        target_device = getattr(self.target, "device", source_ids.device)
        generator = torch.Generator(device=str(target_device)).manual_seed(seed)
        while len(output) < limit and (
            max_rounds is None or rounds < int(max_rounds)
        ):
            rounds += 1
            round_timing_start = dict(timings)
            sched_start = time.perf_counter()
            gate = self._choose_pre_gate(
                context_length=source_ids.numel() + len(output), batch_size=batch_size,
                predicted_gain=self._feedback_gains(
                    feedback, source_ids.numel() + len(output), batch_size,
                ),
                max_kv=limit - len(output),
            )
            self._synchronize()
            timings["scheduler"] += (time.perf_counter() - sched_start) * 1e3
            if gate.kd == 0:
                fallback += 1
                verify_start = time.perf_counter()
                ar_logits = self.target.next_logits(state)
                if stochastic:
                    token = torch.multinomial(ar_logits.softmax(-1), 1, generator=generator)
                else:
                    token = ar_logits.argmax().reshape(1)
                result = type("ARResult", (), {"committed_ids": token, "accepted_length": 0})()
                self.target.commit(state, result)
                self._synchronize()
                timings["verify"] += (time.perf_counter() - verify_start) * 1e3
                feedback.update_ar_latency(timings["verify"] - round_timing_start["verify"])
                output.append(token[0])
                accepted.append(0)
                budgets.append({"kd": 0, "kv": 0})
                if _is_eos(int(token.item()), getattr(self.target, "eos_token_id", None)):
                    break
                continue

            draft_start = time.perf_counter()
            draft = self.drafter.draft(state, gate.kd, source_memory=memory)
            self._synchronize()
            timings["draft"] += (time.perf_counter() - draft_start) * 1e3
            selector_start = time.perf_counter()
            with torch.no_grad():
                selected = self.selector.select(
                    draft.hidden, draft.candidate_ids, draft.candidate_logits,
                    list(getattr(state, "source_ids", source_ids).tolist()) + list(state.generated), ngram,
                    stochastic=stochastic, generator=generator,
                )
            self._synchronize()
            timings["selector"] += (time.perf_counter() - selector_start) * 1e3
            survival_start = time.perf_counter()
            survival_features = self._survival_features(draft, selected).detach()
            head_parameter = next(self.survival_head.parameters())
            head_features = survival_features.to(
                device=head_parameter.device, dtype=head_parameter.dtype,
            )
            survival = self.survival_head.survival(head_features).detach()
            context_length = source_ids.numel() + len(output)
            costs = self._profile_costs(
                gate.kd, gate.kd, batch_size,
                context_length=context_length,
            )
            ar_cost = self._ar_cost(
                feedback, gate.kd, batch_size, context_length,
            )
            choice = self.controller.choose(
                gate.kd, survival, costs, self.config.budget_profiles,
                max_kv=limit - len(output),
                ar_cost=ar_cost, ar_margin=self.config.gate_epsilon,
            )
            self._synchronize()
            timings["survival"] += (time.perf_counter() - survival_start) * 1e3
            if choice.kv <= 0 and force_kv is None:
                # The post-draft measured utility gate can reject a draft
                # after its cost has already been paid.  Commit exactly one
                # target token and expose the paid draft as a fallback round.
                fallback += 1
                verify_start = time.perf_counter()
                ar_logits = self.target.next_logits(state)
                if stochastic:
                    token = torch.multinomial(
                        ar_logits.softmax(-1), 1, generator=generator,
                    )
                else:
                    token = ar_logits.argmax().reshape(1)
                ar_result = type(
                    "ARResult", (), {"committed_ids": token, "accepted_length": 0}
                )()
                self.target.commit(state, ar_result)
                self._synchronize()
                timings["verify"] += (time.perf_counter() - verify_start) * 1e3
                feedback.update_ar_latency(
                    timings["verify"] - round_timing_start["verify"]
                )
                feedback.update(
                    accepted_length=0, proposed_length=max(1, gate.kd),
                    timings_ms={
                        name: timings[name] - round_timing_start[name]
                        for name in ("draft", "selector", "survival", "scheduler")
                    },
                )
                output.append(token[0])
                accepted.append(0)
                budgets.append({"kd": gate.kd, "kv": 0})
                if _is_eos(int(token.item()), getattr(self.target, "eos_token_id", None)):
                    break
                continue
            if force_kv is None:
                kv = min(choice.kv, limit - len(output))
            else:
                # Stage-3 label collection must observe every draft position;
                # it bypasses the not-yet-trained survival/controller choice
                # while keeping the real drafter, selector and verifier path.
                kv = min(int(force_kv), gate.kd, limit - len(output))
            if kv <= 0:
                kv = 1
            verify_start = time.perf_counter()
            proposals = selected.token_ids[:kv]
            if stochastic:
                vocab = int(self.config.vocab_size or self.target.next_logits(state).numel())
                proposal_probs = torch.zeros((kv, vocab), device=selected.q.device)
                proposal_probs.scatter_(1, selected.candidate_ids[:kv], selected.q[:kv])
                result = self.target.verify(
                    state, proposals, stochastic=True, proposal_probs=proposal_probs,
                    generator=generator,
                )
            else:
                result = self.target.verify(state, proposals)
            result = _truncate_result_at_eos(
                result, getattr(self.target, "eos_token_id", None),
            )
            self.target.commit(state, result)
            remaining = limit - len(output)
            new_ids = result.committed_ids[:remaining]
            output.extend(new_ids)
            if (
                len(output) < limit
                and not _is_eos(
                    int(output[-1].item()), getattr(self.target, "eos_token_id", None),
                )
                and getattr(result, "rejected_position", None) is None
            ):
                bonus = self._commit_bonus_token(state, result, stochastic, generator)
                if bonus is not None:
                    output.append(bonus[0])
            self._synchronize()
            timings["verify"] += (time.perf_counter() - verify_start) * 1e3
            feedback.update(
                result.accepted_length, kv,
                {
                    name: timings[name] - round_timing_start[name]
                    for name in ("draft", "selector", "survival", "verify", "scheduler")
                },
            )
            accepted.append(int(result.accepted_length))
            rollout_features.extend(survival_features[:kv].cpu().tolist())
            rollout_labels.extend(float(j < result.accepted_length) for j in range(kv))
            budgets.append({"kd": gate.kd, "kv": kv})
            eos = getattr(self.target, "eos_token_id", None)
            is_eos = output and eos is not None and (
                int(output[-1].item()) in eos if isinstance(eos, (list, tuple, set))
                else int(output[-1].item()) == eos
            )
            if is_eos:
                break
        self._synchronize()
        timings["e2e"] = (time.perf_counter() - started) * 1e3
        tokens = torch.stack(output) if output else torch.empty(0, dtype=torch.long)
        return InferenceResult(
            token_ids=tokens,
            batch_size=batch_size,
            rounds=rounds,
            committed_tokens=int(tokens.numel()),
            accepted_lengths=accepted,
            budgets=budgets,
            fallback_rounds=fallback,
            timing_ms=timings,
            survival_features=rollout_features,
            survival_labels=rollout_labels,
            runtime_feedback=feedback.to_dict(),
        )

    def generate_batch(
        self,
        source_ids: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        max_new_tokens: int | None = None,
        seed: int = 0,
        stochastic: bool = False,
        force_kv: int | None = None,
        max_rounds: int | None = None,
    ) -> list[InferenceResult]:
        """Serve a microbatch while preserving per-request exactness.

        Requests are grouped by their current full-context length before draft
        and verification forwards.  This gives the native Transformers path a
        real batched decode when requests are compatible, while allowing mixed
        prompt lengths and synthetic/legacy adapters to fall back to their
        scalar methods.  Each request still owns its target cache and output
        accounting; the transient stacked cache is never committed directly.
        """
        if torch.is_tensor(source_ids):
            if source_ids.ndim == 1:
                sources = [source_ids]
            elif source_ids.ndim == 2:
                sources = [row for row in source_ids]
            else:
                raise ValueError("source_ids must be [L], [B,L], or a sequence of [L] tensors")
        else:
            sources = list(source_ids)
        if not sources:
            raise ValueError("generate_batch requires at least one request")
        if any(not torch.is_tensor(value) for value in sources):
            raise TypeError("every batch source must be a torch.Tensor")
        sources = [value.flatten() for value in sources]
        if any(value.numel() == 0 for value in sources):
            raise ValueError("every batch source must contain at least one token")
        limit = self.config.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if limit < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if force_kv is not None and int(force_kv) <= 0:
            raise ValueError("force_kv must be positive or None")
        if max_rounds is not None and int(max_rounds) <= 0:
            raise ValueError("max_rounds must be positive or None")

        batch_size = len(sources)
        target_device = torch.device(getattr(self.target, "device", sources[0].device))
        generators = [
            torch.Generator(device=str(target_device)).manual_seed(int(seed) + index)
            for index in range(batch_size)
        ]
        states = []
        memories = []
        ngrams = []
        outputs: list[list[int]] = [[] for _ in sources]
        accepted: list[list[int]] = [[] for _ in sources]
        budgets: list[list[dict[str, int]]] = [[] for _ in sources]
        rollout_features: list[list[list[float]]] = [[] for _ in sources]
        rollout_labels: list[list[float]] = [[] for _ in sources]
        timings = [
            {name: 0.0 for name in (
                "prefill", "draft", "selector", "survival", "verify", "scheduler", "e2e"
            )}
            for _ in sources
        ]
        rounds = [0 for _ in sources]
        fallback = [0 for _ in sources]
        feedbacks = [RuntimeFeedback(alpha=self.config.feedback_alpha) for _ in sources]
        started = time.perf_counter()

        for index, source in enumerate(sources):
            prefill_started = time.perf_counter()
            state = self.target.prefill(source)
            self._synchronize()
            timings[index]["prefill"] = (time.perf_counter() - prefill_started) * 1e3
            states.append(state)
            source_tensor = getattr(state, "source_ids", source)
            source_tensor = source_tensor.to(target_device).flatten()
            ngrams.append(SourceNgramIndex(
                source_tensor.tolist(), self.config.source_ngram_min, self.config.source_ngram_max
            ))
            embeddings = getattr(state, "source_hidden", None)
            if embeddings is None and hasattr(self.target, "source_embeddings"):
                embeddings = self.target.source_embeddings(source_tensor)
            memories.append(SourceMemoryBank.from_source(
                source_tensor, embeddings=embeddings,
                chunk_size=self.config.source_chunk_size, top_r=self.config.source_top_r,
            ))
            if hasattr(state, "source_hidden"):
                state.source_hidden = None

        request_limits = []
        for state in states:
            context_remaining = self._remaining_context_tokens(state)
            request_limits.append(
                limit if context_remaining is None else min(limit, context_remaining)
            )

        def context_length(index: int) -> int:
            state = states[index]
            input_ids = getattr(state, "input_ids", None)
            if input_ids is not None:
                return int(input_ids.numel())
            return int(state.source_ids.numel()) + len(getattr(state, "generated", ()))

        active = {index for index in range(batch_size) if request_limits[index] > 0}
        while active and limit > 0 and (
            max_rounds is None or max(rounds[index] for index in active) < int(max_rounds)
        ):
            round_timing_start = {
                index: dict(timings[index]) for index in active
            }
            # Equal current context and source lengths are sufficient for the
            # native drafter's [B,K_d] inputs and avoid ragged source-memory
            # descriptor tensors.  Verification only needs current length,
            # and is regrouped below after K_v is selected.
            draft_groups: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
            for index in sorted(active):
                draft_groups[(
                    context_length(index),
                    int(states[index].source_ids.numel()),
                )].append(index)

            draft_jobs: list[tuple[list[int], int]] = []
            for indices in draft_groups.values():
                gate_started = time.perf_counter()
                group_gains = [
                    self._feedback_gains(
                        feedbacks[index], context_length(indices[0]), len(indices),
                    )
                    for index in indices
                ]
                profile_kds = sorted({
                    profile.kd for profile in self.config.budget_profiles
                    if profile.kd > 0
                })
                gate = self._choose_pre_gate(
                    context_length=context_length(indices[0]),
                    batch_size=len(indices), predicted_gain={
                        kd: sum(gains.get(kd, 0.0) for gains in group_gains)
                        / max(1, len(group_gains))
                        for kd in profile_kds
                    },
                    max_kv=request_limits[indices[0]] - len(outputs[indices[0]]),
                )
                self._synchronize()
                scheduler_ms = (time.perf_counter() - gate_started) * 1e3
                for index in indices:
                    rounds[index] += 1
                    timings[index]["scheduler"] += scheduler_ms
                if gate.kd <= 0:
                    for index in indices:
                        fallback[index] += 1
                        verify_started = time.perf_counter()
                        logits = self.target.next_logits(states[index])
                        if stochastic:
                            token = torch.multinomial(
                                logits.softmax(-1), 1, generator=generators[index]
                            )
                        else:
                            token = logits.argmax().reshape(1)
                        result = type("ARResult", (), {
                            "committed_ids": token, "accepted_length": 0,
                        })()
                        self.target.commit(states[index], result)
                        self._synchronize()
                        timings[index]["verify"] += (time.perf_counter() - verify_started) * 1e3
                        feedbacks[index].update_ar_latency(
                            timings[index]["verify"] - round_timing_start[index]["verify"]
                        )
                        outputs[index].append(int(token.item()))
                        accepted[index].append(0)
                        budgets[index].append({"kd": 0, "kv": 0})
                    continue
                draft_jobs.append((indices, gate.kd))

            verification_jobs: list[tuple[int, torch.Tensor, torch.Tensor | None, torch.Tensor, int, int]] = []
            for indices, kd in draft_jobs:
                draft_started = time.perf_counter()
                batch_draft = getattr(self.drafter, "draft_batch", None)
                drafts = None
                if batch_draft is not None and len(indices) > 1:
                    try:
                        drafts = batch_draft(
                            [states[index] for index in indices],
                            [memories[index] for index in indices], kd=kd,
                        )
                    except ValueError:
                        # Source-memory chunk counts can differ for otherwise
                        # compatible prompts.  Scalar drafting remains exact.
                        drafts = None
                if drafts is None:
                    drafts = [self.drafter.draft(
                        states[index], kd, source_memory=memories[index]
                    ) for index in indices]
                self._synchronize()
                draft_ms = (time.perf_counter() - draft_started) * 1e3
                for index in indices:
                    timings[index]["draft"] += draft_ms

                for index, draft in zip(indices, drafts):
                    selector_started = time.perf_counter()
                    with torch.no_grad():
                        selected = self.selector.select(
                            draft.hidden, draft.candidate_ids, draft.candidate_logits,
                            list(getattr(states[index], "source_ids", sources[index]).tolist())
                            + list(states[index].generated), ngrams[index],
                            stochastic=stochastic, generator=generators[index],
                        )
                    self._synchronize()
                    timings[index]["selector"] += (time.perf_counter() - selector_started) * 1e3

                    survival_started = time.perf_counter()
                    survival_features = self._survival_features(draft, selected).detach()
                    head_parameter = next(self.survival_head.parameters())
                    survival = self.survival_head.survival(
                        survival_features.to(
                            device=head_parameter.device, dtype=head_parameter.dtype,
                        )
                    ).detach()
                    current_context_length = context_length(index)
                    costs = self._profile_costs(
                        kd, kd, len(indices), context_length=current_context_length,
                    )
                    remaining = request_limits[index] - len(outputs[index])
                    ar_cost = self._ar_cost(
                        feedbacks[index], kd, len(indices), current_context_length,
                    )
                    choice = self.controller.choose(
                        kd, survival, costs, self.config.budget_profiles,
                        max_kv=remaining,
                        ar_cost=ar_cost, ar_margin=self.config.gate_epsilon,
                    )
                    self._synchronize()
                    timings[index]["survival"] += (time.perf_counter() - survival_started) * 1e3
                    if choice.kv <= 0 and force_kv is None:
                        fallback[index] += 1
                        verify_started = time.perf_counter()
                        logits = self.target.next_logits(states[index])
                        if stochastic:
                            token = torch.multinomial(
                                logits.softmax(-1), 1, generator=generators[index],
                            )
                        else:
                            token = logits.argmax().reshape(1)
                        ar_result = type(
                            "ARResult", (), {"committed_ids": token, "accepted_length": 0}
                        )()
                        self.target.commit(states[index], ar_result)
                        self._synchronize()
                        timings[index]["verify"] += (
                            time.perf_counter() - verify_started
                        ) * 1e3
                        feedbacks[index].update_ar_latency(
                            timings[index]["verify"]
                            - round_timing_start[index]["verify"]
                        )
                        feedbacks[index].update(
                            accepted_length=0, proposed_length=max(1, kd),
                            timings_ms={
                                name: timings[index][name]
                                - round_timing_start[index][name]
                                for name in ("draft", "selector", "survival", "scheduler")
                            },
                        )
                        outputs[index].append(int(token.item()))
                        accepted[index].append(0)
                        budgets[index].append({"kd": kd, "kv": 0})
                        continue
                    if force_kv is None:
                        kv = min(int(choice.kv), remaining)
                    else:
                        # Profiling fixes the post-draft verification window
                        # so component timing reflects exactly one supported
                        # serving profile rather than a controller-selected
                        # or multi-round decode.
                        kv = min(int(force_kv), kd, remaining)
                    if kv <= 0:
                        kv = min(1, remaining)
                    proposals = selected.token_ids[:kv]
                    proposal_probs = None
                    if stochastic:
                        vocab = int(self.config.vocab_size or self.target.next_logits(states[index]).numel())
                        proposal_probs = torch.zeros((kv, vocab), device=selected.q.device)
                        proposal_probs.scatter_(1, selected.candidate_ids[:kv], selected.q[:kv])
                    verification_jobs.append((
                        index, proposals, proposal_probs, survival_features[:kv], kd, kv,
                    ))

            verify_groups: defaultdict[tuple[int, int], list[tuple[int, torch.Tensor, torch.Tensor | None, torch.Tensor, int, int]]] = defaultdict(list)
            for job in verification_jobs:
                index, proposals, _, _, _, kv = job
                verify_groups[(context_length(index), kv)].append(job)

            for jobs in verify_groups.values():
                indices = [job[0] for job in jobs]
                verify_started = time.perf_counter()
                batch_verify = getattr(self.target, "verify_batch", None)
                results = None
                if batch_verify is not None and len(jobs) > 1:
                    proposal_stack = torch.stack([job[1] for job in jobs])
                    proposal_probs = None
                    if stochastic:
                        proposal_probs = torch.stack([
                            job[2] for job in jobs if job[2] is not None
                        ])
                    try:
                        results = batch_verify(
                            [states[index] for index in indices], proposal_stack,
                            stochastic=stochastic, proposal_probs=proposal_probs,
                            generator=generators[indices[0]],
                        )
                    except ValueError:
                        # A custom adapter may expose a batch method but reject
                        # a cache layout it cannot stack.  Preserve exactness
                        # with scalar verification for that group.
                        results = None
                if results is None:
                    results = []
                    for index, proposals, proposal_probs, _, _, _ in jobs:
                        kwargs = {"stochastic": stochastic, "generator": generators[index]}
                        if stochastic:
                            kwargs["proposal_probs"] = proposal_probs
                        results.append(self.target.verify(states[index], proposals, **kwargs))
                pending_commits = []
                for job, result in zip(jobs, results):
                    index, proposals, _, features, kd, kv = job
                    result = _truncate_result_at_eos(
                        result, getattr(self.target, "eos_token_id", None),
                    )
                    self.target.commit(states[index], result)
                    new_ids = result.committed_ids[: request_limits[index] - len(outputs[index])]
                    bonus = None
                    proposed_output = outputs[index] + [
                        int(value) for value in new_ids.tolist()
                    ]
                    if (
                        len(proposed_output) < request_limits[index]
                        and not _is_eos(
                            proposed_output[-1], getattr(self.target, "eos_token_id", None),
                        )
                        and getattr(result, "rejected_position", None) is None
                    ):
                        bonus = self._commit_bonus_token(
                            states[index], result, stochastic, generators[index],
                        )
                    pending_commits.append((job, result, new_ids, bonus))
                self._synchronize()
                verify_ms = (time.perf_counter() - verify_started) * 1e3
                for job, result, new_ids, bonus in pending_commits:
                    index, _, _, features, kd, kv = job
                    outputs[index].extend(int(value) for value in new_ids.tolist())
                    if bonus is not None:
                        outputs[index].append(int(bonus[0].item()))
                    accepted[index].append(int(result.accepted_length))
                    budgets[index].append({"kd": kd, "kv": kv})
                    rollout_features[index].extend(features.cpu().tolist())
                    rollout_labels[index].extend(
                        float(position < result.accepted_length) for position in range(kv)
                    )
                    timings[index]["verify"] += verify_ms
                    feedbacks[index].update(
                        result.accepted_length, kv,
                        {
                            name: timings[index][name] - round_timing_start[index][name]
                            for name in ("draft", "selector", "survival", "verify", "scheduler")
                        },
                    )

            eos = getattr(self.target, "eos_token_id", None)
            finished = set()
            for index in active:
                if len(outputs[index]) >= request_limits[index]:
                    finished.add(index)
                    continue
                if outputs[index] and eos is not None:
                    token = outputs[index][-1]
                    is_eos = token in eos if isinstance(eos, (list, tuple, set)) else token == eos
                    if is_eos:
                        finished.add(index)
            active.difference_update(finished)

        total_ms = (time.perf_counter() - started) * 1e3
        results = []
        for index in range(batch_size):
            timings[index]["e2e"] = total_ms
            tokens = torch.tensor(outputs[index], dtype=torch.long, device=target_device)
            results.append(InferenceResult(
                token_ids=tokens, batch_size=batch_size, rounds=rounds[index],
                committed_tokens=len(outputs[index]), accepted_lengths=accepted[index],
                budgets=budgets[index], fallback_rounds=fallback[index],
                timing_ms=timings[index], survival_features=rollout_features[index],
                survival_labels=rollout_labels[index],
                runtime_feedback=feedbacks[index].to_dict(),
            ))
        return results
