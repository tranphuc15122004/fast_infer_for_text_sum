from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.transformers_adapter import TransformersTargetAdapter  # noqa: E402


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.forward_calls = 0
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, past_key_values=None, use_cache=True, return_dict=True, **kwargs):
        self.forward_calls += 1
        del use_cache, kwargs
        prefix = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)
        if past_key_values is not None:
            prefix = past_key_values
        all_ids = torch.cat([prefix, input_ids], dim=1)
        rows = []
        for token in all_ids[:, -input_ids.shape[1]:].flatten().tolist():
            logits = torch.full((self.vocab_size,), -20.0, device=input_ids.device)
            logits[(int(token) + 1) % self.vocab_size] = 20.0
            rows.append(logits)
        output = SimpleNamespace(
            logits=torch.stack(rows).reshape(input_ids.shape[0], input_ids.shape[1], -1),
            past_key_values=all_ids.detach().clone(),
            last_hidden_state=self.embedding(input_ids),
            hidden_states=(self.embedding(input_ids),),
        )
        return output if return_dict else (output.logits, output.past_key_values)


def test_from_pretrained_uses_transformers5_dtype_keyword(monkeypatch, tmp_path: Path) -> None:
    transformers = __import__("transformers")
    calls = {}

    class Tokenizer:
        eos_token_id = 7

    def fake_tokenizer(path, **kwargs):
        calls["tokenizer"] = (path, kwargs)
        return Tokenizer()

    def fake_model(path, **kwargs):
        calls["model"] = (path, kwargs)
        return TinyCausalLM()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tokenizer),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_model),
    )
    TransformersTargetAdapter.from_pretrained(
        tmp_path, device="cpu", dtype="bfloat16", local_files_only=True,
    )
    assert calls["model"][1]["dtype"] is torch.bfloat16
    assert "torch_dtype" not in calls["model"][1]


def test_from_pretrained_falls_back_for_legacy_transformers_dtype(monkeypatch, tmp_path: Path) -> None:
    transformers = __import__("transformers")
    attempts = []

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda path, **kwargs: SimpleNamespace(eos_token_id=7)),
    )

    def legacy_model(path, **kwargs):
        attempts.append(kwargs)
        if "dtype" in kwargs:
            raise TypeError("unexpected keyword argument 'dtype'")
        return TinyCausalLM()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(legacy_model),
    )
    TransformersTargetAdapter.from_pretrained(
        tmp_path, device="cpu", dtype="float32", local_files_only=True,
    )
    assert attempts[0]["dtype"] is torch.float32
    assert attempts[1]["torch_dtype"] is torch.float32


def test_from_pretrained_does_not_mask_internal_dtype_typeerror(monkeypatch, tmp_path: Path) -> None:
    transformers = __import__("transformers")
    attempts = []

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda path, **kwargs: SimpleNamespace(eos_token_id=7)),
    )

    def broken_model(path, **kwargs):
        attempts.append(kwargs)
        raise TypeError("dtype conversion failed inside model loader")

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(broken_model),
    )
    with pytest.raises(TypeError, match="dtype conversion failed"):
        TransformersTargetAdapter.from_pretrained(
            tmp_path, device="cpu", dtype="float32", local_files_only=True,
        )
    assert len(attempts) == 1


def test_transformers_adapter_keeps_full_context_and_commits_only_verified_prefix() -> None:
    target = TransformersTargetAdapter(TinyCausalLM(), device="cpu", eos_token_id=7)
    state = target.prefill(torch.tensor([1, 2]))
    assert state.source_hidden is not None
    assert state.source_hidden.shape == (2, 4)
    result = target.verify(state, torch.tensor([3, 9]))
    assert result.accepted_length == 1
    assert result.committed_ids.tolist() == [3, 4]
    target.commit(state, result)
    assert state.input_ids.tolist() == [1, 2, 3, 4]
    assert state.past_key_values.squeeze(0).tolist() == [1, 2, 3, 4]
    assert target.next_logits(state).argmax().item() == 5


def test_prefill_captures_final_hidden_without_materializing_all_hidden_states() -> None:
    class HookOnlyTarget(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 8
            self.embedding = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8, bias=False)
            self.model = torch.nn.Module()
            self.model.norm = torch.nn.LayerNorm(4)
            self.config = SimpleNamespace(vocab_size=8, hidden_size=4)

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.head

        def forward(self, input_ids, past_key_values=None, output_hidden_states=False,
                    use_cache=True, return_dict=True, **kwargs):
            del past_key_values, use_cache, kwargs
            assert output_hidden_states is False
            hidden = self.model.norm(self.embedding(input_ids))
            logits = self.head(hidden)
            output = SimpleNamespace(
                logits=logits,
                past_key_values=input_ids.detach().clone(),
                hidden_states=None,
            )
            return output if return_dict else (logits, output.past_key_values)

    target = TransformersTargetAdapter(HookOnlyTarget(), device="cpu", eos_token_id=7)
    state = target.prefill(torch.tensor([1, 2]))
    assert state.anchor_hidden.shape == (4,)
    assert state.source_hidden is not None
    assert state.source_hidden.shape == (2, 4)
    result = target.verify(state, state.next_logits.argmax().reshape(1))
    assert result.accepted_length == 1


def test_transformers_adapter_stochastic_path_uses_full_target_distribution() -> None:
    target = TransformersTargetAdapter(TinyCausalLM(), device="cpu", eos_token_id=7)
    state = target.prefill(torch.tensor([1]))
    result = target.verify(
        state,
        torch.tensor([3]),
        stochastic=True,
        proposal_probs=torch.tensor([[0.2] * 8]),
        generator=torch.Generator().manual_seed(5),
    )
    assert result.committed_ids.numel() == 1
    assert result.residual_probs is not None
    assert torch.allclose(result.residual_probs.sum(), torch.tensor(1.0))


def test_transformers_adapter_reuses_all_accepted_transaction_cache() -> None:
    model = TinyCausalLM()
    target = TransformersTargetAdapter(model, device="cpu", eos_token_id=7)
    state = target.prefill(torch.tensor([1, 2]))
    before_verify = model.forward_calls
    result = target.verify(state, torch.tensor([3, 4]))
    assert result.accepted_length == 2
    assert model.forward_calls == before_verify + 1
    target.commit(state, result)
    assert model.forward_calls == before_verify + 1
    assert state.input_ids.tolist() == [1, 2, 3, 4]
    assert state.next_logits.argmax().item() == 5


def test_real_transformers_cache_api_matches_full_recompute() -> None:
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    torch.manual_seed(13)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    ))
    target = TransformersTargetAdapter(model, device="cpu", eos_token_id=31)
    prefix = torch.tensor([1, 2, 3])
    state = target.prefill(prefix)
    proposals = torch.tensor([4, 5])
    result = target.verify(state, proposals)
    target.commit(state, result)
    with torch.no_grad():
        direct = model(input_ids=state.input_ids.unsqueeze(0), use_cache=False, return_dict=True)
    assert torch.allclose(state.next_logits, direct.logits[0, -1], atol=1e-5, rtol=1e-4)


def test_real_transformers_dynamic_cache_batch_commit_reuse() -> None:
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    ))
    target = TransformersTargetAdapter(model, device="cpu", eos_token_id=31)
    sources = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    states = [target.prefill(source) for source in sources]
    proposal_rows = []
    for source in sources:
        probe = target.prefill(source)
        row = []
        for _ in range(2):
            token = target.next_logits(probe).argmax().reshape(1)
            row.append(int(token.item()))
            target.commit(probe, type("R", (), {"committed_ids": token})())
        proposal_rows.append(row)
    results = target.verify_batch(states, torch.tensor(proposal_rows))
    assert all(result.accepted_length == 2 for result in results)
    for state, result in zip(states, results):
        target.commit(state, result)
        with torch.no_grad():
            direct = model(input_ids=state.input_ids.unsqueeze(0), use_cache=False, return_dict=True)
        assert torch.allclose(state.next_logits, direct.logits[0, -1], atol=1e-5, rtol=1e-4)


def test_transformers_adapter_rejects_generation_past_context_limit() -> None:
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4,
    ))
    target = TransformersTargetAdapter(model, device="cpu", eos_token_id=15)
    state = target.prefill(torch.tensor([1, 2, 3]))
    target.commit(state, type("R", (), {"committed_ids": torch.tensor([4])})())
    with pytest.raises(ValueError, match="context limit"):
        target.commit(state, type("R", (), {"committed_ids": torch.tensor([5])})())


def test_transformers_adapter_caps_vanilla_ar_to_context_headroom() -> None:
    model = TinyCausalLM()
    model.config.max_position_embeddings = 4
    target = TransformersTargetAdapter(model, device="cpu", eos_token_id=7)
    generated = target.generate_greedy(torch.tensor([1, 2, 3]), max_new_tokens=8)
    assert generated.numel() == 1


def test_transformers_adapter_batches_verification_for_equal_length_states() -> None:
    target = TransformersTargetAdapter(TinyCausalLM(), device="cpu", eos_token_id=7)
    states = [target.prefill(torch.tensor([1, 2])), target.prefill(torch.tensor([3, 4]))]
    proposals = torch.tensor([[3, 9], [5, 0]])
    results = target.verify_batch(states, proposals)
    assert len(results) == 2
    assert results[0].committed_ids.tolist() == [3, 4]
    assert results[1].committed_ids.tolist() == [5, 6]
    for state, result in zip(states, results):
        target.commit(state, result)
    assert [state.input_ids.tolist() for state in states] == [[1, 2, 3, 4], [3, 4, 5, 6]]


def test_transformers_adapter_batch_reuses_cache_for_all_accepted_rows() -> None:
    model = TinyCausalLM()
    target = TransformersTargetAdapter(model, device="cpu", eos_token_id=7)
    states = [target.prefill(torch.tensor([1, 2])), target.prefill(torch.tensor([3, 4]))]
    before_verify = model.forward_calls
    results = target.verify_batch(states, torch.tensor([[3, 4], [5, 6]]))
    assert all(result.accepted_length == 2 for result in results)
    assert model.forward_calls == before_verify + 1
    for state, result in zip(states, results):
        target.commit(state, result)
    assert model.forward_calls == before_verify + 1
    assert [state.next_logits.argmax().item() for state in states] == [5, 7]
