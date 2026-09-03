from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from SyncSpec.config import SyncSpecConfig  # noqa: E402
from SyncSpec.engine import SyncSpecEngine  # noqa: E402
from SyncSpec.model import SyncSpecDrafter, SyncSpecDrafterConfig  # noqa: E402
from SyncSpec.transformers_adapter import NativeDrafterAdapter, TransformersTargetAdapter  # noqa: E402
from test_syncspec_transformers import TinyCausalLM  # noqa: E402


def test_native_drafter_and_transformers_target_share_engine_contract() -> None:
    target = TransformersTargetAdapter(TinyCausalLM(vocab_size=8, hidden_size=4), device="cpu", eos_token_id=7)
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=8, hidden_size=4, layers=1, heads=2, groups=2, top_m=4, mask_token_id=7
    ))
    model.tie_target_weights(target.model.get_input_embeddings(), target.model.get_output_embeddings())
    drafter = NativeDrafterAdapter(model, target)
    config = SyncSpecConfig(vocab_size=8, hidden_size=4, top_m=4, predicted_spec_gain=0.2,
                            budget_profiles=((0, 0), (4, 2), (4, 4)))
    result = SyncSpecEngine(target, drafter, config).generate(torch.tensor([1, 2]), max_new_tokens=3)
    vanilla = []
    state = target.prefill(torch.tensor([1, 2]))
    for _ in range(3):
        token = target.next_logits(state).argmax().reshape(1)
        vanilla.append(int(token.item()))
        target.commit(state, type("R", (), {"committed_ids": token})())
    assert result.token_ids.tolist() == vanilla
    assert result.status == "ok"


def test_native_drafter_batches_equal_length_states() -> None:
    target = TransformersTargetAdapter(TinyCausalLM(vocab_size=16, hidden_size=8), device="cpu", eos_token_id=15)
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=16, hidden_size=8, layers=1, heads=2, groups=2, top_m=4, mask_token_id=15,
    ))
    model.tie_target_weights(target.model.get_input_embeddings(), target.model.get_output_embeddings())
    states = [target.prefill(torch.tensor([1, 2])), target.prefill(torch.tensor([3, 4]))]
    outputs = NativeDrafterAdapter(model, target).draft_batch(states, [None, None], kd=2)
    assert len(outputs) == 2
    assert all(output.candidate_ids.shape == (2, 4) for output in outputs)


def test_engine_batch_generation_matches_independent_exact_generation() -> None:
    target = TransformersTargetAdapter(TinyCausalLM(vocab_size=16, hidden_size=8), device="cpu", eos_token_id=15)
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=16, hidden_size=8, layers=1, heads=2, groups=2, top_m=4, mask_token_id=15,
    ))
    model.tie_target_weights(target.model.get_input_embeddings(), target.model.get_output_embeddings())
    engine = SyncSpecEngine(
        target, NativeDrafterAdapter(model, target),
        SyncSpecConfig(
            vocab_size=16, hidden_size=8, top_m=4, predicted_spec_gain=0.2,
            budget_profiles=((0, 0), (4, 2), (4, 4)),
        ),
    )
    sources = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    results = engine.generate_batch(sources, max_new_tokens=4)
    expected = [target.generate_greedy(source, max_new_tokens=4).tolist() for source in sources]
    assert [result.token_ids.tolist() for result in results] == expected
    assert all(result.batch_size == 2 for result in results)


def test_engine_batch_generation_handles_mixed_prompt_lengths() -> None:
    target = TransformersTargetAdapter(TinyCausalLM(vocab_size=16, hidden_size=8), device="cpu", eos_token_id=15)
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=16, hidden_size=8, layers=1, heads=2, groups=2, top_m=4, mask_token_id=15,
    ))
    model.tie_target_weights(target.model.get_input_embeddings(), target.model.get_output_embeddings())
    engine = SyncSpecEngine(
        target, NativeDrafterAdapter(model, target),
        SyncSpecConfig(vocab_size=16, hidden_size=8, top_m=4, predicted_spec_gain=0.2,
                       budget_profiles=((0, 0), (4, 2), (4, 4))),
    )
    sources = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    results = engine.generate_batch(sources, max_new_tokens=3)
    expected = [target.generate_greedy(source, max_new_tokens=3).tolist() for source in sources]
    assert [result.token_ids.tolist() for result in results] == expected
    assert all(result.batch_size == 2 for result in results)


def test_native_drafter_matches_vanilla_on_real_llama_cache() -> None:
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    torch.manual_seed(22)
    target = TransformersTargetAdapter(LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    )), device="cpu", eos_token_id=31)
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=32, hidden_size=16, layers=1, heads=2, groups=2, top_m=8, mask_token_id=31
    ))
    model.tie_target_weights(target.model.get_input_embeddings(), target.model.get_output_embeddings())
    engine = SyncSpecEngine(
        target, NativeDrafterAdapter(model, target),
        SyncSpecConfig(vocab_size=32, hidden_size=16, top_m=8, predicted_spec_gain=0.2,
                       budget_profiles=((0, 0), (4, 2), (4, 4))),
    )
    source = torch.tensor([1, 2, 3, 4])
    accelerated = engine.generate(source, max_new_tokens=4)
    vanilla = target.generate_greedy(source, max_new_tokens=4)
    assert accelerated.token_ids.tolist() == vanilla.tolist()


def test_native_drafter_matches_bfloat16_target_dtype() -> None:
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    target_model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    )).to(torch.bfloat16)
    target = TransformersTargetAdapter(target_model, device="cpu", eos_token_id=31)
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=32, hidden_size=16, layers=1, heads=2, groups=2,
        top_m=4, mask_token_id=31,
    ))
    model.tie_target_weights(target.model.get_input_embeddings(), target.model.get_output_embeddings())
    result = NativeDrafterAdapter(model, target).draft(target.prefill(torch.tensor([1, 2, 3])), 2)
    assert result.hidden.dtype == torch.bfloat16
    assert result.candidate_logits.dtype == torch.bfloat16
    engine = SyncSpecEngine(
        target, NativeDrafterAdapter(model, target),
        SyncSpecConfig(vocab_size=32, hidden_size=16, top_m=4,
                       budget_profiles=((0, 0), (2, 1), (2, 2))),
    )
    generated = engine.generate(torch.tensor([1, 2, 3]), max_new_tokens=2)
    assert generated.status == "ok"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/B200 is not available on this host")
def test_real_llama_bfloat16_native_engine_smoke_on_cuda() -> None:
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        return
    target_model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    )).to(device="cuda", dtype=torch.bfloat16)
    target = TransformersTargetAdapter(target_model, device="cuda", eos_token_id=31)
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=32, hidden_size=16, layers=1, heads=2, groups=2,
        top_m=4, mask_token_id=31,
    ))
    model.tie_target_weights(target.model.get_input_embeddings(), target.model.get_output_embeddings())
    engine = SyncSpecEngine(
        target, NativeDrafterAdapter(model, target),
        SyncSpecConfig(
            vocab_size=32, hidden_size=16, top_m=4, device="cuda",
            budget_profiles=((0, 0), (2, 1), (2, 2)), predicted_spec_gain=0.2,
        ),
    )
    result = engine.generate(torch.tensor([1, 2, 3], device="cuda"), max_new_tokens=2)
    assert result.status == "ok"
    assert result.token_ids.is_cuda
    assert result.committed_tokens == 2
    batch_results = engine.generate_batch([
        torch.tensor([1, 2, 3], device="cuda"),
        torch.tensor([4, 5, 6], device="cuda"),
    ], max_new_tokens=2)
    assert len(batch_results) == 2
    assert all(item.token_ids.is_cuda and item.committed_tokens == 2 for item in batch_results)
