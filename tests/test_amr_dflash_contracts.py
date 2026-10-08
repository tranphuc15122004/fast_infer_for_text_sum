"""Behavioral contracts for AMR-DFlash memory and training primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from AMR_DFlash.core import (  # noqa: E402
    AcceptanceSelector,
    ComplementaryCompressor,
    MemoryConfig,
    alignment_kl_loss,
    build_sparse_attention_mask,
    greedy_acceptance,
    preference_loss,
    resolve_greedy_commit,
)
from AMR_DFlash.candidates import generate_candidate_sets  # noqa: E402
from AMR_DFlash.checkpoint import load_memory_checkpoint, save_memory_checkpoint  # noqa: E402
from AMR_DFlash.budget import charge_phase, read_ledger, remaining_gpu_seconds  # noqa: E402
from AMR_DFlash.pipeline import (  # noqa: E402
    build_preferences,
    select_capture_state_indices,
    select_records_for_split,
)
from AMR_DFlash.inference import AMRDFlashEngine  # noqa: E402
from AMR_DFlash.evaluation import evaluate_one_fixed_state  # noqa: E402
from AMR_DFlash.memory import AMRMemory  # noqa: E402
from AMR_DFlash.model import AMRDFlashDraft  # noqa: E402


def test_selector_keeps_recent_guard_inside_raw_budget_and_returns_unique_positions():
    selector = AcceptanceSelector(hidden_size=4, index_dim=3)
    with torch.no_grad():
        selector.key_projection.weight.zero_()
        selector.key_projection.weight[0, 3] = 1.0
        selector.key_projection.weight[1, 1] = 1.0
        selector.key_projection.weight[2, 2] = 1.0
        selector.query_projection[0].weight.zero_()
        selector.query_projection[0].weight[0, 3] = 1.0
        selector.query_projection[2].weight.zero_()
        selector.query_projection[2].weight[0, 0] = 1.0
        selector.position_bias.zero_()
    features = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
          [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]
    )
    positions = torch.arange(6).unsqueeze(0)
    anchor = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    selected = selector.select(features, positions, anchor, raw_budget=3, local_window=2)
    cached_selected = selector.select_encoded(
        selector.encode_keys(features),
        positions,
        anchor,
        raw_budget=3,
        local_window=2,
        query_state=selector.context_query(features),
    )

    assert selected.tolist() == [[3, 4, 5]]
    assert torch.equal(selected, cached_selected)
    assert selected.shape[1] == 3
    assert len(set(selected[0].tolist())) == 3


def test_selector_query_changes_with_committed_context_when_anchor_is_fixed():
    selector = AcceptanceSelector(hidden_size=4, index_dim=3)
    anchor = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    context_a = torch.zeros(1, 6, 4)
    context_a[0, :, 0] = torch.arange(1, 7, dtype=torch.float32)
    context_a[0, :, 1] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    context_a[0, :, 2] = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    context_b = context_a.clone()
    context_b[0, -1, 1] = 2.0
    context_b[0, -2, 2] = 10.0
    positions = torch.arange(6).unsqueeze(0)
    with torch.no_grad():
        selector.key_projection.weight.copy_(torch.eye(3, 4))
        selector.query_projection[0].weight.zero_()
        selector.query_projection[0].weight[0, 0] = 1.0
        selector.query_projection[0].weight[1, 5] = 1.0
        selector.query_projection[0].weight[2, 10] = 1.0
        selector.query_projection[2].weight.copy_(torch.eye(3))
    keys = selector.encode_keys(context_a)

    query_a = selector.context_query(context_a, window=4)
    query_b = selector.context_query(context_b, window=4)
    scores_a = selector.score_keys(keys, anchor, positions, query_state=query_a)
    scores_b = selector.score_keys(keys, anchor, positions, query_state=query_b)
    selected_a = selector.select_encoded(
        keys,
        positions,
        anchor,
        raw_budget=1,
        local_window=0,
        query_state=query_a,
    )
    selected_b = selector.select_encoded(
        keys,
        positions,
        anchor,
        raw_budget=1,
        local_window=0,
        query_state=query_b,
    )

    assert not torch.equal(query_a, query_b)
    assert not torch.allclose(scores_a, scores_b)
    assert not torch.equal(selected_a, selected_b)


def test_capture_state_sampling_is_deterministic_and_spans_the_trajectory():
    assert select_capture_state_indices(20, 6) == [0, 4, 8, 11, 15, 19]
    assert select_capture_state_indices(3, 6) == [0, 1, 2]
    assert select_capture_state_indices(1, 1) == [0]


def test_inference_split_filter_applies_limit_after_document_split_selection():
    records = [
        {"id": "train-1", "raw": {"split": "train"}},
        {"id": "valid-1", "raw": {"split": "validation"}},
        {"id": "train-2", "raw": {"split": "train"}},
        {"id": "holdout-1", "raw": {"split": "holdout"}},
    ]
    config = {"data": {"train_fraction": 0.8, "validation_fraction": 0.1}}

    selected = select_records_for_split(
        records, split="train", config=config, max_samples=1
    )

    assert [record["id"] for record in selected] == ["train-1"]


def test_sparse_attention_mask_preserves_selected_raw_and_live_keys_and_gates_slots():
    mask = build_sparse_attention_mask(
        raw_positions=torch.tensor([[1, 7]]),
        slot_positions=torch.tensor([[4]]),
        live_positions=torch.tensor([[8, 9, 10]]),
        query_length=3,
        slot_gate=-2.0,
        dtype=torch.float32,
    )

    assert mask.shape == (1, 1, 3, 6)
    assert torch.equal(mask[0, 0, 0, :2], torch.zeros(2))
    assert mask[0, 0, 0, 2].item() == pytest.approx(-2.0)
    assert torch.equal(mask[0, 0, 0, 3:], torch.zeros(3))


def test_compressor_only_memory_contains_slots_and_no_raw_positions():
    memory = AMRMemory(
        hidden_size=4,
        config=MemoryConfig(raw_budget=3, num_slots=2, local_window=1, min_context_tokens=0),
        index_dim=2,
    )
    features = torch.randn(1, 5, 4)
    built = memory.build(
        features,
        torch.randn(1, 4),
        torch.arange(5, 21).unsqueeze(0),
        mode="compressor",
        use_cost_gate=False,
    )

    assert built.raw_positions.shape == (1, 0)
    assert built.slot_positions.shape == (1, 2)
    assert built.features.shape == (1, 2, 4)


def test_streaming_compressor_matches_single_pass_and_slot_positions_are_observed():
    torch.manual_seed(3)
    compressor = ComplementaryCompressor(hidden_size=5, num_slots=3)
    features = torch.randn(1, 7, 5)
    positions = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])

    whole = compressor(features, positions)
    state = compressor.empty_state(batch_size=1, device=features.device, dtype=features.dtype)
    compressor.update(state, features[:, :3], positions[:, :3])
    compressor.update(state, features[:, 3:], positions[:, 3:])
    streamed, streamed_positions = compressor.finalize(state)

    assert torch.allclose(whole.values, streamed, atol=1e-6, rtol=1e-6)
    assert torch.equal(whole.positions, streamed_positions)
    assert int(streamed_positions.max()) <= 6


def test_preference_loss_rewards_correct_pair_and_sends_gradient_to_scores():
    positive = torch.tensor([2.0], requires_grad=True)
    negative = torch.tensor([0.0], requires_grad=True)

    loss = preference_loss(positive, negative, torch.ones(1))
    loss.backward()

    assert loss.item() < 0.2
    assert positive.grad.item() < 0
    assert negative.grad.item() > 0


def test_alignment_loss_ignores_invalid_horizon_and_backpropagates_only_valid_rows():
    draft = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [20.0, -20.0]]], requires_grad=True)
    target = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [-20.0, 20.0]]])

    loss = alignment_kl_loss(draft, target, torch.tensor([[True, True, False]]))
    loss.backward()

    assert loss.item() < 0.02
    assert torch.count_nonzero(draft.grad[0, :2]).item() > 0
    assert torch.count_nonzero(draft.grad[0, 2]).item() == 0
    assert target.grad is None


def test_greedy_acceptance_reports_first_mismatch_and_full_acceptance():
    draft = torch.tensor([[10, 11, 12, 13]])
    target = torch.tensor([[10, 99, 12, 13]])

    assert greedy_acceptance(draft, target).accepted == 1
    assert greedy_acceptance(draft, target).first_mismatch == 1
    full = greedy_acceptance(draft, draft)
    assert full.accepted == 4
    assert full.first_mismatch is None


def test_greedy_commit_clips_eos_after_output_cap_and_keeps_pending_correction():
    after_cap = resolve_greedy_commit(
        [10, 11, 99, 13],
        accepted=3,
        correction=42,
        remaining=2,
        eos_token_ids={99},
    )
    before_cap = resolve_greedy_commit(
        [10, 11, 99, 13],
        accepted=3,
        correction=42,
        remaining=4,
        eos_token_ids={99},
    )
    correction_eos = resolve_greedy_commit(
        [10, 12, 13],
        accepted=1,
        correction=99,
        remaining=4,
        eos_token_ids={99},
    )

    assert after_cap.committed_proposals == [10, 11]
    assert after_cap.emitted_tokens == [10, 11]
    assert after_cap.processed_tokens == 3
    assert after_cap.stopped_on_eos is False
    assert before_cap.committed_proposals == [10, 11, 99]
    assert before_cap.emitted_tokens == [10, 11, 99]
    assert before_cap.stopped_on_eos is True
    assert correction_eos.committed_proposals == [10]
    assert correction_eos.emitted_tokens == [10, 99]
    assert correction_eos.stopped_on_eos is True


def test_dense_projected_dflash_matches_original_dense_forward():
    from transformers import Qwen3Config

    from dflash.model import DFlashDraftModel

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config.block_size = 16
    config.dflash_config = {"target_layer_ids": [0, 1]}
    config.num_target_layers = 2
    base = DFlashDraftModel(config).eval()
    adapted = AMRDFlashDraft(base)
    torch.manual_seed(19)
    raw_features = torch.randn(1, 6, 32)
    noise = torch.randn(1, 16, 16)
    positions = torch.arange(22).unsqueeze(0)
    mask = torch.zeros(1, 1, 16, 22)

    expected = base(
        target_hidden=raw_features,
        noise_embedding=noise,
        position_ids=positions,
        attention_mask=mask,
        use_cache=False,
    )
    projected = adapted.project_context(raw_features)
    actual = adapted.forward_projected(
        projected_context=projected,
        noise_embedding=noise,
        position_ids=positions,
        attention_mask=mask,
    )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cached_amr_generation_keeps_target_greedy_tokens_exact():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from dflash.model import DFlashDraftModel

    torch.manual_seed(29)
    target_config = Qwen3Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        eos_token_id=None,
        pad_token_id=0,
    )
    target = Qwen3ForCausalLM(target_config).eval()
    draft_config = Qwen3Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    draft_config.block_size = 16
    draft_config.dflash_config = {"target_layer_ids": [0, 1, 2], "mask_token_id": 1}
    draft_config.num_target_layers = 3
    draft = AMRDFlashDraft(DFlashDraftModel(draft_config).eval())
    memory = AMRMemory(
        hidden_size=16,
        config=MemoryConfig(raw_budget=64, num_slots=0, local_window=0, min_context_tokens=0),
        index_dim=8,
    )
    engine = AMRDFlashEngine(
        target,
        tokenizer=type("Tokenizer", (), {"eos_token_id": None})(),
        draft=draft,
        memory=memory,
        device=torch.device("cpu"),
        mode="dense",
        use_cost_gate=False,
    )
    prompt = torch.tensor([[3, 7, 9, 11]])
    generation = engine.generate(prompt, max_new_tokens=19, capture_states=True)
    actual = generation.output_ids

    expected = prompt.clone()
    for _ in range(19):
        logits = target(input_ids=expected, use_cache=False, return_dict=True).logits[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)
        expected = torch.cat((expected, next_id), dim=1)

    assert torch.equal(actual, expected)
    assert actual.shape[1] == prompt.shape[1] + 19
    assert len(generation.acceptance_lengths) >= 2
    assert any(count < 15 for count in generation.accepted_proposals)
    expected_processed_context = prompt.shape[1] + sum(
        1 + row["committed_proposals"] for row in generation.trace
    )
    assert generation.projected_context is not None
    assert generation.projected_context.shape[1] == expected_processed_context

    reference_new = expected[0, prompt.shape[1] :].tolist()
    stop_index = next(
        index
        for index, token in enumerate(reference_new[1:], start=1)
        if token not in reference_new[:index]
    )
    stop_token = reference_new[stop_index]
    stopped = engine.generate(
        prompt,
        max_new_tokens=19,
        eos_token_ids=[stop_token],
    ).output_ids
    expected_stop = expected[:, : prompt.shape[1] + stop_index + 1]
    assert torch.equal(stopped, expected_stop)

    hybrid_memory = AMRMemory(
        hidden_size=16,
        config=MemoryConfig(raw_budget=3, num_slots=2, local_window=1, min_context_tokens=0),
        index_dim=8,
    )
    hybrid_engine = AMRDFlashEngine(
        target,
        tokenizer=type("Tokenizer", (), {"eos_token_id": None})(),
        draft=draft,
        memory=hybrid_memory,
        device=torch.device("cpu"),
        mode="amr",
        use_cost_gate=False,
    )
    hybrid = hybrid_engine.generate(prompt, max_new_tokens=19).output_ids
    assert torch.equal(hybrid, expected)

    compressor_memory = AMRMemory(
        hidden_size=16,
        config=MemoryConfig(raw_budget=0, num_slots=3, local_window=0, min_context_tokens=0),
        index_dim=8,
    )
    compressor_engine = AMRDFlashEngine(
        target,
        tokenizer=type("Tokenizer", (), {"eos_token_id": None})(),
        draft=draft,
        memory=compressor_memory,
        device=torch.device("cpu"),
        mode="compressor",
        use_cost_gate=False,
    )
    compressed = compressor_engine.generate(prompt, max_new_tokens=19).output_ids
    assert torch.equal(compressed, expected)


def test_fixed_state_evaluator_matches_target_verifier_on_dense_tiny_fixture():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from dflash.model import DFlashDraftModel
    from AMR_DFlash.model import extract_target_features

    torch.manual_seed(41)
    target_config = Qwen3Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        eos_token_id=40,
        pad_token_id=0,
    )
    target = Qwen3ForCausalLM(target_config).eval()
    draft_config = Qwen3Config(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    draft_config.block_size = 16
    draft_config.dflash_config = {"target_layer_ids": [0, 1, 2], "mask_token_id": 1}
    draft_config.num_target_layers = 3
    draft = AMRDFlashDraft(DFlashDraftModel(draft_config).eval())
    context_ids = torch.tensor([[3, 7, 9, 11]])
    with torch.inference_mode():
        context_outputs = target(input_ids=context_ids, output_hidden_states=True, return_dict=True)
        features = draft.project_context(
            extract_target_features(context_outputs, draft.target_layer_ids)
        )
        anchor_id = int(context_outputs.logits[:, -1, :].argmax(dim=-1)[0])
    memory = AMRMemory(
        hidden_size=16,
        config=MemoryConfig(raw_budget=8, num_slots=0, local_window=0, min_context_tokens=0),
        index_dim=8,
    )

    record = evaluate_one_fixed_state(
        target,
        draft,
        memory,
        context_ids=context_ids,
        projected_context=features,
        anchor_id=anchor_id,
        remaining_output_budget=7,
        mode="dense",
        eos_token_ids={40},
    )
    candidate_ids = torch.tensor([[anchor_id, *record["proposal_ids"]]])
    full_logits = target(input_ids=torch.cat((context_ids, candidate_ids), dim=1)).logits
    expected_choices = full_logits[:, context_ids.shape[1] : -1, :].argmax(dim=-1)[0].tolist()

    assert record["target_choices"] == expected_choices
    assert record["accepted_proposals_raw"] == greedy_acceptance(
        torch.tensor([record["proposal_ids"]]), torch.tensor([expected_choices])
    ).accepted
    assert record["raw_tokens"] == context_ids.shape[1]
    assert record["slot_tokens"] == 0


def test_compressor_receives_gradient_through_frozen_dflash_attention():
    from transformers import Qwen3Config

    from dflash.model import DFlashDraftModel

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config.block_size = 16
    config.dflash_config = {"target_layer_ids": [0, 1]}
    config.num_target_layers = 2
    draft = AMRDFlashDraft(DFlashDraftModel(config).eval())
    for parameter in draft.parameters():
        parameter.requires_grad_(False)
    memory = AMRMemory(
        hidden_size=16,
        config=MemoryConfig(raw_budget=3, num_slots=2, local_window=1, min_context_tokens=0),
        index_dim=8,
    )
    features = torch.randn(1, 6, 16)
    anchor = torch.randn(1, 16)
    live_positions = torch.arange(6, 22).unsqueeze(0)
    selected_positions = torch.tensor([[0, 3, 5]])
    built = memory.build(
        features,
        anchor,
        live_positions,
        mode="amr",
        use_cost_gate=False,
        candidate_positions=selected_positions,
    )
    noise = torch.randn(1, 16, 16)
    all_positions = torch.cat((built.positions, live_positions), dim=1)
    output = draft.forward_projected(
        projected_context=built.features,
        noise_embedding=noise,
        position_ids=all_positions,
        attention_mask=built.attention_mask,
    )
    output.square().sum().backward()

    compressor_grad = memory.compressor.value_projection.weight.grad
    assert compressor_grad is not None
    assert torch.isfinite(compressor_grad).all()
    assert compressor_grad.abs().sum().item() > 0
    assert memory.slot_gate.grad is not None
    assert memory.slot_gate.grad.abs().item() > 0
    assert all(parameter.grad is None for parameter in draft.parameters())


def test_candidate_generation_preserves_guard_budget_and_deduplicates_sets():
    candidates = generate_candidate_sets(
        context_length=100,
        raw_budget=12,
        local_window=4,
        seed=23,
        max_candidates=12,
    )

    assert len(candidates) >= 4
    assert all(len(row.positions) == 12 for row in candidates)
    assert all(row.positions[-4:] == (96, 97, 98, 99) for row in candidates)
    assert all(len(set(row.positions)) == len(row.positions) for row in candidates)
    assert len({row.positions for row in candidates}) == len(candidates)
    assert all(max(row.positions) < 100 for row in candidates)


def test_preference_builder_drops_ties_and_censored_candidates():
    rows = [
        {"state_id": "s", "document_id": "d", "split": "train", "candidate_id": "good", "accepted_proposals_committed": 6, "censored": False, "status": "success"},
        {"state_id": "s", "document_id": "d", "split": "train", "candidate_id": "bad", "accepted_proposals_committed": 2, "censored": False, "status": "success"},
        {"state_id": "s", "document_id": "d", "split": "train", "candidate_id": "tie", "accepted_proposals_committed": 6, "censored": False, "status": "success"},
        {"state_id": "s", "document_id": "d", "split": "train", "candidate_id": "censored", "accepted_proposals_committed": 10, "censored": True, "status": "success"},
    ]

    pairs = build_preferences(rows, max_pairs_per_state=8)

    assert len(pairs) == 2
    assert all(pair["reward_positive"] == 6 for pair in pairs)
    assert all(pair["reward_negative"] == 2 for pair in pairs)
    assert sum(pair["weight"] for pair in pairs) == pytest.approx(1.0)


def test_memory_checkpoint_restores_state_and_rejects_fingerprint_mismatch(tmp_path: Path):
    memory = AMRMemory(
        hidden_size=4,
        config=MemoryConfig(raw_budget=3, num_slots=2, local_window=1),
        index_dim=2,
    )
    metadata = {"schema_version": "amr-v0", "target_hash": "target-a", "draft_hash": "draft-a"}
    path = tmp_path / "memory.pt"
    save_memory_checkpoint(path, memory, metadata=metadata, step=7)
    expected = {
        key: value.detach().clone()
        for key, value in memory.state_dict().items()
    }
    with torch.no_grad():
        for parameter in memory.parameters():
            parameter.zero_()

    restored = load_memory_checkpoint(
        path,
        memory,
        expected_metadata={"target_hash": "target-a", "draft_hash": "draft-a"},
        device=torch.device("cpu"),
    )

    assert restored["step"] == 7
    assert all(torch.equal(memory.state_dict()[key], value) for key, value in expected.items())
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_memory_checkpoint(
            path,
            memory,
            expected_metadata={"target_hash": "target-b"},
            device=torch.device("cpu"),
        )


def test_resource_ledger_tracks_wall_time_and_only_charges_gpu_time(tmp_path: Path):
    ledger = charge_phase(
        tmp_path,
        "capture",
        elapsed_seconds=12.5,
        device=torch.device("cpu"),
    )

    assert ledger["total_gpu_seconds"] == 0.0
    assert ledger["total_wall_seconds"] == 12.5
    assert ledger["phases"]["capture"]["wall_seconds"] == 12.5
    assert remaining_gpu_seconds(
        tmp_path,
        24,
        device=torch.device("cpu"),
    ) == float("inf")
    assert read_ledger(tmp_path)["schema_version"] == "amr-v0"
