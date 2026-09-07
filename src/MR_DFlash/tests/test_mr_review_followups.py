"""Regression tests cho các hạn chế còn lại trong review MR-DFlash."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from MR_DFlash.checkpoint import warm_start_draft_model
from MR_DFlash.memory import CSAIndexer, TargetFeatureAdapter
from MR_DFlash.mr_model import MRDFlashDraftModel, MRDFlashJointAttention, MRDraftSpec
from MR_DFlash.model import DraftSpec, DFlashDraftModel
from MR_DFlash.training import (
    OnlineMRDFlashModel,
    build_dflash_additive_mask,
    build_dflash_block_additive_mask,
)


def _spec(*, num_stages: int = 2) -> MRDraftSpec:
    return MRDraftSpec(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        num_hidden_layers=1,
        block_size=4,
        target_layer_ids=[0, 1],
        head_dim=4,
        max_position_embeddings=64,
        use_qk_norm=True,
        num_stages=num_stages,
        hca_compression_ratio=4,
        csa_compression_ratio=2,
        local_window=4,
        csa_top_k=2,
        indexer_dim=4,
    )


def test_target_feature_adapter_normalizes_both_memory_views() -> None:
    adapter = TargetFeatureAdapter(input_dim=8, hidden_size=4).float()
    features = torch.randn(2, 5, 8)
    hca, csa = adapter(features)

    assert isinstance(adapter.hca_norm, nn.Module)
    assert isinstance(adapter.csa_norm, nn.Module)
    assert torch.allclose(hca.float().square().mean(-1), torch.ones(2, 5), atol=2e-4)
    assert torch.allclose(csa.float().square().mean(-1), torch.ones(2, 5), atol=2e-4)


def test_indexer_uses_relu_of_head_dot_product() -> None:
    indexer = CSAIndexer(hidden_size=2, indexer_dim=2, num_heads=1).float()
    with torch.no_grad():
        indexer.q_proj.weight.copy_(torch.eye(2))
        indexer.k_proj.weight.copy_(torch.eye(2))
        indexer.weight_proj.weight.zero_()

    query = torch.tensor([[[1.0, -1.0]]])
    memory = torch.tensor([[[1.0, -1.0], [1.0, 1.0]]])
    scores = indexer.score(query, memory)
    expected = torch.tensor([[[2.0, 0.0]]]) * (2.0 ** -0.5)
    assert torch.allclose(scores, expected)


def test_mr_training_flattens_anchors_into_block_batch() -> None:
    torch.manual_seed(1)
    draft = MRDFlashDraftModel(_spec()).float()
    wrapper = OnlineMRDFlashModel(
        draft,
        target_lm_head=nn.Linear(8, 32, bias=False),
        target_embed_tokens=nn.Embedding(32, 8),
        mask_token_id=31,
        block_size=4,
        num_anchors=2,
        indexer_train_mode="dense",
    ).float()
    seen = []
    for stage in draft.stages:
        stage.joint_attn.register_forward_pre_hook(
            lambda _module, args: seen.append(tuple(args[0].shape))
        )

    input_ids = torch.randint(0, 31, (1, 12))
    hidden_states = torch.randn(1, 12, 16)
    loss_mask = torch.ones(1, 12)
    wrapper._forward_draft_blocks(input_ids, hidden_states, loss_mask)

    assert seen
    assert all(shape == (2, 4, 8) for shape in seen)


def test_block_local_mask_matches_diagonal_of_legacy_mask() -> None:
    keep = torch.tensor([[True, False, True]])
    anchors = torch.tensor([[2, 7, 11]])
    legacy = build_dflash_additive_mask(
        anchors,
        keep,
        S=0,
        block_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    block = build_dflash_block_additive_mask(
        keep,
        block_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = torch.stack(
        [legacy[:, :, start : start + 3, start : start + 3] for start in (0, 3, 6)],
        dim=1,
    ).reshape(3, 1, 3, 3)
    assert torch.equal(block, expected)


def test_memory_flatten_blocks_preserves_each_anchor_local_view() -> None:
    from MR_DFlash.memory import MRTargetMemory

    memory = MRTargetMemory(
        input_dim=8,
        hidden_size=4,
        hca_compression_ratio=4,
        csa_compression_ratio=2,
        local_window=2,
    )
    state = memory.build(
        torch.randn(1, 10, 8),
        query_positions=torch.tensor([[3, 8]]),
    )
    flat = state.flatten_blocks(2)

    # Global memory stays at the original batch; only local memory is
    # materialized per block.
    assert flat.hca.shape[0] == 1
    assert flat.local_hca.shape == (2, 2, 4)
    assert flat.local_positions.tolist() == [[1, 2], [6, 7]]


def test_joint_attention_projects_shared_context_without_query_expansion() -> None:
    attention = MRDFlashJointAttention(
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        rope_theta=10000.0,
        use_qk_norm=True,
        rms_norm_eps=1e-6,
    ).float()
    seen = []
    attention.k_proj.register_forward_pre_hook(
        lambda _module, args: seen.append(tuple(args[0].shape))
    )
    hidden = torch.randn(2, 4, 8)
    context = torch.randn(2, 5, 8)
    positions = torch.arange(4).view(1, 4).expand(2, -1)
    context_positions = torch.arange(5).view(1, 5).expand(2, -1)
    mask = torch.zeros(2, 1, 4, 9)
    output = attention(hidden, context, positions, context_positions, mask)

    assert output.shape == hidden.shape
    assert seen == [(2, 5, 8), (2, 4, 8)]


def test_dflash_checkpoint_maps_into_every_mr_stage_and_native_load_is_strict(tmp_path: Path) -> None:
    dflash_spec = DraftSpec(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        num_hidden_layers=1,
        block_size=4,
        target_layer_ids=[0],
        head_dim=4,
        max_position_embeddings=64,
        use_qk_norm=True,
    )
    dflash = DFlashDraftModel(dflash_spec).float()
    mr = MRDFlashDraftModel(
        MRDraftSpec.from_dflash(
            dflash_spec,
            num_stages=2,
            hca_compression_ratio=4,
            csa_compression_ratio=2,
            local_window=4,
            csa_top_k=2,
            indexer_dim=4,
        )
    ).float()
    source_path = tmp_path / "dflash.pt"
    torch.save({"format": "dflash", "draft_state_dict": dflash.state_dict()}, source_path)

    missing, unexpected = warm_start_draft_model(
        mr, str(source_path), strategy_name="mr_dflash"
    )
    assert missing == []
    assert unexpected == []
    assert torch.equal(
        mr.stages[0].joint_attn.q_proj.weight,
        dflash.layers[0].self_attn.q_proj.weight,
    )
    assert torch.equal(
        mr.stages[1].joint_attn.q_proj.weight,
        dflash.layers[0].self_attn.q_proj.weight,
    )

    native_path = tmp_path / "native.pt"
    state = mr.state_dict()
    state.pop("stages.0.joint_attn.q_proj.weight")
    torch.save({"format": "mr_dflash", "draft_state_dict": state}, native_path)
    with pytest.raises(RuntimeError, match="strict"):
        warm_start_draft_model(mr, str(native_path), strategy_name="mr_dflash")
