from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.model import (  # noqa: E402
    SyncSpecDrafter,
    SyncSpecDrafterConfig,
    build_masked_block,
    top_m_candidates,
)


def test_drafter_block_forward_and_backward_cpu() -> None:
    torch.manual_seed(3)
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=41, hidden_size=16, layers=2, heads=4, groups=4, top_m=8
    ))
    ids = build_masked_block(torch.tensor([7, 8]), kd=5, mask_token_id=40)
    output = model(
        ids,
        target_anchor=torch.randn(2, 16),
        recent_hidden=torch.randn(2, 3, 16),
        source_memory=torch.randn(2, 2, 16),
    )
    assert output.logits.shape == (2, 5, 41)
    assert output.hidden.shape == (2, 5, 16)
    loss = output.logits.square().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_mask_slots_use_learned_sentinel_after_target_tying() -> None:
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=17, hidden_size=8, layers=1, heads=2, groups=2,
        mask_token_id=16,
    )).eval()
    embedding = torch.nn.Embedding(17, 8)
    head = torch.nn.Linear(8, 17, bias=False)
    model.tie_target_weights(embedding, head)
    ids = build_masked_block(torch.tensor([3]), kd=2, mask_token_id=16)
    with torch.no_grad():
        first = model(ids).logits
        embedding.weight[16].fill_(123.0)
        second = model(ids).logits
    assert torch.allclose(first, second)
    assert model.mask_embedding.requires_grad


def test_drafter_accepts_per_row_anchor_offsets() -> None:
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=17, hidden_size=8, layers=1, heads=2, groups=2,
        max_positions=32,
    )).eval()
    ids = build_masked_block(torch.tensor([3, 4]), kd=2, mask_token_id=16)
    output = model(ids, position_offset=torch.tensor([0, 7]))
    assert output.logits.shape == (2, 2, 17)


def test_drafter_rejects_position_offsets_beyond_capacity() -> None:
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=17, hidden_size=8, layers=1, heads=2, groups=2,
        max_positions=8,
    )).eval()
    ids = build_masked_block(torch.tensor([3]), kd=2, mask_token_id=16)
    with pytest.raises(ValueError, match="max_positions"):
        model(ids, position_offset=7)
    with pytest.raises(ValueError, match="max_positions"):
        model(ids.repeat(2, 1), position_offset=torch.tensor([0, 7]))


def test_drafter_config_rejects_invalid_attention_and_position_capacity() -> None:
    with pytest.raises(ValueError, match="heads"):
        SyncSpecDrafterConfig(
            vocab_size=17, hidden_size=8, layers=1, heads=0, groups=2,
        )
    with pytest.raises(ValueError, match="groups"):
        SyncSpecDrafterConfig(
            vocab_size=17, hidden_size=8, layers=1, heads=2, groups=0,
        )
    with pytest.raises(ValueError, match="max_positions"):
        SyncSpecDrafterConfig(
            vocab_size=17, hidden_size=8, layers=1, heads=2, groups=2,
            max_positions=0,
        )
    with pytest.raises(ValueError, match="top_m"):
        SyncSpecDrafterConfig(
            vocab_size=17, hidden_size=8, layers=1, heads=2, groups=2,
            top_m=0,
        )


def test_top_m_and_target_weight_tying(tmp_path: Path) -> None:
    model = SyncSpecDrafter(SyncSpecDrafterConfig(vocab_size=17, hidden_size=8, heads=2, groups=2))
    logits = torch.arange(34, dtype=torch.float32).reshape(2, 17)
    ids, values = top_m_candidates(logits, 4)
    assert ids.shape == values.shape == (2, 4)
    assert ids[0].tolist() == [16, 15, 14, 13]
    with pytest.raises(ValueError, match="top_m"):
        top_m_candidates(logits, 0)

    embedding = torch.nn.Embedding(17, 8)
    head = torch.nn.Linear(8, 17, bias=False)
    model.tie_target_weights(embedding, head)
    assert model.embedding.weight.data_ptr() == embedding.weight.data_ptr()
    assert model.lm_head.weight.data_ptr() == head.weight.data_ptr()
    assert model.embedding.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False

    path = tmp_path / "drafter"
    model.save_pretrained(path)
    restored = SyncSpecDrafter.from_pretrained(path)
    assert restored.config == model.config
    assert torch.allclose(restored.embedding.weight, model.embedding.weight)


def test_tied_checkpoint_can_omit_frozen_target_weights(tmp_path: Path) -> None:
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=17, hidden_size=8, heads=2, groups=2,
    ))
    embedding = torch.nn.Embedding(17, 8)
    head = torch.nn.Linear(8, 17, bias=False)
    model.tie_target_weights(embedding, head)
    path = tmp_path / "compact-drafter"
    model.save_pretrained(path, omit_tied_weights=True)
    state = torch.load(path / "pytorch_model.bin", weights_only=True)
    assert "embedding.weight" not in state
    assert "lm_head.weight" not in state
    metadata = json.loads((path / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert metadata["tied_target_weights"] is True
    restored = SyncSpecDrafter.from_pretrained(path)
    with pytest.raises(RuntimeError, match="tie_target_weights"):
        restored(torch.tensor([[1, 2]]))
    restored.tie_target_weights(embedding, head)
    assert restored(torch.tensor([[1, 2]])).logits.shape == (1, 2, 17)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/B200 is not available on this host")
def test_drafter_cuda_forward() -> None:
    model = SyncSpecDrafter(SyncSpecDrafterConfig(vocab_size=32, hidden_size=16, heads=4, groups=4)).cuda()
    ids = build_masked_block(torch.tensor([4], device="cuda"), 3, 31)
    output = model(ids)
    assert output.logits.is_cuda
