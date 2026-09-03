from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.verifier import (  # noqa: E402
    KVTransaction,
    greedy_verify,
    stochastic_verify,
)


def test_greedy_verifier_commits_longest_prefix_and_correction() -> None:
    proposals = torch.tensor([2, 9, 9])
    target_logits = torch.tensor([
        [0.0, 0.0, 8.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 7.0],
        [0.0, 6.0, 0.0, 0.0, 0.0],
    ])
    result = greedy_verify(proposals, target_logits)
    assert result.accepted_length == 1
    assert result.committed_ids.tolist() == [2, 4]
    assert result.correction_token_id == 4


def test_stochastic_residual_is_normalized_and_transaction_rolls_back() -> None:
    p = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    q = torch.tensor([[0.5, 0.5, 0.0], [0.6, 0.2, 0.2]])
    result = stochastic_verify(
        proposals=torch.tensor([0, 1]),
        target_probs=p,
        proposal_probs=q,
        generator=torch.Generator().manual_seed(3),
    )
    assert result.committed_ids.numel() >= 1
    if result.residual_probs is not None:
        assert torch.allclose(result.residual_probs.sum(), torch.tensor(1.0))
    rejected = stochastic_verify(
        torch.tensor([2]), torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]), generator=torch.Generator().manual_seed(3),
    )
    assert rejected.correction_token_id == 0
    assert torch.allclose(rejected.residual_probs.sum(), torch.tensor(1.0))

    tx = KVTransaction([10, 11])
    snapshot = tx.begin()
    tx.append_uncommitted([12, 13])
    tx.rollback(snapshot)
    assert tx.committed_ids == [10, 11]
    tx.append_uncommitted([12])
    tx.commit()
    assert tx.committed_ids == [10, 11, 12]


def test_stochastic_rejection_sampling_matches_target_distribution() -> None:
    target = torch.tensor([0.70, 0.20, 0.10])
    proposal = torch.tensor([0.10, 0.80, 0.10])
    generator = torch.Generator().manual_seed(17)
    counts = torch.zeros(3, dtype=torch.float32)
    trials = 12000
    for _ in range(trials):
        proposed = torch.multinomial(proposal, 1, generator=generator)
        result = stochastic_verify(
            proposed, target.unsqueeze(0), proposal.unsqueeze(0), generator=generator,
        )
        counts[int(result.committed_ids[0].item())] += 1
    observed = counts / trials
    assert torch.allclose(observed, target, atol=0.025)
