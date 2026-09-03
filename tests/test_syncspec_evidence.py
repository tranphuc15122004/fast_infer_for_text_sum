from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.evidence import SourceMemoryBank, SourceNgramIndex  # noqa: E402


def test_ngram_features_use_source_evidence_without_mutating_candidates() -> None:
    source = [10, 11, 12, 13, 14, 12, 13, 15]
    index = SourceNgramIndex(source, min_n=2, max_n=6)
    history = [10, 11, 12]
    candidates = torch.tensor([13, 99, 15])
    before = candidates.clone()
    features = index.features(history, candidates)

    assert torch.equal(candidates, before)
    assert features.shape == (3, 6)
    assert features[0, 0].item() == 4  # longest suffix 10,11,12,13
    assert features[0, 1].item() >= 1  # occurrence count
    assert features[1, 0].item() == 0
    assert features[:, 4].min().item() >= 0.0
    assert features[:, 4].max().item() <= 1.0


def test_source_memory_has_fixed_chunks_and_deterministic_fallback() -> None:
    ids = torch.arange(20)
    embeddings = torch.arange(20 * 4, dtype=torch.float32).reshape(20, 4)
    bank = SourceMemoryBank.from_source(
        ids, embeddings=embeddings, chunk_size=8, top_r=2
    )
    result = bank.retrieve(torch.ones(4), top_r=2)
    assert result.descriptors.shape == (2, 4)
    assert result.indices.tolist() == [2, 1]
    assert result.status == "ok"

    empty = SourceMemoryBank.from_source(torch.empty(0, dtype=torch.long))
    fallback = empty.retrieve(torch.ones(4), top_r=2)
    assert fallback.status == "fallback_anchor"
    assert fallback.descriptors.shape == (1, 4)
