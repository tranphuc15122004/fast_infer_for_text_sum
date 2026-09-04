"""Stage-0 target-generated trajectory collection."""

from __future__ import annotations

import random
import hashlib
from typing import Iterable

import torch

from .evidence import SourceMemoryBank
from .training import TrajectoryRecord
from .verifier import VerificationResult


class TargetTrajectoryBuilder:
    def __init__(
        self, target, seed: int = 42, metadata: dict | None = None,
        source_chunk_size: int = 128, num_anchors: int = 512,
    ):
        self.target = target
        self.seed = int(seed)
        self.metadata = dict(metadata or {})
        self.source_chunk_size = int(source_chunk_size)
        self.num_anchors = int(num_anchors)
        if self.source_chunk_size <= 0:
            raise ValueError("source_chunk_size must be positive")
        if self.num_anchors <= 0:
            raise ValueError("num_anchors must be positive")

    def build_record(
        self, sample_id: str, source_ids: torch.Tensor, max_new_tokens: int,
        include_logits: bool = False, include_target_features: bool = False,
        include_source_memory: bool = False,
    ) -> TrajectoryRecord:
        limit = int(max_new_tokens)
        if limit < 0:
            raise ValueError("max_new_tokens must be non-negative")
        state = self.target.prefill(source_ids)
        context_remaining = getattr(self.target, "remaining_context_tokens", None)
        if context_remaining is not None:
            remaining = context_remaining(state)
            if remaining is not None:
                limit = min(limit, max(0, int(remaining)))
        generated: list[int] = []
        anchor_token_ids: list[int] = []
        logits_rows: list[list[float]] = []
        target_features: list[list[float]] = []
        recent_hidden_rows: list[list[list[float]]] = []
        recent_hidden_complete = True
        source_memory = None
        source_memory_bank = None
        source_hidden = getattr(state, "source_hidden", None)
        if include_source_memory and source_hidden is not None:
            source_memory_bank = SourceMemoryBank.from_source(
                source_ids, embeddings=source_hidden,
                chunk_size=self.source_chunk_size,
            )
            source_memory = source_memory_bank.descriptors.detach().float().cpu().tolist()
        for _ in range(limit):
            anchor_token_ids.append(
                generated[-1] if generated else int(source_ids.flatten()[-1].item())
            )
            logits = self.target.next_logits(state).detach()
            token = int(logits.argmax().item())
            generated.append(token)
            if include_logits:
                logits_rows.append(logits.float().cpu().tolist())
            if include_target_features and hasattr(state, "anchor_hidden"):
                target_features.append(state.anchor_hidden.detach().float().cpu().flatten().tolist())
            if include_target_features:
                recent_hidden = getattr(state, "recent_hidden", None)
                if recent_hidden is None:
                    recent_hidden_complete = False
                else:
                    recent_tensor = recent_hidden.detach().float().cpu()
                    if recent_tensor.ndim == 1:
                        recent_tensor = recent_tensor.unsqueeze(0)
                    if recent_tensor.ndim != 2:
                        raise ValueError("target recent_hidden must have shape [R, D]")
                    recent_hidden_rows.append(recent_tensor.tolist())
            self.target.commit(
                state,
                VerificationResult(torch.tensor([token], device=logits.device), 0),
            )
            eos = getattr(self.target, "eos_token_id", None)
            if eos is not None and (token in eos if isinstance(eos, (list, tuple, set)) else token == eos):
                break
        stable_id = int.from_bytes(hashlib.sha256(str(sample_id).encode()).digest()[:8], "big")
        random_state = random.Random(self.seed ^ stable_id)
        anchor_count = min(self.num_anchors, len(generated))
        anchors = sorted(random_state.sample(range(len(generated)), anchor_count)) if anchor_count else []
        metadata = {
            "target_generated": True,
            "context_length": int(source_ids.numel()),
            "source_boundaries": [{"start": 0, "end": int(source_ids.numel()), "kind": "prompt"}],
            "prompt_template": "summarize-v1",
            "seed": self.seed,
            "eos_token_id": getattr(self.target, "eos_token_id", None),
            "decoding": {"strategy": "greedy", "max_new_tokens": limit},
            "trajectory_contract_version": 2,
            "trajectory_contract": "dflash-anchor-state-v2",
            "num_anchors_requested": self.num_anchors,
            "anchor_token_positions": [int(anchor) for anchor in anchors],
        }
        # The physical DFlash block starts at the position of the committed
        # anchor token.  For target suffix index ``a``, that position is the
        # prompt's last position plus ``a``.
        metadata["anchor_position_offsets"] = [
            max(0, int(source_ids.numel()) - 1) + int(anchor)
            for anchor in anchors
        ]
        if include_source_memory:
            if source_memory_bank is None:
                metadata["source_memory_source"] = "unavailable"
            else:
                metadata["source_memory_source"] = "target_final_hidden"
                metadata["source_memory_chunk_size"] = self.source_chunk_size
                metadata["source_memory_chunk_offsets"] = [
                    [int(start), int(end)]
                    for start, end in source_memory_bank.chunk_offsets
                ]
        stored_target_features = None
        if include_target_features and target_features:
            selected = [
                target_features[anchor]
                for anchor in anchors
                if 0 <= int(anchor) < len(target_features)
            ]
            if len(selected) == len(anchors):
                stored_target_features = selected
                metadata["target_feature_positions"] = [int(anchor) for anchor in anchors]
            else:
                # Keep a partially available legacy-compatible cache rather
                # than fabricating features for anchors the target did not
                # expose.
                stored_target_features = target_features
        stored_recent_hidden = None
        if (
            include_target_features
            and recent_hidden_complete
            and len(recent_hidden_rows) == len(generated)
        ):
            stored_recent_hidden = [recent_hidden_rows[anchor] for anchor in anchors]
            metadata["recent_hidden_positions"] = [int(anchor) for anchor in anchors]
            metadata["recent_hidden_available"] = True
        elif include_target_features:
            metadata["recent_hidden_available"] = False
        metadata.update(self.metadata)
        return TrajectoryRecord(
            sample_id=str(sample_id),
            source_ids=source_ids.detach().cpu().flatten().tolist(),
            target_ids=generated,
            anchors=anchors,
            anchor_token_ids=[anchor_token_ids[anchor] for anchor in anchors],
            target_logits=logits_rows if include_logits else None,
            target_features=stored_target_features,
            target_recent_hidden=stored_recent_hidden,
            source_memory=source_memory,
            metadata=metadata,
            contract_version=2,
        )

    def build_records(
        self, samples: Iterable, max_new_tokens: int, include_logits: bool = False,
        include_target_features: bool = False, include_source_memory: bool = False,
    ) -> list[TrajectoryRecord]:
        records = []
        for sample in samples:
            if isinstance(sample, dict):
                sample_id = sample.get("id", sample.get("sample_id", len(records)))
                source_ids = sample["source_ids"]
            else:
                sample_id, source_ids = sample
            records.append(self.build_record(
                sample_id, source_ids, max_new_tokens, include_logits,
                include_target_features, include_source_memory,
            ))
        return records
