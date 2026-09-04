"""Trajectory cache and trainable losses for SyncSpec stages 0–4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

from .model import top_m_candidates
from .evidence import SourceMemoryBank, SourceNgramIndex
from .survival import survival_from_hazard


@dataclass(eq=True)
class TrajectoryRecord:
    sample_id: str
    source_ids: list[int]
    target_ids: list[int]
    anchors: list[int] = field(default_factory=list)
    anchor_token_ids: list[int] | None = None
    target_logits: list[list[float]] | None = None
    target_features: list[list[float]] | None = None
    target_recent_hidden: list[list[list[float]]] | None = None
    source_memory: list[list[float]] | None = None
    metadata: dict = field(default_factory=dict)
    contract_version: int = 2

    def to_json(self, fingerprint: str) -> dict:
        return {"schema_version": 1, "fingerprint": fingerprint, **asdict(self)}

    @classmethod
    def from_json(cls, raw: dict) -> "TrajectoryRecord":
        legacy = any(
            key not in raw
            for key in ("contract_version", "anchor_token_ids", "target_recent_hidden")
        )
        if legacy:
            warnings.warn(
                "loading legacy trajectory contract without explicit DFlash "
                "anchor tokens/recent hidden; downstream code must use its "
                "documented compatibility fallback",
                UserWarning,
                stacklevel=2,
            )
        allowed = {
            "sample_id", "source_ids", "target_ids", "anchors", "target_logits",
            "anchor_token_ids", "target_features", "target_recent_hidden",
            "source_memory", "metadata", "contract_version",
        }
        values = {key: raw[key] for key in allowed if key in raw}
        if legacy:
            values["contract_version"] = 1
        return cls(**values)


class TrajectoryCache:
    def __init__(self, path: Path | str, fingerprint: str):
        self.path = Path(path)
        self.fingerprint = fingerprint

    @property
    def _torch_format(self) -> bool:
        return self.path.suffix.lower() in {".pt", ".pth", ".torch"}

    def _load_torch_payload(self) -> dict:
        try:
            payload = torch.load(
                self.path, map_location="cpu", weights_only=True,
            )
        except Exception as exc:
            raise ValueError(f"invalid torch trajectory cache: {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("invalid torch trajectory cache schema")
        if not isinstance(payload.get("fingerprint"), str):
            raise ValueError("torch trajectory cache has no fingerprint")
        if not isinstance(payload.get("records"), list):
            raise ValueError("torch trajectory cache has no records list")
        return payload

    @staticmethod
    def _torch_record_payload(record: TrajectoryRecord) -> dict:
        raw = asdict(record)
        for field_name in (
            "target_logits", "target_features", "target_recent_hidden", "source_memory",
        ):
            value = raw.get(field_name)
            if value is not None and not torch.is_tensor(value):
                raw[field_name] = torch.as_tensor(value, dtype=torch.float32)
        return raw

    @staticmethod
    def _restore_torch_record(raw: dict) -> dict:
        restored = dict(raw)
        for field_name in (
            "target_logits", "target_features", "target_recent_hidden", "source_memory",
        ):
            value = restored.get(field_name)
            if torch.is_tensor(value):
                restored[field_name] = value.tolist()
        return restored

    def read_fingerprint(self) -> str:
        """Read only the cache fingerprint without materializing records."""
        if self._torch_format:
            return str(self._load_torch_payload()["fingerprint"])
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("schema_version") != 1:
                    raise ValueError("unsupported trajectory cache schema")
                fingerprint = raw.get("fingerprint")
                if not isinstance(fingerprint, str) or not fingerprint:
                    raise ValueError("trajectory cache has no fingerprint")
                return fingerprint
        raise ValueError("trajectory cache is empty")

    def write(self, records: Iterable[TrajectoryRecord], append: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._torch_format:
            existing = list(self.read()) if append and self.path.exists() else []
            existing_ids = {record.sample_id for record in existing}
            rows = existing[:]
            for record in records:
                if record.sample_id in existing_ids:
                    continue
                rows.append(record)
                existing_ids.add(record.sample_id)
            payload = {
                "schema_version": 1,
                "fingerprint": self.fingerprint,
                "records": [self._torch_record_payload(record) for record in rows],
            }
            temporary = self.path.with_name(self.path.name + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(self.path)
            return
        existing_ids: set[str] = set()
        if append and self.path.exists():
            existing_ids = {record.sample_id for record in self.read()}
        mode = "a" if append and self.path.exists() else "w"
        with self.path.open(mode, encoding="utf-8") as stream:
            for record in records:
                if record.sample_id in existing_ids:
                    continue
                stream.write(json.dumps(record.to_json(self.fingerprint), ensure_ascii=False) + "\n")
                existing_ids.add(record.sample_id)

    def read(self) -> Iterator[TrajectoryRecord]:
        if self._torch_format:
            payload = self._load_torch_payload()
            if payload["fingerprint"] != self.fingerprint:
                raise ValueError("trajectory fingerprint mismatch")
            for raw in payload["records"]:
                if not isinstance(raw, dict):
                    raise ValueError("invalid record in torch trajectory cache")
                yield TrajectoryRecord.from_json(self._restore_torch_record(raw))
            return
        with self.path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("schema_version") != 1:
                    raise ValueError(
                        f"unsupported trajectory cache schema at line {line_no}"
                    )
                if raw.get("fingerprint") != self.fingerprint:
                    raise ValueError(f"trajectory fingerprint mismatch at line {line_no}")
                yield TrajectoryRecord.from_json(raw)


def diffusion_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    position_weight: torch.Tensor | None = None,
    teacher_logits: torch.Tensor | None = None,
    kl_weight: float = 0.0,
    rank_margin: float = 0.0,
    rank_weight: float = 0.0,
    rank_top_m: int = 16,
) -> torch.Tensor:
    if logits.shape[:-1] != target_ids.shape:
        raise ValueError("logits and target_ids shape mismatch")
    if teacher_logits is not None and teacher_logits.shape != logits.shape:
        raise ValueError("teacher_logits and student logits shape mismatch")
    if target_ids.numel() and (
        int(target_ids.min().item()) < 0 or int(target_ids.max().item()) >= logits.shape[-1]
    ):
        raise ValueError("target_ids contain a token outside the vocabulary")
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction="none").reshape_as(target_ids)
    if valid_mask is None:
        valid_mask = torch.ones_like(losses, dtype=torch.bool)
    weights = valid_mask.to(losses.dtype)
    if position_weight is not None:
        weights = weights * position_weight.to(losses.device, losses.dtype)
    denominator = weights.sum().clamp_min(1.0)
    total = (losses * weights).sum() / denominator

    if teacher_logits is not None and float(kl_weight) > 0.0:
        teacher = teacher_logits.detach().float()
        student_log_probs = F.log_softmax(logits.float(), dim=-1)
        teacher_probs = F.softmax(teacher, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        total = total + float(kl_weight) * (kl * weights.float()).sum() / denominator.float()

    if float(rank_weight) > 0.0 and float(rank_margin) > 0.0:
        top_m = int(rank_top_m)
        if top_m <= 0:
            raise ValueError("rank_top_m must be positive")
        if logits.shape[-1] > top_m + 1:
            non_target = logits.float().clone()
            non_target.scatter_(-1, target_ids.unsqueeze(-1), float("-inf"))
            boundary = non_target.topk(k=min(top_m + 1, logits.shape[-1] - 1), dim=-1).values[..., -1]
            target_logit = logits.float().gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            margin = F.relu(float(rank_margin) - target_logit + boundary)
            total = total + float(rank_weight) * (margin * weights.float()).sum() / denominator.float()
    return total


def dflash_position_weights(
    kd: int, gamma: float = 7.0, device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return DFlash's exponential weights over future proposal slots."""
    if int(kd) <= 0:
        raise ValueError("kd must be positive")
    if float(gamma) <= 0.0:
        raise ValueError("DFlash loss-decay gamma must be positive")
    return torch.exp(
        -torch.arange(int(kd), dtype=torch.float32, device=device) / float(gamma)
    )


def _target_logits_batch(
    records: list[TrajectoryRecord], anchor_indices: list[int], kd: int,
    vocab_size: int, device: str | torch.device,
) -> torch.Tensor | None:
    """Build an optional teacher-logit block from Stage-0 cached logits."""
    rows = []
    for record, anchor in zip(records, anchor_indices):
        if not record.target_logits:
            return None
        values = record.target_logits[int(anchor): int(anchor) + int(kd)]
        if len(values) != int(kd) or any(len(row) != int(vocab_size) for row in values):
            return None
        rows.append(torch.tensor(values, dtype=torch.float32, device=device))
    return torch.stack(rows) if rows else None


def selector_loss(
    candidate_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    teacher_forcing: float = 1.0,
) -> torch.Tensor:
    if candidate_logits.shape != candidate_ids.shape or candidate_logits.shape[:-1] != target_ids.shape:
        raise ValueError("selector candidate/target shape mismatch")
    matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    has_target = matches.any(dim=-1)
    safe_target = matches.to(candidate_logits.dtype).argmax(dim=-1)
    losses = F.cross_entropy(candidate_logits.reshape(-1, candidate_logits.shape[-1]), safe_target.reshape(-1), reduction="none").reshape_as(safe_target)
    mask = has_target
    if valid_mask is not None:
        mask = mask & valid_mask.bool()
    # Teacher-forcing is tracked as a scalar schedule; missing candidates are
    # always masked instead of silently teaching an impossible target.
    scale = float(max(0.0, min(1.0, teacher_forcing)))
    return (losses * mask.to(losses.dtype)).sum() / mask.to(losses.dtype).sum().clamp_min(1.0) * scale


def survival_loss(hazard: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if hazard.shape != labels.shape:
        raise ValueError("hazard and labels shape mismatch")
    # The head emits discrete hazards h_j; the supervision target is the
    # cumulative prefix-survival z_j=P(A >= j) from the v1.1 contract.
    survival = survival_from_hazard(hazard)
    loss = F.binary_cross_entropy(
        survival.clamp(1e-6, 1 - 1e-6), labels.to(hazard.dtype), reduction="none",
    )
    if valid_mask is not None:
        loss = loss * valid_mask.to(loss.dtype)
        return loss.sum() / valid_mask.to(loss.dtype).sum().clamp_min(1.0)
    return loss.mean()


def calibration_metrics(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> dict[str, float]:
    p = probabilities.detach().float().flatten().clamp(0, 1)
    y = labels.detach().float().flatten()
    if p.numel() != y.numel() or p.numel() == 0:
        raise ValueError("probabilities and labels must be non-empty and equal length")
    brier = float((p - y).square().mean().item())
    ece = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for index in range(bins):
        mask = (p >= edges[index]) & (p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1])
        if mask.any():
            ece += float(mask.float().mean().item()) * abs(float(p[mask].mean() - y[mask].mean()))
    return {"ece": ece, "brier": brier}


def collect_on_policy_survival_examples(
    engine, samples: Iterable, max_new_tokens: int, seed: int = 0,
    force_kv: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect survival labels after real selector + exact-verifier rollouts.

    ``force_kv`` is used during initial head training so a random, untrained
    controller cannot hide suffix draft positions from the label set.
    """
    feature_rows: list[list[float]] = []
    label_rows: list[float] = []
    for offset, sample in enumerate(samples):
        sample_id, source_ids = sample if not isinstance(sample, dict) else (
            sample.get("id", offset), sample["source_ids"]
        )
        del sample_id
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "seed": seed + offset,
        }
        if force_kv is not None:
            generate_kwargs["force_kv"] = int(force_kv)
        result = engine.generate(source_ids, **generate_kwargs)
        if result.survival_features and result.survival_labels:
            feature_rows.extend(result.survival_features)
            label_rows.extend(result.survival_labels)
            continue
        for budget, accepted in zip(result.budgets, result.accepted_lengths):
            kd, kv = int(budget["kd"]), int(budget["kv"])
            if kv <= 0:
                continue
            for position in range(kv):
                feature_rows.append([
                    (position + 1) / max(1, kv),
                    float(position < accepted),
                    float(kd),
                    float(kv),
                    float(result.fallback_rounds),
                    float(source_ids.numel()),
                    float(result.rounds),
                    1.0,
                ])
                label_rows.append(float(position < accepted))
    if not feature_rows:
        return torch.empty((0, 8)), torch.empty((0,))
    return torch.tensor(feature_rows, dtype=torch.float32), torch.tensor(label_rows, dtype=torch.float32)


def cache_fingerprint(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def artifact_fingerprint(path: Path | str) -> str:
    """Fingerprint a local model/checkpoint directory with bounded I/O.

    Small metadata/tokenizer files are hashed in full.  Large weight files use
    their relative path, size, and beginning/end chunks so Stage 0 does not
    spend minutes rereading a multi-shard model merely to construct a cache
    key.  An unresolved model identifier remains stable and is deliberately
    distinct from a local artifact.
    """
    root = Path(path)
    if not root.exists():
        return cache_fingerprint("unresolved-artifact", str(path))
    files = [root] if root.is_file() else sorted(
        item for item in root.rglob("*") if item.is_file()
    )
    digest = hashlib.sha256()
    if not files:
        digest.update(b"empty-artifact")
    for item in files:
        relative = item.name if root.is_file() else str(item.relative_to(root))
        stat = item.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            if stat.st_size <= 8 * 1024 * 1024:
                digest.update(stream.read())
            else:
                chunk = 64 * 1024
                digest.update(stream.read(chunk))
                stream.seek(max(0, stat.st_size - chunk))
                digest.update(stream.read(chunk))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def build_stage1_batch(
    records: Iterable[TrajectoryRecord], kd: int, mask_token_id: int,
    device: str | torch.device = "cpu", anchor_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a DFlash block: one committed anchor plus ``kd`` future slots.

    ``kd`` is intentionally the public proposal count.  The physical input
    has length ``kd + 1``; slot zero is the committed token and is excluded
    from the loss.  Target suffix index ``a`` therefore uses labels
    ``target_ids[a:a+kd]`` for physical slots ``1..kd``.
    """
    rows = list(records)
    if not rows or kd <= 0:
        raise ValueError("stage-1 batch requires records and K_d > 0")
    if anchor_indices is not None and len(anchor_indices) != len(rows):
        raise ValueError("one anchor index is required per trajectory record")
    physical_length = int(kd) + 1
    masked = torch.full(
        (len(rows), physical_length), int(mask_token_id), dtype=torch.long, device=device,
    )
    targets = torch.zeros((len(rows), physical_length), dtype=torch.long, device=device)
    valid = torch.zeros((len(rows), physical_length), dtype=torch.bool, device=device)
    for row, record in enumerate(rows):
        anchor = (
            anchor_indices[row]
            if anchor_indices is not None
            else (record.anchors[0] if record.anchors else 0)
        )
        anchor = min(max(0, int(anchor)), len(record.target_ids))
        anchor_token = None
        if record.anchor_token_ids:
            stored_positions = record.metadata.get("anchor_token_positions")
            if (
                isinstance(stored_positions, list)
                and len(stored_positions) == len(record.anchor_token_ids)
                and int(anchor) in [int(value) for value in stored_positions]
            ):
                anchor_token = record.anchor_token_ids[stored_positions.index(int(anchor))]
            elif len(record.anchor_token_ids) == len(record.anchors):
                try:
                    anchor_token = record.anchor_token_ids[record.anchors.index(anchor)]
                except ValueError:
                    anchor_token = None
        if anchor_token is None:
            # Legacy caches can derive the state token because Stage 0's
            # anchor index is the number of already committed target tokens.
            warnings.warn(
                "deriving anchor token from legacy trajectory fields",
                UserWarning,
                stacklevel=2,
            )
            if anchor == 0:
                if not record.source_ids:
                    raise ValueError("trajectory record needs source_ids for anchor_token")
                anchor_token = record.source_ids[-1]
            elif anchor <= len(record.target_ids):
                anchor_token = record.target_ids[anchor - 1]
            else:
                raise ValueError("trajectory anchor index cannot derive anchor_token")
        masked[row, 0] = int(anchor_token)
        targets[row, 0] = int(anchor_token)
        values = record.target_ids[anchor : anchor + kd]
        if values:
            targets[row, 1 : 1 + len(values)] = torch.tensor(
                values, dtype=torch.long, device=device,
            )
            valid[row, 1 : 1 + len(values)] = True
    return masked, targets, valid


def sample_anchor_indices(
    records: Iterable[TrajectoryRecord], generator: torch.Generator | None = None,
) -> list[int]:
    """Sample one stored target-generation anchor per record reproducibly."""
    rows = list(records)
    result = []
    for record in rows:
        anchors = record.anchors or [0]
        if generator is None or len(anchors) == 1:
            position = 0
        else:
            position = int(torch.randint(
                len(anchors), (), generator=generator, device="cpu",
            ).item())
        result.append(int(anchors[position]))
    return result


def anchor_position_offsets(
    records: Iterable[TrajectoryRecord], anchor_indices: Iterable[int],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return absolute target-context positions for Anchor-Offset training.

    An anchor index is measured inside the target-generated suffix.  The
    physical block starts at the last committed token, so its absolute
    position is ``prompt_length - 1 + anchor``.  Metadata is preferred
    because it preserves the exact tokenized prompt length used by Stage 0;
    old caches fall back to ``source_ids``.
    """
    rows = list(records)
    anchors = list(anchor_indices)
    if len(rows) != len(anchors):
        raise ValueError("one anchor index is required per trajectory record")
    offsets = []
    for record, anchor in zip(rows, anchors):
        context_length = int(record.metadata.get("context_length", len(record.source_ids)))
        stored = record.metadata.get("anchor_position_offsets")
        stored_positions = record.metadata.get("anchor_position_offsets_positions")
        if not isinstance(stored_positions, list):
            stored_positions = record.metadata.get("anchor_token_positions")
        if isinstance(stored, list):
            if isinstance(stored_positions, list) and int(anchor) in [
                int(value) for value in stored_positions
            ]:
                anchor_slot = [int(value) for value in stored_positions].index(int(anchor))
                if anchor_slot < len(stored):
                    offsets.append(max(0, int(stored[anchor_slot])))
                    continue
            if len(stored) == len(record.anchors) and int(anchor) in record.anchors:
                anchor_slot = record.anchors.index(int(anchor))
                if anchor_slot < len(stored):
                    offsets.append(max(0, int(stored[anchor_slot])))
                    continue
        offsets.append(max(0, context_length - 1) + int(anchor))
    return torch.tensor(offsets, dtype=torch.long, device=device)


def _target_anchor_batch(
    records: list[TrajectoryRecord], anchor_indices: list[int], hidden_size: int,
    device: str | torch.device,
) -> torch.Tensor | None:
    values = []
    for record, anchor in zip(records, anchor_indices):
        if not record.target_features:
            return None
        stored_positions = record.metadata.get("target_feature_positions")
        if (
            isinstance(stored_positions, list)
            and len(stored_positions) == len(record.target_features)
        ):
            try:
                position = stored_positions.index(int(anchor))
            except ValueError:
                return None
        else:
            # Older caches stored one feature for every target suffix token.
            position = min(max(0, int(anchor)), len(record.target_features) - 1)
        values.append(record.target_features[position])
    if not values or not all(len(value) == int(hidden_size) for value in values):
        return None
    return torch.tensor(values, dtype=torch.float32, device=device)


def _target_recent_hidden_batch(
    records: list[TrajectoryRecord], anchor_indices: list[int], hidden_size: int,
    device: str | torch.device,
) -> torch.Tensor | None:
    """Load and left-pad serving-equivalent target recent-hidden windows."""
    values: list[torch.Tensor] = []
    for record, anchor in zip(records, anchor_indices):
        if not record.target_recent_hidden:
            return None
        stored_positions = record.metadata.get("recent_hidden_positions")
        if (
            isinstance(stored_positions, list)
            and len(stored_positions) == len(record.target_recent_hidden)
        ):
            try:
                position = stored_positions.index(int(anchor))
            except ValueError:
                return None
        elif len(record.target_recent_hidden) == len(record.target_ids):
            position = min(max(0, int(anchor)), len(record.target_recent_hidden) - 1)
        elif len(record.target_recent_hidden) == len(record.anchors):
            try:
                position = record.anchors.index(int(anchor))
            except ValueError:
                return None
        else:
            return None
        value = torch.tensor(
            record.target_recent_hidden[position], dtype=torch.float32, device=device,
        )
        if value.ndim != 2 or value.shape[-1] != int(hidden_size) or value.shape[0] == 0:
            raise ValueError("target_recent_hidden must contain non-empty [R, hidden] windows")
        values.append(value)
    if not values:
        return None
    width = max(int(value.shape[0]) for value in values)
    result = torch.zeros(
        (len(values), width, int(hidden_size)), dtype=torch.float32, device=device,
    )
    for row, value in enumerate(values):
        result[row, -value.shape[0] :] = value
    return result


def _expand_anchor_rows(
    records: Iterable[TrajectoryRecord], kd: int | None = None,
    max_positions: int | None = None,
) -> list[tuple[TrajectoryRecord, int]]:
    """Flatten eligible stored target states for DFlash-style batches.

    A physical block starts at ``prompt_len - 1 + anchor`` and has ``kd + 1``
    positions.  Dropping rows that would exceed the drafter positional table
    keeps long-context training aligned with the runtime capacity guard.
    """
    if kd is not None and int(kd) <= 0:
        raise ValueError("kd must be positive when filtering anchor rows")
    if max_positions is not None and int(max_positions) <= 0:
        raise ValueError("max_positions must be positive when filtering anchor rows")
    expanded = []
    for record in records:
        anchors = record.anchors or [0]
        for anchor in anchors:
            anchor = int(anchor)
            # SpecForge's DFlash sampler only considers positions for which
            # the current forward has a fully supervised target suffix.  The
            # offline cache has the same information in ``target_ids``.
            if kd is not None and (anchor < 0 or anchor + int(kd) > len(record.target_ids)):
                continue
            if kd is not None and max_positions is not None:
                offset = int(anchor_position_offsets([record], [anchor])[0].item())
                if offset + int(kd) + 1 > int(max_positions):
                    continue
            expanded.append((record, anchor))
    return expanded


def _sample_random_anchor_rows(
    records: Iterable[TrajectoryRecord], kd: int,
    max_positions: int | None, num_anchors: int,
    generator: torch.Generator | None = None,
    allow_truncated_suffix: bool = False,
) -> list[tuple[TrajectoryRecord, int]]:
    """Sample eligible DFlash anchor states independently for one forward.

    SpecForge samples valid anchor positions from each sequence on every
    training forward, rather than flattening all positions once and cycling
    through that static list.  SyncSpec's Stage-0 cache is offline, so the
    equivalent eligible set is the cached anchor states whose full ``kd``
    target suffix and physical ``kd + 1`` block fit.  Selected positions are
    returned in sequence order, matching SpecForge's sorted sampled anchors.
    """
    if int(kd) <= 0:
        raise ValueError("kd must be positive when sampling anchor rows")
    if int(num_anchors) <= 0:
        raise ValueError("num_anchors must be positive")
    if max_positions is not None and int(max_positions) <= 0:
        raise ValueError("max_positions must be positive when sampling anchor rows")

    sampled: list[tuple[TrajectoryRecord, int]] = []
    for record in records:
        eligible = _expand_anchor_rows(
            [record], kd=int(kd), max_positions=max_positions,
        )
        if not eligible and allow_truncated_suffix:
            # A target can terminate before K_d because of EOS.  Keep the
            # state only when at least one future target remains; build_stage1
            # marks the unavailable tail invalid.  The normal path above is
            # the strict SpecForge-style full-suffix eligibility rule.
            eligible = [
                (record, anchor) for record, anchor in _expand_anchor_rows(
                    [record], kd=None, max_positions=max_positions,
                ) if int(anchor) < len(record.target_ids)
            ]
        # A malformed cache should not make the same anchor appear multiple
        # times in a single DFlash batch.
        unique: list[tuple[TrajectoryRecord, int]] = []
        seen: set[int] = set()
        for row in eligible:
            if row[1] not in seen:
                unique.append(row)
                seen.add(row[1])
        if not unique:
            continue
        width = min(int(num_anchors), len(unique))
        permutation = torch.randperm(
            len(unique), generator=generator, device="cpu",
        )[:width].tolist()
        selected = [unique[index] for index in permutation]
        sampled.extend(sorted(selected, key=lambda row: row[1]))
    return sampled


def _source_memory_batch(
    model, records: list[TrajectoryRecord], anchor_tensor: torch.Tensor | None,
    device: str | torch.device, top_r: int = 8, chunk_size: int = 128,
) -> torch.Tensor | None:
    """Build the same bounded source-memory shape used by serving."""
    memories = []
    for row, record in enumerate(records):
        source = torch.tensor(record.source_ids, dtype=torch.long, device=device)
        if source.numel() == 0:
            return None
        cached = record.source_memory
        if cached:
            descriptors = torch.tensor(cached, dtype=torch.float32, device=device)
            if descriptors.ndim != 2 or descriptors.shape[0] == 0:
                raise ValueError("cached source_memory must be a non-empty [chunks, hidden] matrix")
            raw_offsets = record.metadata.get("source_memory_chunk_offsets")
            if isinstance(raw_offsets, list) and len(raw_offsets) == descriptors.shape[0]:
                offsets = tuple((int(pair[0]), int(pair[1])) for pair in raw_offsets)
            else:
                offsets = tuple(
                    (start, min(int(source.numel()), start + int(chunk_size)))
                    for start in range(0, int(source.numel()), int(chunk_size))
                )
            if len(offsets) != descriptors.shape[0]:
                raise ValueError("cached source_memory chunk metadata does not match descriptors")
            bank = SourceMemoryBank(descriptors, offsets, source, top_r=top_r)
        else:
            # Legacy/synthetic caches may not carry target-derived descriptors.
            # Keep their bounded embedding fallback, while the real B200 Stage
            # 0 path writes ``source_memory`` so train and serving agree.
            embeddings = model.embedding(source)
            bank = SourceMemoryBank.from_source(
                source, embeddings=embeddings, chunk_size=chunk_size, top_r=top_r,
            )
        # A cache may intentionally omit target anchor features to save space.
        # In that case use the target-tied embedding of the last source token
        # as a deterministic retrieval query; never depend on the legacy
        # ``embeddings`` local from the uncached branch.
        query = (
            anchor_tensor[row]
            if anchor_tensor is not None
            else model.embedding(source[-1].reshape(1))[0]
        )
        memories.append(bank.retrieve(query, top_r=top_r).descriptors)
    if not memories:
        return None
    width = max(int(value.shape[-1]) for value in memories)
    count = max(1, max(int(value.shape[0]) for value in memories))
    padded = []
    for value in memories:
        value = value.to(device=device, dtype=torch.float32)
        if value.shape[-1] < width:
            value = torch.nn.functional.pad(value, (0, width - value.shape[-1]))
        if value.shape[0] < count:
            value = torch.nn.functional.pad(value, (0, 0, 0, count - value.shape[0]))
        padded.append(value)
    return torch.stack(padded)


class SyncSpecTrainer:
    """Minimal but real PyTorch trainers shared by CPU smoke and B200 runs."""

    def __init__(
        self, model, device: str | torch.device = "cpu", learning_rate: float = 1e-4,
        grad_accumulation_steps: int = 1, grad_clip_norm: float | None = 1.0,
        amp: bool = True, seed: int = 42, log_path: Path | str | None = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.learning_rate = float(learning_rate)
        self.grad_accumulation_steps = int(grad_accumulation_steps)
        if self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive")
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        if self.grad_clip_norm is not None and self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm must be non-negative or None")
        self.amp = bool(amp) and self.device.type == "cuda"
        self.seed = int(seed)
        self._anchor_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.completed_steps = 0
        self._pending_optimizer_state = None
        self._last_optimizer_state = None
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _clip(self, parameters) -> None:
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm)

    def _make_optimizer(self, parameters):
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)
        if self._pending_optimizer_state is not None:
            try:
                optimizer.load_state_dict(self._pending_optimizer_state)
            except (KeyError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    "checkpoint optimizer state is incompatible with this trainer"
                ) from exc
            self._pending_optimizer_state = None
        return optimizer

    def _remember_optimizer(self, optimizer, steps: int) -> None:
        self._last_optimizer_state = optimizer.state_dict()
        self.completed_steps += max(0, int(steps))

    def _record_step(
        self, phase: str, step: int, loss: torch.Tensor | float,
        started: float, tokens: int,
    ) -> None:
        if self.log_path is None:
            return
        elapsed = max(time.perf_counter() - float(started), 1e-9)
        value = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        row = {
            "phase": str(phase), "step": int(step), "loss": value,
            "step_time_s": elapsed, "tokens": int(tokens),
            "throughput_tokens_per_s": float(tokens) / elapsed,
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    @staticmethod
    def _batch_indices(
        total: int, batch_size: int | None, step: int,
        generator: torch.Generator | None = None, randomize: bool = False,
    ) -> list[int]:
        """Return row indices for one optimizer step.

        The default remains deterministic for survival and legacy callers.
        DFlash-style callers opt into a fresh random batch with the trainer's
        seeded CPU generator.
        """
        total = int(total)
        if total <= 0:
            raise ValueError("training dataset must contain at least one row")
        if randomize:
            size = total if batch_size is None or int(batch_size) <= 0 else min(
                int(batch_size), total,
            )
            return torch.randperm(total, generator=generator, device="cpu")[:size].tolist()
        if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= total:
            return list(range(total))
        size = int(batch_size)
        start = (int(step) * size) % total
        return [(start + offset) % total for offset in range(size)]

    def load_training_state(self, path: Path | str) -> None:
        """Load optimizer/step state for an already-loaded model checkpoint."""
        root = Path(path)
        state_path = root / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"trainer state not found: {state_path}")
        metadata = json.loads(state_path.read_text(encoding="utf-8"))
        self.completed_steps = int(metadata.get("completed_steps", 0))
        optimizer_path = root / "optimizer_state.pt"
        if optimizer_path.is_file():
            self._pending_optimizer_state = torch.load(
                optimizer_path, map_location=self.device, weights_only=True,
            )
        rng_path = root / "anchor_rng_state.pt"
        if rng_path.is_file():
            rng_state = torch.load(rng_path, map_location="cpu", weights_only=True)
            if not torch.is_tensor(rng_state):
                raise ValueError("anchor RNG checkpoint must contain a tensor state")
            self._anchor_generator.set_state(rng_state)

    def fit_diffusion(
        self, records: Iterable[TrajectoryRecord], kd: int, mask_token_id: int,
        steps: int = 100, position_weight: torch.Tensor | None = None,
        kl_weight: float = 0.0, rank_margin: float = 0.0,
        rank_weight: float = 0.0, rank_top_m: int = 16,
        batch_size: int | None = None, loss_decay_gamma: float | None = 7.0,
        num_anchors: int = 512,
    ) -> dict[str, float | int]:
        rows = list(records)
        if int(num_anchors) <= 0:
            raise ValueError("num_anchors must be positive")
        strict_rows = [
            (record, _expand_anchor_rows(
                [record], kd=kd, max_positions=self.model.config.max_positions,
            )) for record in rows
        ]
        eligible_records = [record for record, eligible in strict_rows if eligible]
        truncated_suffix_fallback = False
        if not eligible_records:
            # Preserve the useful CPU/tiny-model path when EOS truncates every
            # trajectory before K_d.  Full-length training never enters this
            # branch and remains strict like SpecForge.
            partial_rows = [
                (record, [
                    row for row in _expand_anchor_rows(
                        [record], kd=None, max_positions=self.model.config.max_positions,
                    ) if int(row[1]) < len(record.target_ids)
                ]) for record in rows
            ]
            eligible_records = [record for record, eligible in partial_rows if eligible]
            truncated_suffix_fallback = bool(eligible_records)
            eligible_anchor_count = sum(len(eligible) for _, eligible in partial_rows if eligible)
        else:
            eligible_anchor_count = sum(len(eligible) for _, eligible in strict_rows if eligible)
        if not eligible_records:
            raise ValueError(
                "diffusion training requires an anchor with enough drafter positional headroom"
            )
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError("batch_size must be positive or None")
        if loss_decay_gamma is not None and float(loss_decay_gamma) < 0.0:
            raise ValueError("loss_decay_gamma must be non-negative or None")
        if position_weight is None and loss_decay_gamma not in (None, 0.0):
            position_weight = dflash_position_weights(
                kd, gamma=float(loss_decay_gamma), device=self.device,
            )
        optimizer = self._make_optimizer(self.model.parameters())
        self.model.train()
        last = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step in range(int(steps)):
            step_started = time.perf_counter()
            record_indices = self._batch_indices(
                len(eligible_records), batch_size, step,
            )
            selected_rows = _sample_random_anchor_rows(
                [eligible_records[index] for index in record_indices],
                kd=kd, max_positions=self.model.config.max_positions,
                num_anchors=num_anchors, generator=self._anchor_generator,
                allow_truncated_suffix=truncated_suffix_fallback,
            )
            if not selected_rows:
                raise RuntimeError("DFlash anchor sampler produced an empty forward batch")
            batch_rows = [item[0] for item in selected_rows]
            anchor_indices = [item[1] for item in selected_rows]
            masked, targets, valid = build_stage1_batch(
                batch_rows, kd, mask_token_id, self.device, anchor_indices=anchor_indices,
            )
            anchor_tensor = _target_anchor_batch(
                batch_rows, anchor_indices, self.model.config.hidden_size, self.device,
            )
            recent_hidden = _target_recent_hidden_batch(
                batch_rows, anchor_indices, self.model.config.hidden_size, self.device,
            )
            source_memory = _source_memory_batch(
                self.model, batch_rows, anchor_tensor, self.device,
            )
            position_offsets = anchor_position_offsets(batch_rows, anchor_indices, self.device)
            teacher_logits = _target_logits_batch(
                batch_rows, anchor_indices, kd, self.model.config.vocab_size, self.device,
            ) if float(kl_weight) > 0.0 else None
            with self._autocast():
                output = self.model(
                    masked, target_anchor=anchor_tensor, source_memory=source_memory,
                    recent_hidden=recent_hidden, position_offset=position_offsets,
                )
                last_tensor = diffusion_loss(
                    output.logits[:, 1:], targets[:, 1:], valid[:, 1:], position_weight,
                    teacher_logits=teacher_logits, kl_weight=kl_weight,
                    rank_margin=rank_margin, rank_weight=rank_weight,
                    rank_top_m=rank_top_m,
                )
            (last_tensor / self.grad_accumulation_steps).backward()
            if (step + 1) % self.grad_accumulation_steps == 0 or step + 1 == int(steps):
                self._clip(self.model.parameters())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            last = float(last_tensor.detach().item())
            self._record_step(
                "diffusion", step + 1, last_tensor, step_started,
                int(valid[:, 1:].sum().item()),
            )
        self.model.eval()
        self._remember_optimizer(optimizer, steps)
        return {
            "stage": "diffusion", "steps": int(steps), "loss": last,
            "completed_steps": self.completed_steps,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "batch_size": (
                min(int(batch_size), len(eligible_records))
                if batch_size is not None else len(eligible_records)
            ),
            "anchor_count": eligible_anchor_count,
            "eligible_anchor_count": eligible_anchor_count,
            "num_anchors": int(num_anchors),
            "random_anchor_sampling": True,
            "truncated_suffix_fallback": truncated_suffix_fallback,
            "physical_block_size": int(kd) + 1,
            "anchor_slot_excluded": True,
            "loss_decay_gamma": (
                None if loss_decay_gamma is None else float(loss_decay_gamma)
            ),
            "recent_hidden_available": all(
                record.target_recent_hidden is not None for record in eligible_records
            ),
            "kl_weight": float(kl_weight), "rank_weight": float(rank_weight),
            "rank_margin": float(rank_margin), "rank_top_m": int(rank_top_m),
        }

    def fit_selector(
        self, candidate_logits: torch.Tensor, candidate_ids: torch.Tensor,
        target_ids: torch.Tensor, valid_mask: torch.Tensor | None = None,
        steps: int = 100, teacher_forcing: float = 1.0,
    ) -> dict[str, float | int]:
        parameter = torch.nn.Parameter(candidate_logits.detach().to(self.device).clone())
        optimizer = torch.optim.AdamW([parameter], lr=self.learning_rate)
        target_ids, candidate_ids = target_ids.to(self.device), candidate_ids.to(self.device)
        valid_mask = valid_mask.to(self.device) if valid_mask is not None else None
        last = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step in range(int(steps)):
            step_started = time.perf_counter()
            with self._autocast():
                loss = selector_loss(parameter, candidate_ids, target_ids, valid_mask, teacher_forcing)
            (loss / self.grad_accumulation_steps).backward()
            if (step + 1) % self.grad_accumulation_steps == 0 or step + 1 == int(steps):
                self._clip([parameter])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            last = float(loss.detach().item())
            self._record_step(
                "selector", step + 1, loss, step_started,
                int(target_ids.shape[0]),
            )
        return {
            "stage": "selector", "steps": int(steps), "loss": last,
            "grad_accumulation_steps": self.grad_accumulation_steps,
        }

    def fit_selector_module(
        self, selector, hidden: torch.Tensor, candidate_ids: torch.Tensor,
        candidate_logits: torch.Tensor, target_ids: torch.Tensor, source_index,
        history: list[int] | list[list[int]] | None = None,
        valid_mask: torch.Tensor | None = None,
        steps: int = 100, teacher_forcing_start: float = 1.0,
        teacher_forcing_end: float = 0.5,
        batch_size: int | None = None, randomize_batches: bool = False,
    ) -> dict[str, float | int]:
        """Train selector weights with the exact serving Top-M contract."""
        selector = selector.to(self.device)
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError("batch_size must be positive or None")
        if hidden.ndim == 2:
            if candidate_ids.ndim != 2 or candidate_logits.shape != candidate_ids.shape:
                raise ValueError("selector inputs must have matching [K,M] shapes")
            if target_ids.shape != hidden.shape[:1]:
                raise ValueError("selector target shape mismatch")
            hidden_rows = [hidden.detach()]
            candidate_id_rows = [candidate_ids.detach()]
            candidate_logit_rows = [candidate_logits.detach()]
            target_rows = [target_ids.detach()]
            index_rows = [source_index]
            history_rows = [history or []]
            mask_rows = [valid_mask.detach().bool() if valid_mask is not None else None]
            if mask_rows[0] is not None and mask_rows[0].shape != target_ids.shape:
                raise ValueError("selector valid mask shape mismatch")
        elif hidden.ndim == 3:
            if candidate_ids.ndim != 3 or candidate_logits.ndim != 3 or target_ids.ndim != 2:
                raise ValueError("batched selector inputs must be [B,K,D], [B,K,M], and [B,K]")
            batch = hidden.shape[0]
            if candidate_ids.shape[:2] != hidden.shape[:2] or candidate_logits.shape != candidate_ids.shape:
                raise ValueError("batched selector lattice shape mismatch")
            if target_ids.shape != hidden.shape[:2]:
                raise ValueError("batched selector target shape mismatch")
            if isinstance(source_index, (list, tuple)):
                if len(source_index) != batch:
                    raise ValueError("one source index is required per selector lattice")
                index_rows = list(source_index)
            else:
                index_rows = [source_index] * batch
            if not history:
                history_rows = [[] for _ in range(batch)]
            elif history and isinstance(history[0], int):
                history_rows = [list(history)] * batch
            else:
                history_rows = list(history)
                if len(history_rows) != batch:
                    raise ValueError("one history is required per selector lattice")
            hidden_rows = [row.detach() for row in hidden.unbind(0)]
            candidate_id_rows = [row.detach() for row in candidate_ids.unbind(0)]
            candidate_logit_rows = [row.detach() for row in candidate_logits.unbind(0)]
            target_rows = [row.detach() for row in target_ids.unbind(0)]
            if valid_mask is None:
                mask_rows = [None] * batch
            else:
                valid_mask = valid_mask.detach().bool()
                if valid_mask.shape != target_ids.shape:
                    raise ValueError("batched selector valid mask shape mismatch")
                mask_rows = list(valid_mask.unbind(0))
        else:
            raise ValueError("hidden must be [K,D] or [B,K,D]")
        if len(hidden_rows) == 0:
            raise ValueError("selector training requires at least one lattice")
        optimizer = torch.optim.AdamW(selector.parameters(), lr=self.learning_rate)
        last = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step in range(int(steps)):
            step_started = time.perf_counter()
            progress = step / max(1, int(steps) - 1)
            teacher_forcing = float(teacher_forcing_start) + progress * (
                float(teacher_forcing_end) - float(teacher_forcing_start)
            )
            with self._autocast():
                numerator = None
                denominator = 0.0
                indices = self._batch_indices(
                    len(hidden_rows), batch_size, step,
                    generator=self._anchor_generator,
                    randomize=bool(randomize_batches),
                )
                for index in indices:
                    hidden_row = hidden_rows[index].to(self.device)
                    ids_row = candidate_id_rows[index].to(self.device)
                    logits_row = candidate_logit_rows[index].to(self.device)
                    targets_row = target_rows[index].to(self.device)
                    index_row = index_rows[index]
                    history_row = history_rows[index]
                    mask_row = mask_rows[index]
                    if mask_row is not None:
                        mask_row = mask_row.to(self.device)
                    output = selector.select(
                        hidden_row, ids_row, logits_row, history_row, index_row,
                        target_ids=targets_row, teacher_forcing=teacher_forcing,
                    )
                    matches = output.candidate_ids.eq(targets_row.unsqueeze(-1))
                    available = matches.any(dim=-1)
                    if mask_row is not None:
                        available = available & mask_row
                    target_index = matches.to(output.q.dtype).argmax(dim=-1)
                    token_prob = output.q.gather(-1, target_index.unsqueeze(-1)).squeeze(-1)
                    loss_values = -token_prob.clamp_min(1e-8).log()
                    weight = available.to(loss_values.dtype)
                    term = (loss_values * weight).sum()
                    numerator = term if numerator is None else numerator + term
                    denominator += float(weight.sum().item())
                loss = numerator / max(1.0, denominator)
            (loss / self.grad_accumulation_steps).backward()
            if (step + 1) % self.grad_accumulation_steps == 0 or step + 1 == int(steps):
                self._clip(selector.parameters())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            last = float(loss.detach().item())
            self._record_step(
                "selector", step + 1, loss, step_started,
                sum(int(hidden_rows[index].shape[0]) for index in indices),
            )
        selector.eval()
        return {
            "stage": "selector", "steps": int(steps), "loss": last,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "batch_size": (
                min(int(batch_size), len(hidden_rows))
                if batch_size is not None else len(hidden_rows)
            ),
            "randomize_batches": bool(randomize_batches),
        }

    def fit_survival(
        self, head, features: torch.Tensor, labels: torch.Tensor, steps: int = 100,
        batch_size: int | None = None,
    ) -> dict[str, float | int]:
        head = head.to(self.device)
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError("batch_size must be positive or None")
        if features.ndim < 2 or labels.shape != features.shape[:1]:
            raise ValueError("survival features must be [N,F] and labels [N]")
        features = features.detach()
        labels = labels.detach()
        if features.shape[0] == 0:
            raise ValueError("survival training requires at least one row")
        optimizer = torch.optim.AdamW(head.parameters(), lr=self.learning_rate)
        last = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step in range(int(steps)):
            step_started = time.perf_counter()
            indices = self._batch_indices(features.shape[0], batch_size, step)
            batch_indices = torch.tensor(indices, dtype=torch.long, device=features.device)
            batch_features = features.index_select(0, batch_indices).to(self.device)
            batch_labels = labels.index_select(0, batch_indices).to(self.device)
            with self._autocast():
                loss = survival_loss(head(batch_features), batch_labels)
            (loss / self.grad_accumulation_steps).backward()
            if (step + 1) % self.grad_accumulation_steps == 0 or step + 1 == int(steps):
                self._clip(head.parameters())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            last = float(loss.detach().item())
            self._record_step(
                "survival", step + 1, loss, step_started, len(indices),
            )
        return {
            "stage": "survival", "steps": int(steps), "loss": last,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "batch_size": (
                min(int(batch_size), features.shape[0])
                if batch_size is not None else features.shape[0]
            ),
        }

    def fit_joint(
        self, records: Iterable[TrajectoryRecord], kd: int, mask_token_id: int,
        selector, survival_head=None, survival_features: torch.Tensor | None = None,
        survival_labels: torch.Tensor | None = None, steps: int = 100,
        selector_weight: float = 0.1, survival_weight: float = 0.1,
        batch_size: int | None = None, loss_decay_gamma: float | None = 7.0,
        num_anchors: int = 512,
    ) -> dict[str, float | int]:
        """Optional low-LR joint refinement after the staged warm start.

        Candidate IDs remain the serving Top-M lattice. The diffusion and
        selector terms are differentiable through the drafter hidden/logit
        values; candidate misses are masked. Survival contributes only when
        caller-provided on-policy features/labels are available.
        """
        rows = list(records)
        if int(num_anchors) <= 0:
            raise ValueError("num_anchors must be positive")
        strict_rows = [
            (record, _expand_anchor_rows(
                [record], kd=kd, max_positions=self.model.config.max_positions,
            )) for record in rows
        ]
        eligible_records = [record for record, eligible in strict_rows if eligible]
        truncated_suffix_fallback = False
        if not eligible_records:
            partial_rows = [
                (record, [
                    row for row in _expand_anchor_rows(
                        [record], kd=None, max_positions=self.model.config.max_positions,
                    ) if int(row[1]) < len(record.target_ids)
                ]) for record in rows
            ]
            eligible_records = [record for record, eligible in partial_rows if eligible]
            truncated_suffix_fallback = bool(eligible_records)
            eligible_anchor_count = sum(len(eligible) for _, eligible in partial_rows if eligible)
        else:
            eligible_anchor_count = sum(len(eligible) for _, eligible in strict_rows if eligible)
        if not eligible_records:
            raise ValueError(
                "joint training requires an anchor with enough drafter positional headroom"
            )
        if batch_size is not None and int(batch_size) <= 0:
            raise ValueError("batch_size must be positive or None")
        if loss_decay_gamma is not None and float(loss_decay_gamma) < 0.0:
            raise ValueError("loss_decay_gamma must be non-negative or None")
        position_weight = None
        if loss_decay_gamma not in (None, 0.0):
            position_weight = dflash_position_weights(
                kd, gamma=float(loss_decay_gamma), device=self.device,
            )
        selector = selector.to(self.device)
        if survival_head is not None:
            survival_head = survival_head.to(self.device)
        if (survival_features is None) != (survival_labels is None):
            raise ValueError("survival features and labels must be provided together")
        if survival_features is not None and survival_features.shape[0] != survival_labels.shape[0]:
            raise ValueError("survival features and labels must have equal length")
        if survival_features is not None and survival_features.shape[0] == 0:
            raise ValueError("survival training requires at least one row")
        if survival_features is not None:
            survival_features = survival_features.detach()
            survival_labels = survival_labels.detach()
        parameters = [
            parameter for module in (self.model, selector, survival_head)
            if module is not None for parameter in module.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("joint fine-tuning requires trainable parameters")
        optimizer = self._make_optimizer(parameters)
        self.model.train()
        selector.train()
        if survival_head is not None:
            survival_head.train()
        last_total = last_diffusion = last_selector = last_survival = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step in range(int(steps)):
            step_started = time.perf_counter()
            record_indices = self._batch_indices(
                len(eligible_records), batch_size, step,
            )
            selected_rows = _sample_random_anchor_rows(
                [eligible_records[index] for index in record_indices],
                kd=kd, max_positions=self.model.config.max_positions,
                num_anchors=num_anchors, generator=self._anchor_generator,
                allow_truncated_suffix=truncated_suffix_fallback,
            )
            if not selected_rows:
                raise RuntimeError("DFlash anchor sampler produced an empty forward batch")
            batch_rows = [item[0] for item in selected_rows]
            batch_anchor_indices = [item[1] for item in selected_rows]
            batch_source_indices = [
                SourceNgramIndex(record.source_ids) for record in batch_rows
            ]
            batch_histories = [
                record.source_ids + record.target_ids[: int(anchor)]
                for record, anchor in selected_rows
            ]
            masked, targets, valid = build_stage1_batch(
                batch_rows, kd, mask_token_id, self.device,
                anchor_indices=batch_anchor_indices,
            )
            anchor_tensor = _target_anchor_batch(
                batch_rows, batch_anchor_indices, self.model.config.hidden_size, self.device,
            )
            recent_hidden = _target_recent_hidden_batch(
                batch_rows, batch_anchor_indices, self.model.config.hidden_size, self.device,
            )
            source_memory = _source_memory_batch(
                self.model, batch_rows, anchor_tensor, self.device,
            )
            position_offsets = anchor_position_offsets(
                batch_rows, batch_anchor_indices, self.device,
            )
            with self._autocast():
                output = self.model(
                    masked, target_anchor=anchor_tensor, source_memory=source_memory,
                    recent_hidden=recent_hidden, position_offset=position_offsets,
                )
                future_logits = output.logits[:, 1:]
                future_hidden = output.hidden[:, 1:]
                future_targets = targets[:, 1:]
                future_valid = valid[:, 1:]
                diffusion = diffusion_loss(
                    future_logits, future_targets, future_valid, position_weight,
                )
                candidate_ids, candidate_logits = top_m_candidates(
                    future_logits, self.model.config.top_m,
                )
                numerator = None
                denominator = 0.0
                for local_row, (hidden_row, ids_row, logits_row, targets_row, valid_row) in enumerate(zip(
                    future_hidden, candidate_ids, candidate_logits,
                    future_targets, future_valid,
                )):
                    selected = selector.select(
                        hidden_row, ids_row, logits_row, batch_histories[local_row],
                        batch_source_indices[local_row],
                        target_ids=targets_row, teacher_forcing=0.5,
                    )
                    matches = selected.candidate_ids.eq(targets_row.unsqueeze(-1))
                    available = matches.any(dim=-1) & valid_row.bool()
                    target_index = matches.to(selected.q.dtype).argmax(dim=-1)
                    token_prob = selected.q.gather(-1, target_index.unsqueeze(-1)).squeeze(-1)
                    values = -token_prob.clamp_min(1e-8).log()
                    weight = available.to(values.dtype)
                    term = (values * weight).sum()
                    numerator = term if numerator is None else numerator + term
                    denominator += float(weight.sum().item())
                selector_loss_value = numerator / max(1.0, denominator)
                total = diffusion + float(selector_weight) * selector_loss_value
                if survival_head is not None and survival_features is not None:
                    survival_indices = self._batch_indices(
                        survival_features.shape[0], batch_size, step,
                    )
                    survival_index_tensor = torch.tensor(
                        survival_indices, dtype=torch.long, device=survival_features.device,
                    )
                    survival_loss_value = survival_loss(
                        survival_head(
                            survival_features.index_select(0, survival_index_tensor).to(self.device)
                        ),
                        survival_labels.index_select(0, survival_index_tensor).to(self.device),
                    )
                    total = total + float(survival_weight) * survival_loss_value
                else:
                    survival_loss_value = total.new_zeros(())
            (total / self.grad_accumulation_steps).backward()
            if (step + 1) % self.grad_accumulation_steps == 0 or step + 1 == int(steps):
                self._clip(parameters)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            last_total = float(total.detach().item())
            last_diffusion = float(diffusion.detach().item())
            last_selector = float(selector_loss_value.detach().item())
            last_survival = float(survival_loss_value.detach().item())
            self._record_step(
                "joint", step + 1, total, step_started,
                int(future_valid.sum().item()),
            )
        self.model.eval()
        selector.eval()
        if survival_head is not None:
            survival_head.eval()
        self._remember_optimizer(optimizer, steps)
        return {
            "stage": "joint_finetune", "steps": int(steps), "loss": last_total,
            "completed_steps": self.completed_steps,
            "diffusion_loss": last_diffusion, "selector_loss": last_selector,
            "survival_loss": last_survival,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "batch_size": (
                min(int(batch_size), len(eligible_records))
                if batch_size is not None else len(eligible_records)
            ),
            "anchor_count": eligible_anchor_count,
            "eligible_anchor_count": eligible_anchor_count,
            "num_anchors": int(num_anchors),
            "random_anchor_sampling": True,
            "truncated_suffix_fallback": truncated_suffix_fallback,
            "physical_block_size": int(kd) + 1,
            "anchor_slot_excluded": True,
            "loss_decay_gamma": (
                None if loss_decay_gamma is None else float(loss_decay_gamma)
            ),
        }

    def save_checkpoint(self, path: Path | str) -> None:
        self.model.save_pretrained(
            path, omit_tied_weights=bool(getattr(self.model, "_tied_embedding", False))
        )
        state_path = Path(path) / "trainer_state.json"
        state_path.write_text(
            json.dumps({
                "learning_rate": self.learning_rate, "device": str(self.device),
                "seed": self.seed,
                "completed_steps": self.completed_steps,
                "grad_accumulation_steps": self.grad_accumulation_steps,
                "grad_clip_norm": self.grad_clip_norm, "amp": self.amp,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        if self._last_optimizer_state is not None:
            torch.save(self._last_optimizer_state, Path(path) / "optimizer_state.pt")
        torch.save(self._anchor_generator.get_state(), Path(path) / "anchor_rng_state.pt")
