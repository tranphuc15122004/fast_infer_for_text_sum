"""Dataset tokenized nhẹ cho online target-feature extraction.

Feature hidden states không được ghi ra disk ở mode này. Các shard chỉ chứa
``input_ids``, ``loss_mask`` và độ dài; target frozen sẽ tạo feature trong
trainer ngay trước khi gọi DFlash objective.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


TOKENIZED_SCHEMA_VERSION = "mr_dflash_tokenized_v1"


def _read_manifest(path: Path) -> Dict[str, Any]:
    manifest = path / "manifest.json" if path.is_dir() else path
    with manifest.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"tokenized manifest phải là object: {manifest}")
    if value.get("schema_version") != TOKENIZED_SCHEMA_VERSION:
        raise ValueError(
            "tokenized manifest không tương thích: "
            f"{value.get('schema_version')!r} != {TOKENIZED_SCHEMA_VERSION!r}"
        )
    return value


class TokenizedDFlashDataset:
    """Lazy đọc các shard ``.pt`` sinh bởi ``tokenize_dataset.py``.

    Mỗi shard có dạng ``{"samples": [{"input_ids": Tensor,
    "loss_mask": Tensor, "length": int, "id": str}, ...]}``. Chỉ một số
    shard gần nhất được giữ trong cache process để tránh mở toàn bộ dataset.
    """

    def __init__(
        self,
        path: str,
        *,
        sample_limit: Optional[int] = None,
        pad_token_id: int = 0,
        shard_cache_size: int = 2,
        expected_target_model: Optional[str] = None,
        expected_feature_layer_ids: Optional[Sequence[int]] = None,
        expected_max_length: Optional[int] = None,
        expected_supervision_mode: Optional[str] = None,
    ) -> None:
        root = Path(path)
        self.manifest = _read_manifest(root)
        if expected_target_model is not None and self.manifest.get("target_model") != expected_target_model:
            raise ValueError(
                "tokenized target_model không khớp: "
                f"{self.manifest.get('target_model')!r} != {expected_target_model!r}"
            )
        if expected_feature_layer_ids is not None:
            actual_layers = [int(x) for x in self.manifest.get("feature_layer_ids", [])]
            expected_layers = [int(x) for x in expected_feature_layer_ids]
            if actual_layers != expected_layers:
                raise ValueError(
                    f"tokenized feature_layer_ids không khớp: {actual_layers} != {expected_layers}"
                )
        if expected_max_length is not None and int(self.manifest.get("max_length", -1)) != int(expected_max_length):
            raise ValueError("tokenized max_length không khớp config")
        if expected_supervision_mode is not None and self.manifest.get("supervision_mode") != expected_supervision_mode:
            raise ValueError("tokenized supervision_mode không khớp config")
        raw_shards = self.manifest.get("shards", [])
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError(f"manifest không có shards: {root}")
        self.root = root if root.is_dir() else root.parent
        self.pad_token_id = int(pad_token_id)
        self._cache: OrderedDict[int, List[Dict[str, Any]]] = OrderedDict()
        self._cache_size = max(1, int(shard_cache_size))
        self._shards: List[Dict[str, Any]] = []
        self._cumulative: List[int] = []
        total = 0
        for item in raw_shards:
            if not isinstance(item, dict) or "path" not in item:
                raise ValueError("mỗi shard trong manifest cần path")
            count = int(item.get("count", 0))
            if count < 0:
                raise ValueError("shard count không được âm")
            shard = dict(item)
            shard["count"] = count
            self._shards.append(shard)
            total += count
            self._cumulative.append(total)
        if sample_limit is not None:
            total = min(total, int(sample_limit))
        self._length = total
        self.lengths: List[int] = []
        for index in range(self._length):
            self.lengths.append(int(self._get_raw(index).get("length", 0)))

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_id = 0
        while index >= self._cumulative[shard_id]:
            shard_id += 1
        previous = 0 if shard_id == 0 else self._cumulative[shard_id - 1]
        return shard_id, index - previous

    def _load_shard(self, shard_id: int) -> List[Dict[str, Any]]:
        cached = self._cache.get(shard_id)
        if cached is not None:
            self._cache.move_to_end(shard_id)
            return cached
        path = self.root / str(self._shards[shard_id]["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise ValueError(f"shard không có list samples: {path}")
        self._cache[shard_id] = samples
        self._cache.move_to_end(shard_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return samples

    def _get_raw(self, index: int) -> Dict[str, Any]:
        shard_id, local_index = self._locate(index)
        samples = self._load_shard(shard_id)
        return samples[local_index]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = self._get_raw(index)
        input_ids = torch.as_tensor(raw["input_ids"], dtype=torch.long).flatten()
        loss_mask = torch.as_tensor(raw["loss_mask"], dtype=torch.float32).flatten()
        if input_ids.numel() != loss_mask.numel():
            raise ValueError(f"sample {raw.get('id', index)!r} lệch input/loss mask")
        return {
            "id": str(raw.get("id", index)),
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            "length": int(input_ids.numel()),
        }

    def collate(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("không thể collate batch rỗng")
        max_len = max(int(item["input_ids"].numel()) for item in features)
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        ids: List[str] = []
        lengths: List[int] = []
        for row, item in enumerate(features):
            length = int(item["input_ids"].numel())
            input_ids[row, :length] = item["input_ids"]
            loss_mask[row, :length] = item["loss_mask"]
            attention_mask[row, :length] = 1
            ids.append(str(item.get("id", row)))
            lengths.append(length)
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "ids": ids,
        }


def write_tokenized_manifest(
    output_dir: str | Path,
    *,
    shards: Iterable[Dict[str, Any]],
    num_samples: int,
    target_model: str,
    feature_layer_ids: Sequence[int],
    chat_template: str,
    max_length: int,
    supervision_mode: str,
    tokenizer_name: Optional[str] = None,
    target_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Ghi manifest provenance và trả payload đã ghi."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TOKENIZED_SCHEMA_VERSION,
        "target_model": target_model,
        "feature_layer_ids": [int(x) for x in feature_layer_ids],
        "chat_template": chat_template,
        "max_length": int(max_length),
        "supervision_mode": supervision_mode,
        "tokenizer_name": tokenizer_name,
        "target_revision": target_revision,
        "num_samples": int(num_samples),
        "shards": list(shards),
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


__all__ = [
    "TOKENIZED_SCHEMA_VERSION",
    "TokenizedDFlashDataset",
    "write_tokenized_manifest",
]
