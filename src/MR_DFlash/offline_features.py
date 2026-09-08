"""Feature store dạng shard cho target cache offline.

Cache legacy của project lưu một ``.ckpt`` cho mỗi mẫu. Cách đó phù hợp cho
smoke test nhưng tạo quá nhiều inode và không thuận tiện khi cache hàng chục
nghìn mẫu. Module này lưu nhiều mẫu biến độ dài trong một shard ``.pt`` và
ghi manifest có provenance. Hidden states vẫn được giữ tại *mọi offset hợp
lệ* của sample; chỉ phần padding không được lưu.

Schema sample trong shard:

``{"id", "input_ids", "loss_mask", "hidden_states", "length"}``

Các tensor được lưu trên CPU. Dataset đọc lazy theo shard và có LRU cache để
không nạp toàn bộ feature store vào RAM.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .data import (
    build_dflash_collator,
    load_feature_manifest,
    normalize_offline_sample,
    validate_feature_manifest,
)


SHARDED_FEATURE_SCHEMA_VERSION = "mr_dflash_feature_sharded_v1"
SHARDED_FEATURE_MANIFEST_FILENAME = "manifest.json"


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cpu_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa sample trước khi ghi, không giữ tensor GPU trong cache."""
    required = ("id", "input_ids", "loss_mask", "hidden_states")
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"feature cache sample thiếu key: {missing}")
    input_ids = torch.as_tensor(sample["input_ids"], dtype=torch.long).flatten()
    loss_mask = torch.as_tensor(sample["loss_mask"], dtype=torch.float32).flatten()
    hidden = torch.as_tensor(sample["hidden_states"]).detach().cpu()
    if hidden.dim() == 3 and hidden.shape[0] == 1:
        hidden = hidden.squeeze(0)
    if hidden.dim() != 2:
        raise ValueError(
            "hidden_states cache phải có dạng [seq, feature_width], "
            f"got {tuple(hidden.shape)}"
        )
    if input_ids.numel() != loss_mask.numel() or input_ids.numel() != hidden.shape[0]:
        raise ValueError(
            f"sample {sample.get('id')!r} lệch sequence: "
            f"input={input_ids.numel()}, mask={loss_mask.numel()}, "
            f"hidden={hidden.shape[0]}"
        )
    if input_ids.numel() < 1:
        raise ValueError(f"sample {sample.get('id')!r} rỗng")
    return {
        "id": str(sample["id"]),
        "input_ids": input_ids.cpu(),
        "loss_mask": loss_mask.cpu(),
        "hidden_states": hidden.cpu(),
        "length": int(input_ids.numel()),
    }


def write_feature_shard(path: str | Path, samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Ghi một shard atomically và trả metadata của shard."""
    target = Path(path)
    prepared = [_cpu_sample(sample) for sample in samples]
    if not prepared:
        raise ValueError("không thể ghi feature shard rỗng")
    ids = [sample["id"] for sample in prepared]
    if len(ids) != len(set(ids)):
        raise ValueError("sample id bị trùng trong cùng feature shard")
    temporary = target.with_name(f".{target.name}.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": prepared}, temporary)
    os.replace(temporary, target)
    return {
        "path": target.name,
        "count": len(prepared),
        "ids": ids,
        "lengths": [sample["length"] for sample in prepared],
    }


def write_sharded_feature_manifest(
    output_dir: str | Path,
    *,
    target_model_path: str,
    feature_layer_ids: Sequence[int],
    hidden_size: int,
    feature_width: int,
    max_length: int,
    requested_torch_dtype: str,
    shards: Sequence[Dict[str, Any]],
    sample_ids: Sequence[str],
    source_data_path: Optional[str] = None,
    target_revision: Optional[str] = None,
    stored_feature_dtype: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ghi manifest schema ``mr_dflash_feature_sharded_v1``."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema_version": SHARDED_FEATURE_SCHEMA_VERSION,
        "cache_type": "target_hidden_states",
        "target_model_path": target_model_path,
        "target_revision": target_revision,
        "feature_layer_ids": [int(x) for x in feature_layer_ids],
        "hidden_size": int(hidden_size),
        "feature_width": int(feature_width),
        "requested_torch_dtype": requested_torch_dtype,
        "stored_feature_dtype": stored_feature_dtype,
        "max_length": int(max_length),
        "num_samples": len(sample_ids),
        "sample_ids": [str(x) for x in sample_ids],
        "shards": [dict(item) for item in shards],
        "source_data_path": source_data_path,
        "stats": dict(stats or {}),
    }
    _atomic_json_write(root / SHARDED_FEATURE_MANIFEST_FILENAME, payload)
    return payload


def is_sharded_feature_store(path: str | Path) -> bool:
    """Trả về True khi ``path`` là cache shard đã có manifest đúng schema."""
    root = Path(path)
    manifest = root if root.is_file() else root / SHARDED_FEATURE_MANIFEST_FILENAME
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("schema_version") == SHARDED_FEATURE_SCHEMA_VERSION


def sharded_feature_store_ready(path: str | Path) -> bool:
    """Kiểm tra manifest và toàn bộ shard được khai báo đều tồn tại."""
    root = Path(path)
    if not is_sharded_feature_store(root):
        return False
    try:
        manifest = load_feature_manifest(str(root))
        if root.is_file():
            root = root.parent
        shards = manifest.get("shards", [])
        if not isinstance(shards, list) or not shards:
            return False
        return all((root / str(item["path"])).is_file() for item in shards)
    except (KeyError, OSError, TypeError, ValueError):
        return False


class ShardedFeatureWriter:
    """Incremental/resumable writer cho cache target hidden states.

    ``resume=True`` dùng ``sample_ids`` trong manifest làm index nhẹ; không
    cần load các tensor hidden khổng lồ để biết mẫu nào đã hoàn tất. Một cache
    chưa có manifest nhưng có shard cũ sẽ được phục hồi bằng cách đọc shard
    một lần.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        shard_size: int,
        target_model_path: str,
        feature_layer_ids: Sequence[int],
        hidden_size: int,
        feature_width: int,
        max_length: int,
        requested_torch_dtype: str,
        source_data_path: Optional[str] = None,
        target_revision: Optional[str] = None,
        resume: bool = False,
    ) -> None:
        if int(shard_size) < 1:
            raise ValueError("shard_size phải >= 1")
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.metadata = {
            "target_model_path": target_model_path,
            "feature_layer_ids": [int(x) for x in feature_layer_ids],
            "hidden_size": int(hidden_size),
            "feature_width": int(feature_width),
            "max_length": int(max_length),
            "requested_torch_dtype": requested_torch_dtype,
            "source_data_path": source_data_path,
            "target_revision": target_revision,
        }
        manifest_path = self.root / SHARDED_FEATURE_MANIFEST_FILENAME
        existing_pt = sorted(self.root.glob("shard_*.pt"))
        if not resume and (manifest_path.exists() or existing_pt):
            raise FileExistsError(
                f"feature cache đã tồn tại: {self.root}; dùng --resume hoặc thư mục mới"
            )
        self.shards: List[Dict[str, Any]] = []
        self.sample_ids: List[str] = []
        manifest: Optional[Dict[str, Any]] = None
        if resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SHARDED_FEATURE_SCHEMA_VERSION:
                raise ValueError("manifest cache không phải sharded feature schema")
            for key in (
                "target_model_path",
                "feature_layer_ids",
                "hidden_size",
                "feature_width",
                "max_length",
            ):
                if manifest.get(key) != self.metadata[key]:
                    raise ValueError(
                        f"metadata cache không khớp ở {key}: "
                        f"{manifest.get(key)!r} != {self.metadata[key]!r}"
                    )
            self.shards = [dict(item) for item in manifest.get("shards", [])]
            self.sample_ids = [str(x) for x in manifest.get("sample_ids", [])]
            missing_shards = [
                str(item.get("path"))
                for item in self.shards
                if not (self.root / str(item.get("path"))).is_file()
            ]
            if missing_shards:
                raise FileNotFoundError(
                    "feature cache thiếu shard khi resume: "
                    + ", ".join(missing_shards[:3])
                )
            # flush() ghi shard trước rồi mới ghi manifest. Nếu tiến trình bị
            # dừng giữa hai thao tác, phục hồi shard mồ côi thay vì ghi đè nó
            # ở lần --resume kế tiếp.
            listed_names = {str(item.get("path")) for item in self.shards}
            for shard_path in existing_pt:
                if shard_path.name in listed_names:
                    continue
                payload = torch.load(shard_path, map_location="cpu", weights_only=False)
                samples = payload.get("samples") if isinstance(payload, dict) else payload
                if not isinstance(samples, list):
                    raise ValueError(f"shard không có list samples: {shard_path}")
                ids = [str(item["id"]) for item in samples]
                if set(ids) & set(self.sample_ids):
                    raise ValueError(f"shard mồ côi chứa sample id đã có: {shard_path}")
                lengths = [int(item.get("length", len(item["input_ids"]))) for item in samples]
                self.shards.append({"path": shard_path.name, "count": len(ids), "ids": ids, "lengths": lengths})
                self.sample_ids.extend(ids)
        elif resume and existing_pt:
            # Recovery path cho cache bị dừng trước lần ghi manifest đầu tiên.
            for shard_path in existing_pt:
                payload = torch.load(shard_path, map_location="cpu", weights_only=False)
                samples = payload.get("samples") if isinstance(payload, dict) else payload
                if not isinstance(samples, list):
                    raise ValueError(f"shard không có list samples: {shard_path}")
                ids = [str(item["id"]) for item in samples]
                lengths = [int(item.get("length", len(item["input_ids"]))) for item in samples]
                self.shards.append({"path": shard_path.name, "count": len(ids), "ids": ids, "lengths": lengths})
                self.sample_ids.extend(ids)
        self.existing_ids = set(self.sample_ids)
        self._pending: List[Dict[str, Any]] = []
        self._pending_ids: set[str] = set()
        self._stored_dtype: Optional[str] = (
            str(manifest.get("stored_feature_dtype"))
            if manifest is not None and manifest.get("stored_feature_dtype")
            else None
        )
        self.stats: Dict[str, Any] = {}

    @property
    def total_samples(self) -> int:
        """Số sample đã ghi hoặc đang chờ flush."""
        return len(self.sample_ids) + len(self._pending)

    def add(self, sample: Dict[str, Any]) -> bool:
        """Thêm sample; trả False nếu id đã có khi resume."""
        prepared = _cpu_sample(sample)
        sample_id = prepared["id"]
        if sample_id in self.existing_ids or sample_id in self._pending_ids:
            return False
        if prepared["hidden_states"].shape[-1] != self.metadata["feature_width"]:
            raise ValueError(
                f"sample {sample_id!r} có feature_width="
                f"{prepared['hidden_states'].shape[-1]}, "
                f"kỳ vọng {self.metadata['feature_width']}"
            )
        self._pending.append(prepared)
        self._pending_ids.add(sample_id)
        if self._stored_dtype is None:
            self._stored_dtype = str(prepared["hidden_states"].dtype).replace("torch.", "")
        if len(self._pending) >= self.shard_size:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._pending:
            return
        name = f"shard_{len(self.shards):05d}.pt"
        descriptor = write_feature_shard(self.root / name, self._pending)
        self.shards.append(descriptor)
        ids = [str(item["id"]) for item in self._pending]
        self.sample_ids.extend(ids)
        self.existing_ids.update(ids)
        self._pending = []
        self._pending_ids.clear()
        self._write_manifest()

    def _write_manifest(self) -> Dict[str, Any]:
        return write_sharded_feature_manifest(
            self.root,
            **self.metadata,
            stored_feature_dtype=self._stored_dtype,
            shards=self.shards,
            sample_ids=self.sample_ids,
            stats=self.stats,
        )

    def close(self, *, stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if stats is not None:
            self.stats = dict(stats)
        self.flush()
        return self._write_manifest()


class ShardedDFlashFeatureDataset:
    """Dataset lazy-load theo shard, tương thích DFlashFeatureDataset."""

    def __init__(
        self,
        hidden_states_path: str,
        *,
        max_len: int = 3072,
        shard_cache_size: int = 2,
        sample_limit: Optional[int] = None,
        expected_feature_width: Optional[int] = None,
        expected_feature_layer_ids: Optional[Sequence[int]] = None,
        expected_target_model_path: Optional[str] = None,
        expected_max_length: Optional[int] = None,
    ) -> None:
        root = Path(hidden_states_path)
        self.root = root if root.is_dir() else root.parent
        self.manifest = load_feature_manifest(str(root))
        if self.manifest.get("schema_version") != SHARDED_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "feature manifest không phải sharded schema: "
                f"{self.manifest.get('schema_version')!r}"
            )
        validate_feature_manifest(
            {
                **self.manifest,
                # validator chung kiểm schema legacy; đưa về schema sharded
                # sau khi kiểm thủ công ở trên.
                "schema_version": "mr_dflash_feature_v1",
            },
            expected_feature_width=expected_feature_width,
            expected_feature_layer_ids=expected_feature_layer_ids,
            expected_target_model_path=expected_target_model_path,
        )
        if expected_max_length is not None and int(self.manifest.get("max_length", -1)) != int(expected_max_length):
            raise ValueError(
                "feature cache max_length không khớp model: "
                f"{self.manifest.get('max_length')} != {expected_max_length}"
            )
        raw_shards = self.manifest.get("shards", [])
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError(f"manifest không có shards: {root}")
        self.max_len = int(max_len)
        self.expected_feature_width = expected_feature_width
        self._cache: OrderedDict[int, List[Dict[str, Any]]] = OrderedDict()
        self._cache_size = max(1, int(shard_cache_size))
        self._shards = [dict(item) for item in raw_shards]
        self._cumulative: List[int] = []
        total = 0
        self.lengths: List[int] = []
        for shard in self._shards:
            count = int(shard.get("count", 0))
            if count < 0:
                raise ValueError("shard count không được âm")
            shard_lengths = shard.get("lengths")
            if isinstance(shard_lengths, list) and len(shard_lengths) == count:
                self.lengths.extend(int(x) for x in shard_lengths)
            else:
                self.lengths.extend(self._load_shard_lengths(len(self._cumulative)))
            total += count
            self._cumulative.append(total)
        if len(self.lengths) != total:
            raise ValueError("manifest shard lengths không khớp count")
        if sample_limit is not None:
            total = min(total, int(sample_limit))
            self.lengths = self.lengths[:total]
        self._length = total

    def _load_shard_lengths(self, shard_id: int) -> List[int]:
        path = self.root / str(self._shards[shard_id]["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise ValueError(f"shard không có list samples: {path}")
        return [int(item.get("length", len(item["input_ids"]))) for item in samples]

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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        shard_id, local_index = self._locate(index)
        raw = self._load_shard(shard_id)[local_index]
        return normalize_offline_sample(
            {
                "input_ids": torch.as_tensor(raw["input_ids"]),
                "loss_mask": torch.as_tensor(raw["loss_mask"]),
                "hidden_states": torch.as_tensor(raw["hidden_states"]),
            },
            self.max_len,
            expected_feature_width=self.expected_feature_width,
        )

    def collate(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return build_dflash_collator()(features)


__all__ = [
    "SHARDED_FEATURE_SCHEMA_VERSION",
    "SHARDED_FEATURE_MANIFEST_FILENAME",
    "ShardedDFlashFeatureDataset",
    "ShardedFeatureWriter",
    "is_sharded_feature_store",
    "sharded_feature_store_ready",
    "write_feature_shard",
    "write_sharded_feature_manifest",
]
