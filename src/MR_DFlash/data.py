"""Dữ liệu train DFlash offline: loss_mask, dataset, feature store, collator.

Tương ứng các mảnh của SpecForge:
- ``specforge/data/loss_mask.py``: ``has_consecutive_supervised_tokens``.
- ``specforge/algorithms/common/dflash_family_data.py``: normalizer offline +
  collator DFlash-family.
- ``specforge/runtime/data_plane/offline_reader.py`` + feature store: đọc thư
  mục file feature (đuôi ``.ckpt``) một cách deterministic.

Feature contract DFlash offline: mỗi file lưu ``input_ids``, ``loss_mask``,
``hidden_states`` (concat tại các target layer; chiều cuối = feature).
Loss mask được dựng từ hội thoại: đánh dấu vùng *nội dung* mỗi assistant turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# loss mask predicates
# --------------------------------------------------------------------------- #

def has_consecutive_supervised_tokens(loss_mask: Any) -> bool:
    """Một mẫu có ít nhất hai token được supervise liên tiếp (yêu cầu DFlash)."""
    values = loss_mask.tolist() if hasattr(loss_mask, "tolist") else list(loss_mask)
    if values and isinstance(values[0], Sequence) and not isinstance(values[0], (int, float)):
        if len(values) != 1:
            raise ValueError("kỳ vọng một hàng loss-mask")
        values = list(values[0])
    return any(
        bool(a) and bool(b) for a, b in zip(values, values[1:])
    )


# --------------------------------------------------------------------------- #
# Render hội thoại → input_ids + assistant loss mask
# --------------------------------------------------------------------------- #

def _encode_prefix(conv, tokenizer) -> List[int]:
    ids = tokenizer.apply_chat_template(
        conv,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    if isinstance(ids, dict):  # transformers >= 5 có thể trả BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def _find_content_window(
    added: List[int],
    content: List[int],
) -> Optional[Tuple[int, int]]:
    """Tìm vị trí nội dung assistant trong đoạn token mới được thêm."""
    if not content:
        return None
    n, m = len(added), len(content)
    for start in range(n - m + 1):
        if added[start : start + m] == content:
            return start, start + m
    return None


def render_conversation(
    conversation: List[Dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """Render hội thoại → (input_ids, assistant_mask).

    Dùng prefix-diff: với mỗi assistant turn k, encode prefix tới turn đó và
    prefix trước nó; đoạn chênh lệch chứa header + nội dung assistant. Nội dung
    thật được định vị bằng cách align token của riêng message content
    (``add_special_tokens=False``). Không tìm thấy → fallback đánh dấu cả đoạn
    chênh lệch (trừ token đóng template cuối).
    """
    ids: List[int] = []
    mask: List[int] = []
    prev_conv: List[Dict[str, str]] = []
    assistant_idx = 0
    for idx, msg in enumerate(conversation):
        role = str(msg.get("role", "")).strip().lower()
        if role not in ("user", "assistant", "system", "tool"):
            continue
        next_conv = prev_conv + [msg]
        next_ids = _encode_prefix(next_conv, tokenizer)
        if role == "assistant":
            added = next_ids[len(ids):]
            content = tokenizer(
                str(msg.get("content", "")),
                add_special_tokens=False,
            )["input_ids"]
            window = _find_content_window(added, content)
            if window is None:
                # Fallback: cả đoạn thêm, bỏ token đóng cuối nếu là end token.
                lo, hi = 0, len(added)
                end_candidates = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ]
                if hi > 0 and added[hi - 1] in end_candidates and added[hi - 1] is not None:
                    hi -= 1
            else:
                lo, hi = window
            base = len(ids)
            assistant_mask = [0] * len(next_ids)
            for pos in range(base + lo, base + hi):
                assistant_mask[pos] = 1
            ids = next_ids
            mask = assistant_mask
            assistant_idx += 1
        else:
            ids = next_ids
            mask = mask + [0] * (len(next_ids) - len(mask))
        prev_conv = next_conv

    if len(ids) > max_length:
        ids = ids[:max_length]
        mask = mask[:max_length]
    return ids, mask


def build_sample(
    row: Dict[str, Any],
    tokenizer: Any,
    max_length: int,
) -> Optional[Dict[str, Any]]:
    """Chuyển một dòng jsonl → dict sample (hoặc None nếu không hợp lệ)."""
    sample_id = str(row.get("id", ""))
    conversations = row.get("conversations")
    if not conversations:
        return None
    try:
        input_ids, mask = render_conversation(conversations, tokenizer, max_length)
    except Exception:
        return None
    if len(input_ids) < 3 or not has_consecutive_supervised_tokens(mask):
        return None
    return {"id": sample_id, "input_ids": input_ids, "loss_mask": mask}


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# --------------------------------------------------------------------------- #
# Offline feature store (.ckpt per mẫu)
# --------------------------------------------------------------------------- #

_FEATURE_SUFFIXES = (".ckpt", ".ckpt.gz")


def list_feature_files(path: str) -> List[str]:
    """Liệt kê deterministic (sorted) các file feature dưới ``path``."""
    if Path(path).is_file():
        return [str(Path(path).resolve())]
    files: List[str] = []
    for root, _dirs, names in Path(path).walk():
        for name in names:
            if name.endswith(_FEATURE_SUFFIXES):
                files.append(str((root / name).resolve()))
    files.sort()
    return files


def save_feature_file(path: str, tensors: Dict[str, torch.Tensor]) -> None:
    """Ghi một sample feature (CPU tensors) ra file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value.detach().cpu() for key, value in tensors.items()}
    torch.save(payload, path)


def load_feature_file(path: str) -> Dict[str, torch.Tensor]:
    """Đọc một sample feature file."""
    return torch.load(path, map_location="cpu", weights_only=True)


@dataclass
class FeatureRef:
    """Tham chiếu metadata-tới-một file feature (không mở tensor)."""

    sample_id: str
    path: str
    run_id: str = "offline"
    num_tokens: int = 0
    estimated_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def read_feature_refs(hidden_states_path: str, run_id: str = "offline") -> List[FeatureRef]:
    """Liệt kê refs theo thứ tự deterministic (không đọc nội dung tensor)."""
    return [
        FeatureRef(sample_id=f"{run_id}:{i:08d}", path=path, run_id=run_id)
        for i, path in enumerate(list_feature_files(hidden_states_path))
    ]


# --------------------------------------------------------------------------- #
# Normalizer + collator DFlash-family
# --------------------------------------------------------------------------- #

def normalize_offline_sample(raw: Dict[str, torch.Tensor], max_len: int) -> Dict[str, torch.Tensor]:
    """Chuẩn hoá một sample feature: truncate theo max_len + kiểm tra thẳng hàng."""
    input_ids = raw["input_ids"][:max_len].unsqueeze(0) if raw["input_ids"].dim() == 1 else raw["input_ids"][:, :max_len]
    loss_mask = raw["loss_mask"][:max_len].unsqueeze(0) if raw["loss_mask"].dim() == 1 else raw["loss_mask"][:, :max_len]

    hidden = raw["hidden_states"]
    if hidden.dim() == 3:
        if hidden.shape[0] != 1:
            raise ValueError(
                f"hidden_states offline phải [seq, width] hoặc [1, seq, width], got {tuple(hidden.shape)}"
            )
        hidden = hidden.squeeze(0)
    if hidden.dim() != 2:
        raise ValueError(f"hidden_states offline phải 2D/3D, got {tuple(hidden.shape)}")
    hidden_states = hidden[:max_len].unsqueeze(0)

    lengths = {input_ids.shape[1], loss_mask.shape[1], hidden_states.shape[1]}
    if len(lengths) != 1:
        raise ValueError(
            "feature offline lệch độ dài sau truncate: "
            f"input_ids={input_ids.shape[1]}, loss_mask={loss_mask.shape[1]}, "
            f"hidden_states={hidden_states.shape[1]}"
        )
    if not has_consecutive_supervised_tokens(loss_mask[0]):
        raise ValueError(
            "offline DFlash-family yêu cầu hai token được supervise liên tiếp"
        )
    return {
        "input_ids": input_ids.long(),
        "loss_mask": loss_mask.float(),
        "hidden_states": hidden_states,
    }


def pad_and_concatenate_features(
    features: List[Dict[str, torch.Tensor]],
    *,
    sequence_axes: Dict[str, int],
    required_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Zero-pad theo chiều sequence dài nhất trong batch rồi concat theo batch."""
    if not features:
        raise ValueError("không thể collate batch rỗng")
    required = tuple(required_keys)
    missing = [
        (i, key) for i, feature in enumerate(features)
        for key in required if key not in feature
    ]
    if missing:
        raise KeyError(f"feature batch thiếu key: {missing}")
    max_length = max(int(feature["input_ids"].shape[-1]) for feature in features)

    batch: Dict[str, torch.Tensor] = {}
    for key in required:
        axis = sequence_axes[key]
        padded = []
        for feature in features:
            tensor = feature[key]
            length = int(tensor.shape[axis])
            if length > max_length:
                raise ValueError(
                    f"feature {key!r} dài {length} vượt input_ids {max_length}"
                )
            if length < max_length:
                shape = list(tensor.shape)
                shape[axis] = max_length - length
                tensor = torch.cat([tensor, tensor.new_zeros(shape)], dim=axis)
            padded.append(tensor)
        batch[key] = torch.cat(padded, dim=0)
    return batch


def build_dflash_collator():
    """Collator chuẩn cho DFlash-family (input_ids/loss_mask/hidden_states)."""

    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
            },
            required_keys=("input_ids", "loss_mask", "hidden_states"),
        )

    return collate


# --------------------------------------------------------------------------- #
# Dataset huấn luyện từ feature files
# --------------------------------------------------------------------------- #

class DFlashFeatureDataset:
    """Dataset đọc từng file feature (lazy per-sample) + truncate theo max_len.

    Có thể trả batch qua ``collate()`` (cùng collator dùng cho capture mới).
    """

    def __init__(
        self,
        hidden_states_path: str,
        *,
        max_len: int = 3072,
        run_id: str = "offline",
        sample_limit: Optional[int] = None,
    ) -> None:
        self.refs = read_feature_refs(hidden_states_path, run_id=run_id)
        if sample_limit is not None:
            self.refs = self.refs[:sample_limit]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        raw = load_feature_file(self.refs[index].path)
        return normalize_offline_sample(raw, self.max_len)

    def collate(self, features) -> Dict[str, torch.Tensor]:
        return build_dflash_collator()(features)


__all__ = [
    "FeatureRef",
    "DFlashFeatureDataset",
    "has_consecutive_supervised_tokens",
    "iter_jsonl",
    "build_sample",
    "render_conversation",
    "list_feature_files",
    "save_feature_file",
    "load_feature_file",
    "read_feature_refs",
    "normalize_offline_sample",
    "pad_and_concatenate_features",
    "build_dflash_collator",
]
