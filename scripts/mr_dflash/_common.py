"""Tiện ích chung cho các script chuẩn bị pilot MR-DFlash."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} phải là JSON object")
            yield value


def read_records(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Đọc JSON array (`.json`) hoặc JSONL (`.jsonl`) thành các record.

    ShareGPT trên server là một JSON array lớn, trong khi ArXiv là JSONL.
    Tách primitive này khỏi ``read_jsonl`` để tránh coi cả array là một dòng
    JSON object. File JSON array được load một lần có chủ đích: nó chỉ chứa
    metadata/text thô, không chứa hidden states; bước pilot vẫn ghi artifact
    chuẩn hóa theo từng dòng ra thư mục mới.
    """
    source = Path(path)
    if source.suffix.lower() != ".json":
        yield from read_jsonl(source)
        return
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        # Một số mirror bọc records dưới data/records; vẫn chấp nhận JSON
        # object đơn để smoke fixture và báo lỗi rõ nếu schema bất thường.
        for key in ("data", "records", "conversations"):
            nested = value.get(key)
            if isinstance(nested, list) and key != "conversations":
                value = nested
                break
        else:
            value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{source} phải là JSON array hoặc JSON object")
    for index, row in enumerate(value, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{source}[{index}] phải là JSON object")
        yield row


def join_text(value: Any, *, separator: str = "\n\n") -> str:
    """Chuẩn hóa text dạng string/list thành text thuần.

    ArXiv ``text`` và ``summary`` là list các đoạn/câu. Không dùng ``str``
    trực tiếp vì nó tạo Python-list literal trong prompt.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return join_text(value[key], separator=separator)
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = [join_text(item, separator=separator).strip() for item in value]
        return separator.join(part for part in parts if part)
    return str(value)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return count


def stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def write_json(path: str | Path, value: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def append_jsonl_durable(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Append một batch JSONL và fsync để hỗ trợ resume sau mất tiến trình."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        for row in rows:
            handle.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def resolve_parallel_work_root(output: str | Path, mode: str) -> Path:
    """Resolve a visible worker directory, preserving legacy resume paths.

    New runs use a visible, descriptive directory such as
    ``parallel_regenerate_train``.  Older runs used the hidden
    ``.parallel_regenerate_train`` convention; if that directory is the only
    existing work-root, keep using it so an interrupted legacy run remains
    resumable instead of silently launching a duplicate job.
    """
    if mode not in {"regenerate", "cache"}:
        raise ValueError(f"parallel mode không hợp lệ: {mode!r}")
    output_path = Path(output)
    visible = output_path.parent / f"parallel_{mode}_{output_path.stem}"
    legacy = output_path.parent / f".parallel_{mode}_{output_path.stem}"
    if visible.exists():
        return visible
    if legacy.exists():
        return legacy
    return visible


def normalize_role(role: Any) -> str:
    value = str(role or "").strip().lower()
    aliases = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "model": "assistant",
    }
    return aliases.get(value, value)


def content_of(message: Dict[str, Any]) -> str:
    value = message.get("content", message.get("value", ""))
    if isinstance(value, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in value)
    return str(value or "")


def canonical_prompt(sample_id: str, source: str, messages: List[Dict[str, str]], **metadata: Any) -> Dict[str, Any]:
    if not messages or not any(m.get("role") == "user" for m in messages):
        raise ValueError("prompt canonical cần ít nhất một user message")
    return {
        "id": str(sample_id),
        "source": str(source),
        "conversations": messages,
        "metadata": metadata,
    }
