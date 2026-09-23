#!/usr/bin/env python3
"""Quét và mô tả raw dataset Phase 1, không build hay lấy mẫu ngẫu nhiên.

Script này cố ý đứng ngoài pipeline prepare/regenerate/cache. Nó đọc raw
ShareGPT và ArXiv theo thứ tự input, ghi audit record cho từng JSON object,
ghi lỗi parse/schema riêng, rồi tạo thống kê và biểu đồ để quyết định cách
build dataset ở bước sau.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from _common import content_of, join_text, normalize_role, stable_id
    from prepare_arxiv import _document
    from prepare_sharegpt import _messages
except ImportError:  # pragma: no cover - hỗ trợ import package ngoài CLI
    from scripts.mr_dflash._common import content_of, join_text, normalize_role, stable_id
    from scripts.mr_dflash.prepare_arxiv import _document
    from scripts.mr_dflash.prepare_sharegpt import _messages


class _ChunkedJSONStream:
    """Đọc một JSON value lớn bằng chunk, không load cả array vào RAM."""

    def __init__(self, handle, *, chunk_size: int = 1 << 20) -> None:
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _fill(self) -> bool:
        if self.eof:
            return False
        chunk = self.handle.read(self.chunk_size)
        if chunk == "":
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def _ensure(self) -> bool:
        while self.position >= len(self.buffer) and not self.eof:
            self.buffer = ""
            self.position = 0
            self._fill()
        return self.position < len(self.buffer)

    def skip_whitespace(self) -> bool:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return True
            if not self._fill():
                return False

    def peek(self) -> str | None:
        return self.buffer[self.position] if self.skip_whitespace() else None

    def take(self) -> str | None:
        if not self.skip_whitespace():
            return None
        value = self.buffer[self.position]
        self.position += 1
        return value

    def decode(self) -> Any:
        """Decode value tại cursor; refill khi lỗi chỉ vì value còn thiếu."""

        while True:
            if not self._ensure():
                raise json.JSONDecodeError("unexpected end of input", "", 0)
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                # Bỏ phần trước cursor, giữ phần value đang đọc và đọc thêm.
                self.buffer = self.buffer[self.position :]
                self.position = 0
                self._fill()
                continue
            self.position = end
            return value


def _iter_json_array(path: Path) -> Iterator[tuple[int, Any, str | None]]:
    """Yield ``(index, value, error)`` cho JSON array/object lớn."""

    with path.open("r", encoding="utf-8") as handle:
        stream = _ChunkedJSONStream(handle)
        first = stream.peek()
        if first == "[":
            stream.take()
            index = 0
            while True:
                token = stream.peek()
                if token == "]":
                    stream.take()
                    break
                if token is None:
                    yield index, None, "unexpected end of JSON array"
                    break
                try:
                    value = stream.decode()
                except json.JSONDecodeError as exc:
                    yield index, None, f"{exc.msg}"
                    break
                yield index, value, None
                index += 1
                delimiter = stream.take()
                if delimiter == "]":
                    break
                if delimiter != ",":
                    yield index, None, f"expected ',' or ']', got {delimiter!r}"
                    break
            return

        if first is None:
            return
        try:
            value = stream.decode()
        except json.JSONDecodeError as exc:
            yield 0, None, exc.msg
            return
        yield 0, value, None
        if stream.peek() is not None:
            yield 1, None, "trailing data after JSON value"


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Any, str | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            if not raw.strip():
                continue
            try:
                yield index, json.loads(raw), None
            except json.JSONDecodeError as exc:
                yield index, None, exc.msg


def iter_source_entries(path: str | Path, source: str) -> Iterator[dict[str, Any]]:
    """Iterate raw records in source order, including parse-error events."""

    source_path = Path(path)
    iterator = _iter_jsonl(source_path) if source_path.suffix.lower() == ".jsonl" else _iter_json_array(source_path)
    for source_index, value, error in iterator:
        yield {
            "source": source,
            "source_index": source_index,
            "value": value,
            "error": error,
        }


def analyze_source(path: str | Path, source: str, tokenizer: Any = None) -> Iterator[dict[str, Any]]:
    """Yield audit records for one source without shuffling or truncating."""

    for event in iter_source_entries(path, source):
        value = event["value"]
        if event["error"] is not None:
            yield {
                "kind": "parse_error",
                "source": source,
                "source_index": event["source_index"],
                "error": event["error"],
            }
            continue
        if not isinstance(value, dict):
            yield {
                "kind": "invalid_record",
                "source": source,
                "source_index": event["source_index"],
                "reason": "record_is_not_object",
            }
            continue
        yield _audit_record(value, source, event["source_index"], tokenizer)


def _token_length(tokenizer: Any, text: str, messages: list[dict[str, str]] | None = None) -> tuple[int, str]:
    if tokenizer is None:
        return len(text), "characters_fallback"
    if messages and hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            if isinstance(encoded, dict):
                encoded = encoded.get("input_ids", encoded)
            if hasattr(encoded, "shape"):
                return int(encoded.shape[-1]), "tokenizer_chat_template"
            return len(encoded), "tokenizer_chat_template"
        except Exception:  # noqa: BLE001 - fallback tokenizer vẫn hữu ích cho audit
            pass
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return len(encoded), "tokenizer"


def _audit_record(row: dict[str, Any], source: str, source_index: int, tokenizer: Any) -> dict[str, Any]:
    raw_id = row.get("id") or row.get("conversation_id") or row.get("paper_id") or row.get("arxiv_id")
    sample_id = str(raw_id) if raw_id is not None else stable_id(source, source_index)
    if source == "sharegpt":
        raw_messages = row.get("conversations", row.get("messages", []))
        messages = _messages(row)
        raw_turns = len(raw_messages) if isinstance(raw_messages, list) else 0
        if not messages:
            return {
                "kind": "record",
                "source": source,
                "source_index": source_index,
                "id": sample_id,
                "valid": False,
                "reason": "missing_user_prompt",
                "turn_count": raw_turns,
            }
        text = "\n\n".join(item["content"] for item in messages)
        token_length, length_unit = _token_length(tokenizer, text, messages)
        return {
            "kind": "record",
            "source": source,
            "source_index": source_index,
            "id": sample_id,
            "valid": True,
            "token_length": token_length,
            "length_unit": length_unit,
            "char_length": len(text),
            "turn_count": raw_turns,
            "retained_turn_count": len(messages),
            "user_turn_count": sum(item["role"] == "user" for item in messages),
            "assistant_turn_count": sum(item["role"] == "assistant" for item in messages),
        }

    document = _document(row).strip()
    if not document:
        return {
            "kind": "record",
            "source": source,
            "source_index": source_index,
            "id": sample_id,
            "valid": False,
            "reason": "missing_document",
        }
    prompt = "Summarize the following scientific document:\n\n" + document
    token_length, length_unit = _token_length(tokenizer, prompt)
    reference = join_text(row.get("summary")).strip()
    return {
        "kind": "record",
        "source": source,
        "source_index": source_index,
        "id": sample_id,
        "valid": True,
        "token_length": token_length,
        "length_unit": length_unit,
        "char_length": len(document),
        "prompt_char_length": len(prompt),
        "document_has_reference": bool(reference),
        "reference_char_length": len(reference),
    }


def _quantile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "quantiles": {f"p{int(p * 100)}": None for p in (0.50, 0.75, 0.90, 0.95, 0.99)},
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "quantiles": {
            f"p{int(p * 100)}": _quantile(values, p)
            for p in (0.50, 0.75, 0.90, 0.95, 0.99)
        },
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_plot(path: Path, plotter) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(9, 5.5), dpi=160)
    try:
        plotter(plt)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def _write_figures(
    output: Path,
    lengths_by_source: dict[str, list[int]],
    source_counts: dict[str, dict[str, int]],
    sharegpt_turns: list[int],
    arxiv_docs: list[int],
) -> dict[str, str]:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return {}

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    palette = {"sharegpt": "#0072B2", "arxiv": "#D55E00"}
    # Chỉ giữ các scalar cần vẽ; không giữ raw text hoặc toàn bộ audit record
    # trong RAM khi source có hàng triệu dòng.
    by_source = lengths_by_source

    distribution = figures / "token_length_distribution.png"
    def draw_distribution(plt):
        for source in ("sharegpt", "arxiv"):
            values = by_source.get(source, [])
            if values:
                plt.hist(values, bins=40, alpha=0.65, label=source, color=palette[source])
        plt.xlabel("Độ dài prompt (token hoặc fallback ký tự)")
        plt.ylabel("Số mẫu")
        plt.title("Phân phối độ dài prompt hợp lệ")
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
    _write_plot(distribution, draw_distribution)

    ecdf = figures / "token_length_ecdf.png"
    def draw_ecdf(plt):
        for source in ("sharegpt", "arxiv"):
            values = sorted(by_source.get(source, []))
            if values:
                y = [(index + 1) / len(values) for index in range(len(values))]
                plt.step(values, y, where="post", label=source, color=palette[source])
        plt.xlabel("Độ dài prompt (token hoặc fallback ký tự)")
        plt.ylabel("ECDF")
        plt.ylim(0, 1.02)
        plt.title("ECDF độ dài prompt hợp lệ")
        plt.legend()
        plt.grid(alpha=0.25)
    _write_plot(ecdf, draw_ecdf)

    composition = figures / "source_composition.png"
    def draw_composition(plt):
        sources = ["sharegpt", "arxiv"]
        valid_counts = [source_counts.get(source, {}).get("valid", 0) for source in sources]
        invalid_counts = [source_counts.get(source, {}).get("invalid", 0) for source in sources]
        positions = list(range(len(sources)))
        plt.bar(positions, valid_counts, label="valid", color="#009E73")
        plt.bar(positions, invalid_counts, bottom=valid_counts, label="invalid", color="#CC79A7")
        plt.xticks(positions, sources)
        plt.ylabel("Số JSON object")
        plt.title("Thành phần dữ liệu theo source")
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
    _write_plot(composition, draw_composition)

    structure = figures / "conversation_structure.png"
    def draw_structure(plt):
        if share_turns:
            plt.hist(share_turns, bins=max(1, min(20, max(share_turns) - min(share_turns) + 1)), alpha=0.65, label="ShareGPT turns", color=palette["sharegpt"])
        if arxiv_docs:
            plt.twinx().hist(arxiv_docs, bins=30, histtype="step", linewidth=2, label="ArXiv chars", color=palette["arxiv"])
        plt.xlabel("Turn count (ShareGPT) / document length (ArXiv)")
        plt.ylabel("Số mẫu")
        plt.title("Cấu trúc hội thoại và văn bản")
        plt.grid(axis="y", alpha=0.25)
    _write_plot(structure, draw_structure)

    return {name: str(figures / name) for name in (
        "token_length_distribution.png",
        "token_length_ecdf.png",
        "source_composition.png",
        "conversation_structure.png",
    )}


def analyze_dataset(
    sharegpt_source: str | Path,
    arxiv_source: str | Path,
    output_root: str | Path,
    tokenizer: Any = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Quét đủ hai raw source theo thứ tự, không random sampling.

    ``max_records`` chỉ dành cho smoke test; khi bỏ trống, không có giới hạn.
    Nó đếm JSON object đã đọc, không đếm dòng trống hay lỗi parse không tạo
    được object.
    """

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    issues_path = output / "issues.jsonl"
    records_path.unlink(missing_ok=True)
    issues_path.unlink(missing_ok=True)

    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"scanned": 0, "valid": 0, "invalid": 0})
    id_counts: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []
    lengths: list[int] = []
    lengths_by_source: dict[str, list[int]] = defaultdict(list)
    sharegpt_turns: list[int] = []
    arxiv_docs: list[int] = []
    scanned_rows = 0

    with records_path.open("w", encoding="utf-8") as record_handle, issues_path.open("w", encoding="utf-8") as issue_handle:
        for source, path in (("sharegpt", sharegpt_source), ("arxiv", arxiv_source)):
            for audit in analyze_source(path, source, tokenizer):
                if audit["kind"] == "parse_error":
                    issue = {
                        "kind": "malformed_json",
                        "source": source,
                        "source_index": audit["source_index"],
                        "message": audit["error"],
                    }
                    issues.append(issue)
                    issue_handle.write(json.dumps(issue, ensure_ascii=False, sort_keys=True) + "\n")
                    continue

                scanned_rows += 1
                source_counts[source]["scanned"] += 1
                sample_id = audit.get("id")
                id_counts[str(sample_id)] += 1
                if audit["kind"] == "invalid_record":
                    audit["valid"] = False
                    audit["reason"] = audit.get("reason", "invalid_record")
                if audit.get("valid") and id_counts[str(sample_id)] > 1:
                    audit["valid"] = False
                    audit["reason"] = "duplicate_id"
                if audit.get("valid"):
                    source_counts[source]["valid"] += 1
                    lengths.append(int(audit["token_length"]))
                    lengths_by_source[source].append(int(audit["token_length"]))
                else:
                    source_counts[source]["invalid"] += 1
                    issue = {
                        "kind": "invalid_record",
                        "source": source,
                        "source_index": audit["source_index"],
                        "id": sample_id,
                        "reason": audit.get("reason", "invalid_record"),
                    }
                    issues.append(issue)
                    issue_handle.write(json.dumps(issue, ensure_ascii=False, sort_keys=True) + "\n")
                record_handle.write(json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n")
                if audit.get("source") == "sharegpt" and audit.get("turn_count") is not None:
                    sharegpt_turns.append(int(audit["turn_count"]))
                if audit.get("source") == "arxiv" and audit.get("char_length") is not None:
                    arxiv_docs.append(int(audit["char_length"]))
                if max_records is not None and scanned_rows >= max_records:
                    break
            if max_records is not None and scanned_rows >= max_records:
                break

    duplicate_ids = {key: count for key, count in id_counts.items() if count > 1}
    source_summary = {
        source: {
            **counts,
            "length_stats": _length_stats(lengths_by_source.get(source, [])),
        }
        for source, counts in sorted(source_counts.items())
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "scan_order": ["sharegpt", "arxiv"],
        "sampling": "none",
        "scanned_rows": scanned_rows,
        "valid_rows": len(lengths),
        "invalid_rows": scanned_rows - len(lengths),
        "parse_error_count": sum(issue["kind"] == "malformed_json" for issue in issues),
        "duplicate_ids": duplicate_ids,
        "source_counts": {source: counts["scanned"] for source, counts in sorted(source_counts.items())},
        "sources": source_summary,
        "length_stats": _length_stats(lengths),
        "artifacts": {
            "records": str(records_path),
            "issues": str(issues_path),
        },
    }
    summary["figures"] = _write_figures(
        output,
        lengths_by_source,
        source_counts,
        sharegpt_turns,
        arxiv_docs,
    )
    _write_json(output / "summary.json", summary)
    return summary


def _load_tokenizer(path: str | None) -> Any:
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharegpt-source", required=True, type=Path)
    parser.add_argument("--arxiv-source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tokenizer", default=None, help="Tokenizer local; không truy cập internet")
    parser.add_argument("--max-records", type=int, default=None, help="Chỉ dùng smoke test; bỏ trống để quét toàn bộ")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Giữ contract offline; tokenizer luôn được load local-only",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    for path in (args.sharegpt_source, args.arxiv_source):
        if not path.is_file():
            parser.error(f"Không tìm thấy raw source: {path}")
    summary = analyze_dataset(
        sharegpt_source=args.sharegpt_source,
        arxiv_source=args.arxiv_source,
        output_root=args.output_root,
        tokenizer=_load_tokenizer(args.tokenizer),
        max_records=args.max_records,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
