#!/usr/bin/env python3
"""Trích một tập con đại diện từ các file normalized JSONL.

Script này chỉ đọc dữ liệu đã chuẩn hóa, không tải lại raw dataset và không
thay đổi nội dung record. Với mỗi file ``*_test.jsonl``, các record được xếp
theo độ dài document rồi lấy các điểm cách đều trong phân phối đó. Vì vậy tập
con vẫn bao phủ từ tài liệu ngắn đến tài liệu dài, thay vì phụ thuộc vào thứ
tự dòng hoặc một lần random cụ thể.

Ví dụ:

    python data/extract_representative_samples.py \
        --normalized-dir data/normalized \
        --output-dir data/representative_100 \
        --samples-per-dataset 100

Mặc định độ dài được tính bằng số từ (``document.split()``), không cần
tokenizer hay network. Nếu muốn dùng đúng token length của một tokenizer:

    python data/extract_representative_samples.py \
        --length-metric tokens \
        --tokenizer Qwen/Qwen3-4B

Các file output chứa nguyên schema của input. Script tạo một file JSONL riêng
cho từng dataset và thêm ``manifest.json`` để ghi thống kê. Record có
``document`` hoặc ``reference`` rỗng sẽ được bỏ qua giống chính sách normalize
của prepare_data.py; số lượng bị bỏ qua được ghi trong manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


CORE_FIELDS = ("id", "dataset", "document", "reference")
PREFERRED_DATASET_ORDER = (
    "govreport",
    "multinews",
    "cnn_dailymail",
    "xsum",
)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Đọc JSONL và báo lỗi với vị trí chính xác nếu file không hợp lệ."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON không hợp lệ tại {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"Record tại {path}:{line_number} phải là JSON object"
                )
            yield row


def load_normalized_file(
    path: Path,
) -> tuple[str, list[dict[str, Any]], int]:
    """Load và kiểm tra một file normalized.

    ``source_split`` được kiểm tra nếu có. Một số normalized file cũ trong
    repo không có trường này, vì vậy đây là trường tùy chọn để script vẫn
    dùng được với cả hai phiên bản output của prepare_data.py.
    """

    input_rows = list(iter_jsonl(path))
    if not input_rows:
        raise ValueError(f"File normalized rỗng: {path}")

    dataset: str | None = None
    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    skipped_empty = 0

    for line_number, row in enumerate(input_rows, start=1):
        missing = [field for field in CORE_FIELDS if field not in row]
        if missing:
            raise KeyError(
                f"{path}:{line_number} thiếu trường normalized: {missing}"
            )

        current_dataset = str(row["dataset"])
        if not current_dataset:
            raise ValueError(f"{path}:{line_number} có dataset rỗng")
        if dataset is None:
            dataset = current_dataset
        elif current_dataset != dataset:
            raise ValueError(
                f"{path}:{line_number} trộn nhiều dataset: "
                f"{dataset!r} và {current_dataset!r}"
            )

        if "source_split" in row and row["source_split"] != "test":
            raise ValueError(
                f"{path}:{line_number} có source_split khác test: "
                f"{row['source_split']!r}"
            )

        row_id = str(row["id"])
        if not row_id:
            raise ValueError(f"{path}:{line_number} có id rỗng")
        if row_id in seen_ids:
            raise ValueError(f"{path}:{line_number} trùng id {row_id!r}")
        seen_ids.add(row_id)

        if any(
            not isinstance(row[field], str) or not row[field].strip()
            for field in ("document", "reference")
        ):
            skipped_empty += 1
            continue
        rows.append(row)

    assert dataset is not None
    if not rows:
        raise ValueError(f"File normalized không còn record hợp lệ: {path}")
    return dataset, rows, skipped_empty


def evenly_spaced_indices(population_size: int, sample_size: int) -> list[int]:
    """Lấy index cách đều, tương tự chiến lược debug của prepare_data.py."""

    if sample_size <= 0:
        raise ValueError("sample_size phải > 0")
    if population_size <= 0:
        return []
    if sample_size >= population_size:
        return list(range(population_size))
    if sample_size == 1:
        return [population_size // 2]
    return [
        round(i * (population_size - 1) / (sample_size - 1))
        for i in range(sample_size)
    ]


def make_length_function(
    metric: str,
    *,
    tokenizer_name: str | None,
    tokenizer_batch_size: int,
) -> Callable[[Sequence[dict[str, Any]]], list[int]]:
    """Tạo hàm tính độ dài cho toàn bộ rows của một dataset."""

    if metric == "chars":
        return lambda rows: [len(str(row["document"])) for row in rows]
    if metric == "words":
        return lambda rows: [
            len(str(row["document"]).split()) for row in rows
        ]
    if metric != "tokens":
        raise ValueError(f"length metric không hỗ trợ: {metric}")
    if not tokenizer_name:
        raise ValueError(
            "--length-metric tokens yêu cầu thêm --tokenizer, ví dụ "
            "--tokenizer Qwen/Qwen3-4B"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Cần transformers để dùng --length-metric tokens; "
            "hoặc bỏ tùy chọn này để dùng word length mặc định."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        use_fast=True,
    )

    def token_lengths(rows: Sequence[dict[str, Any]]) -> list[int]:
        lengths: list[int] = []
        for start in range(0, len(rows), tokenizer_batch_size):
            batch = rows[start : start + tokenizer_batch_size]
            encoded = tokenizer(
                [str(row["document"]) for row in batch],
                add_special_tokens=False,
                truncation=False,
                return_length=True,
            )
            lengths.extend(int(length) for length in encoded["length"])
        return lengths

    return token_lengths


def select_representative(
    rows: Sequence[dict[str, Any]],
    lengths: Sequence[int],
    sample_size: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Chọn đều theo độ dài và trả output theo thứ tự gốc của dataset."""

    if len(rows) != len(lengths):
        raise ValueError("Số rows và số length không khớp")

    ranked = sorted(
        enumerate(rows),
        key=lambda item: (
            int(lengths[item[0]]),
            len(str(item[1]["document"])),
            str(item[1]["id"]),
            item[0],
        ),
    )
    chosen_ranked_positions = evenly_spaced_indices(
        len(ranked),
        min(sample_size, len(ranked)),
    )
    chosen = [ranked[position] for position in chosen_ranked_positions]
    chosen.sort(key=lambda item: item[0])

    return (
        [row for _, row in chosen],
        [int(lengths[index]) for index, _ in chosen],
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def dataset_order(datasets: Iterable[str]) -> list[str]:
    known = [
        dataset for dataset in PREFERRED_DATASET_ORDER if dataset in datasets
    ]
    unknown = sorted(set(datasets) - set(known))
    return known + unknown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trích tập mẫu đại diện từ các file normalized JSONL; "
            "mặc định 100 mẫu cho mỗi dataset."
        )
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("data/normalized"),
        help="Thư mục chứa các file *_test.jsonl đã normalized.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/representative_100"),
        help="Thư mục ghi sample theo dataset, file gộp và manifest.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=100,
        help="Số mẫu mục tiêu cho mỗi dataset; dataset ít hơn sẽ lấy toàn bộ.",
    )
    parser.add_argument(
        "--pattern",
        default="*_test.jsonl",
        help="Glob dùng để tìm input trong normalized-dir.",
    )
    parser.add_argument(
        "--length-metric",
        choices=("words", "chars", "tokens"),
        default="words",
        help="Độ dài dùng để stratify: words (mặc định), chars hoặc tokens.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tên/path tokenizer, bắt buộc khi --length-metric tokens.",
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=8,
        help="Batch size khi tính token length.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Cho phép ghi đè output đã tồn tại.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset phải > 0")
    if args.tokenizer_batch_size <= 0:
        raise ValueError("--tokenizer-batch-size phải > 0")
    if args.length_metric == "tokens" and not args.tokenizer:
        raise ValueError("--length-metric tokens yêu cầu --tokenizer")


def main() -> None:
    args = parse_args()
    validate_args(args)

    normalized_dir = args.normalized_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not normalized_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy normalized directory: {normalized_dir}"
        )
    if output_dir == normalized_dir:
        raise ValueError("--output-dir phải khác --normalized-dir")

    input_files = sorted(
        path
        for path in normalized_dir.glob(args.pattern)
        if path.is_file()
    )
    if not input_files:
        raise FileNotFoundError(
            f"Không tìm thấy file {args.pattern!r} trong {normalized_dir}"
        )

    length_function = make_length_function(
        args.length_metric,
        tokenizer_name=args.tokenizer,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )

    per_dataset: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for input_file in input_files:
        dataset, rows, skipped_empty = load_normalized_file(input_file)
        if dataset in per_dataset:
            raise ValueError(
                f"Dataset {dataset!r} xuất hiện ở nhiều file normalized; "
                "hãy lọc input bằng --pattern cụ thể."
            )

        lengths = length_function(rows)
        selected, selected_lengths = select_representative(
            rows,
            lengths,
            args.samples_per_dataset,
        )
        per_dataset[dataset] = selected
        stats[dataset] = {
            "input_file": str(input_file),
            "input_count": len(rows) + skipped_empty,
            "skipped_empty_count": skipped_empty,
            "selected_count": len(selected),
            "min_length": min(lengths),
            "median_length": sorted(lengths)[len(lengths) // 2],
            "max_length": max(lengths),
            "selected_lengths": selected_lengths,
        }

    ordered_datasets = dataset_order(per_dataset)
    output_files = [
        output_dir / f"{dataset}_representative.jsonl"
        for dataset in ordered_datasets
    ]
    output_files.append(output_dir / "manifest.json")
    if not args.force:
        existing = [path for path in output_files if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Output đã tồn tại: {names}. Dùng --force để ghi đè."
            )

    for dataset in ordered_datasets:
        rows = per_dataset[dataset]
        write_jsonl(
            output_dir / f"{dataset}_representative.jsonl",
            rows,
        )

    manifest = {
        "schema_version": 1,
        "source_dir": str(normalized_dir),
        "selection": {
            "strategy": "evenly_spaced_after_sorting_by_document_length",
            "samples_per_dataset": args.samples_per_dataset,
            "length_metric": args.length_metric,
            "tokenizer": args.tokenizer,
        },
        "datasets": {dataset: stats[dataset] for dataset in ordered_datasets},
        "total_selected_count": sum(
            stats[dataset]["selected_count"] for dataset in ordered_datasets
        ),
    }
    (output_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total_selected = sum(
        stats[dataset]["selected_count"] for dataset in ordered_datasets
    )
    print(
        f"Đã trích {total_selected} mẫu vào "
        f"{len(ordered_datasets)} file riêng theo dataset:"
    )
    for dataset in ordered_datasets:
        info = stats[dataset]
        skipped = info["skipped_empty_count"]
        skipped_note = f"; bỏ qua {skipped} record rỗng" if skipped else ""
        print(
            f"  {dataset:16s} {info['selected_count']:>3d}/"
            f"{info['input_count']:<5d} mẫu; độ dài "
            f"{info['min_length']}..{info['max_length']}"
            f"{skipped_note}"
        )
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
