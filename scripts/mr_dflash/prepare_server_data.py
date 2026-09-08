"""Chuẩn hóa hai nguồn dữ liệu trên B200 vào layout pilot MR-DFlash.

Script này chỉ làm:

``ShareGPT JSON + ArXiv JSONL -> normalized JSONL -> train/val/test prompts``.

Regenerate bằng target và tokenize là các bước riêng vì chúng tốn GPU/thời
gian hơn. Raw source chỉ được đọc; artifact mới được ghi dưới
``--output-root``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import REPO_ROOT


DEFAULT_SHAREGPT_COUNT = 50000
DEFAULT_ARXIV_COUNT = 50000
DEFAULT_SHAREGPT_SOURCE = (
    "/workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_ARXIV_SOURCE = (
    "/workspace/storage-shared/nlp/dungdx4/datasets/arxiv/train.label.jsonl"
)


def _run(script: str, arguments: list[str], *, dry_run: bool) -> None:
    command = [sys.executable, str(Path(__file__).with_name(script)), *arguments]
    print("[prepare_server_data]", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Chuẩn hóa ShareGPT JSON và ArXiv JSONL cho MR-DFlash"
    )
    parser.add_argument("--sharegpt-source", default=DEFAULT_SHAREGPT_SOURCE)
    parser.add_argument("--arxiv-source", default=DEFAULT_ARXIV_SOURCE)
    parser.add_argument("--output-root", default="data/mr_dflash_pilot")
    parser.add_argument("--sharegpt-count", type=int, default=DEFAULT_SHAREGPT_COUNT)
    parser.add_argument("--arxiv-count", type=int, default=DEFAULT_ARXIV_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="in command và không đọc source/ghi artifact",
    )
    args = parser.parse_args(argv)
    if args.sharegpt_count < 1 or args.arxiv_count < 1:
        raise ValueError("sharegpt-count và arxiv-count phải dương")
    if not args.dry_run:
        for name in (args.sharegpt_source, args.arxiv_source):
            if not Path(name).is_file():
                raise FileNotFoundError(
                    f"không tìm thấy source {name!r}; kiểm tra mount server"
                )

    root = Path(args.output_root)
    normalized = root / "normalized"
    share_output = normalized / "sharegpt_prompts.jsonl"
    arxiv_output = normalized / "arxiv_prompts.jsonl"
    resume_args = ["--resume"] if args.resume else []

    share_args = [
        "--input",
        args.sharegpt_source,
        "--output",
        str(share_output),
        "--limit",
        str(args.sharegpt_count),
        *resume_args,
    ]
    if args.tokenizer:
        arxiv_args = [
            "--input",
            args.arxiv_source,
            "--output",
            str(arxiv_output),
            "--limit",
            str(args.arxiv_count),
            "--tokenizer",
            args.tokenizer,
            *resume_args,
        ]
    else:
        arxiv_args = [
            "--input",
            args.arxiv_source,
            "--output",
            str(arxiv_output),
            "--limit",
            str(args.arxiv_count),
            *resume_args,
        ]
    _run("prepare_sharegpt.py", share_args, dry_run=args.dry_run)
    _run("prepare_arxiv.py", arxiv_args, dry_run=args.dry_run)

    build_args = [
        "--sharegpt",
        str(share_output),
        "--arxiv",
        str(arxiv_output),
        "--source-sharegpt-input",
        args.sharegpt_source,
        "--source-arxiv-input",
        args.arxiv_source,
        "--output-root",
        str(root),
        "--sharegpt-count",
        str(args.sharegpt_count),
        "--arxiv-count",
        str(args.arxiv_count),
        "--seed",
        str(args.seed),
    ]
    if args.allow_short:
        build_args.append("--allow-short")
    _run("build_pilot_dataset.py", build_args, dry_run=args.dry_run)
    if args.dry_run:
        print("[prepare_server_data] dry-run; chưa đọc source và chưa ghi artifact")
    else:
        print(
            "[prepare_server_data] normalized/split hoàn tất; "
            "bước tiếp theo là regenerate_pilot.py rồi tokenize_dataset.py"
        )


if __name__ == "__main__":
    main()
