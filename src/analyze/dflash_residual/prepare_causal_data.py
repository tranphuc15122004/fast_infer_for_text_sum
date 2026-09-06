"""Prepare deterministic local inputs for E16/E17 causal diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_gsm8k(source: str | Path, output: str | Path, count: int) -> int:
    rows = []
    for index, line in enumerate(Path(source).read_text(encoding="utf-8").splitlines()):
        if not line.strip() or len(rows) >= count:
            continue
        item = json.loads(line)
        rows.append({
            "id": index,
            "dataset": "canonical",
            "prompt": str(item["question"]) + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        })
    if len(rows) < count:
        raise ValueError(f"requested {count} GSM8K rows but found {len(rows)}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def prepare_multinews_state_pairs(source: str | Path, output: str | Path, count: int) -> int:
    rows = []
    for index, line in enumerate(Path(source).read_text(encoding="utf-8").splitlines()):
        if not line.strip() or len(rows) >= count:
            continue
        item = json.loads(line)
        conversations = item.get("conversations", [])
        user = next((turn.get("content", "") for turn in conversations if turn.get("role") == "user"), "")
        assistant = next((turn.get("content", "") for turn in conversations if turn.get("role") == "assistant"), "")
        if not user or not assistant:
            continue
        rows.append({
            "id": item.get("id", index),
            "dataset": "multi_news",
            "document": user,
            "reference": assistant,
        })
    if len(rows) < count:
        raise ValueError(f"requested {count} Multi-News rows but found {len(rows)}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def prepare_conversation_state_pairs(source: str | Path, output: str | Path, count: int, dataset_filter: str | None = None) -> int:
    rows = []
    for index, line in enumerate(Path(source).read_text(encoding="utf-8").splitlines()):
        if not line.strip() or len(rows) >= count:
            continue
        item = json.loads(line)
        if dataset_filter and item.get("dataset") != dataset_filter:
            continue
        conversations = item.get("conversations", [])
        user = next((turn.get("content", "") for turn in conversations if turn.get("role") == "user"), "")
        assistant = next((turn.get("content", "") for turn in conversations if turn.get("role") == "assistant"), "")
        if not user or not assistant:
            continue
        rows.append({
            "id": item.get("id", index),
            "dataset": item.get("dataset", "summary"),
            "document": user,
            "reference": assistant,
        })
    if len(rows) < count:
        raise ValueError(f"requested {count} conversation rows but found {len(rows)}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--mode", choices=("gsm8k", "multinews_state", "conversations_state"), default="gsm8k")
    parser.add_argument("--dataset-filter", default=None)
    args = parser.parse_args()
    if args.mode == "gsm8k":
        print(prepare_gsm8k(args.source, args.output, args.count))
    elif args.mode == "multinews_state":
        print(prepare_multinews_state_pairs(args.source, args.output, args.count))
    else:
        print(prepare_conversation_state_pairs(args.source, args.output, args.count, args.dataset_filter))


if __name__ == "__main__":
    main()
