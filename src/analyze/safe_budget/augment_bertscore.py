"""Add local BERTScore metrics to semantic-selection JSONL records.

This implementation is intentionally dependency-light for the external T4
Conda environment: it uses only ``torch`` and ``transformers`` and requires a
local encoder snapshot.  It computes the standard token-level cosine matching
precision, recall, and F1 used by BERTScore (without IDF weighting).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModel, AutoTokenizer


def _chunks(values: Sequence[str], size: int) -> Iterable[tuple[int, Sequence[str]]]:
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


class LocalBERTScorer:
    """BERTScore-compatible local scorer with explicit offline behavior."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        dtype: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
        model_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }.get(dtype)
        if dtype != "auto" and model_dtype is None:
            raise ValueError(f"unknown dtype: {dtype}")
        if model_dtype is None:
            model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
            dtype=model_dtype,
        ).to(self.device)
        self.model.eval()

        special_ids = set(self.tokenizer.all_special_ids or [])
        self._special_ids = special_ids

    def _encode(self, texts: Sequence[str]) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        with torch.inference_mode():
            for _, batch in _chunks(texts, self.batch_size):
                encoded = self.tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                hidden = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                for row in range(hidden.shape[0]):
                    keep = attention_mask[row].bool()
                    if self._special_ids:
                        special = torch.zeros_like(keep)
                        for token_id in self._special_ids:
                            special |= input_ids[row].eq(token_id)
                        keep &= ~special
                    values = hidden[row][keep]
                    if values.shape[0] == 0:
                        values = hidden[row, :1]
                    result.append(torch.nn.functional.normalize(values, p=2, dim=-1))
        return result

    @staticmethod
    def _pair(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float, float]:
        similarity = candidate @ reference.transpose(0, 1)
        precision = similarity.max(dim=1).values.mean()
        recall = similarity.max(dim=0).values.mean()
        denominator = precision + recall
        f1 = (2.0 * precision * recall / denominator) if denominator > 0 else precision.new_zeros(())
        return float(precision.item()), float(recall.item()), float(f1.item())

    def score(self, candidates: Sequence[str], references: Sequence[str]) -> list[dict[str, float]]:
        if len(candidates) != len(references):
            raise ValueError("candidate/reference lengths differ")
        candidate_embeddings = self._encode(candidates)
        reference_embeddings = self._encode(references)
        return [
            dict(zip(("bertscore_p", "bertscore_r", "bertscore_f1"), self._pair(c, r)))
            for c, r in zip(candidate_embeddings, reference_embeddings)
        ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float16", "float32", "bfloat16"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    rows: list[dict] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{args.input}:{line_no}: expected object")
                rows.append(value)

    valid_indices = [
        i for i, row in enumerate(rows)
        if str(row.get("summary", "")).strip() and str(row.get("reference", "")).strip()
    ]
    scorer = LocalBERTScorer(
        str(args.model),
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    scores = scorer.score(
        [str(rows[i]["summary"]) for i in valid_indices],
        [str(rows[i]["reference"]) for i in valid_indices],
    )
    for index, score in zip(valid_indices, scores):
        rows[index].update(score)
        rows[index]["bertscore_model"] = str(args.model)
        rows[index]["bertscore_implementation"] = "local_token_cosine_no_idf"
        rows[index]["bertscore_max_length"] = args.max_length

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "scored_rows": len(valid_indices),
        "model": str(args.model),
        "implementation": "local_token_cosine_no_idf",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
