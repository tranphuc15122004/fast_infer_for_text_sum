"""GPU semantic source-support diagnostic for the recorded DFlash lattice.

This is intentionally a frozen-lattice analysis.  It embeds source chunks and
the decoded text of each recorded candidate token, then attaches the maximum
cosine support to each row.  It does not add candidates, run DFlash2, or train
an extra selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from .io import read_trace_jsonl, write_metrics_bundle
from .report import render_markdown_report
from .source_disambiguation import analyze_source_ladder


def _load_records(paths: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("id", record.get("sample_id", record.get("document_id")))
            if sample_id is not None:
                records[str(sample_id)] = record
    return records


def _chunks(text: str, words_per_chunk: int, stride: int, max_chunks: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    starts = list(range(0, max(1, len(words) - words_per_chunk + 1), stride))
    final_start = max(0, len(words) - words_per_chunk)
    if starts[-1] != final_start:
        starts.append(final_start)
    chunks = [" ".join(words[start:start + words_per_chunk]) for start in starts]
    if len(chunks) > max_chunks:
        positions = torch.linspace(0, len(chunks) - 1, max_chunks).round().long().tolist()
        chunks = [chunks[index] for index in positions]
    return chunks


@torch.inference_mode()
def _embed(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        outputs.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, int(model.config.hidden_size)))


def run_semantic_diagnostic(
    *,
    trace: str,
    source_jsonl: Sequence[str],
    target_tokenizer_path: str,
    encoder_path: str,
    output: str,
    dataset_filter: str | None = None,
    device_name: str = "cuda",
    lambda_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    words_per_chunk: int = 96,
    chunk_stride: int = 72,
    max_chunks_per_sample: int = 128,
    candidate_batch_size: int = 128,
    source_batch_size: int = 64,
) -> dict[str, Any]:
    rows = [row for row in read_trace_jsonl(trace, dataset_filter=dataset_filter) if row.get("status") == "ok"]
    records = _load_records(source_jsonl)
    missing = sorted({str(row.get("sample_id")) for row in rows} - set(records))
    if missing:
        raise ValueError(f"missing source records for {len(missing)} trace samples")
    target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_path, local_files_only=True)
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    model = AutoModel.from_pretrained(encoder_path, local_files_only=True)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()

    candidate_ids = sorted({int(token) for row in rows for token in row["candidate_token_ids"]})
    candidate_texts = []
    for token in candidate_ids:
        decoded = target_tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
        candidate_texts.append(decoded if decoded else f"token_{token}")
    candidate_embeddings = _embed(
        candidate_texts,
        encoder_tokenizer,
        model,
        device,
        batch_size=candidate_batch_size,
        max_length=32,
    )
    candidate_lookup = {token: index for index, token in enumerate(candidate_ids)}

    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    sample_metadata: dict[str, Any] = {}
    for sample_id, sample_rows in sorted(rows_by_sample.items()):
        record = records[sample_id]
        text = record.get("document", record.get("context", record.get("text", "")))
        source_chunks = _chunks(text, words_per_chunk, chunk_stride, max_chunks_per_sample)
        source_embeddings = _embed(
            source_chunks,
            encoder_tokenizer,
            model,
            device,
            batch_size=source_batch_size,
            max_length=256,
        )
        sample_candidates = sorted({int(token) for row in sample_rows for token in row["candidate_token_ids"]})
        candidate_indices = torch.tensor([candidate_lookup[token] for token in sample_candidates], dtype=torch.long)
        support = (candidate_embeddings[candidate_indices] @ source_embeddings.T).max(dim=1).values.tolist()
        support_by_token = dict(zip(sample_candidates, support))
        for row in sample_rows:
            row["candidate_source_semantic_scores"] = [support_by_token[int(token)] for token in row["candidate_token_ids"]]
        sample_metadata[sample_id] = {
            "source_chunks": len(source_chunks),
            "trace_rows": len(sample_rows),
        }

    metrics = analyze_source_ladder(rows, lambda_values=lambda_values)
    metrics["experiment"] = "E7_semantic"
    metrics["selector_scope"] = "frozen_recorded_top16_lattice"
    metrics["semantic_encoder"] = {
        "path": encoder_path,
        "device": str(device),
        "candidate_tokens": len(candidate_ids),
        "samples": len(sample_metadata),
        "words_per_chunk": words_per_chunk,
        "chunk_stride": chunk_stride,
        "max_chunks_per_sample": max_chunks_per_sample,
    }
    metrics["semantic_sample_metadata"] = sample_metadata
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_bundle(output_dir / "e7", metrics, report=render_markdown_report("e7_semantic", metrics))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--source-jsonl", action="append", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-filter", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambda-values", type=lambda value: tuple(float(item) for item in value.split(",")), default=(0.0, 0.25, 0.5, 1.0, 2.0))
    args = parser.parse_args()
    metrics = run_semantic_diagnostic(
        trace=args.trace,
        source_jsonl=args.source_jsonl,
        target_tokenizer_path=args.target_tokenizer,
        encoder_path=args.encoder,
        output=args.output,
        dataset_filter=args.dataset_filter,
        device_name=args.device,
        lambda_values=args.lambda_values,
    )
    print(json.dumps({"status": metrics.get("status"), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
