"""Run a small model-backed E0 position-relocation control.

The three documents contain the same unique evidence sentence at the beginning,
middle, or end of an otherwise repeated source.  The target model, prompt,
decoding and trace settings stay fixed; only the evidence position changes.
This is a confounder diagnostic, not an additional H1--H5 success criterion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .report import summarize_position_relocation, write_json
from .trace_target import (
    DEFAULT_INSTRUCTION,
    _tokenizer_ids,
    generate_target_trace,
    load_local_model,
    locate_subsequence,
    render_document_prompt,
)


EVIDENCE = (
    "EVIDENCE ANCHOR: The audited program delivered exactly eighty-seven "
    "percent of its planned service coverage in fiscal year twenty twenty-four."
)
FILLER = (
    "Background context describes the agency process, reporting schedule, "
    "review controls, budget assumptions, and implementation milestones. "
)


def relocation_documents() -> list[dict[str, str]]:
    """Return equal-structure documents with one anchor at three positions."""

    filler = FILLER * 18
    return [
        {"id": "e0_begin", "relocation_case": "begin", "document": EVIDENCE + "\n" + filler},
        {
            "id": "e0_middle",
            "relocation_case": "middle",
            "document": FILLER * 9 + EVIDENCE + "\n" + FILLER * 9,
        },
        {"id": "e0_end", "relocation_case": "end", "document": filler + EVIDENCE},
    ]


def _evidence_token_span(
    tokenizer: Any,
    document: str,
    evidence: str,
    source_token_ids: Sequence[int],
) -> tuple[int, int]:
    """Locate the anchor in source-token coordinates with offset fallback."""

    evidence_ids = _tokenizer_ids(tokenizer, evidence)
    source_start = locate_subsequence(source_token_ids, evidence_ids)
    if source_start is not None:
        return source_start, source_start + len(evidence_ids)

    char_start = document.index(evidence)
    char_end = char_start + len(evidence)
    content = DEFAULT_INSTRUCTION + document
    try:
        encoded = tokenizer(
            content,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"] if isinstance(encoded, Mapping) else encoded.offset_mapping
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        prefix_tokens = len(_tokenizer_ids(tokenizer, DEFAULT_INSTRUCTION))
        selected = [
            index - prefix_tokens
            for index, (start, end) in enumerate(offsets)
            if end > len(DEFAULT_INSTRUCTION) + char_start
            and start < len(DEFAULT_INSTRUCTION) + char_end
        ]
        if selected:
            return min(selected), max(selected) + 1
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    raise ValueError("could not locate evidence span in source token ids")


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty result directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_local_model(
        args.model, device=args.device, dtype=args.dtype
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for record in relocation_documents():
        sample_id = record["id"]
        try:
            rendered = render_document_prompt(tokenizer, record["document"])
            row = generate_target_trace(
                model,
                tokenizer,
                rendered,
                sample_id=sample_id,
                document_id=sample_id,
                max_new_tokens=args.max_new_tokens,
                chunk_size=args.chunk_size,
                skip_source_tokens=args.skip_source_tokens,
                device=device,
                prefill_chunk_size=args.prefill_chunk_size,
                sensitivity_chunk_sizes=(args.chunk_size,),
                sink_sizes=(args.skip_source_tokens,),
            )
            span_start, span_end = _evidence_token_span(
                tokenizer,
                record["document"],
                EVIDENCE,
                row["source_token_ids"],
            )
            row.update({
                "relocation_case": record["relocation_case"],
                "evidence_token_start": span_start,
                "evidence_token_end": span_end,
                "evidence_skip_source_tokens": args.skip_source_tokens,
                "evidence_chunk_size": args.chunk_size,
                "evidence_text": EVIDENCE,
            })
        except Exception as exc:
            row = {
                "schema_version": "groundsync.target.v1",
                "status": "error",
                "sample_id": sample_id,
                "document_id": sample_id,
                "relocation_case": record["relocation_case"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(f"e0_relocation {len(rows)}/3 case={record['relocation_case']}", flush=True)

    trace_path = output_dir / "target_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_position_relocation(
        rows,
        variants=(
            f"raw_chunk_{args.chunk_size}",
            f"nosink_{args.skip_source_tokens}_chunk_{args.chunk_size}",
        ),
    )
    write_json(output_dir / "e0_position_relocation.json", summary)
    write_json(output_dir / "run_manifest.json", {
        "schema_version": "groundsync.e0.manifest.v1",
        "experiment": "position_relocation",
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "requested_cases": 3,
        "ok_cases": sum(row.get("status") == "ok" for row in rows),
        "max_new_tokens": args.max_new_tokens,
        "chunk_size": args.chunk_size,
        "skip_source_tokens": args.skip_source_tokens,
        "prefill_chunk_size": args.prefill_chunk_size,
        "elapsed_s": time.perf_counter() - started,
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--skip-source-tokens", type=int, default=8)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
