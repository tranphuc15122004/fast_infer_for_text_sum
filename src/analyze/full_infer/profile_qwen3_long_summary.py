#!/usr/bin/env python3
"""Profile single-model Qwen3-4B long-summary inference.

The profiler deliberately avoids speculative decoding.  It measures one
target model at a time and reports the phases that are observable from the
Transformers forward API:

* tokenization and CPU->GPU input transfer;
* prefill, which includes the first forward pass and creation/writes of the
  Hugging Face KV cache;
* the first one-token decode, where the existing KV cache is first read;
* remaining decode steps and post-processing;
* peak allocated/reserved GPU memory and the KV-cache tensor size.

The first decode step is reported separately because Transformers does not
expose a standalone "KV-cache load" event.  A cache is read as part of that
one-token forward, so calling it an independent kernel time would be
misleading.  The output includes both raw per-repeat JSONL and median summary
CSV/JSON files plus PNG charts generated with Pillow (no matplotlib needed).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_WORD_MARKS = (256, 512, 1024, 2048, 3072)
PLOT_COMPONENTS = (
    "tokenize_ms",
    "input_transfer_ms",
    "prefill_ms",
    "kv_cache_first_read_ms",
    "decode_rest_ms",
    "postprocess_ms",
    "unattributed_ms",
)


def truncate_words(text: str, max_words: int) -> str:
    """Return a deterministic whitespace-normalized prefix of ``max_words``."""

    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    return " ".join(str(text).split()[:max_words])


def select_source_row(rows: Sequence[Mapping[str, Any]], word_mark: int) -> Mapping[str, Any]:
    """Choose the shortest source document that can cover a word mark."""

    if not rows:
        raise ValueError("source dataset is empty")
    if word_mark <= 0:
        raise ValueError("word_mark must be > 0")
    ranked = sorted(
        rows,
        key=lambda row: (len(str(row.get("document", "")).split()), str(row.get("id", ""))),
    )
    for row in ranked:
        if len(str(row.get("document", "")).split()) >= word_mark:
            return row
    return ranked[-1]


def component_ratios(components: Mapping[str, float]) -> dict[str, float]:
    """Normalize measured phase times to fractions of the profiled total."""

    values = {name: max(float(components.get(name, 0.0)), 0.0) for name in PLOT_COMPONENTS}
    # Accept compact callers that only have an aggregate decode measurement.
    # The runtime profiler supplies the more useful first-read/rest split.
    if (
        values["kv_cache_first_read_ms"] == 0.0
        and values["decode_rest_ms"] == 0.0
        and float(components.get("decode_ms", 0.0)) > 0.0
    ):
        values["decode_rest_ms"] = float(components["decode_ms"])
    total = sum(values.values())
    if total <= 0:
        return {name: 0.0 for name in PLOT_COMPONENTS}
    ratios = {name: value / total for name, value in values.items()}
    # Keep the serialized sum stable after floating-point formatting.
    last = PLOT_COMPONENTS[-1]
    ratios[last] = 1.0 - sum(ratios[name] for name in PLOT_COMPONENTS[:-1])
    return ratios


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            if not str(value.get("document", "")).strip():
                continue
            rows.append(value)
    if not rows:
        raise ValueError(f"No usable documents found in {path}")
    return rows


def _sync_cuda(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def cache_nbytes(cache: Any, torch: Any) -> int:
    """Best-effort byte count for tuple and DynamicCache implementations."""

    seen: set[int] = set()

    def visit(value: Any) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(value, torch.Tensor):
            return int(value.numel() * value.element_size())
        if isinstance(value, Mapping):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        total = 0
        for attribute in ("key_cache", "value_cache", "layers", "keys", "values", "key", "value"):
            if hasattr(value, attribute):
                total += visit(getattr(value, attribute))
        return total

    return visit(cache)


def _model_call(model: Any, torch: Any, **kwargs: Any) -> Any:
    """Call Qwen3 while keeping compatibility with older Transformers APIs."""

    try:
        return model(logits_to_keep=1, **kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        return model(**kwargs)


def _tokenize_prompt(tokenizer: Any, prompt: str, torch: Any) -> Any:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )


def measure_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    """Measure one greedy generation after the model has been loaded."""

    total_start = time.perf_counter()

    tokenize_start = time.perf_counter()
    input_ids = _tokenize_prompt(tokenizer, prompt, torch)
    tokenize_ms = (time.perf_counter() - tokenize_start) * 1000.0
    if isinstance(input_ids, Mapping):
        input_ids = input_ids["input_ids"]

    transfer_start = time.perf_counter()
    input_ids = input_ids.to(device)
    _sync_cuda(torch, device)
    input_transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        _sync_cuda(torch, device)
        prefill_start = time.perf_counter()
        outputs = _model_call(
            model,
            torch,
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        _sync_cuda(torch, device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

        past_key_values = outputs.past_key_values
        cache_bytes = cache_nbytes(past_key_values, torch)
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated = [next_token]
        eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
        first_decode_ms = 0.0
        decode_rest_ms = 0.0
        decode_forward_tokens = 0

        if int(next_token[0, 0]) in eos_ids:
            decode_token_count = 1
        else:
            decode_token_count = 1
            for step in range(1, max_new_tokens):
                _sync_cuda(torch, device)
                decode_start = time.perf_counter()
                step_outputs = _model_call(
                    model,
                    torch,
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                _sync_cuda(torch, device)
                elapsed_ms = (time.perf_counter() - decode_start) * 1000.0
                if step == 1:
                    first_decode_ms = elapsed_ms
                else:
                    decode_rest_ms += elapsed_ms
                decode_forward_tokens += 1
                past_key_values = step_outputs.past_key_values
                next_token = step_outputs.logits[:, -1:, :].argmax(dim=-1)
                generated.append(next_token)
                decode_token_count += 1
                if int(next_token[0, 0]) in eos_ids:
                    break

        decode_ms = first_decode_ms + decode_rest_ms
        output_ids = torch.cat(generated, dim=1)
        postprocess_start = time.perf_counter()
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0
    measured = (
        tokenize_ms
        + input_transfer_ms
        + prefill_ms
        + first_decode_ms
        + decode_rest_ms
        + postprocess_ms
    )
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

    return {
        "status": "ok",
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": decode_token_count,
        "tokenize_ms": tokenize_ms,
        "input_transfer_ms": input_transfer_ms,
        "prefill_ms": prefill_ms,
        "kv_cache_build_ms": prefill_ms,
        "kv_cache_first_read_ms": first_decode_ms,
        "decode_rest_ms": decode_rest_ms,
        "decode_ms": decode_ms,
        "decode_forward_tokens": decode_forward_tokens,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
        "unattributed_ms": max(total_ms - measured, 0.0),
        "kv_cache_bytes": cache_bytes,
        "kv_cache_mb": cache_bytes / (1024**2),
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "summary_text": decoded,
    }


def _median_row(rows: Sequence[Mapping[str, Any]], *, word_mark: int) -> dict[str, Any]:
    numeric_fields = (
        "input_tokens",
        "output_tokens",
        "tokenize_ms",
        "input_transfer_ms",
        "prefill_ms",
        "kv_cache_build_ms",
        "kv_cache_first_read_ms",
        "decode_rest_ms",
        "decode_ms",
        "decode_forward_tokens",
        "postprocess_ms",
        "total_ms",
        "unattributed_ms",
        "kv_cache_bytes",
        "kv_cache_mb",
        "peak_allocated_mb",
        "peak_reserved_mb",
    )
    result: dict[str, Any] = {"word_mark": word_mark, "status": "ok"}
    for field in numeric_fields:
        result[field] = statistics.median(float(row[field]) for row in rows)
    result["component_ratios"] = component_ratios(result)
    result["repeats_ok"] = len(rows)
    return result


def _font(size: int) -> Any:
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _write_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    percent: bool = False,
    ylabel: str = "ms",
) -> None:
    """Draw a dependency-free PNG chart using Pillow."""

    from PIL import Image, ImageDraw

    width, height = 1600, 900
    left, top, right, bottom = 120, 95, 350, 135
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(30)
    label_font = _font(22)
    small_font = _font(18)
    draw.text((left, 25), title, fill=(25, 25, 25), font=title_font)

    plot_left, plot_top = left, top
    plot_right, plot_bottom = width - right, height - bottom
    values = list(series.values())
    max_value = 1.0 if percent else max((sum(row) for row in zip(*values)), default=1.0)
    if not percent:
        max_value = max(max_value * 1.15, 1.0)
    else:
        max_value = 1.0

    def y_for(value: float) -> int:
        return plot_bottom - int((value / max_value) * (plot_bottom - plot_top))

    for tick in range(0, 6):
        value = max_value * tick / 5
        y = y_for(value)
        draw.line((plot_left, y, plot_right, y), fill=(225, 225, 225), width=1)
        label = f"{value * 100:.0f}%" if percent else f"{value:.0f}"
        draw.text((20, y - 12), label, fill=(80, 80, 80), font=small_font)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(50, 50, 50), width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(50, 50, 50), width=2)

    palette = [
        (52, 101, 164),
        (87, 160, 211),
        (231, 138, 76),
        (224, 90, 90),
        (116, 168, 85),
        (156, 112, 180),
        (150, 150, 150),
    ]
    bar_width = max(25, int((plot_right - plot_left) / max(len(labels) * 1.6, 1)))
    step = (plot_right - plot_left) / max(len(labels), 1)
    for index, label in enumerate(labels):
        x_center = int(plot_left + step * (index + 0.5))
        if percent:
            cumulative = 0.0
            for color_index, (name, values_for_name) in enumerate(series.items()):
                value = float(values_for_name[index])
                y0, y1 = y_for(cumulative), y_for(cumulative + value)
                draw.rectangle(
                    (x_center - bar_width // 2, y1, x_center + bar_width // 2, y0),
                    fill=palette[color_index % len(palette)],
                )
                cumulative += value
        else:
            cumulative = 0.0
            for color_index, (name, values_for_name) in enumerate(series.items()):
                value = float(values_for_name[index])
                y0, y1 = y_for(cumulative), y_for(cumulative + value)
                draw.rectangle(
                    (x_center - bar_width // 2, y1, x_center + bar_width // 2, y0),
                    fill=palette[color_index % len(palette)],
                )
                cumulative += value
        draw.text((x_center - 35, plot_bottom + 18), label, fill=(50, 50, 50), font=label_font)

    legend_x = plot_right + 35
    legend_y = plot_top + 20
    for color_index, name in enumerate(series):
        y = legend_y + color_index * 40
        draw.rectangle((legend_x, y, legend_x + 24, y + 24), fill=palette[color_index % len(palette)])
        draw.text((legend_x + 35, y - 2), name.replace("_ms", ""), fill=(50, 50, 50), font=small_font)
    draw.text((plot_left, height - 55), "Input length (words)", fill=(70, 70, 70), font=label_font)
    draw.text((plot_right - 40, plot_bottom + 62), ylabel, fill=(70, 70, 70), font=small_font)
    image.save(path)


def _write_line_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    ylabel: str,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1600, 900
    left, top, right, bottom = 120, 95, 350, 135
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 25), title, fill=(25, 25, 25), font=_font(30))
    plot_left, plot_top = left, top
    plot_right, plot_bottom = width - right, height - bottom
    ymax = max((max(values) for values in series.values() if values), default=1.0) * 1.15
    ymax = max(ymax, 1.0)

    def point(index: int, value: float) -> tuple[int, int]:
        x = plot_left + int((plot_right - plot_left) * index / max(len(labels) - 1, 1))
        y = plot_bottom - int(value / ymax * (plot_bottom - plot_top))
        return x, y

    for tick in range(0, 6):
        value = ymax * tick / 5
        y = plot_bottom - int(value / ymax * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill=(225, 225, 225), width=1)
        draw.text((20, y - 12), f"{value:.0f}", fill=(80, 80, 80), font=_font(18))
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(50, 50, 50), width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(50, 50, 50), width=2)
    palette = [(52, 101, 164), (231, 138, 76), (116, 168, 85), (156, 112, 180)]
    for color_index, (name, values) in enumerate(series.items()):
        points = [point(i, float(value)) for i, value in enumerate(values)]
        if len(points) > 1:
            draw.line(points, fill=palette[color_index % len(palette)], width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=palette[color_index % len(palette)])
    for index, label in enumerate(labels):
        x, _ = point(index, 0)
        draw.text((x - 35, plot_bottom + 18), label, fill=(50, 50, 50), font=_font(22))
    legend_x = plot_right + 35
    for color_index, name in enumerate(series):
        y = plot_top + 20 + color_index * 40
        draw.line((legend_x, y + 12, legend_x + 25, y + 12), fill=palette[color_index % len(palette)], width=5)
        draw.text((legend_x + 35, y - 2), name, fill=(50, 50, 50), font=_font(18))
    draw.text((plot_left, height - 55), "Input length (words)", fill=(70, 70, 70), font=_font(22))
    draw.text((plot_right - 40, plot_bottom + 62), ylabel, fill=(70, 70, 70), font=_font(18))
    image.save(path)


def write_plots(
    summary_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    model_load_ms: float | None = None,
) -> list[Path]:
    """Generate the PNG artifacts requested for the analysis."""

    valid = [row for row in summary_rows if row.get("status") == "ok"]
    if not valid:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(int(row["word_mark"])) for row in valid]
    absolute = {
        name: [float(row.get(name, 0.0)) for row in valid]
        for name in PLOT_COMPONENTS
    }
    ratios = {
        name: [float(row.get("component_ratios", {}).get(name, 0.0)) for row in valid]
        for name in PLOT_COMPONENTS
    }
    paths = [
        output_dir / "phase_time_stacked.png",
        output_dir / "phase_share_100pct.png",
        output_dir / "phase_time_by_length.png",
        output_dir / "memory_by_length.png",
    ]
    _write_chart(
        paths[0],
        "Qwen3-4B inference phase time by input length",
        labels,
        absolute,
        ylabel="milliseconds",
    )
    _write_chart(
        paths[1],
        "Qwen3-4B inference phase share",
        labels,
        ratios,
        percent=True,
        ylabel="share",
    )
    _write_line_chart(
        paths[2],
        "Qwen3-4B phase time trends",
        labels,
        {
            "prefill": absolute["prefill_ms"],
            "KV first read": absolute["kv_cache_first_read_ms"],
            "decode rest": absolute["decode_rest_ms"],
        },
        ylabel="milliseconds",
    )
    _write_line_chart(
        paths[3],
        "Qwen3-4B memory by input length",
        labels,
        {
            "peak allocated MB": [float(row.get("peak_allocated_mb", 0.0)) for row in valid],
            "peak reserved MB": [float(row.get("peak_reserved_mb", 0.0)) for row in valid],
            "KV cache MB": [float(row.get("kv_cache_mb", 0.0)) for row in valid],
        },
        ylabel="MB",
    )
    if model_load_ms is not None:
        model_path = output_dir / "model_load_vs_sample_total.png"
        _write_line_chart(
            model_path,
            "One-time model load vs per-sample total",
            labels,
            {
                "model load (one-time)": [float(model_load_ms)] * len(valid),
                "sample total": [float(row.get("total_ms", 0.0)) for row in valid],
            },
            ylabel="milliseconds",
        )
        paths.append(model_path)
    return paths


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "word_mark", "status", "input_tokens", "output_tokens", "repeats_ok",
        "tokenize_ms", "input_transfer_ms", "prefill_ms", "kv_cache_build_ms",
        "kv_cache_first_read_ms", "decode_rest_ms", "decode_ms", "postprocess_ms",
        "total_ms", "unattributed_ms", "kv_cache_mb", "peak_allocated_mb",
        "peak_reserved_mb",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def run_profile(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-4B profiler requires a visible CUDA GPU")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This profiler is intended for a CUDA device")

    source_rows = _load_jsonl(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    _sync_cuda(torch, device)
    model_load_ms = (time.perf_counter() - model_load_start) * 1000.0
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Warm up the same single-model path once; model load is still reported
    # separately and never folded into per-sample percentages.
    warmup_ids = _tokenize_prompt(tokenizer, "Summarize: warmup", torch).to(device)
    for _ in range(max(args.warmup_runs, 0)):
        with torch.inference_mode():
            _ = _model_call(model, torch, input_ids=warmup_ids, use_cache=True, return_dict=True)
        _sync_cuda(torch, device)
    del warmup_ids
    gc.collect()
    torch.cuda.empty_cache()

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for word_mark in args.word_marks:
        source = select_source_row(source_rows, word_mark)
        source_text = str(source["document"])
        effective_words = min(word_mark, len(source_text.split()))
        text = truncate_words(source_text, effective_words)
        prompt = (
            "Summarize the following document faithfully and concisely. "
            "Return only the summary.\n\nDocument:\n" + text
        )
        ok_rows: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            gc.collect()
            torch.cuda.empty_cache()
            try:
                measured = measure_one(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    torch=torch,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raw_rows.append({
                    "word_mark": word_mark,
                    "effective_words": effective_words,
                    "source_id": source.get("id"),
                    "repeat": repeat,
                    "status": "oom",
                    "error": str(exc).splitlines()[-1] if str(exc) else "CUDA out of memory",
                })
                break
            measured.update({
                "word_mark": word_mark,
                "effective_words": effective_words,
                "source_id": source.get("id"),
                "repeat": repeat,
            })
            raw_rows.append(measured)
            ok_rows.append(measured)
        if ok_rows:
            summary = _median_row(ok_rows, word_mark=word_mark)
            summary.update({
                "effective_words": effective_words,
                "source_id": source.get("id"),
                "status": "ok",
            })
        else:
            summary = {
                "word_mark": word_mark,
                "effective_words": effective_words,
                "source_id": source.get("id"),
                "status": "oom",
                "repeats_ok": 0,
            }
        summary_rows.append(summary)

    metadata = {
        "model": args.model,
        "device": str(device),
        "dtype": "float16",
        "attn_implementation": args.attn_implementation,
        "input_file": args.input,
        "word_marks": args.word_marks,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "model_load_ms": model_load_ms,
        "phase_semantics": {
            "prefill_ms": "first full-input forward; includes KV cache creation/writes",
            "kv_cache_first_read_ms": "first one-token decode forward using existing KV cache",
            "decode_rest_ms": "remaining one-token decode forwards",
            "ratios": "normalized over measured per-sample phases, excluding model_load_ms",
        },
    }
    _write_jsonl(output_dir / "measurements.jsonl", raw_rows)
    _write_jsonl(output_dir / "summary.jsonl", summary_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_paths = write_plots(summary_rows, output_dir, model_load_ms=model_load_ms)
    print(json.dumps({"metadata": metadata, "plots": [str(path) for path in plot_paths],
                      "summary": summary_rows}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--input", default="data/representative_100/govreport_representative.jsonl")
    parser.add_argument("--output-dir", default="src/analyze/full_infer/results")
    parser.add_argument("--word-marks", type=int, nargs="+", default=list(DEFAULT_WORD_MARKS))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "eager"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_profile(args))


if __name__ == "__main__":
    main()
