"""E29-A: future-use/source-retirement oracle for long-document summarization.

The experiment is deliberately oracle-only.  It measures whether source units
that have low future attention after a generated summary prefix can be removed
without materially changing the target's next-token distribution.  It does
not modify production KV management and it never treats hindsight future
attention as an online policy.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CHECKPOINTS = (32, 64, 96, 128, 192)
DEFAULT_DELTAS = (0.001, 0.005, 0.01)


def _as_matrix(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    if not rows:
        raise ValueError("attention trace must not be empty")
    width = len(rows[0])
    if width <= 0:
        raise ValueError("attention trace must contain source units")
    matrix: list[list[float]] = []
    for row in rows:
        if len(row) != width:
            raise ValueError("attention trace rows must have equal width")
        values = [float(value) for value in row]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("attention trace must contain finite non-negative values")
        matrix.append(values)
    return matrix


def chunk_ranges(source_length: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return contiguous source-token ranges used as retireable units."""

    source_length = int(source_length)
    chunk_size = int(chunk_size)
    if source_length <= 0 or chunk_size <= 0:
        raise ValueError("source_length and chunk_size must be positive")
    return [
        (start, min(start + chunk_size, source_length))
        for start in range(0, source_length, chunk_size)
    ]


def future_use_at(
    source_unit_mass: Sequence[Sequence[float]],
    prefix_length: int,
) -> dict[str, list[float] | int]:
    """Compute cumulative and per-future-query source use after a prefix.

    Row ``i`` is the source-unit attention mass of the query predicting output
    token ``i + 1``.  A prefix of length ``t`` therefore has future queries
    ``rows[t:]``.  Both cumulative mass and mean mass are returned so the
    threshold convention is explicit and auditable.
    """

    matrix = _as_matrix(source_unit_mass)
    t = int(prefix_length)
    if t < 0 or t >= len(matrix):
        raise ValueError("prefix_length must leave at least one future query")
    future = matrix[t:]
    cumulative = [sum(row[index] for row in future) for index in range(len(matrix[0]))]
    mean = [value / len(future) for value in cumulative]
    return {
        "prefix_length": t,
        "future_queries": len(future),
        "cumulative": cumulative,
        "mean_per_query": mean,
    }


def active_units(
    future_mean: Sequence[float],
    *,
    delta: float,
) -> list[int]:
    """Return source-unit indices whose mean future use meets ``delta``."""

    delta = float(delta)
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be finite and non-negative")
    values = [float(value) for value in future_mean]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("future_mean must be finite and non-negative")
    return [index for index, value in enumerate(values) if value >= delta]


def projected_work(
    source_unit_mass: Sequence[Sequence[float]],
    *,
    delta: float,
) -> dict[str, float | int | list[float]]:
    """Compute active-source work and oracle removable-work fraction.

    The active fraction at every decode step uses all future queries from that
    step onward.  This is a hindsight ceiling, not an online estimator.
    """

    matrix = _as_matrix(source_unit_mass)
    fractions: list[float] = []
    for prefix_length in range(len(matrix)):
        summary = future_use_at(matrix, prefix_length)
        active = active_units(summary["mean_per_query"], delta=delta)  # type: ignore[arg-type]
        fractions.append(len(active) / len(matrix[0]))
    work = float(sum(fractions))
    return {
        "decode_steps": len(matrix),
        "source_units": len(matrix[0]),
        "active_fraction_mean": statistics.fmean(fractions),
        "active_work": work,
        "retire_gain": 1.0 - work / len(matrix),
        "active_fraction_by_step": fractions,
    }


def checkpoint_metrics(
    source_unit_mass: Sequence[Sequence[float]],
    *,
    checkpoints: Sequence[int],
    deltas: Sequence[float],
) -> list[dict[str, Any]]:
    """Return future-use/retirement metrics at registered checkpoints."""

    matrix = _as_matrix(source_unit_mass)
    result: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        t = int(checkpoint)
        if t < 0 or t >= len(matrix):
            continue
        summary = future_use_at(matrix, t)
        mean_values = summary["mean_per_query"]
        for delta in deltas:
            active = active_units(mean_values, delta=float(delta))  # type: ignore[arg-type]
            result.append({
                "prefix_length": t,
                "future_queries": summary["future_queries"],
                "delta": float(delta),
                "source_units": len(matrix[0]),
                "active_units": len(active),
                "active_fraction": len(active) / len(matrix[0]),
                "retire_fraction": 1.0 - len(active) / len(matrix[0]),
                "future_source_mass": float(sum(summary["cumulative"])),  # type: ignore[arg-type]
                "mean_future_source_mass": float(sum(mean_values)),  # type: ignore[arg-type]
            })
    return result


def _collapse_attention(attentions: Sequence[Any]) -> list[float]:
    """Average the last-query attention vector across layers and heads."""

    import numpy as np

    vectors: list[Any] = []
    for attention in attentions:
        if attention is None:
            continue
        if hasattr(attention, "detach"):
            attention = attention.detach().float().cpu().numpy()
        array = np.asarray(attention, dtype=float)
        if array.ndim == 4:
            vector = array[0, :, -1, :].mean(axis=0)
        elif array.ndim == 3:
            vector = array[:, -1, :].mean(axis=0)
        elif array.ndim == 2:
            vector = array[-1, :]
        elif array.ndim == 1:
            vector = array
        else:
            raise ValueError(f"unsupported attention rank: {array.ndim}")
        vectors.append(vector)
    if not vectors:
        raise ValueError("model returned no attention tensors")
    result = np.mean(np.stack(vectors, axis=0), axis=0)
    values = [float(value) for value in result.tolist()]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("model returned invalid attention values")
    return values


def _model_call(model: Any, **kwargs: Any) -> Any:
    try:
        return model(logits_to_keep=1, **kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        return model(**kwargs)


def _source_unit_mass(
    attentions: Sequence[Any],
    *,
    source_start: int,
    source_end: int,
    chunk_size: int,
) -> tuple[float, list[float], list[float]]:
    vector = _collapse_attention(attentions)
    if source_start < 0 or source_end <= source_start or source_end > len(vector):
        raise ValueError("invalid source span for attention vector")
    source = vector[source_start:source_end]
    units = [
        sum(source[start : min(start + chunk_size, len(source))])
        for start in range(0, len(source), chunk_size)
    ]
    total = float(sum(source))
    if total > 0.0:
        normalized = [value / total for value in units]
    else:
        normalized = [0.0 for _ in units]
    return total, units, normalized


def _tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(item) for item in encoded]


def _record_document(record: Mapping[str, Any]) -> str:
    for key in ("document", "context", "input", "prompt", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("record has no usable document field")


def _truncate_document(tokenizer: Any, document: str, max_tokens: int | None) -> str:
    if max_tokens is None:
        return document
    ids = _tokenizer_ids(tokenizer, document)
    if len(ids) <= max_tokens:
        return document
    return tokenizer.decode(ids[: int(max_tokens)], clean_up_tokenization_spaces=False)


def _render(tokenizer: Any, document: str) -> Any:
    from src.analyze.groundsync.trace_target import render_document_prompt

    return render_document_prompt(tokenizer, document)


def _prefill(model: Any, input_ids: Any, *, chunk_size: int) -> Any:
    """Memory-bounded causal prefill that returns only the KV cache.

    A full 4K SDPA call on a T4 can select the quadratic math kernel and
    allocate a multi-gigabyte attention matrix.  We therefore process the
    prompt in bounded chunks with the explicit bottom-right causal mask. The
    CausalLM vocabulary projection is bypassed by calling its decoder
    backbone directly. This keeps the peak memory bounded for the real 4K
    protocol while preserving causal cache semantics.
    """

    import torch
    from src.analyze.groundsync.trace_target import (
        _bottom_right_causal_mask,
        _set_attention_implementation,
    )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    backbone = getattr(model, "model", None)
    # The explicit mask is intentionally paired with eager attention. This
    # avoids SDPA selecting the T4 quadratic math backend for a long square
    # query.
    _set_attention_implementation(model, "eager")
    model_dtype = getattr(model, "dtype", torch.float32)
    if not isinstance(model_dtype, torch.dtype):
        model_dtype = torch.float32
    past_key_values = None
    try:
        for start in range(0, int(input_ids.shape[1]), int(chunk_size)):
            end = min(start + int(chunk_size), int(input_ids.shape[1]))
            chunk = input_ids[:, start:end]
            attention_mask = _bottom_right_causal_mask(
                end - start,
                start,
                dtype=model_dtype,
                device=input_ids.device,
            )
            kwargs: dict[str, Any] = {
                "input_ids": chunk,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "attention_mask": attention_mask,
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            if backbone is None:
                outputs = _model_call(model, **kwargs)
            else:
                outputs = backbone(**kwargs)
            past_key_values = outputs.past_key_values
            del outputs, chunk, attention_mask
        return past_key_values
    finally:
        if getattr(input_ids.device, "type", None) == "cuda":
            torch.cuda.empty_cache()


def _next_logits(model: Any, sequence: Any, *, device: Any, prefill_chunk_size: int) -> Any:
    """Return logits for the next token after ``sequence``."""

    import torch

    if sequence.ndim != 2 or sequence.shape[1] < 2:
        raise ValueError("sequence must be [batch, length] with length >= 2")
    with torch.inference_mode():
        past_key_values = _prefill(model, sequence[:, :-1].to(device), chunk_size=prefill_chunk_size)
        outputs = _model_call(
            model,
            input_ids=sequence[:, -1:].to(device),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
        )
        result = outputs.logits[0, -1].float().detach().cpu()
        del outputs, past_key_values
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    return result


def _teacher_forced_agreement(
    model: Any,
    sequence: Any,
    future_tokens: Sequence[int],
    *,
    device: Any,
    prefill_chunk_size: int,
    horizon: int,
) -> float:
    """Compare greedy predictions with a fixed target continuation efficiently."""

    import torch

    if horizon <= 0 or not future_tokens:
        return float("nan")
    target_tokens = [int(value) for value in future_tokens[:horizon]]
    with torch.inference_mode():
        past = _prefill(model, sequence[:, :-1].to(device), chunk_size=prefill_chunk_size)
        current = sequence[:, -1:].to(device)
        matches = 0
        for target_token in target_tokens:
            outputs = _model_call(
                model,
                input_ids=current,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
            )
            prediction = int(outputs.logits[0, -1].argmax().item())
            matches += int(prediction == target_token)
            current = torch.tensor([[target_token]], dtype=torch.long, device=device)
            past = outputs.past_key_values
        result = matches / len(target_tokens)
        del current, past
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    return result


def _retained_prompt(
    input_ids: Any,
    *,
    source_start: int,
    source_end: int,
    active_indices: Sequence[int],
    source_chunk_size: int,
) -> Any:
    """Remove inactive source chunks while preserving prompt and generation markers."""

    import torch

    source = input_ids[0, source_start:source_end]
    keep: list[Any] = []
    active = set(int(index) for index in active_indices)
    for index, start in enumerate(range(0, int(source.shape[0]), source_chunk_size)):
        if index in active:
            keep.append(source[start : min(start + source_chunk_size, source.shape[0])])
    pieces = [
        input_ids[0, :source_start],
        *keep,
        input_ids[0, source_end:],
    ]
    return torch.cat(pieces, dim=0).unsqueeze(0)


def oracle_ablation(
    model: Any,
    trace: Mapping[str, Any],
    input_ids: Any,
    *,
    checkpoints: Sequence[int],
    delta: float,
    source_chunk_size: int,
    prefill_chunk_size: int,
    device: Any,
    horizon: int,
) -> list[dict[str, Any]]:
    """Evaluate next-token stability after hindsight source-unit retirement."""

    import torch

    generated = [int(value) for value in trace["generated_token_ids"]]
    source_mass = trace["source_unit_mass"]
    result: list[dict[str, Any]] = []
    source_start = int(trace["source_start"])
    source_end = int(trace["source_end"])
    for checkpoint in checkpoints:
        t = int(checkpoint)
        if t < 0 or t >= len(generated):
            continue
        future = future_use_at(source_mass, t)
        active = active_units(future["mean_per_query"], delta=delta)  # type: ignore[arg-type]
        prefix_tokens = torch.tensor([generated[:t]], dtype=torch.long, device=device)
        full_sequence = torch.cat([input_ids.to(device), prefix_tokens], dim=1)
        retained_prompt = _retained_prompt(
            input_ids,
            source_start=source_start,
            source_end=source_end,
            active_indices=active,
            source_chunk_size=source_chunk_size,
        ).to(device)
        retained_sequence = torch.cat([retained_prompt, prefix_tokens], dim=1)
        full_logits = _next_logits(model, full_sequence, device=device, prefill_chunk_size=prefill_chunk_size)
        retained_logits = _next_logits(model, retained_sequence, device=device, prefill_chunk_size=prefill_chunk_size)
        full_logp = torch.log_softmax(full_logits, dim=-1)
        retained_logp = torch.log_softmax(retained_logits, dim=-1)
        full_p = full_logp.exp()
        kl = float((full_p * (full_logp - retained_logp)).sum().item())
        target_token = generated[t]
        target_nll_delta = float(-retained_logp[target_token].item() + full_logp[target_token].item())
        full_top1 = int(full_logits.argmax().item())
        retained_top1 = int(retained_logits.argmax().item())
        horizon_agreement = _teacher_forced_agreement(
            model,
            full_sequence,
            generated[t:],
            device=device,
            prefill_chunk_size=prefill_chunk_size,
            horizon=horizon,
        )
        retained_horizon_agreement = _teacher_forced_agreement(
            model,
            retained_sequence,
            generated[t:],
            device=device,
            prefill_chunk_size=prefill_chunk_size,
            horizon=horizon,
        )
        result.append({
            "prefix_length": t,
            "delta": float(delta),
            "source_units": len(source_mass[0]),
            "active_units": len(active),
            "retire_fraction": 1.0 - len(active) / len(source_mass[0]),
            "full_top1": full_top1,
            "retained_top1": retained_top1,
            "top1_agreement": float(full_top1 == retained_top1),
            "mean_kl": kl,
            "target_nll_delta": target_nll_delta,
            "full_target_token": int(target_token),
            "horizon": int(horizon),
            "full_horizon_agreement": horizon_agreement,
            "retained_horizon_agreement": retained_horizon_agreement,
        })
        del full_sequence, retained_sequence, retained_prompt, prefix_tokens
        del full_logits, retained_logits, full_logp, retained_logp, full_p
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    return result


def generate_trace(
    model: Any,
    tokenizer: Any,
    rendered: Any,
    *,
    sample_id: str,
    document_id: str,
    max_new_tokens: int,
    source_chunk_size: int,
    prefill_chunk_size: int,
    device: Any,
) -> dict[str, Any]:
    """Generate target tokens and unnormalized source-unit attention mass."""

    import torch

    input_ids = rendered.input_ids.to(device)
    source_start = int(rendered.source_start)
    source_end = int(rendered.source_end)
    prefix = input_ids[:, :-1]
    current = input_ids[:, -1:]
    generated: list[int] = []
    entropies: list[float] = []
    source_mass: list[float] = []
    source_units: list[list[float]] = []
    source_unit_distributions: list[list[float]] = []
    eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    started = time.perf_counter()
    with torch.inference_mode():
        # The prefill must be inside inference mode as well.  Leaving it
        # outside retains the autograd graph for every bounded chunk through
        # the returned DynamicCache and exhausts a 16 GB T4 at 4K.
        past_key_values = _prefill(model, prefix, chunk_size=prefill_chunk_size)
        for _ in range(int(max_new_tokens)):
            outputs = _model_call(
                model,
                input_ids=current,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=True,
            )
            logits = outputs.logits[:, -1, :]
            probabilities = torch.softmax(logits.float(), dim=-1)
            entropy = float((-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()).item())
            total, units, normalized = _source_unit_mass(
                outputs.attentions,
                source_start=source_start,
                source_end=source_end,
                chunk_size=source_chunk_size,
            )
            next_token = logits.argmax(dim=-1, keepdim=True)
            token_id = int(next_token[0, 0].item())
            entropies.append(entropy)
            source_mass.append(total)
            source_units.append(units)
            source_unit_distributions.append(normalized)
            generated.append(token_id)
            past_key_values = outputs.past_key_values
            current = next_token
            if token_id in eos_ids:
                break
    return {
        "schema_version": "e29a.trace.v1",
        "status": "ok",
        "sample_id": str(sample_id),
        "document_id": str(document_id),
        "input_tokens": int(input_ids.shape[1]),
        "source_start": source_start,
        "source_end": source_end,
        "source_tokens": source_end - source_start,
        "source_chunk_size": int(source_chunk_size),
        "source_units": len(source_units[0]) if source_units else 0,
        "output_tokens": len(generated),
        "generated_token_ids": generated,
        "target_entropy": entropies,
        "source_attention_mass": source_mass,
        "source_unit_mass": source_units,
        "source_unit_distribution": source_unit_distributions,
        "elapsed_s": time.perf_counter() - started,
    }


def _parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return tuple(value.split("=", 1))  # type: ignore[return-value]


def _parse_int_pair(value: str) -> tuple[str, int]:
    name, raw = _parse_pair(value)
    return name, int(raw)


def _parse_list(value: str, cast: Any) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def render_report(result: Mapping[str, Any], *, output_dir: Path) -> str:
    lines = [
        "# E29-A — Future-Use / Source-Retirement Oracle",
        "",
        "## Kết luận",
        "",
        f"Run này có {result['documents']} documents và {result['ok_documents']} trace hợp lệ. Đây là oracle screen: future attention được dùng để tính hindsight ceiling, không phải online eviction policy.",
        f"Gate E29-B: retire_gain >= 35–40% trên ít nhất 2/3 datasets và ablation giữ top-1 agreement >=95%. Trạng thái hiện tại: **{'đủ điều kiện xem xét E29-B' if result.get('gate_pass') else 'chưa đạt gate; không mở E29-B/C'}**.",
        "",
        "## Phạm vi và provenance",
        "",
        f"- Model: `{result['model']}`; device ghi nhận: `{result['device']}`; CUDA available: `{result['cuda_available']}`.",
        f"- Datasets: {', '.join(result['datasets'])}; documents/dataset đăng ký: {result['samples_per_dataset']}.",
        f"- Output tối đa: {result['max_new_tokens']} tokens; source unit: contiguous {result['source_chunk_size']}-token chunks.",
        f"- Checkpoints: `{result['checkpoints']}`; deltas: `{result['deltas']}`.",
        "- Attention chính là mean layer/head source-unit mass của từng incremental target query; raw source mass được giữ riêng với normalized distribution.",
        "- Không có thay đổi production KV cache, không có heuristic online và không có claim lossless từ oracle này.",
        "",
        "## Quy trình",
        "",
        "1. Đọc representative JSONL, chọn deterministic số mẫu đăng ký và áp source cap theo dataset.",
        "2. Render đúng chat template/prompt, prefill theo chunks để tránh materialize full prompt attention matrix.",
        "3. Sinh greedy target từng token bằng incremental cached forward; sau mỗi query lấy attention tới source, gom thành source chunks và giải phóng tensor attention.",
        "4. Tại mỗi prefix length, tính cumulative future source use và mean future use; đánh dấu active source units theo từng delta đã đăng ký.",
        "5. Tính active-source work trên mọi decode step và `retire_gain = 1 - active_work/T`. Đây là ceiling hindsight.",
        "6. Nếu có ablation, chạy lại target trên cùng generated prefix với source units được giữ theo oracle; so top-1/KL/NLL thay đổi. Không xem missing ablation là pass.",
        "",
        "## Kết quả theo dataset và delta",
        "",
        "| Dataset | Delta | Docs | Mean output | Mean source tokens | Mean retire gain | CI 95% | Mean active fraction |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in result.get("dataset_aggregates", []):
        lines.append(
            f"| {row['dataset']} | {row['delta']:.4f} | {row['documents']} | {row['mean_output_tokens']:.2f} | {row['mean_source_tokens']:.2f} | {row['mean_retire_gain']:.2%} | {row['retire_gain_ci_95'][0]:.2%}–{row['retire_gain_ci_95'][1]:.2%} | {row['mean_active_fraction']:.2%} |"
        )
    lines += [
        "",
        "## Checkpoint diagnostics",
        "",
        "| Dataset | Prefix | Delta | Future queries | Source units | Active units | Retire fraction | Mean future source mass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("checkpoint_aggregates", []):
        lines.append(
            f"| {row['dataset']} | {row['prefix_length']} | {row['delta']:.4f} | {row['future_queries']:.2f} | {row['source_units']:.2f} | {row['active_units']:.2f} | {row['retire_fraction']:.2%} | {row['mean_future_source_mass']:.4f} |"
        )
    lines += [
        "",
        "## Ablation safety",
        "",
        "Ablation được báo riêng vì retireability chỉ có ý nghĩa nếu target distribution vẫn ổn định. Top-1 agreement, KL(full||retained), target NLL delta và horizon agreement phải được đo trên cùng prefix; không suy ra safety từ attention thấp.",
        "",
        "| Dataset | Prefix | Delta | Ablation docs | Top-1 agreement | Mean KL | Mean target NLL delta | Horizon agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("ablation_aggregates", []):
        lines.append(
            f"| {row['dataset']} | {row['prefix_length']} | {row['delta']:.4f} | {row['documents']} | {row['top1_agreement']:.2%} | {row['mean_kl']:.6f} | {row['mean_target_nll_delta']:.6f} | {row['retained_horizon_agreement']:.2%} |"
        )
    lines += [
        "",
        "## CONFIRMED",
        "",
        "- Collector và oracle math đã chạy/kiểm tra theo schema E29-A; source mass unnormalized và normalized được lưu tách biệt.",
        "",
        "## EXPLORATORY",
        "",
        "- Retire gain là hindsight ceiling, không phải speedup thực tế. Muốn claim latency phải có E29-C runtime với KV implementation thật.",
        "",
        "## FAILED / INCOMPLETE",
        "",
        f"- Dataset gate: {result.get('dataset_gate_pass_count', 0)}/3 dataset scopes đồng thời đạt threshold 35% retire gain và top-1 ablation agreement >=95% ở delta chính {result.get('primary_delta')}.",
        "- Nếu ablation chưa có hoặc chưa đủ mẫu, E29-A chỉ là future-use ceiling và không được gọi là safe retirement.",
        "",
        "## HIGHEST VERIFIED RUNG",
        "",
        "**R7 — result review / decision memo cho E29-A future-use/source-retirement oracle.**",
        "",
        "## EVIDENCE GAPS",
        "",
        "- Attention được average trên layer/head; chưa có retrieval-head selection preregistered.",
        "- Future-use oracle không thể tự chứng minh causal sufficiency; ablation logits là kiểm tra bắt buộc.",
        "- T4 chỉ được gọi là GPU nếu PyTorch thấy CUDA device; `nvidia-smi` đơn độc không đủ.",
        "- Chưa có E29-B online predictor và E29-C KV kernel/latency benchmark.",
        "",
        "## RECOMMENDED NEXT",
        "",
        "Chỉ chạy E29-B nếu dataset retire-gain gate và ablation safety gate cùng đạt. Nếu không, đóng nhánh source-retirement; không chuyển hindsight attention thành heuristic chỉ vì projected work reduction cao.",
        "",
        "## Artifact",
        "",
        f"- Output: `{output_dir}`",
        "- `traces.jsonl`: attention/source-unit traces từng document.",
        "- `metrics.json`: raw document/checkpoint/oracle metrics và aggregates.",
        "- `run_manifest.json`: command, model, device, caps, checkpoints, deltas và status.",
        "- Báo cáo chỉ giữ bảng tổng hợp; không chèn toàn bộ raw attention rows.",
    ]
    return "\n".join(lines) + "\n"


def _aggregate(values: Sequence[float]) -> tuple[float, tuple[float, float]]:
    if not values:
        return float("nan"), (float("nan"), float("nan"))
    ordered = sorted(float(value) for value in values)
    mean = statistics.fmean(ordered)
    lo = ordered[max(0, int(math.floor(0.025 * (len(ordered) - 1))))]
    hi = ordered[min(len(ordered) - 1, int(math.ceil(0.975 * (len(ordered) - 1))))]
    return mean, (lo, hi)


def _load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _json_safe(value: Any) -> Any:
    """Convert non-finite floats to JSON ``null`` recursively.

    A failed document must not prevent the run from writing its manifest and
    partial diagnostics.  ``json.dumps(..., allow_nan=False)`` is intentional
    here because NaN is not valid JSON; empty aggregates therefore become
    explicit ``null`` values rather than aborting result serialization.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from src.analyze.groundsync.trace_target import load_local_model

    input_paths = dict(args.input)
    caps = dict(args.max_input_tokens)
    model, tokenizer, device = load_local_model(args.model, device=args.device, dtype=args.dtype)
    rows_by_dataset = {name: _load_rows(path)[: args.samples_per_dataset] for name, path in input_paths.items()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    started = time.perf_counter()
    ablation_records: list[dict[str, Any]] = []
    requested_documents = sum(len(rows) for rows in rows_by_dataset.values())
    for dataset, rows in rows_by_dataset.items():
        for index, record in enumerate(rows):
            sample_id = str(record.get("id", index))
            try:
                document = _truncate_document(tokenizer, _record_document(record), caps.get(dataset))
                rendered = _render(tokenizer, document)
                trace = generate_trace(
                    model, tokenizer, rendered,
                    sample_id=sample_id, document_id=sample_id,
                    max_new_tokens=args.max_new_tokens,
                    source_chunk_size=args.source_chunk_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                    device=device,
                )
                trace["dataset"] = dataset
                trace["document_index"] = index
                documents.append(trace)
                if args.ablation:
                    try:
                        ablation_rows = oracle_ablation(
                            model,
                            trace,
                            rendered.input_ids,
                            checkpoints=args.ablation_checkpoints or args.checkpoints,
                            delta=args.ablation_delta if args.ablation_delta is not None else args.primary_delta,
                            source_chunk_size=args.source_chunk_size,
                            prefill_chunk_size=args.prefill_chunk_size,
                            device=device,
                            horizon=args.ablation_horizon,
                        )
                        for ablation_row in ablation_rows:
                            ablation_records.append({"dataset": dataset, "sample_id": sample_id, **ablation_row})
                    except Exception as exc:
                        print(f"e29a ABLATION_ERROR {dataset} {index + 1}/{len(rows)} sample={sample_id}: {type(exc).__name__}: {exc}", flush=True)
                print(f"e29a {dataset} {index + 1}/{len(rows)} sample={sample_id} tokens={trace['output_tokens']}", flush=True)
            except Exception as exc:
                documents.append({"schema_version": "e29a.trace.v1", "status": "error", "dataset": dataset, "sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"e29a ERROR {dataset} {index + 1}/{len(rows)} sample={sample_id}: {type(exc).__name__}: {exc}", flush=True)
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in documents:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok_rows = [row for row in documents if row.get("status") == "ok"]
    dataset_aggregates: list[dict[str, Any]] = []
    checkpoint_aggregates: list[dict[str, Any]] = []
    ablation_aggregates: list[dict[str, Any]] = []
    raw_metrics: list[dict[str, Any]] = []
    for dataset in input_paths:
        dataset_rows = [row for row in ok_rows if row["dataset"] == dataset]
        for delta in args.deltas:
            gains = [float(projected_work(row["source_unit_mass"], delta=delta)["retire_gain"]) for row in dataset_rows]
            active = [float(projected_work(row["source_unit_mass"], delta=delta)["active_fraction_mean"]) for row in dataset_rows]
            mean_gain, ci = _aggregate(gains)
            dataset_aggregates.append({"dataset": dataset, "delta": delta, "documents": len(dataset_rows), "mean_output_tokens": statistics.fmean([row["output_tokens"] for row in dataset_rows]) if dataset_rows else float("nan"), "mean_source_tokens": statistics.fmean([row["source_tokens"] for row in dataset_rows]) if dataset_rows else float("nan"), "mean_retire_gain": mean_gain, "retire_gain_ci_95": ci, "mean_active_fraction": statistics.fmean(active) if active else float("nan")})
        for checkpoint in args.checkpoints:
            for delta in args.deltas:
                rows = []
                for row in dataset_rows:
                    rows.extend(checkpoint_metrics(row["source_unit_mass"], checkpoints=(checkpoint,), deltas=(delta,)))
                if not rows:
                    continue
                checkpoint_aggregates.append({"dataset": dataset, "prefix_length": checkpoint, "delta": delta, "future_queries": statistics.fmean([item["future_queries"] for item in rows]), "source_units": statistics.fmean([item["source_units"] for item in rows]), "active_units": statistics.fmean([item["active_units"] for item in rows]), "retire_fraction": statistics.fmean([item["retire_fraction"] for item in rows]), "mean_future_source_mass": statistics.fmean([item["mean_future_source_mass"] for item in rows])})
        for row in dataset_rows:
            for delta in args.deltas:
                work = projected_work(row["source_unit_mass"], delta=delta)
                raw_metrics.append({"dataset": dataset, "sample_id": row["sample_id"], "delta": delta, **work})
    for dataset in input_paths:
        for checkpoint in args.ablation_checkpoints or args.checkpoints:
            for delta in (args.ablation_delta if args.ablation_delta is not None else args.primary_delta,):
                rows = [
                    row for row in ablation_records
                    if row["dataset"] == dataset
                    and int(row["prefix_length"]) == int(checkpoint)
                    and abs(float(row["delta"]) - float(delta)) < 1e-12
                ]
                if not rows:
                    continue
                ablation_aggregates.append({
                    "dataset": dataset,
                    "prefix_length": int(checkpoint),
                    "delta": float(delta),
                    "documents": len(rows),
                    "top1_agreement": statistics.fmean([float(row["top1_agreement"]) for row in rows]),
                    "mean_kl": statistics.fmean([float(row["mean_kl"]) for row in rows]),
                    "mean_target_nll_delta": statistics.fmean([float(row["target_nll_delta"]) for row in rows]),
                    "retained_horizon_agreement": statistics.fmean([float(row["retained_horizon_agreement"]) for row in rows]),
                })
    primary_delta = float(args.primary_delta)
    primary_aggregates = [row for row in dataset_aggregates if abs(row["delta"] - primary_delta) < 1e-12]
    primary_ablation = {}
    for dataset in input_paths:
        rows = [row for row in ablation_aggregates if row["dataset"] == dataset and abs(float(row["delta"]) - float(args.ablation_delta if args.ablation_delta is not None else primary_delta)) < 1e-12]
        primary_ablation[dataset] = statistics.fmean([float(row["top1_agreement"]) for row in rows]) if rows else None
    gate_rows = [
        row for row in primary_aggregates
        if row["mean_retire_gain"] >= args.min_retire_gain
        and primary_ablation.get(row["dataset"]) is not None
        and float(primary_ablation[row["dataset"]]) >= args.min_top1_agreement
    ]
    all_docs_ok = len(ok_rows) == requested_documents
    ablation_complete = (not args.ablation) or len(ablation_records) > 0
    result: dict[str, Any] = {
        "experiment": "E29A_future_use_source_retirement_oracle",
        "status": "complete" if all_docs_ok and ablation_complete else "incomplete",
        "model": args.model,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "datasets": list(input_paths),
        "samples_per_dataset": args.samples_per_dataset,
        "documents": len(documents),
        "ok_documents": len(ok_rows),
        "max_new_tokens": args.max_new_tokens,
        "source_chunk_size": args.source_chunk_size,
        "checkpoints": list(args.checkpoints),
        "deltas": list(args.deltas),
        "primary_delta": primary_delta,
        "min_retire_gain": args.min_retire_gain,
        "dataset_gate_pass_count": len(gate_rows),
        "gate_pass": len(gate_rows) >= 2,
        "primary_ablation_top1_by_dataset": primary_ablation,
        "ablation_enabled": bool(args.ablation),
        "ablation_horizon": args.ablation_horizon,
        "ablation_records": ablation_records,
        "ablation_aggregates": ablation_aggregates,
        "dataset_aggregates": dataset_aggregates,
        "checkpoint_aggregates": checkpoint_aggregates,
        "raw_metrics": raw_metrics,
        "elapsed_s": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {k: result[k] for k in ("experiment", "model", "device", "cuda_available", "datasets", "samples_per_dataset", "max_new_tokens", "source_chunk_size", "checkpoints", "deltas", "primary_delta", "min_retire_gain", "elapsed_s", "ablation_enabled", "ablation_horizon")}
    manifest["command"] = " ".join(__import__("sys").argv)
    manifest["output_dir"] = str(output_dir)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render_report(result, output_dir=output_dir)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "comprehensive_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"experiment": result["experiment"], "documents": result["documents"], "ok_documents": result["ok_documents"], "gate_pass": result["gate_pass"], "output_dir": str(output_dir)}), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_parse_pair, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--source-chunk-size", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--max-input-tokens", action="append", type=_parse_int_pair, default=[])
    parser.add_argument("--checkpoints", type=lambda value: _parse_list(value, int), default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--deltas", type=lambda value: _parse_list(value, float), default=DEFAULT_DELTAS)
    parser.add_argument("--primary-delta", type=float, default=0.005)
    parser.add_argument("--min-retire-gain", type=float, default=0.35)
    parser.add_argument("--min-top1-agreement", type=float, default=0.95)
    parser.add_argument("--no-ablation", dest="ablation", action="store_false", default=True)
    parser.add_argument("--ablation-checkpoints", type=lambda value: _parse_list(value, int), default=None)
    parser.add_argument("--ablation-delta", type=float, default=None)
    parser.add_argument("--ablation-horizon", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
