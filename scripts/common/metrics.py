"""Bộ metric đầy đủ cho benchmark trên bộ dữ liệu canonical LongBench.

Bổ sung cho common/rouge.py (ROUGE-1/2/L F1): các metric semantic mở rộng
(ROUGE P/R/F đầy đủ, ROUGE-Lsum, BLEU-1..4, length ratio) + các hàm tổng hợp
metric tốc độ (mean/median/p90/std) từ schema §13 (baseline_repo_guide.md).

Module pure-Python, không cần cài thêm package ngoài venv chung. Dùng chung cho:

  * scripts/infer_*.py        - ghi metric semantic mở rộng vào từng record;
  * scripts/collect_metrics.py - tổng hợp toàn bộ run thành báo cáo.

Quy ước key (flat, theo schema §13):
  rouge1_p / rouge1_r / rouge1_f / rouge2_* / rougeL_*   (ROUGE P/R/F)
  rougeLsum_p / rougeLsum_r / rougeLsum_f                (ROUGE-Lsum)
  bleu1 .. bleu4                                         (BLEU-n, smoothed)
  length_ratio                                           (hyp_tokens / ref_tokens)
"""

from __future__ import annotations

import math
import re
import statistics
from difflib import SequenceMatcher
from collections import Counter
from typing import Mapping, Optional, Sequence

from common.rouge import _f1, _lcs_length, rouge_all, tokenize

# Các key tốc độ (số) trong schema §13 dùng để tổng hợp.
SPEED_KEYS = [
    "input_tokens",
    "retained_tokens",
    "output_tokens",
    "selector_latency_ms",
    "prefill_ms",
    "decode_ms",
    "ttft_ms",
    "tpot_ms",
    "e2e_ms",
    "pipeline_e2e_ms",
    "throughput_tok_s",
    "decode_throughput_tok_s",
    "qps",
    "peak_memory_gb",
]

# Key speculative decoding (speculative methods, §13).
SPEC_KEYS = [
    "avg_accept_length",
    "acceptance_rate",
    "draft_latency_ms",
    "verification_latency_ms",
    "rejected_draft_ratio",
]

# Paired speedup metrics.  The numerator is the dense/reference timing and
# the denominator is the optimized timing.  These are intentionally kept
# separate from SPEED_KEYS because a speedup is only valid when both timings
# were measured for the same source sample/configuration.
SPEEDUP_TIMINGS = {
    "esr": ("dense_e2e_ms", ("pipeline_e2e_ms", "e2e_ms")),
    "dsr": ("dense_decode_ms", ("decode_ms",)),
    "prefill_speedup": ("dense_prefill_ms", ("prefill_ms",)),
    "ttft_speedup": ("dense_ttft_ms", ("pipeline_ttft_ms", "ttft_ms")),
}

# Key tỷ lệ bổ sung (tính từ record).
DERIVED_KEYS = [
    "retained_ratio",      # retained_tokens / input_tokens
    "compression_ratio",   # output_tokens / input_tokens (tóm tắt ngắn hơn doc)
]


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------

def mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def p90(values: Sequence[float]) -> float:
    """Percentile 90 (latency tail) — trả 0.0 khi rỗng."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(math.ceil(0.90 * len(ordered)) - 1)))
    return ordered[k]


def std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def agg_numeric(values: Sequence[float]) -> Optional[dict]:
    """{mean, median, p90, std} từ danh sách số; None nếu không có giá trị."""
    clean = [float(v) for v in values if v is not None]
    clean = [v for v in clean if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return None
    return {
        "mean": round(mean(clean), 4),
        "median": round(median(clean), 4),
        "p90": round(p90(clean), 4),
        "std": round(std(clean), 4),
    }


# --------------------------------------------------------------------------
# BLEU (pure-Python, smoothed)
# --------------------------------------------------------------------------

def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(hyp: str, ref: str, max_n: int = 4) -> dict:
    """BLEU-1..4 (smoothed, kèm brevity penalty).

    Trả về {"bleu1": .., "bleu2": .., "bleu3": .., "bleu4": ..} với
    BLEU-n = BP * exp((1/n) * Σ log p_k). Smoothing đơn giản (floor epsilon
    1e-9) để kết quả luôn hữu hạn kể cả khi không có n-gram nào trùng — cần
    thiết cho summarization (ref dài, hyp ngắn).
    """
    h_tokens = tokenize(hyp)
    r_tokens = tokenize(ref)
    if not h_tokens or not r_tokens:
        return {f"bleu{n}": 0.0 for n in range(1, max_n + 1)}

    brevity = math.exp(1.0 - len(r_tokens) / len(h_tokens)) if len(h_tokens) < len(r_tokens) else 1.0

    precisions: list[float] = []
    out: dict = {}
    for n in range(1, max_n + 1):
        h_counts = _ngram_counts(h_tokens, n)
        r_counts = _ngram_counts(r_tokens, n)
        hyp_total = sum(h_counts.values())
        if hyp_total == 0:
            precisions.append(1e-9)
        else:
            overlap = sum((h_counts & r_counts).values())
            precisions.append(max(overlap / hyp_total, 1e-9))
        gm = math.exp(sum(math.log(p) for p in precisions) / n)
        out[f"bleu{n}"] = round(brevity * gm, 4)
    return out


# --------------------------------------------------------------------------
# ROUGE-Lsum (sentence-level LCS)
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"[\n。！？!?；;]")


def sentence_split(text: str) -> list[str]:
    """Tách câu đơn giản (newline / dấu câu kết thúc) để tính ROUGE-Lsum."""
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def rouge_lsum(hyp: str, ref: str) -> dict:
    """ROUGE-Lsum: LCS trên chuỗi *câu* (chuẩn summarization, Lin & Hovy 2003)."""
    h_sents = sentence_split(hyp)
    r_sents = sentence_split(ref)
    if not h_sents or not r_sents:
        return {"p": 0.0, "r": 0.0, "f": 0.0}

    lcs = _lcs_length(h_sents, r_sents)
    precision = lcs / len(h_sents)
    recall = lcs / len(r_sents)
    return {"p": precision, "r": recall, "f": _f1(precision, recall)}


# --------------------------------------------------------------------------
# Semantic metric set (ghi vào record)
# --------------------------------------------------------------------------

def semantic_scores(hyp: str, ref: str) -> dict:
    """Toàn bộ metric semantic flat: ROUGE P/R/F, ROUGE-Lsum, BLEU-1..4,
    length ratio. Trả về dict key phẳng theo quy ước ở docstring module."""
    all_rouge = rouge_all(hyp, ref)  # {"rouge-1": {p,r,f}, ...}
    lsum = rouge_lsum(hyp, ref)

    scores: dict = {}
    for name, base in (("rouge1", "rouge-1"), ("rouge2", "rouge-2"), ("rougeL", "rouge-l")):
        s = all_rouge[base]
        scores[f"{name}_p"] = round(s["p"], 4)
        scores[f"{name}_r"] = round(s["r"], 4)
        scores[f"{name}_f"] = round(s["f"], 4)
    scores["rougeLsum_p"] = round(lsum["p"], 4)
    scores["rougeLsum_r"] = round(lsum["r"], 4)
    scores["rougeLsum_f"] = round(lsum["f"], 4)
    scores.update(bleu(hyp, ref))

    h_len = len(tokenize(hyp))
    r_len = len(tokenize(ref))
    scores["length_ratio"] = round(h_len / r_len, 4) if r_len else 0.0
    return scores


def add_semantic(
    record: dict,
    hyp: str,
    ref: Optional[str],
    *,
    prefix: str = "",
) -> Optional[dict]:
    """Ghi toàn bộ metric semantic vào record nếu có reference.

    prefix cho phép nhiều text trong cùng record (vd GemFilter:
    gemfilter_text không prefix, base_text dùng prefix "base_").
    """
    if not ref or not str(ref).strip() or not hyp or not str(hyp).strip():
        return None
    scores = semantic_scores(hyp, str(ref))
    for key, value in scores.items():
        record[prefix + key] = value
    return scores


def aggregate_semantic(
    records: Sequence[Mapping],
    *,
    prefix: str = "",
) -> dict:
    """Trung bình mọi key semantic (*_p/_r/_f, bleu*, length_ratio) trên các
    record có giá trị. Key giữ prefix."""
    out: dict = {}
    keys: list[str] = []
    for r in records:
        for k in r:
            if not k.startswith(prefix):
                continue
            tail = k[len(prefix):]
            if tail.endswith(("_p", "_r", "_f")) or tail.startswith("bleu") or tail == "length_ratio":
                if k not in keys:
                    keys.append(k)
    for k in keys:
        values = [float(r[k]) for r in records if r.get(k) is not None]
        if values:
            out[k] = round(mean(values), 4)
    return out


# --------------------------------------------------------------------------
# Code-completion metrics
# --------------------------------------------------------------------------

def normalize_code_output(text: str) -> str:
    """Normalize harmless formatting before comparing a code continuation."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.rstrip() for line in lines if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def code_completion_scores(hyp: str, ref: str) -> dict:
    """Return exact-match and edit similarity; intentionally no ROUGE keys."""
    normalized_hyp = normalize_code_output(hyp)
    normalized_ref = normalize_code_output(ref)
    return {
        "code_exact_match": float(normalized_hyp == normalized_ref),
        "code_edit_similarity": round(
            SequenceMatcher(None, normalized_hyp, normalized_ref).ratio(), 4
        ),
    }


def add_code_completion(
    record: dict,
    hyp: str,
    ref: Optional[str],
    *,
    prefix: str = "",
) -> Optional[dict]:
    if not ref or not str(ref).strip() or not hyp or not str(hyp).strip():
        return None
    scores = code_completion_scores(hyp, str(ref))
    for key, value in scores.items():
        record[prefix + key] = value
    return scores


def aggregate_code_completion(
    records: Sequence[Mapping],
    *,
    prefix: str = "",
) -> dict:
    """Aggregate code-completion scores without mixing in ROUGE metrics."""
    keys = {
        key
        for record in records
        for key in record
        if key.startswith(prefix + "code_")
    }
    return {
        key: round(mean([float(record[key]) for record in records if key in record]), 4)
        for key in sorted(keys)
        if any(key in record for record in records)
    }


# --------------------------------------------------------------------------
# Speed aggregation (schema §13)
# --------------------------------------------------------------------------

def record_derived(record: Mapping) -> dict:
    """Các tỷ lệ suy ra từ record (retained_ratio, compression_ratio)."""
    out: dict = {}
    input_tokens = record.get("input_tokens")
    retained = record.get("retained_tokens")
    output_tokens = record.get("output_tokens")
    if input_tokens and retained is not None:
        out["retained_ratio"] = round(float(retained) / float(input_tokens), 4)
    if input_tokens and output_tokens:
        out["compression_ratio"] = round(float(output_tokens) / float(input_tokens), 4)
    return out


def aggregate_speed(
    records: Sequence[Mapping],
    *,
    keys: Sequence[str] = SPEED_KEYS,
    include_derived: bool = True,
) -> dict:
    """{key: {mean, median, p90, std}} cho từng key số trong danh sách."""
    out: dict = {}
    for key in keys:
        values = [float(r[key]) for r in records if r.get(key) is not None]
        agg = agg_numeric(values)
        if agg:
            out[key] = agg
    # tỷ lệ suy ra từ record (chỉ cho nhóm key tốc độ chính)
    if include_derived:
        for key in DERIVED_KEYS:
            values = [float(record_derived(r)[key]) for r in records if key in record_derived(r)]
            agg = agg_numeric(values)
            if agg:
                out[key] = agg
    return out


def aggregate_weighted_throughput(records: Sequence[Mapping]) -> dict:
    """Compute total generated tokens divided by total recorded wall time."""

    successful = [
        record
        for record in records
        if record.get("status", "success") == "success"
        and record.get("scope") != "aggregate"
        and record.get("type") != "summary"
    ]
    result: dict[str, float | int] = {}
    for name, timing_key, tokens_key in (
        ("decode_tok_s", "decode_ms", "output_tokens"),
        ("e2e_tok_s", "e2e_ms", "output_tokens"),
    ):
        pairs: list[tuple[float, float]] = []
        for record in successful:
            try:
                tokens = float(record.get(tokens_key) or 0)
                elapsed_ms = float(record.get(timing_key) or 0)
            except (TypeError, ValueError):
                continue
            if tokens > 0 and elapsed_ms > 0:
                pairs.append((tokens, elapsed_ms))
        if not pairs:
            continue
        total_tokens = sum(tokens for tokens, _ in pairs)
        total_ms = sum(elapsed for _, elapsed in pairs)
        result[name] = round(total_tokens * 1000.0 / total_ms, 4)
        result[f"{name}_tokens"] = int(total_tokens)
        result[f"{name}_time_ms"] = round(total_ms, 4)
    return result


def _acceptance_length_from_record(record: Mapping) -> Optional[float]:
    """Read one sample's mean accepted length from supported trace formats.

    Baselines do not expose one identical field: EAGLE emits ``accept_length``,
    DFlash emits block-level ``acceptance_lengths`` and SpecExtend emits
    ``accept_length_list``.  A precomputed per-sample ``avg_accept_length`` is
    preferred when present.  The returned value is always the mean for one
    sample; aggregation across samples is deliberately done by
    :func:`mean_acceptance_length` so long generations do not get extra weight.
    """
    value = record.get("avg_accept_length")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)

    value = record.get("accept_length")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)

    for key in ("acceptance_lengths", "accept_length_list"):
        values = record.get(key)
        if not isinstance(values, (list, tuple)):
            continue
        numeric = [
            float(item)
            for item in values
            if isinstance(item, (int, float)) and math.isfinite(float(item))
        ]
        if numeric:
            return mean(numeric)
    return None


def mean_acceptance_length(records: Sequence[Mapping]) -> Optional[float]:
    """Return mean acceptance length (paper ``τ``) over successful samples.

    Failed rows, aggregate/summary rows and rows without an acceptance trace
    are excluded.  The result is ``None`` when the run did not expose a valid
    acceptance trace, rather than silently treating it as ``1``.
    """
    values: list[float] = []
    for record in records:
        if record.get("status", "success") != "success":
            continue
        if record.get("type") == "summary" or record.get("scope") == "aggregate":
            continue
        value = _acceptance_length_from_record(record)
        if value is not None:
            values.append(value)
    return mean(values) if values else None


def aggregate_speculative(records: Sequence[Mapping]) -> dict:
    """{key: {mean, median, p90, std}} cho các key speculative decoding."""
    out = aggregate_speed(records, keys=SPEC_KEYS, include_derived=False)
    tau = mean_acceptance_length(records)
    if tau is not None:
        values = []
        for record in records:
            if record.get("status", "success") != "success":
                continue
            if record.get("type") == "summary" or record.get("scope") == "aggregate":
                continue
            value = _acceptance_length_from_record(record)
            if value is not None:
                values.append(value)
        out["avg_accept_length"] = agg_numeric(values) or {}
    return out


def aggregate_output_parity(records: Sequence[Mapping]) -> dict:
    """Summarize exact generated-token agreement without gating speedups."""

    available = [
        record
        for record in records
        if record.get("status", "success") == "success"
        and record.get("output_parity_available") is True
    ]
    if not available:
        return {}

    def mean_field(key: str) -> float | None:
        values = [
            float(record[key])
            for record in available
            if isinstance(record.get(key), (int, float))
        ]
        return round(mean(values), 4) if values else None

    def bool_rate(key: str) -> float | None:
        values = [record[key] for record in available if isinstance(record.get(key), bool)]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "samples": len(available),
        "fixed_continuation_exact_match_rate": bool_rate(
            "fixed_continuation_exact_match"
        ),
        "fixed_continuation_token_match_ratio": mean_field(
            "fixed_continuation_token_match_ratio"
        ),
        "quality_prefix_exact_match_rate": bool_rate("quality_prefix_exact_match"),
        "quality_prefix_token_match_ratio": mean_field(
            "quality_prefix_token_match_ratio"
        ),
    }


def _first_positive(record: Mapping, keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return None


def aggregate_speedup(records: Sequence[Mapping]) -> dict:
    """Calculate paired speedups as ``mean(dense) / mean(method)``.

    Only records carrying both sides of a timing pair contribute.  Returning
    a scalar ratio (rather than averaging per-record ratios) makes the report
    agree with the benchmark-level definition and avoids tiny samples with
    unusually short requests dominating the result.

    The result contains only metrics with at least one complete positive pair;
    a missing timing is therefore reported as unavailable instead of being
    treated as zero.
    """
    out: dict = {}
    for name, (dense_key, method_keys) in SPEEDUP_TIMINGS.items():
        dense_values: list[float] = []
        method_values: list[float] = []
        for record in records:
            if record.get("speedup_valid") is False:
                continue
            dense = _first_positive(record, (dense_key,))
            method = _first_positive(record, method_keys)
            if dense is None or method is None:
                continue
            dense_values.append(dense)
            method_values.append(method)
        if method_values:
            out[name] = round(mean(dense_values) / mean(method_values), 4)
    return out


def speedup_pair_coverage(records: Sequence[Mapping]) -> dict:
    """Report how many records provide valid dense/method timing pairs."""

    coverage: dict = {}
    for name, (dense_key, method_keys) in SPEEDUP_TIMINGS.items():
        available = 0
        valid = 0
        for record in records:
            dense = _first_positive(record, (dense_key,))
            method = _first_positive(record, method_keys)
            if dense is None or method is None:
                continue
            available += 1
            if record.get("speedup_valid") is not False:
                valid += 1
        coverage[name] = {
            "available": bool(available),
            "valid": valid,
            "total": len(records),
            "ratio": round(valid / len(records), 4) if records else 0.0,
        }
    return coverage
