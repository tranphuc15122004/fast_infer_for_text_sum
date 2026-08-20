"""
Inference/benchmark runner for Qwen3-4B + semantic-selection baselines.

Baselines supported
-------------------
    full       : no context selection (reference inference baseline)
    random     : deterministic pseudo-random sentence selection
    lead       : leading-sentence selection
    tfidf      : TF-IDF centroid relevance
    textrank   : TF-IDF graph + weighted PageRank
    mmr        : sentence embeddings + Maximal Marginal Relevance

The runner is designed for long-document summarization experiments. For every
document it:

    1. Runs full-context Qwen3-4B once.
    2. Applies each requested selector under the same target-token budget.
    3. Runs the SAME Qwen3-4B target with greedy decoding.
    4. Records selector cost, prompt size, TTFT, prefill, decode, TPOT,
       target-model E2E, selector-inclusive E2E, GPU memory, and speedup
       against the full-context baseline.
    5. Writes one JSON object per (document, selector, budget) to JSONL.
    6. Writes aggregate P50/P95/P99 statistics to <output>.summary.json.

Important
---------
Qwen3 "thinking" is disabled by default for summarization when the tokenizer's
chat template supports `enable_thinking=False`. This avoids measuring hidden
reasoning tokens as part of the summarization output.

Inference is greedy and batch-size 1. This is intentional for the first
controlled baseline suite. Continuous-batching/QPS experiments should be added
as a separate serving benchmark (e.g. vLLM/SGLang), not mixed into this runner.

Recommended package layout
--------------------------
semantic_selection/
    __init__.py
    base.py
    lead.py
    random.py
    tfidf.py
    textrank.py
    mmr.py
    infer.py

Because a file named `random.py` can shadow Python's stdlib `random` module,
the safest invocation is:

    python -m semantic_selection.infer ...

This file also contains a direct-script compatibility loader that removes the
local folder from sys.path before importing PyTorch, so:

    python infer.py ...

is supported as well.

Examples
--------
Single text file, one budget:

    python -m semantic_selection.infer \
        --input sample.txt \
        --output results/qwen3_4b.jsonl \
        --token-budgets 2048 \
        --selectors random lead tfidf textrank mmr

JSONL dataset with several budgets:

    python -m semantic_selection.infer \
        --input data/govreport.jsonl \
        --document-field document \
        --id-field id \
        --reference-field summary \
        --token-budgets 512 1024 2048 \
        --max-new-tokens 256 \
        --output results/govreport_qwen3_4b.jsonl

Retention-ratio experiment:

    python -m semantic_selection.infer \
        --input data.jsonl \
        --retention-ratios 0.10 0.20 0.30 0.50 \
        --output results/retention_sweep.jsonl

Input formats
-------------
.txt:
    Entire file is one document.

.jsonl:
    One JSON object per line.

.json:
    A JSON list of objects, a single object, or {"data": [...]}.

Expected fields default to:
    id
    document
    reference

Only `document` is required. Reference summaries are copied to the output for a
later quality-metric pass; quality metrics are intentionally not computed here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Protect direct-script execution from local random.py shadowing stdlib random.
# Do this before importing torch/numpy/transformers.
# ---------------------------------------------------------------------------

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_SCRIPT_DIR = Path(__file__).resolve().parent
_DIRECT_SCRIPT = __package__ in (None, "")

if _DIRECT_SCRIPT:
    cleaned_path: List[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except Exception:
            cleaned_path.append(entry)
            continue

        if resolved != _SCRIPT_DIR:
            cleaned_path.append(entry)

    sys.path[:] = cleaned_path


import statistics
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Shared ROUGE metrics (scripts/common/rouge.py, dependency-free)
# ---------------------------------------------------------------------------


def _load_shared_rouge():
    """Import the repo-level pure-Python ROUGE from scripts/common/rouge.py."""
    # externals/Sematic_selection/infer.py -> repo root = parents[1]
    repo_root = _SCRIPT_DIR.resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from common import rouge  # noqa: PLC0415, E402  # type: ignore[import-not-found]

        return rouge
    except Exception:  # pragma: no cover - standalone usage outside the repo
        return None


_rouge = _load_shared_rouge()
ROUGE_AVAILABLE = _rouge is not None


# ---------------------------------------------------------------------------
# Local/package imports
# ---------------------------------------------------------------------------


def _load_local_module(module_name: str, filename: str):
    """Load a sibling Python file without re-adding its directory to sys.path."""
    path = _SCRIPT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required module file not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if _DIRECT_SCRIPT:
    _base = _load_local_module("base", "base.py")
    _lead = _load_local_module("_semantic_lead", "lead.py")
    _random = _load_local_module("_semantic_random_selector", "random.py")
    _tfidf = _load_local_module("_semantic_tfidf", "tfidf.py")
    _textrank = _load_local_module("_semantic_textrank", "textrank.py")
    _mmr = _load_local_module("_semantic_mmr", "mmr.py")

    SelectionResult = _base.SelectionResult
    resolve_local_model_path = _base.resolve_local_model_path
    summarize_latencies_ms = _base.summarize_latencies_ms

    LeadSelector = _lead.LeadSelector
    RandomSelector = _random.RandomSelector
    TFIDFCentroidSelector = _tfidf.TFIDFCentroidSelector
    TextRankSelector = _textrank.TextRankSelector
    MMRSelector = _mmr.MMRSelector
else:
    from .base import (
        SelectionResult,
        resolve_local_model_path,
        summarize_latencies_ms,
    )
    from .lead import LeadSelector
    from .random import RandomSelector
    from .tfidf import TFIDFCentroidSelector
    from .textrank import TextRankSelector
    from .mmr import MMRSelector


DEFAULT_MODEL = "Qwen/Qwen3-4B"

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise summarization assistant. "
    "Write faithful summaries grounded only in the provided document."
)

DEFAULT_INSTRUCTION = (
    "Summarize the following document faithfully and concisely. "
    "Preserve important facts, entities, numbers, and relationships. "
    "Do not introduce information that is not supported by the document."
)

VALID_SELECTORS = (
    "random",
    "lead",
    "tfidf",
    "textrank",
    "mmr",
)


# ============================================================================
# Generic helpers
# ============================================================================


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")

    data = sorted(float(v) for v in values)

    if len(data) == 1:
        return data[0]

    position = (len(data) - 1) * q / 100.0
    lo = math.floor(position)
    hi = math.ceil(position)

    if lo == hi:
        return data[lo]

    frac = position - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def _mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [float(v) for v in values if v is not None]

    if not cleaned:
        return None

    return statistics.fmean(cleaned)


def _json_safe(value: Any) -> Any:
    """Convert common NumPy/Torch scalar types into JSON-serializable values."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return value


# ============================================================================
# Dataset I/O
# ============================================================================


@dataclass
class Example:
    example_id: str
    document: str
    reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_examples(
    path: Path,
    *,
    document_field: str,
    id_field: str,
    reference_field: str,
    limit: Optional[int] = None,
) -> List[Example]:
    """Load .txt, .json, or .jsonl input into a common structure."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8").strip()
        raw_records: List[Any] = [
            {
                id_field: path.stem,
                document_field: text,
            }
        ]

    elif suffix == ".jsonl":
        raw_records = []

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}"
                    ) from exc

    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(payload, list):
            raw_records = payload
        elif (
            isinstance(payload, dict)
            and isinstance(payload.get("data"), list)
        ):
            raw_records = payload["data"]
        elif isinstance(payload, dict):
            raw_records = [payload]
        else:
            raise ValueError(
                f"Unsupported JSON structure in {path}: "
                f"{type(payload).__name__}"
            )

    else:
        raise ValueError(
            f"Unsupported input extension {suffix!r}. "
            "Use .txt, .json, or .jsonl."
        )

    examples: List[Example] = []

    for index, record in enumerate(raw_records):
        if limit is not None and len(examples) >= limit:
            break

        if isinstance(record, str):
            document = record
            example_id = str(index)
            reference = None
            metadata: Dict[str, Any] = {}
        elif isinstance(record, Mapping):
            if document_field not in record:
                raise KeyError(
                    f"Record {index} does not contain document field "
                    f"{document_field!r}"
                )

            document = str(record[document_field])
            example_id = str(record.get(id_field, index))

            raw_reference = record.get(reference_field)
            reference = (
                None
                if raw_reference is None
                else str(raw_reference)
            )

            metadata = {
                str(k): v
                for k, v in record.items()
                if k not in {
                    document_field,
                    id_field,
                    reference_field,
                }
            }
        else:
            raise TypeError(
                f"Unsupported record type at index {index}: "
                f"{type(record).__name__}"
            )

        document = document.strip()

        if not document:
            continue

        examples.append(
            Example(
                example_id=example_id,
                document=document,
                reference=reference,
                metadata=metadata,
            )
        )

    if not examples:
        raise ValueError(f"No usable documents found in {path}")

    return examples


# ============================================================================
# Qwen3 target model
# ============================================================================


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def choose_dtype(
    device: torch.device,
    dtype_arg: str,
) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32

    if dtype_arg == "float16":
        return torch.float16

    if dtype_arg == "bfloat16":
        return torch.bfloat16

    if dtype_arg != "auto":
        raise ValueError(f"Unknown dtype {dtype_arg!r}")

    if device.type != "cuda":
        return torch.float32

    # T4-class GPUs should normally use FP16; Ampere+ can use BF16 when
    # PyTorch reports native support.
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def load_target(
    model_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
    attn_implementation: str,
):
    """Load the cached Qwen3 target model and tokenizer once."""
    model_source = (
        resolve_local_model_path(model_name)
        if local_files_only
        else model_name
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }

    if attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **model_kwargs,
    )

    model.to(device)
    model.eval()

    return tokenizer, model


def build_chat_prompt(
    tokenizer,
    document: str,
    *,
    system_prompt: str,
    instruction: str,
    disable_thinking: bool,
) -> str:
    """Build a Qwen3 chat prompt while disabling thinking when supported."""
    messages = []

    if system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": system_prompt.strip(),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                f"{instruction.strip()}\n\n"
                f"DOCUMENT:\n{document.strip()}"
            ),
        }
    )

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    if disable_thinking:
        template_kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(
            messages,
            **template_kwargs,
        )
    except TypeError:
        # Older transformers/chat-template implementation without the Qwen3
        # `enable_thinking` extension.
        template_kwargs.pop("enable_thinking", None)

        return tokenizer.apply_chat_template(
            messages,
            **template_kwargs,
        )


def get_eos_token_ids(tokenizer, model) -> set[int]:
    ids: set[int] = set()

    tokenizer_eos = tokenizer.eos_token_id

    if tokenizer_eos is not None:
        if isinstance(tokenizer_eos, int):
            ids.add(tokenizer_eos)
        else:
            ids.update(int(x) for x in tokenizer_eos)

    generation_eos = getattr(
        getattr(model, "generation_config", None),
        "eos_token_id",
        None,
    )

    if generation_eos is not None:
        if isinstance(generation_eos, int):
            ids.add(generation_eos)
        else:
            ids.update(int(x) for x in generation_eos)

    return ids


@dataclass
class GenerationMetrics:
    summary: str

    content_tokens: int
    prompt_tokens: int
    output_tokens: int

    prompt_build_ms: float
    target_tokenize_ms: float

    prefill_ms: float
    decode_ms: float
    model_e2e_ms: float

    target_request_ttft_ms: float
    target_request_e2e_ms: float

    tpot_ms: Optional[float]
    decode_step_p50_ms: Optional[float]
    decode_step_p95_ms: Optional[float]
    decode_step_p99_ms: Optional[float]

    output_tokens_per_second: Optional[float]

    text_decode_ms: float

    peak_gpu_allocated_mb: Optional[float]
    peak_gpu_reserved_mb: Optional[float]

    stop_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@torch.inference_mode()
def generate_greedy_profiled(
    model,
    tokenizer,
    selected_document: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    system_prompt: str,
    instruction: str,
    disable_thinking: bool,
) -> GenerationMetrics:
    """
    Greedy Qwen3 decoding with explicit prefill/decode timing.

    The manual loop is used instead of `model.generate()` because it exposes a
    clean prefill/decode boundary and request-level TTFT without streamer-thread
    noise. The same loop is used for every selector and the full-context run.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    # ------------------------------
    # Prompt construction
    # ------------------------------
    prompt_start = time.perf_counter()

    prompt_text = build_chat_prompt(
        tokenizer,
        selected_document,
        system_prompt=system_prompt,
        instruction=instruction,
        disable_thinking=disable_thinking,
    )

    prompt_build_ms = _elapsed_ms(prompt_start)

    # ------------------------------
    # Tokenization
    # ------------------------------
    tokenize_start = time.perf_counter()

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    target_tokenize_ms = _elapsed_ms(tokenize_start)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    prompt_tokens = int(input_ids.shape[1])
    content_tokens = len(
        tokenizer.encode(
            selected_document,
            add_special_tokens=False,
        )
    )

    model_limit = getattr(
        getattr(model, "config", None),
        "max_position_embeddings",
        None,
    )

    if (
        isinstance(model_limit, int)
        and model_limit > 0
        and prompt_tokens + max_new_tokens > model_limit
    ):
        raise ValueError(
            "Prompt + generation exceeds model context limit: "
            f"{prompt_tokens} + {max_new_tokens} > {model_limit}. "
            "Reduce the selection budget or --max-new-tokens."
        )

    eos_ids = get_eos_token_ids(tokenizer, model)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ------------------------------
    # Prefill / first token
    # ------------------------------
    _sync_cuda(device)
    prefill_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    _sync_cuda(device)
    prefill_ms = _elapsed_ms(prefill_start)

    generated: List[int] = [
        int(next_token.item())
    ]

    past_key_values = outputs.past_key_values
    del outputs

    stop_reason = "max_new_tokens"

    if generated[-1] in eos_ids:
        stop_reason = "eos"

    # ------------------------------
    # Autoregressive decode
    # ------------------------------
    decode_step_ms: List[float] = []
    decode_total_start = time.perf_counter()

    while (
        len(generated) < max_new_tokens
        and stop_reason != "eos"
    ):
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        _sync_cuda(device)
        step_start = time.perf_counter()

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        past_key_values = outputs.past_key_values

        _sync_cuda(device)
        step_ms = _elapsed_ms(step_start)
        decode_step_ms.append(step_ms)

        token_id = int(next_token.item())
        generated.append(token_id)

        del outputs

        if token_id in eos_ids:
            stop_reason = "eos"

    _sync_cuda(device)
    decode_ms = _elapsed_ms(decode_total_start)

    # If the first token already ended generation there was no decode loop.
    if len(generated) == 1:
        decode_ms = 0.0

    model_e2e_ms = prefill_ms + decode_ms

    # ------------------------------
    # Decode output IDs to text
    # ------------------------------
    text_decode_start = time.perf_counter()

    summary = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    text_decode_ms = _elapsed_ms(text_decode_start)

    # Request-level metrics exclude semantic selection but include CPU prompt
    # build/tokenization and final token->text conversion.
    target_request_ttft_ms = (
        prompt_build_ms
        + target_tokenize_ms
        + prefill_ms
    )

    target_request_e2e_ms = (
        prompt_build_ms
        + target_tokenize_ms
        + model_e2e_ms
        + text_decode_ms
    )

    output_tokens = len(generated)

    # The first output token is produced by prefill. TPOT convention therefore
    # averages the remaining token-step latency across output_tokens - 1.
    if output_tokens > 1:
        tpot_ms: Optional[float] = decode_ms / (output_tokens - 1)
    else:
        tpot_ms = None

    if decode_step_ms:
        step_p50 = _percentile(decode_step_ms, 50.0)
        step_p95 = _percentile(decode_step_ms, 95.0)
        step_p99 = _percentile(decode_step_ms, 99.0)
    else:
        step_p50 = None
        step_p95 = None
        step_p99 = None

    if model_e2e_ms > 0.0:
        output_tps: Optional[float] = (
            output_tokens / (model_e2e_ms / 1000.0)
        )
    else:
        output_tps = None

    if device.type == "cuda":
        peak_allocated = (
            torch.cuda.max_memory_allocated(device)
            / (1024.0 ** 2)
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device)
            / (1024.0 ** 2)
        )
    else:
        peak_allocated = None
        peak_reserved = None

    return GenerationMetrics(
        summary=summary,
        content_tokens=content_tokens,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prompt_build_ms=prompt_build_ms,
        target_tokenize_ms=target_tokenize_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        model_e2e_ms=model_e2e_ms,
        target_request_ttft_ms=target_request_ttft_ms,
        target_request_e2e_ms=target_request_e2e_ms,
        tpot_ms=tpot_ms,
        decode_step_p50_ms=step_p50,
        decode_step_p95_ms=step_p95,
        decode_step_p99_ms=step_p99,
        output_tokens_per_second=output_tps,
        text_decode_ms=text_decode_ms,
        peak_gpu_allocated_mb=peak_allocated,
        peak_gpu_reserved_mb=peak_reserved,
        stop_reason=stop_reason,
    )


# ============================================================================
# Selectors
# ============================================================================


def initialize_selectors(
    names: Sequence[str],
    *,
    tokenizer,
    tokenizer_name: str,
    random_seed: int,
    mmr_lambda: float,
    embedding_model_name: str,
    embedding_device: str,
    embedding_local_files_only: bool,
    textrank_top_k_neighbors: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Initialize each selector exactly once.

    Startup/model-loading time is reported separately and is intentionally NOT
    added to per-request E2E latency because the production assumption is that
    selector models remain resident between requests.
    """
    selectors: Dict[str, Any] = {}
    init_ms: Dict[str, float] = {}

    common = {
        "tokenizer": tokenizer,
        "tokenizer_name": tokenizer_name,
        "local_files_only": True,
    }

    for name in names:
        start = time.perf_counter()

        if name == "random":
            selector = RandomSelector(
                seed=random_seed,
                **common,
            )

        elif name == "lead":
            selector = LeadSelector(
                **common,
            )

        elif name == "tfidf":
            selector = TFIDFCentroidSelector(
                **common,
            )

        elif name == "textrank":
            selector = TextRankSelector(
                top_k_neighbors=textrank_top_k_neighbors,
                **common,
            )

        elif name == "mmr":
            selector = MMRSelector(
                embedding_model_name=embedding_model_name,
                embedding_device=embedding_device,
                embedding_local_files_only=(
                    embedding_local_files_only
                ),
                mmr_lambda=mmr_lambda,
                **common,
            )

        else:
            raise ValueError(
                f"Unknown selector {name!r}; valid values: "
                f"{', '.join(VALID_SELECTORS)}"
            )

        selectors[name] = selector
        init_ms[name] = _elapsed_ms(start)

    return selectors, init_ms


def make_full_context_result(
    document: str,
    tokenizer,
) -> SelectionResult:
    """Create a SelectionResult-compatible object for the no-selection run."""
    original_tokens = len(
        tokenizer.encode(
            document,
            add_special_tokens=False,
        )
    )

    return SelectionResult(
        selected_text=document,
        selected_sentences=[],
        selected_indices=[],
        selected_tokens=original_tokens,
        original_tokens=original_tokens,
        token_budget=original_tokens,
        retention_ratio=1.0,
        selection_time_ms=0.0,
        selector_name="full",
        metadata={
            "num_sentences": 0,
            "num_selected_sentences": 0,
            "selection_skipped": True,
            "selection_total_wall_time_ms": 0.0,
            "selection_preprocess_overhead_ms": 0.0,
        },
    )


def resolve_budgets(
    original_tokens: int,
    *,
    token_budgets: Optional[Sequence[int]],
    retention_ratios: Optional[Sequence[float]],
) -> List[Tuple[str, int, Optional[float]]]:
    """
    Return (budget_label, token_budget, requested_retention_ratio) tuples.
    """
    resolved: List[Tuple[str, int, Optional[float]]] = []

    if token_budgets:
        for budget in token_budgets:
            if budget <= 0:
                raise ValueError("All token budgets must be > 0")

            resolved.append(
                (
                    f"tokens_{budget}",
                    int(budget),
                    None,
                )
            )

    elif retention_ratios:
        for ratio in retention_ratios:
            if not 0.0 < ratio <= 1.0:
                raise ValueError(
                    "Retention ratios must be in (0, 1]"
                )

            budget = max(
                1,
                int(round(original_tokens * ratio)),
            )

            resolved.append(
                (
                    f"ratio_{ratio:g}",
                    budget,
                    float(ratio),
                )
            )
    else:
        resolved.append(
            ("tokens_2048", 2048, None)
        )

    return resolved


# ============================================================================
# Row construction and aggregation
# ============================================================================


def make_result_row(
    *,
    example: Example,
    selector_name: str,
    budget_label: str,
    requested_retention_ratio: Optional[float],
    selection: SelectionResult,
    generation: GenerationMetrics,
    selector_init_ms: float,
    baseline_generation: Optional[GenerationMetrics],
    save_selected_text: bool,
    compute_rouge: bool = False,
) -> Dict[str, Any]:
    """
    Build one flat JSONL record.

    Pipeline E2E uses preprocessing-inclusive selector wall time when available.
    """
    selector_total_ms = float(
        selection.metadata.get(
            "selection_total_wall_time_ms",
            selection.selection_time_ms,
        )
    )

    selector_preprocess_ms = float(
        selection.metadata.get(
            "selection_preprocess_overhead_ms",
            max(
                selector_total_ms - selection.selection_time_ms,
                0.0,
            ),
        )
    )

    pipeline_ttft_ms = (
        selector_total_ms
        + generation.target_request_ttft_ms
    )

    pipeline_e2e_ms = (
        selector_total_ms
        + generation.target_request_e2e_ms
    )

    if baseline_generation is not None:
        baseline_e2e = (
            baseline_generation.target_request_e2e_ms
        )
        baseline_ttft = (
            baseline_generation.target_request_ttft_ms
        )

        e2e_speedup = (
            baseline_e2e / pipeline_e2e_ms
            if pipeline_e2e_ms > 0.0
            else None
        )

        ttft_speedup = (
            baseline_ttft / pipeline_ttft_ms
            if pipeline_ttft_ms > 0.0
            else None
        )

        latency_saved_ms = (
            baseline_e2e - pipeline_e2e_ms
        )

        latency_reduction_ratio = (
            latency_saved_ms / baseline_e2e
            if baseline_e2e > 0.0
            else None
        )
    else:
        baseline_e2e = None
        baseline_ttft = None
        e2e_speedup = None
        ttft_speedup = None
        latency_saved_ms = None
        latency_reduction_ratio = None

    row: Dict[str, Any] = {
        # Identity
        "example_id": example.example_id,
        "selector": selector_name,
        "budget_label": budget_label,
        "requested_retention_ratio": requested_retention_ratio,

        # Selector startup vs per-request selector cost
        "selector_init_ms": selector_init_ms,
        "selection_algorithm_ms": selection.selection_time_ms,
        "selection_total_wall_ms": selector_total_ms,
        "selection_preprocess_overhead_ms": selector_preprocess_ms,

        # Content reduction
        "original_tokens": selection.original_tokens,
        "selected_tokens": selection.selected_tokens,
        "tokens_saved": selection.tokens_saved,
        "token_budget": selection.token_budget,
        "retention_ratio": selection.retention_ratio,
        "token_reduction_ratio": (
            selection.token_reduction_ratio
        ),
        "compression_ratio": selection.compression_ratio,
        "budget_utilization": selection.budget_utilization,
        "original_sentences": (
            selection.original_sentence_count
        ),
        "selected_sentences": (
            selection.selected_sentence_count
        ),
        "sentence_retention_ratio": (
            selection.sentence_retention_ratio
        ),
        "selector_input_tokens_per_second": (
            selection.selector_input_tokens_per_second
        ),
        "tokens_saved_per_algorithm_ms": (
            selection.tokens_saved_per_ms
        ),

        # Actual target prompt/output sizes
        "target_content_tokens": generation.content_tokens,
        "target_prompt_tokens": generation.prompt_tokens,
        "output_tokens": generation.output_tokens,

        # CPU request overhead
        "prompt_build_ms": generation.prompt_build_ms,
        "target_tokenize_ms": generation.target_tokenize_ms,
        "text_decode_ms": generation.text_decode_ms,

        # Target model
        "prefill_ms": generation.prefill_ms,
        "decode_ms": generation.decode_ms,
        "model_e2e_ms": generation.model_e2e_ms,
        "target_request_ttft_ms": (
            generation.target_request_ttft_ms
        ),
        "target_request_e2e_ms": (
            generation.target_request_e2e_ms
        ),
        "tpot_ms": generation.tpot_ms,
        "decode_step_p50_ms": (
            generation.decode_step_p50_ms
        ),
        "decode_step_p95_ms": (
            generation.decode_step_p95_ms
        ),
        "decode_step_p99_ms": (
            generation.decode_step_p99_ms
        ),
        "output_tokens_per_second": (
            generation.output_tokens_per_second
        ),
        "peak_gpu_allocated_mb": (
            generation.peak_gpu_allocated_mb
        ),
        "peak_gpu_reserved_mb": (
            generation.peak_gpu_reserved_mb
        ),
        "stop_reason": generation.stop_reason,

        # True selector-inclusive pipeline metrics
        "pipeline_ttft_ms": pipeline_ttft_ms,
        "pipeline_e2e_ms": pipeline_e2e_ms,

        # Full-context comparison
        "baseline_full_ttft_ms": baseline_ttft,
        "baseline_full_e2e_ms": baseline_e2e,
        "baseline_full_prefill_ms": (
            baseline_generation.prefill_ms
            if baseline_generation is not None
            else None
        ),
        "baseline_full_decode_ms": (
            baseline_generation.decode_ms
            if baseline_generation is not None
            else None
        ),
        "ttft_speedup_vs_full": ttft_speedup,
        "e2e_speedup_vs_full": e2e_speedup,
        "prefill_speedup_vs_full": (
            baseline_generation.prefill_ms / generation.prefill_ms
            if baseline_generation is not None and generation.prefill_ms > 0.0
            else None
        ),
        "decode_speedup_vs_full": (
            baseline_generation.decode_ms / generation.decode_ms
            if baseline_generation is not None and generation.decode_ms > 0.0
            else None
        ),
        "net_latency_saved_ms": latency_saved_ms,
        "net_latency_reduction_ratio": (
            latency_reduction_ratio
        ),

        # Output for later ROUGE/BERTScore/factuality evaluation
        "summary": generation.summary,
        "reference": example.reference,
    }

    if save_selected_text:
        row["selected_text"] = selection.selected_text

    # ROUGE-1/2/L vs reference summary (only when --rouge and a reference
    # exists; uses the shared dependency-free scripts/common/rouge.py).
    if (
        compute_rouge
        and ROUGE_AVAILABLE
        and example.reference
        and generation.summary
    ):
        _rouge.add_rouge(row, generation.summary, example.reference)

    # Preserve simple input metadata such as dataset labels/split/domain.
    if example.metadata:
        row["example_metadata"] = example.metadata

    return _json_safe(row)


def aggregate_results(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate request metrics by selector + budget label."""
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}

    for row in rows:
        key = (
            str(row["selector"]),
            str(row["budget_label"]),
        )
        groups.setdefault(key, []).append(row)

    aggregate: Dict[str, Any] = {
        "groups": []
    }

    for (selector, budget_label), group in sorted(groups.items()):
        def ratio_of_means(
            dense_key: str,
            method_key: str,
        ) -> Optional[float]:
            pairs = [
                (float(row[dense_key]), float(row[method_key]))
                for row in group
                if row.get(dense_key) is not None
                and row.get(method_key) is not None
                and float(row[dense_key]) > 0.0
                and float(row[method_key]) > 0.0
            ]
            if not pairs:
                return None
            return round(
                statistics.fmean(d for d, _ in pairs)
                / statistics.fmean(m for _, m in pairs),
                4,
            )

        selection_latencies = [
            float(r["selection_total_wall_ms"])
            for r in group
        ]

        prefill_latencies = [
            float(r["prefill_ms"])
            for r in group
        ]

        pipeline_ttft = [
            float(r["pipeline_ttft_ms"])
            for r in group
        ]

        pipeline_e2e = [
            float(r["pipeline_e2e_ms"])
            for r in group
        ]

        selection_summary = summarize_latencies_ms(
            selection_latencies
        )
        prefill_summary = summarize_latencies_ms(
            prefill_latencies
        )
        ttft_summary = summarize_latencies_ms(
            pipeline_ttft
        )
        e2e_summary = summarize_latencies_ms(
            pipeline_e2e
        )

        group_summary = {
            "selector": selector,
            "budget_label": budget_label,
            "count": len(group),

            "mean_original_tokens": statistics.fmean(
                float(r["original_tokens"])
                for r in group
            ),
            "mean_selected_tokens": statistics.fmean(
                float(r["selected_tokens"])
                for r in group
            ),
            "mean_output_tokens": statistics.fmean(
                float(r["output_tokens"])
                for r in group
            ),
            "mean_retention_ratio": statistics.fmean(
                float(r["retention_ratio"])
                for r in group
            ),
            "mean_token_reduction_ratio": statistics.fmean(
                float(r["token_reduction_ratio"])
                for r in group
            ),

            "selection_latency_ms": (
                selection_summary.to_dict()
            ),
            "prefill_latency_ms": (
                prefill_summary.to_dict()
            ),
            "pipeline_ttft_ms": (
                ttft_summary.to_dict()
            ),
            "pipeline_e2e_ms": (
                e2e_summary.to_dict()
            ),

            "mean_tpot_ms": _mean_or_none(
                r.get("tpot_ms")
                for r in group
            ),
            "mean_output_tokens_per_second": _mean_or_none(
                r.get("output_tokens_per_second")
                for r in group
            ),
            "mean_peak_gpu_allocated_mb": _mean_or_none(
                r.get("peak_gpu_allocated_mb")
                for r in group
            ),
            "mean_e2e_speedup_vs_full": _mean_or_none(
                r.get("e2e_speedup_vs_full")
                for r in group
            ),
            "mean_ttft_speedup_vs_full": _mean_or_none(
                r.get("ttft_speedup_vs_full")
                for r in group
            ),
            "mean_prefill_speedup_vs_full": _mean_or_none(
                r.get("prefill_speedup_vs_full")
                for r in group
            ),
            "mean_decode_speedup_vs_full": _mean_or_none(
                r.get("decode_speedup_vs_full")
                for r in group
            ),
            # Benchmark-level ratios use ratio of means, matching
            # scripts/common/metrics.py and avoiding short requests
            # dominating the aggregate.
            "esr": ratio_of_means(
                "baseline_full_e2e_ms",
                "pipeline_e2e_ms",
            ),
            "dsr": ratio_of_means(
                "baseline_full_decode_ms",
                "decode_ms",
            ),
            "prefill_speedup": ratio_of_means(
                "baseline_full_prefill_ms",
                "prefill_ms",
            ),
            "ttft_speedup": ratio_of_means(
                "baseline_full_ttft_ms",
                "pipeline_ttft_ms",
            ),
            "mean_net_latency_reduction_ratio": _mean_or_none(
                r.get("net_latency_reduction_ratio")
                for r in group
            ),

            # ROUGE quality (only present when --rouge + references exist)
            "mean_rouge1": _mean_or_none(
                r.get("rouge1")
                for r in group
            ),
            "mean_rouge2": _mean_or_none(
                r.get("rouge2")
                for r in group
            ),
            "mean_rougeL": _mean_or_none(
                r.get("rougeL")
                for r in group
            ),
        }

        aggregate["groups"].append(
            _json_safe(group_summary)
        )

    return aggregate


# ============================================================================
# Warm-up
# ============================================================================


def warmup_target(
    model,
    tokenizer,
    *,
    device: torch.device,
    rounds: int,
    disable_thinking: bool,
) -> None:
    if rounds <= 0:
        return

    warmup_document = (
        "Large language models can summarize long documents. "
        "This is a short warm-up request used only to initialize GPU kernels."
    )

    for _ in range(rounds):
        _ = generate_greedy_profiled(
            model,
            tokenizer,
            warmup_document,
            device=device,
            max_new_tokens=4,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            instruction="Summarize the document in one short sentence.",
            disable_thinking=disable_thinking,
        )


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3-4B summarization inference with lightweight "
            "semantic-selection baselines."
        )
    )

    # Data
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=".txt, .json, or .jsonl input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--document-field",
        default="document",
    )
    parser.add_argument(
        "--id-field",
        default="id",
    )
    parser.add_argument(
        "--reference-field",
        default="reference",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save-selected-text",
        action="store_true",
        help="Store selected context in JSONL (can make files large).",
    )
    parser.add_argument(
        "--rouge",
        action="store_true",
        help=(
            "Compute ROUGE-1/2/L (F1) against the reference summary and store "
            "per-row rouge1/rouge2/rougeL plus per-group means. Uses the "
            "dependency-free implementation in scripts/common/rouge.py."
        ),
    )

    # Baselines / budgets
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=VALID_SELECTORS,
        default=list(VALID_SELECTORS),
    )

    budget_group = parser.add_mutually_exclusive_group()

    budget_group.add_argument(
        "--token-budgets",
        nargs="+",
        type=int,
        default=None,
        help="Absolute selected-context token budgets.",
    )

    budget_group.add_argument(
        "--retention-ratios",
        nargs="+",
        type=float,
        default=None,
        help="Per-document fractions of source tokens to retain.",
    )

    # Qwen target
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "Allow Hugging Face downloads for Qwen. "
            "Default uses local cache only."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:0, cpu, ...",
    )
    parser.add_argument(
        "--dtype",
        choices=[
            "auto",
            "float16",
            "bfloat16",
            "float32",
        ],
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=[
            "auto",
            "eager",
            "sdpa",
            "flash_attention_2",
        ],
        default="auto",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable Qwen3 thinking mode. Default is disabled for "
            "summarization latency experiments."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
    )

    # Random
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    # MMR
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--embedding-device",
        default="cpu",
        help=(
            "Sentence encoder device. CPU is recommended to keep "
            "selector compute separate from Qwen GPU inference."
        ),
    )
    parser.add_argument(
        "--embedding-allow-download",
        action="store_true",
        help=(
            "Allow downloading MiniLM/sentence encoder. "
            "Default requires it in the local cache."
        ),
    )

    # TextRank
    parser.add_argument(
        "--textrank-top-k-neighbors",
        type=int,
        default=None,
        help=(
            "Optional graph sparsification for documents with many sentences."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0")

    if args.rouge and not ROUGE_AVAILABLE:
        print(
            "[rouge] WARNING: shared scripts/common/rouge.py not importable "
            "— ROUGE will be skipped"
        )

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")

    if not 0.0 <= args.mmr_lambda <= 1.0:
        raise ValueError("--mmr-lambda must be in [0, 1]")

    examples = load_examples(
        args.input,
        document_field=args.document_field,
        id_field=args.id_field,
        reference_field=args.reference_field,
        limit=args.limit,
    )

    device = choose_device(args.device)
    dtype = choose_dtype(device, args.dtype)

    print(
        f"[target] model={args.model} device={device} dtype={dtype}"
    )
    print(
        f"[data] {len(examples)} documents from {args.input}"
    )

    load_start = time.perf_counter()

    tokenizer, model = load_target(
        args.model,
        device=device,
        dtype=dtype,
        local_files_only=not args.allow_download,
        attn_implementation=args.attn_implementation,
    )

    target_load_ms = _elapsed_ms(load_start)

    print(
        f"[target] loaded in {target_load_ms / 1000.0:.2f}s"
    )

    selectors, selector_init_ms = initialize_selectors(
        args.selectors,
        tokenizer=tokenizer,
        tokenizer_name=args.model,
        random_seed=args.random_seed,
        mmr_lambda=args.mmr_lambda,
        embedding_model_name=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_local_files_only=(
            not args.embedding_allow_download
        ),
        textrank_top_k_neighbors=(
            args.textrank_top_k_neighbors
        ),
    )

    for name in args.selectors:
        print(
            f"[selector] {name}: init "
            f"{selector_init_ms[name]:.2f} ms"
        )

    if args.warmup_rounds > 0:
        print(
            f"[target] warm-up x{args.warmup_rounds}"
        )

        warmup_target(
            model,
            tokenizer,
            device=device,
            rounds=args.warmup_rounds,
            disable_thinking=not args.enable_thinking,
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: List[Dict[str, Any]] = []

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as writer:

        for example_index, example in enumerate(
            examples,
            start=1,
        ):
            print(
                f"[{example_index}/{len(examples)}] "
                f"id={example.example_id}"
            )

            # --------------------------------------------------------------
            # Full-context reference run
            # --------------------------------------------------------------
            full_selection = make_full_context_result(
                example.document,
                tokenizer,
            )

            full_generation = generate_greedy_profiled(
                model,
                tokenizer,
                example.document,
                device=device,
                max_new_tokens=args.max_new_tokens,
                system_prompt=args.system_prompt,
                instruction=args.instruction,
                disable_thinking=not args.enable_thinking,
            )

            full_row = make_result_row(
                example=example,
                selector_name="full",
                budget_label="full",
                requested_retention_ratio=1.0,
                selection=full_selection,
                generation=full_generation,
                selector_init_ms=0.0,
                baseline_generation=full_generation,
                save_selected_text=args.save_selected_text,
                compute_rouge=args.rouge,
            )

            rows.append(full_row)
            writer.write(
                json.dumps(
                    full_row,
                    ensure_ascii=False,
                )
                + "\n"
            )
            writer.flush()

            print(
                "  full: "
                f"prompt={full_generation.prompt_tokens} "
                f"out={full_generation.output_tokens} "
                f"TTFT={full_generation.target_request_ttft_ms:.1f}ms "
                f"E2E={full_generation.target_request_e2e_ms:.1f}ms"
            )

            # Budget is based on document-content tokens, not chat-template
            # tokens, because semantic selectors operate on source content.
            budgets = resolve_budgets(
                full_selection.original_tokens,
                token_budgets=args.token_budgets,
                retention_ratios=args.retention_ratios,
            )

            # --------------------------------------------------------------
            # Selected-context runs
            # --------------------------------------------------------------
            for budget_label, token_budget, requested_ratio in budgets:

                for selector_name in args.selectors:
                    selector = selectors[selector_name]

                    selection = selector.select_profiled(
                        example.document,
                        token_budget=token_budget,
                    )

                    generation = generate_greedy_profiled(
                        model,
                        tokenizer,
                        selection.selected_text,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        system_prompt=args.system_prompt,
                        instruction=args.instruction,
                        disable_thinking=not args.enable_thinking,
                    )

                    row = make_result_row(
                        example=example,
                        selector_name=selector_name,
                        budget_label=budget_label,
                        requested_retention_ratio=requested_ratio,
                        selection=selection,
                        generation=generation,
                        selector_init_ms=selector_init_ms[
                            selector_name
                        ],
                        baseline_generation=full_generation,
                        save_selected_text=args.save_selected_text,
                        compute_rouge=args.rouge,
                    )

                    rows.append(row)

                    writer.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    writer.flush()

                    speedup = row["e2e_speedup_vs_full"]

                    speedup_text = (
                        f"{speedup:.3f}x"
                        if speedup is not None
                        else "n/a"
                    )

                    print(
                        f"  {selector_name:8s} "
                        f"budget={token_budget:5d} "
                        f"keep={selection.retention_ratio:.3f} "
                        f"sel={row['selection_total_wall_ms']:.1f}ms "
                        f"prefill={generation.prefill_ms:.1f}ms "
                        f"E2E={row['pipeline_e2e_ms']:.1f}ms "
                        f"speedup={speedup_text}"
                    )

    # ----------------------------------------------------------------------
    # Aggregate summary
    # ----------------------------------------------------------------------
    aggregate = aggregate_results(rows)

    aggregate["run"] = {
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "disable_thinking": not args.enable_thinking,
        "target_load_ms": target_load_ms,
        "warmup_rounds": args.warmup_rounds,
        "selectors": list(args.selectors),
        "token_budgets": args.token_budgets,
        "retention_ratios": args.retention_ratios,
        "num_documents": len(examples),
        "random_seed": args.random_seed,
        "mmr_lambda": args.mmr_lambda,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "selector_init_ms": selector_init_ms,
        "rouge": args.rouge,
    }

    summary_path = args.output.with_suffix(
        args.output.suffix + ".summary.json"
    )

    summary_path.write_text(
        json.dumps(
            _json_safe(aggregate),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"[done] request rows: {args.output}")
    print(f"[done] aggregate:    {summary_path}")


if __name__ == "__main__":
    main()
