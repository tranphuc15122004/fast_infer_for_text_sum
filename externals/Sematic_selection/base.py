"""
Base interfaces and utilities for lightweight semantic-selection baselines.

Designed for long-document summarization experiments where every selector is
evaluated under the same token budget and the downstream target model uses the
Qwen3-4B tokenizer.

The base class handles:
    - loading a cached tokenizer (offline by default),
    - sentence segmentation,
    - exact token counting,
    - budget enforcement,
    - restoration of original document order,
    - selector latency measurement,
    - standardized result objects,
    - batch selection.

Subclasses only need to implement `_select_priority(...)`, which returns
sentence indices in the order the selector would like to keep them.

Example subclass
----------------
class LeadSelector(BaseSemanticSelector):
    def _select_priority(self, sentences, token_counts, token_budget, **kwargs):
        return list(range(len(sentences)))
"""

from __future__ import annotations

import re
import os
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from transformers import AutoTokenizer, PreTrainedTokenizerBase


SentenceSplitter = Callable[[str], List[str]]


def resolve_local_model_path(model_name: str) -> str:
    """Resolve a cached Hub model ID to a snapshot directory when possible.

    Recent ``transformers`` versions may still query the Hub for model
    metadata even with ``local_files_only=True`` when given a model ID. Using
    the concrete snapshot path avoids that network access and makes offline
    benchmark runs deterministic.
    """
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate)

    if "/" not in model_name:
        return model_name

    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path.home() / ".cache" / "huggingface" / "hub",
        )
    )
    repo_cache = cache_root / (
        "models--" + model_name.replace("/", "--")
    )
    snapshots = repo_cache / "snapshots"

    if snapshots.is_dir():
        candidates = [path for path in snapshots.iterdir() if path.is_dir()]
        if candidates:
            return str(max(candidates, key=lambda path: path.stat().st_mtime))

    return model_name


@dataclass
class SentenceUnit:
    """One sentence extracted from the original document."""

    index: int
    text: str
    token_count: int


@dataclass
class SelectionResult:
    """Standardized output returned by every semantic selector."""

    selected_text: str
    selected_sentences: List[str]
    selected_indices: List[int]

    selected_tokens: int
    original_tokens: int
    token_budget: int

    retention_ratio: float
    selection_time_ms: float

    selector_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        """Number of source tokens removed by the selector."""
        return max(self.original_tokens - self.selected_tokens, 0)

    @property
    def token_reduction_ratio(self) -> float:
        """Fraction of source tokens removed by the selector."""
        if self.original_tokens <= 0:
            return 0.0
        return self.tokens_saved / self.original_tokens

    @property
    def compression_ratio(self) -> Optional[float]:
        """Original/selected token ratio, or ``None`` for empty output."""
        if self.selected_tokens <= 0:
            return None
        return self.original_tokens / self.selected_tokens

    @property
    def budget_utilization(self) -> float:
        """Fraction of the requested token budget that was used."""
        if self.token_budget <= 0:
            return 0.0
        return self.selected_tokens / self.token_budget

    @property
    def original_sentence_count(self) -> int:
        """Number of sentences available to the selector."""
        return int(self.metadata.get("num_sentences", 0))

    @property
    def selected_sentence_count(self) -> int:
        """Number of sentences retained by the selector."""
        return int(
            self.metadata.get(
                "num_selected_sentences",
                len(self.selected_indices),
            )
        )

    @property
    def sentence_retention_ratio(self) -> float:
        """Fraction of source sentences retained by the selector."""
        if self.original_sentence_count <= 0:
            return 0.0
        return self.selected_sentence_count / self.original_sentence_count

    @property
    def selector_input_tokens_per_second(self) -> Optional[float]:
        """Source-token processing rate for the measured selector phase."""
        if self.selection_time_ms <= 0.0:
            return None
        return self.original_tokens / (self.selection_time_ms / 1000.0)

    @property
    def tokens_saved_per_ms(self) -> Optional[float]:
        """Token reduction per measured selector millisecond."""
        if self.selection_time_ms <= 0.0:
            return None
        return self.tokens_saved / self.selection_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatencySummary:
    """Small JSON-friendly percentile summary used by the runner."""

    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    minimum: float
    maximum: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_latencies_ms(values: Sequence[float]) -> LatencySummary:
    """Summarize non-empty latency samples in milliseconds."""
    if not values:
        raise ValueError("values cannot be empty")

    data = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        if len(data) == 1:
            return data[0]

        position = (len(data) - 1) * q / 100.0
        lower = int(position)
        upper = min(lower + 1, len(data) - 1)
        fraction = position - lower
        return data[lower] * (1.0 - fraction) + data[upper] * fraction

    return LatencySummary(
        count=len(data),
        mean=statistics.fmean(data),
        p50=percentile(50.0),
        p95=percentile(95.0),
        p99=percentile(99.0),
        minimum=data[0],
        maximum=data[-1],
    )


class BaseSemanticSelector(ABC):
    """
    Abstract base class for sentence-level semantic/context selectors.

    Parameters
    ----------
    tokenizer_name:
        Hugging Face model/tokenizer identifier. The default is Qwen3-4B.
        Only the tokenizer is loaded; the 4B model weights are NOT loaded.

    local_files_only:
        If True, Hugging Face is forced to use the local cache. This is
        recommended for the current experiment because Qwen3-4B is already
        cached on the server.

    preserve_order:
        Ranking-based methods such as TF-IDF, TextRank, and MMR usually select
        sentences in score order. For summarization, the final selected context
        should normally be restored to the document's original order.

    separator:
        Text inserted between selected sentences before sending them to the
        downstream target model.

    sentence_splitter:
        Optional custom sentence-splitting callable. If omitted, a lightweight
        regex-based splitter is used. This deliberately avoids NLTK/spaCy
        downloads and makes the baseline easy to deploy.

    add_special_tokens:
        Whether token-budget accounting includes special tokens. For prompt
        content selection this should normally be False.

    allow_partial_fallback:
        If no complete sentence can fit inside a very small token budget, keep
        a truncated prefix of the first preferred sentence rather than return
        an empty context.
    """

    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen3-4B",
        *,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        local_files_only: bool = True,
        preserve_order: bool = True,
        separator: str = "\n",
        sentence_splitter: Optional[SentenceSplitter] = None,
        add_special_tokens: bool = False,
        allow_partial_fallback: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.local_files_only = local_files_only
        self.preserve_order = preserve_order
        self.separator = separator
        self.sentence_splitter = sentence_splitter
        self.add_special_tokens = add_special_tokens
        self.allow_partial_fallback = allow_partial_fallback

        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    resolve_local_model_path(tokenizer_name)
                    if local_files_only
                    else tokenizer_name,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                    use_fast=True,
                )
            except Exception as exc:
                mode = "local cache only" if local_files_only else "local cache/network"
                raise RuntimeError(
                    f"Could not load tokenizer '{tokenizer_name}' using {mode}. "
                    "If Qwen3-4B is cached under another Hugging Face model ID or "
                    "a local path, pass it via tokenizer_name=... ."
                ) from exc

        self.tokenizer = tokenizer

    @property
    def name(self) -> str:
        """Human-readable selector name used in experiment logs."""
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        document: str,
        token_budget: int,
        **kwargs: Any,
    ) -> SelectionResult:
        """
        Select document content under a strict target-tokenizer budget.

        The selector-specific ranking/selection logic is implemented by
        `_select_priority`. The base class then enforces the budget and returns
        selected sentences in original document order (by default).
        """
        if not isinstance(document, str):
            raise TypeError(f"document must be str, got {type(document).__name__}")

        if token_budget <= 0:
            raise ValueError(f"token_budget must be > 0, got {token_budget}")

        document = document.strip()
        if not document:
            return SelectionResult(
                selected_text="",
                selected_sentences=[],
                selected_indices=[],
                selected_tokens=0,
                original_tokens=0,
                token_budget=token_budget,
                retention_ratio=0.0,
                selection_time_ms=0.0,
                selector_name=self.name,
                metadata={"num_sentences": 0},
            )

        original_tokens = self.count_tokens(document)
        sentences = self.build_sentence_units(document)

        if not sentences:
            return SelectionResult(
                selected_text="",
                selected_sentences=[],
                selected_indices=[],
                selected_tokens=0,
                original_tokens=original_tokens,
                token_budget=token_budget,
                retention_ratio=0.0,
                selection_time_ms=0.0,
                selector_name=self.name,
                metadata={"num_sentences": 0},
            )

        # No reduction is required. This avoids unnecessary selector work and
        # gives every method identical behavior when budget >= document length.
        if original_tokens <= token_budget:
            return SelectionResult(
                selected_text=document,
                selected_sentences=[s.text for s in sentences],
                selected_indices=[s.index for s in sentences],
                selected_tokens=original_tokens,
                original_tokens=original_tokens,
                token_budget=token_budget,
                retention_ratio=1.0,
                selection_time_ms=0.0,
                selector_name=self.name,
                metadata={
                    "num_sentences": len(sentences),
                    "selection_skipped": True,
                },
            )

        sentence_texts = [s.text for s in sentences]
        sentence_token_counts = [s.token_count for s in sentences]

        start = time.perf_counter()
        priority = self._select_priority(
            sentences=sentence_texts,
            token_counts=sentence_token_counts,
            token_budget=token_budget,
            **kwargs,
        )

        priority = self._validate_priority(priority, len(sentences))
        selected_priority = self._fit_priority_to_budget(
            priority=priority,
            sentences=sentences,
            token_budget=token_budget,
        )

        partial_text: Optional[str] = None
        partial_index: Optional[int] = None

        if not selected_priority and self.allow_partial_fallback and priority:
            partial_index = priority[0]
            partial_text = self.truncate_to_token_budget(
                sentences[partial_index].text,
                token_budget,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if partial_text is not None:
            selected_text = partial_text
            selected_sentences = [partial_text]
            selected_indices = [partial_index] if partial_index is not None else []
        else:
            output_indices = (
                sorted(selected_priority)
                if self.preserve_order
                else list(selected_priority)
            )
            selected_sentences = [sentences[i].text for i in output_indices]
            selected_indices = output_indices
            selected_text = self.separator.join(selected_sentences)

        # Sum-of-sentence token counts can differ slightly from tokenizing the
        # concatenated text, so always report the exact downstream token count.
        selected_tokens = self.count_tokens(selected_text)

        # Defensive exact-budget enforcement.
        if selected_tokens > token_budget:
            selected_text = self.truncate_to_token_budget(selected_text, token_budget)
            selected_tokens = self.count_tokens(selected_text)

        retention_ratio = (
            selected_tokens / original_tokens if original_tokens > 0 else 0.0
        )

        return SelectionResult(
            selected_text=selected_text,
            selected_sentences=selected_sentences,
            selected_indices=selected_indices,
            selected_tokens=selected_tokens,
            original_tokens=original_tokens,
            token_budget=token_budget,
            retention_ratio=retention_ratio,
            selection_time_ms=elapsed_ms,
            selector_name=self.name,
            metadata={
                "num_sentences": len(sentences),
                "num_selected_sentences": len(selected_indices),
                "priority_indices": priority,
                "preserve_order": self.preserve_order,
            },
        )

    def select_batch(
        self,
        documents: Sequence[str],
        token_budget: int,
        **kwargs: Any,
    ) -> List[SelectionResult]:
        """
        Apply the selector to multiple documents.

        This generic implementation is intentionally simple. Embedding-based
        subclasses may override it later to batch encoder inference efficiently.
        """
        return [
            self.select(document, token_budget=token_budget, **kwargs)
            for document in documents
        ]

    def select_profiled(
        self,
        document: str,
        token_budget: int,
        **kwargs: Any,
    ) -> SelectionResult:
        """Select while also measuring preprocessing and ranking overhead.

        ``selection_time_ms`` remains the selector-specific ranking/embedding
        phase measured by :meth:`select`.  The additional metadata field
        ``selection_total_wall_time_ms`` covers the complete per-request
        selector path, including sentence splitting and token counting.
        """
        start = time.perf_counter()
        result = self.select(
            document=document,
            token_budget=token_budget,
            **kwargs,
        )
        total_ms = (time.perf_counter() - start) * 1000.0
        result.metadata["selection_total_wall_time_ms"] = total_ms
        result.metadata["selection_preprocess_overhead_ms"] = max(
            total_ms - result.selection_time_ms,
            0.0,
        )
        return result

    # ------------------------------------------------------------------
    # Required selector-specific method
    # ------------------------------------------------------------------

    @abstractmethod
    def _select_priority(
        self,
        sentences: Sequence[str],
        token_counts: Sequence[int],
        token_budget: int,
        **kwargs: Any,
    ) -> Sequence[int]:
        """
        Return sentence indices in descending keep priority.

        Examples
        --------
        Lead-k:
            return [0, 1, 2, ..., n-1]

        TF-IDF:
            return indices sorted by relevance score descending.

        MMR:
            return the greedy MMR selection order.

        Notes
        -----
        Budget enforcement is centralized in BaseSemanticSelector, so
        subclasses should focus on ranking/selection logic only.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens exactly with the target model tokenizer."""
        if not text:
            return 0

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=self.add_special_tokens,
            )
        )

    def tokenize(self, text: str) -> List[int]:
        """Return token IDs using the same settings as budget accounting."""
        return self.tokenizer.encode(
            text,
            add_special_tokens=self.add_special_tokens,
        )

    def truncate_to_token_budget(self, text: str, token_budget: int) -> str:
        """Truncate text to at most `token_budget` target-model tokens."""
        if token_budget <= 0 or not text:
            return ""

        token_ids = self.tokenize(text)
        if len(token_ids) <= token_budget:
            return text

        token_ids = token_ids[:token_budget]
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def split_sentences(self, document: str) -> List[str]:
        """
        Lightweight sentence segmentation without external model downloads.

        The default rule splits on:
            - one or more blank/new lines,
            - whitespace following common sentence-ending punctuation.

        For production-quality multilingual segmentation, inject a custom
        `sentence_splitter` when constructing the selector.
        """
        if self.sentence_splitter is not None:
            raw = self.sentence_splitter(document)
        else:
            normalized = re.sub(r"\r\n?", "\n", document.strip())
            raw = re.split(
                r"(?:\n+|(?<=[.!?。！？])\s+)",
                normalized,
            )

        sentences = []
        for sentence in raw:
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def build_sentence_units(self, document: str) -> List[SentenceUnit]:
        """Split a document and attach exact token counts to each sentence."""
        sentence_texts = self.split_sentences(document)

        return [
            SentenceUnit(
                index=i,
                text=text,
                token_count=self.count_tokens(text),
            )
            for i, text in enumerate(sentence_texts)
        ]

    def _fit_priority_to_budget(
        self,
        priority: Sequence[int],
        sentences: Sequence[SentenceUnit],
        token_budget: int,
    ) -> List[int]:
        """
        Greedily keep preferred complete sentences while respecting budget.

        A sentence that does not fit is skipped rather than terminating the
        search. This is important for score-based selectors because a later,
        shorter sentence may still fit the remaining budget.
        """
        chosen: List[int] = []

        for idx in priority:
            candidate = chosen + [idx]

            output_indices = (
                sorted(candidate)
                if self.preserve_order
                else candidate
            )
            candidate_text = self.separator.join(
                sentences[i].text for i in output_indices
            )

            if self.count_tokens(candidate_text) <= token_budget:
                chosen.append(idx)

        return chosen

    @staticmethod
    def _validate_priority(
        priority: Iterable[int],
        num_sentences: int,
    ) -> List[int]:
        """Validate, de-duplicate, and normalize selector output."""
        if priority is None:
            raise ValueError("_select_priority() returned None")

        normalized: List[int] = []
        seen = set()

        for idx in priority:
            if not isinstance(idx, int):
                raise TypeError(
                    "_select_priority() must return integer sentence indices; "
                    f"got {type(idx).__name__}"
                )

            if idx < 0 or idx >= num_sentences:
                raise IndexError(
                    f"Sentence index {idx} out of range for {num_sentences} sentences"
                )

            if idx not in seen:
                seen.add(idx)
                normalized.append(idx)

        return normalized
