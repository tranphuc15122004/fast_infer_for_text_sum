"""
Embedding + Maximal Marginal Relevance (MMR) baseline.

This selector is intended for lightweight semantic content selection before
long-document summarization. It embeds sentences with a small sentence encoder,
represents the document by the centroid of its sentence embeddings, and greedily
selects sentences that balance:

    relevance to the document
        vs.
    redundancy with already-selected sentences.

MMR objective
-------------
For candidate sentence i and selected set S:

    score(i) =
        lambda * sim(e_i, e_doc)
        - (1 - lambda) * max_{j in S} sim(e_i, e_j)

where cosine similarity is used and sentence embeddings are L2-normalized.

The target token budget is always measured with the tokenizer configured in
BaseSemanticSelector (Qwen3-4B by default).

Important benchmarking detail
-----------------------------
The sentence-encoder model is loaded in __init__, while BaseSemanticSelector
starts timing only inside select(). Therefore model-loading/warm-up cost is not
included in per-document selector latency, but sentence embedding inference is.

Example
-------
from mmr import MMRSelector

selector = MMRSelector(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_device="cpu",
    mmr_lambda=0.7,
)

result = selector.select(document, token_budget=2048)

print(result.selected_text)
print(result.selection_time_ms)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

try:
    # Package-style import.
    from .base import BaseSemanticSelector, SelectionResult
except ImportError:
    # Local-folder import.
    from base import BaseSemanticSelector, SelectionResult


class MMRSelector(BaseSemanticSelector):
    """
    Sentence-embedding + MMR selector.

    Parameters
    ----------
    embedding_model_name:
        SentenceTransformer checkpoint used for semantic sentence embeddings.

    embedding_model:
        Optional pre-loaded SentenceTransformer-compatible object. Supplying one
        is useful when several selectors share the same encoder or when the
        experiment runner manages model placement explicitly.

    embedding_device:
        Device used by the sentence encoder. ``"cpu"`` is the default because
        this baseline is intended to represent a lightweight non-LLM selector
        without consuming target-model GPU memory. Set to ``"cuda"`` only when
        that is part of the experiment protocol.

    embedding_local_files_only:
        If True, the sentence encoder must already exist in the local HF cache.
        This setting is separate from BaseSemanticSelector.local_files_only,
        which controls the Qwen tokenizer.

    mmr_lambda:
        Trade-off in [0, 1].
        - 1.0: pure relevance ranking.
        - 0.0: maximal diversity after the first sentence.
        A common starting value is 0.7.

    embedding_batch_size:
        Batch size for sentence embedding inference.

    centroid_weighting:
        ``"uniform"`` computes the mean of sentence embeddings.
        ``"token"`` weights each sentence by its Qwen-token count.

    All remaining keyword arguments are forwarded to BaseSemanticSelector.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        embedding_model: Optional[Any] = None,
        embedding_device: str = "cpu",
        embedding_local_files_only: bool = False,
        mmr_lambda: float = 0.7,
        embedding_batch_size: int = 64,
        centroid_weighting: str = "uniform",
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)

        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(
                f"mmr_lambda must be in [0, 1], got {mmr_lambda}"
            )

        if embedding_batch_size <= 0:
            raise ValueError(
                "embedding_batch_size must be > 0, "
                f"got {embedding_batch_size}"
            )

        if centroid_weighting not in {"uniform", "token"}:
            raise ValueError(
                "centroid_weighting must be either 'uniform' or 'token', "
                f"got {centroid_weighting!r}"
            )

        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.embedding_local_files_only = embedding_local_files_only
        self.mmr_lambda = float(mmr_lambda)
        self.embedding_batch_size = int(embedding_batch_size)
        self.centroid_weighting = centroid_weighting

        if embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "MMRSelector requires `sentence-transformers`. "
                    "Install it with `uv add sentence-transformers` (or the "
                    "equivalent package-manager command), then instantiate the "
                    "selector again."
                ) from exc

            try:
                embedding_model = SentenceTransformer(
                    embedding_model_name,
                    device=embedding_device,
                    local_files_only=embedding_local_files_only,
                )
            except Exception as exc:
                cache_hint = (
                    "local cache only"
                    if embedding_local_files_only
                    else "local cache/network"
                )
                raise RuntimeError(
                    f"Could not load sentence encoder "
                    f"'{embedding_model_name}' using {cache_hint}. "
                    "Pass another checkpoint/path through "
                    "`embedding_model_name`, or inject a pre-loaded encoder via "
                    "`embedding_model=`."
                ) from exc

        self.embedding_model = embedding_model

    @property
    def name(self) -> str:
        return "embedding_mmr"

    def select(
        self,
        document: str,
        token_budget: int,
        **kwargs: Any,
    ) -> SelectionResult:
        """Run MMR selection and attach method configuration to metadata."""
        result = super().select(
            document=document,
            token_budget=token_budget,
            **kwargs,
        )
        result.metadata.update(
            {
                "embedding_model": self.embedding_model_name,
                "embedding_device": self.embedding_device,
                "mmr_lambda": self.mmr_lambda,
                "embedding_batch_size": self.embedding_batch_size,
                "centroid_weighting": self.centroid_weighting,
            }
        )
        return result

    def _select_priority(
        self,
        sentences: Sequence[str],
        token_counts: Sequence[int],
        token_budget: int,
        **kwargs: Any,
    ) -> Sequence[int]:
        """
        Return a budget-aware greedy MMR selection order.

        ``mmr_lambda`` can optionally be overridden per call:

            selector.select(text, 2048, mmr_lambda=0.5)
        """
        if not sentences:
            return []

        mmr_lambda = float(kwargs.get("mmr_lambda", self.mmr_lambda))
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(
                f"mmr_lambda must be in [0, 1], got {mmr_lambda}"
            )

        embeddings = self._encode_sentences(sentences)
        doc_embedding = self._document_centroid(
            embeddings=embeddings,
            token_counts=token_counts,
        )

        # Embeddings and centroid are normalized, so dot product == cosine sim.
        relevance = embeddings @ doc_embedding

        selected: list[int] = []
        remaining = set(range(len(sentences)))

        # Approximate separator cost during greedy budget allocation. The base
        # class performs exact re-tokenization afterward, so the final result is
        # still guaranteed not to exceed token_budget.
        separator_tokens = self.count_tokens(self.separator)
        remaining_budget = token_budget

        while remaining:
            eligible = []
            for idx in remaining:
                extra_separator = separator_tokens if selected else 0
                estimated_cost = token_counts[idx] + extra_separator
                if estimated_cost <= remaining_budget:
                    eligible.append(idx)

            if not eligible:
                break

            best_idx = None
            best_key = None

            for idx in eligible:
                if selected:
                    redundancy = float(
                        np.max(embeddings[selected] @ embeddings[idx])
                    )
                else:
                    redundancy = 0.0

                score = (
                    mmr_lambda * float(relevance[idx])
                    - (1.0 - mmr_lambda) * redundancy
                )

                # Deterministic tie-breaking:
                # 1) higher MMR score,
                # 2) higher relevance,
                # 3) earlier document position.
                key = (score, float(relevance[idx]), -idx)

                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx

            assert best_idx is not None

            extra_separator = separator_tokens if selected else 0
            remaining_budget -= token_counts[best_idx] + extra_separator
            selected.append(best_idx)
            remaining.remove(best_idx)

        # If every complete sentence exceeds the budget, return the most
        # relevant sentence so BaseSemanticSelector can use its partial-sentence
        # fallback instead of returning an empty context.
        if not selected:
            best = max(
                range(len(sentences)),
                key=lambda i: (float(relevance[i]), -i),
            )
            return [best]

        return selected

    def _encode_sentences(
        self,
        sentences: Sequence[str],
    ) -> np.ndarray:
        """Encode and L2-normalize all sentences in one batched call."""
        embeddings = self.embedding_model.encode(
            list(sentences),
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(
                "Sentence encoder must return a 2-D array shaped "
                f"[num_sentences, dim], got {embeddings.shape}"
            )

        if embeddings.shape[0] != len(sentences):
            raise ValueError(
                "Sentence encoder returned a different number of embeddings "
                f"({embeddings.shape[0]}) than input sentences "
                f"({len(sentences)})."
            )

        return embeddings

    def _document_centroid(
        self,
        embeddings: np.ndarray,
        token_counts: Sequence[int],
    ) -> np.ndarray:
        """
        Build and normalize a document representation from sentence embeddings.

        Encoding the entire long document directly with MiniLM would truncate
        it at the encoder's context limit. Using the sentence centroid avoids
        that long-document truncation.
        """
        if self.centroid_weighting == "uniform":
            centroid = embeddings.mean(axis=0)
        else:
            weights = np.asarray(token_counts, dtype=np.float32)
            weights = np.maximum(weights, 1.0)
            weights = weights / weights.sum()
            centroid = np.sum(embeddings * weights[:, None], axis=0)

        norm = float(np.linalg.norm(centroid))
        if norm <= 1e-12:
            # Extremely unlikely, but keeps the method numerically safe.
            centroid = np.zeros_like(centroid, dtype=np.float32)
        else:
            centroid = centroid / norm

        return centroid.astype(np.float32, copy=False)