"""
TF-IDF centroid baseline for lightweight content selection.

The selector represents every sentence with a TF-IDF vector, computes a
document centroid from all sentence vectors, and ranks sentences by cosine
similarity to that centroid.

This is a cheap, CPU-only lexical relevance baseline. It uses no neural model.
The final token budget is still enforced by BaseSemanticSelector with the
target-model tokenizer (Qwen3-4B by default).

Scoring
-------
Let v_i be the TF-IDF vector for sentence i and c the document centroid:

    c = sum_i w_i v_i / sum_i w_i

Then:

    score(s_i) = cosine(v_i, c)

where w_i is either 1 (uniform) or the sentence's target-token count.

Example
-------
from tfidf import TFIDFCentroidSelector

selector = TFIDFCentroidSelector()
result = selector.select(document, token_budget=2048)

print(result.selected_text)
print(result.selection_time_ms)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .base import BaseSemanticSelector, SelectionResult
except ImportError:
    from base import BaseSemanticSelector, SelectionResult


class TFIDFCentroidSelector(BaseSemanticSelector):
    """
    Rank sentences by TF-IDF similarity to the document centroid.

    Parameters
    ----------
    lowercase:
        Lowercase text before TF-IDF extraction.

    ngram_range:
        Word n-gram range used by TfidfVectorizer. ``(1, 1)`` is the classical
        lightweight default.

    min_df:
        Minimum document/sentence frequency for vocabulary terms.

    max_df:
        Maximum document/sentence frequency for vocabulary terms.

    sublinear_tf:
        Use ``1 + log(tf)`` instead of raw term frequency.

    centroid_weighting:
        ``"uniform"`` averages sentence vectors equally.
        ``"token"`` weights them by Qwen target-token count.

    max_features:
        Optional vocabulary cap. ``None`` preserves all observed features.

    All remaining keyword arguments are forwarded to BaseSemanticSelector.
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        sublinear_tf: bool = True,
        centroid_weighting: str = "uniform",
        max_features: int | None = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)

        if centroid_weighting not in {"uniform", "token"}:
            raise ValueError(
                "centroid_weighting must be 'uniform' or 'token', "
                f"got {centroid_weighting!r}"
            )

        if ngram_range[0] <= 0 or ngram_range[1] < ngram_range[0]:
            raise ValueError(f"Invalid ngram_range: {ngram_range}")

        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.centroid_weighting = centroid_weighting
        self.max_features = max_features

    @property
    def name(self) -> str:
        return "tfidf_centroid"

    def select(
        self,
        document: str,
        token_budget: int,
        **kwargs: Any,
    ) -> SelectionResult:
        result = super().select(
            document=document,
            token_budget=token_budget,
            **kwargs,
        )
        result.metadata.update(
            {
                "representation": "tfidf",
                "centroid_weighting": self.centroid_weighting,
                "ngram_range": self.ngram_range,
                "lowercase": self.lowercase,
                "sublinear_tf": self.sublinear_tf,
                "max_features": self.max_features,
            }
        )
        return result

    def _make_vectorizer(self) -> TfidfVectorizer:
        """
        Construct a language-agnostic TF-IDF vectorizer.

        We intentionally do not use an English stop-word list because the
        benchmark may include multilingual documents. The Unicode-aware default
        token pattern keeps words with at least two alphanumeric characters.
        """
        return TfidfVectorizer(
            lowercase=self.lowercase,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            max_features=self.max_features,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
        )

    def _select_priority(
        self,
        sentences: Sequence[str],
        token_counts: Sequence[int],
        token_budget: int,
        **kwargs: Any,
    ) -> Sequence[int]:
        del token_budget, kwargs

        n = len(sentences)
        if n <= 1:
            return list(range(n))

        vectorizer = self._make_vectorizer()

        try:
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # Empty vocabulary, e.g. punctuation-only or extremely unusual text.
            # Deterministic positional fallback keeps the baseline usable.
            return list(range(n))

        if matrix.shape[1] == 0 or matrix.nnz == 0:
            return list(range(n))

        centroid = self._compute_centroid(
            matrix=matrix,
            token_counts=token_counts,
        )

        centroid_norm = float(sparse.linalg.norm(centroid))
        if centroid_norm <= 1e-12:
            return list(range(n))

        # Sentence rows are already L2-normalized by TfidfVectorizer.
        # Divide by centroid norm to obtain cosine similarity.
        scores = (matrix @ centroid.T).toarray().reshape(-1) / centroid_norm

        # Deterministic ranking:
        #   1) higher similarity to centroid,
        #   2) earlier sentence position.
        priority = sorted(
            range(n),
            key=lambda i: (-float(scores[i]), i),
        )

        return priority

    def _compute_centroid(
        self,
        matrix: sparse.csr_matrix,
        token_counts: Sequence[int],
    ) -> sparse.csr_matrix:
        """Return a 1 x vocab sparse centroid vector."""
        if self.centroid_weighting == "uniform":
            centroid = matrix.mean(axis=0)
            return sparse.csr_matrix(centroid)

        weights = np.asarray(token_counts, dtype=np.float64)
        weights = np.maximum(weights, 1.0)
        weights /= weights.sum()

        centroid = matrix.multiply(weights[:, None]).sum(axis=0)
        return sparse.csr_matrix(centroid)
