"""
TextRank baseline for lightweight sentence selection.

This implementation follows the standard graph-based TextRank idea:

    sentence -> TF-IDF representation
             -> weighted sentence-similarity graph
             -> PageRank centrality
             -> sentence priority
             -> target-token budget enforcement

It is CPU-only and requires no neural model. Cosine similarities between
L2-normalized TF-IDF sentence vectors are used as graph edge weights.

The implementation includes its own vectorized weighted PageRank routine rather
than depending on NetworkX, keeping the baseline small and making the measured
selector latency easier to interpret.

Example
-------
from textrank import TextRankSelector

selector = TextRankSelector()
result = selector.select(document, token_budget=2048)

print(result.selected_text)
print(result.metadata["pagerank_iterations"])
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .base import BaseSemanticSelector, SelectionResult
except ImportError:
    from base import BaseSemanticSelector, SelectionResult


class TextRankSelector(BaseSemanticSelector):
    """
    Weighted TextRank sentence selector.

    Parameters
    ----------
    damping:
        PageRank damping factor. Classical PageRank commonly uses 0.85.

    max_iter:
        Maximum PageRank iterations.

    tol:
        Convergence tolerance on L1 rank-vector change.

    similarity_threshold:
        Remove graph edges with cosine similarity <= this threshold.
        ``0.0`` retains all positive-similarity edges.

    top_k_neighbors:
        Optional graph sparsification. If set, each sentence keeps only its
        strongest K neighbors before PageRank. ``None`` uses the standard dense
        positive-similarity graph.

    lowercase, ngram_range, min_df, max_df, sublinear_tf, max_features:
        TF-IDF configuration. Defaults are intentionally lightweight and
        language-agnostic (no English-only stop-word list).

    All remaining keyword arguments are forwarded to BaseSemanticSelector.
    """

    def __init__(
        self,
        *,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        similarity_threshold: float = 0.0,
        top_k_neighbors: int | None = None,
        lowercase: bool = True,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        sublinear_tf: bool = True,
        max_features: int | None = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)

        if not 0.0 < damping < 1.0:
            raise ValueError(f"damping must be in (0, 1), got {damping}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if similarity_threshold < 0.0:
            raise ValueError(
                "similarity_threshold must be >= 0, "
                f"got {similarity_threshold}"
            )
        if top_k_neighbors is not None and top_k_neighbors <= 0:
            raise ValueError(
                f"top_k_neighbors must be > 0 or None, got {top_k_neighbors}"
            )
        if ngram_range[0] <= 0 or ngram_range[1] < ngram_range[0]:
            raise ValueError(f"Invalid ngram_range: {ngram_range}")

        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.similarity_threshold = float(similarity_threshold)
        self.top_k_neighbors = top_k_neighbors

        self.lowercase = lowercase
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features

        # Updated on every non-trivial selection call and exposed in metadata.
        self._last_pagerank_iterations = 0
        self._last_pagerank_converged = True
        self._last_num_edges = 0

    @property
    def name(self) -> str:
        return "textrank"

    def select(
        self,
        document: str,
        token_budget: int,
        **kwargs: Any,
    ) -> SelectionResult:
        # Reset call-specific diagnostic state. This matters when selection is
        # skipped because the full document already fits the requested budget.
        self._last_pagerank_iterations = 0
        self._last_pagerank_converged = True
        self._last_num_edges = 0

        result = super().select(
            document=document,
            token_budget=token_budget,
            **kwargs,
        )

        result.metadata.update(
            {
                "representation": "tfidf_cosine",
                "damping": self.damping,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "similarity_threshold": self.similarity_threshold,
                "top_k_neighbors": self.top_k_neighbors,
                "pagerank_iterations": self._last_pagerank_iterations,
                "pagerank_converged": self._last_pagerank_converged,
                "graph_edges": self._last_num_edges,
                "ngram_range": self.ngram_range,
            }
        )
        return result

    def _make_vectorizer(self) -> TfidfVectorizer:
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
        del token_counts, token_budget, kwargs

        n = len(sentences)
        if n <= 1:
            return list(range(n))

        vectorizer = self._make_vectorizer()

        try:
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # Empty vocabulary: use deterministic positional fallback.
            return list(range(n))

        if matrix.shape[1] == 0 or matrix.nnz == 0:
            return list(range(n))

        # TfidfVectorizer rows are L2-normalized, so X @ X.T is cosine sim.
        similarity = (matrix @ matrix.T).toarray().astype(np.float64, copy=False)

        # No self-loops in TextRank graph.
        np.fill_diagonal(similarity, 0.0)

        if self.similarity_threshold > 0.0:
            similarity[similarity <= self.similarity_threshold] = 0.0
        else:
            # Avoid tiny numerical negatives if any backend produces them.
            similarity[similarity < 0.0] = 0.0

        if self.top_k_neighbors is not None and n > 1:
            similarity = self._keep_top_k_neighbors(
                similarity,
                self.top_k_neighbors,
            )

        # Symmetrize after optional directed top-k pruning. TextRank sentence
        # similarity is naturally an undirected relation.
        similarity = np.maximum(similarity, similarity.T)
        np.fill_diagonal(similarity, 0.0)

        self._last_num_edges = int(np.count_nonzero(np.triu(similarity, k=1)))

        if self._last_num_edges == 0:
            # No lexical overlap across sentences. All PageRank scores would be
            # equal, so use original order as deterministic tie-break/fallback.
            return list(range(n))

        ranks, iterations, converged = self._weighted_pagerank(similarity)
        self._last_pagerank_iterations = iterations
        self._last_pagerank_converged = converged

        # Deterministic ranking:
        #   1) higher TextRank score,
        #   2) earlier document position.
        priority = sorted(
            range(n),
            key=lambda i: (-float(ranks[i]), i),
        )
        return priority

    @staticmethod
    def _keep_top_k_neighbors(
        similarity: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Keep at most K strongest outgoing edges per sentence."""
        n = similarity.shape[0]
        if k >= n - 1:
            return similarity

        pruned = np.zeros_like(similarity)

        for i in range(n):
            row = similarity[i]
            positive = np.flatnonzero(row > 0.0)

            if positive.size <= k:
                pruned[i, positive] = row[positive]
                continue

            # argpartition avoids a full O(n log n) sort for every row.
            local_values = row[positive]
            keep_local = np.argpartition(local_values, -k)[-k:]
            keep = positive[keep_local]
            pruned[i, keep] = row[keep]

        return pruned

    def _weighted_pagerank(
        self,
        adjacency: np.ndarray,
    ) -> tuple[np.ndarray, int, bool]:
        """
        Weighted PageRank with proper dangling-node handling.

        For weighted adjacency W, transition probability from i -> j is:

            P_ij = W_ij / sum_j W_ij

        Dangling nodes distribute their probability uniformly across all nodes.
        """
        n = adjacency.shape[0]
        row_sums = adjacency.sum(axis=1)

        transition = np.zeros_like(adjacency, dtype=np.float64)
        non_dangling = row_sums > 0.0

        transition[non_dangling] = (
            adjacency[non_dangling] / row_sums[non_dangling, None]
        )

        rank = np.full(n, 1.0 / n, dtype=np.float64)
        teleport = (1.0 - self.damping) / n

        for iteration in range(1, self.max_iter + 1):
            dangling_mass = float(rank[~non_dangling].sum()) / n

            new_rank = (
                teleport
                + self.damping
                * (
                    transition.T @ rank
                    + dangling_mass
                )
            )

            # Normalize against accumulated floating-point drift.
            total = float(new_rank.sum())
            if total > 0.0:
                new_rank /= total

            delta = float(np.abs(new_rank - rank).sum())
            rank = new_rank

            if delta < self.tol:
                return rank, iteration, True

        return rank, self.max_iter, False