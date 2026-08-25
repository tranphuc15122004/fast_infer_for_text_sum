"""
Random sentence-selection baseline.

This is a deterministic pseudo-random lower/sanity baseline for semantic
selection experiments. Given the same document and seed, it produces the same
sentence priority across runs and machines.

Why deterministic?
------------------
A random selector is useful for answering:

    "Does a semantic method outperform arbitrary context reduction at the same
     token budget?"

For a fair experiment, its output should not silently change when dataset order,
worker count, or process scheduling changes. This implementation therefore
derives a SHA-256 score for every sentence from:

    global seed + document fingerprint + sentence index

and sorts sentences by that score.

No neural model or semantic representation is used.

Note on the filename
--------------------
Because this module is named ``random.py``, prefer importing it as part of a
package, e.g. ``from semantic_selection.random import RandomSelector``. Running
a local file named random.py directly from its own directory can shadow Python's
standard-library ``random`` module for unrelated dependencies.

Example
-------
from random import RandomSelector

selector = RandomSelector(seed=42)
result = selector.select(document, token_budget=2048)
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

try:
    # Package-style import.
    from .base import BaseSemanticSelector, SelectionResult
except ImportError:
    # Local-folder import.
    from base import BaseSemanticSelector, SelectionResult


class RandomSelector(BaseSemanticSelector):
    """
    Deterministic pseudo-random sentence selector.

    Parameters
    ----------
    seed:
        Global experiment seed. Different seeds produce different random
        sentence orderings; the same seed/document pair is fully reproducible.

    All remaining keyword arguments are forwarded to BaseSemanticSelector.
    """

    def __init__(
        self,
        seed: int = 42,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        self.seed = int(seed)

    @property
    def name(self) -> str:
        return "random"

    def select(
        self,
        document: str,
        token_budget: int,
        **kwargs: Any,
    ) -> SelectionResult:
        """Run random selection and store the effective seed in metadata."""
        effective_seed = int(kwargs.get("seed", self.seed))

        result = super().select(
            document=document,
            token_budget=token_budget,
            **kwargs,
        )

        result.metadata.update(
            {
                "seed": effective_seed,
                "randomization": "sha256_deterministic_priority",
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
        Return a reproducible pseudo-random permutation of sentence indices.

        Budget enforcement remains centralized in BaseSemanticSelector, so the
        random baseline is compared under exactly the same Qwen token budget as
        Lead, TF-IDF, TextRank, MMR, and later selectors.
        """
        del token_counts, token_budget

        effective_seed = int(kwargs.get("seed", self.seed))

        # Use the full sentence sequence to derive a stable document identity.
        # SHA-256 is stable across Python processes unlike built-in hash().
        doc_hasher = hashlib.sha256()
        for idx, sentence in enumerate(sentences):
            doc_hasher.update(str(idx).encode("utf-8"))
            doc_hasher.update(b"\x00")
            doc_hasher.update(sentence.encode("utf-8"))
            doc_hasher.update(b"\x00")
        document_digest = doc_hasher.hexdigest()

        def pseudo_random_key(idx: int) -> bytes:
            payload = (
                f"{effective_seed}|{document_digest}|{idx}".encode("utf-8")
            )
            return hashlib.sha256(payload).digest()

        return sorted(
            range(len(sentences)),
            key=pseudo_random_key,
        )