"""
Lead baseline for lightweight semantic/content selection.

The Lead selector keeps sentences in their original document order until the
shared token budget is exhausted. It is intentionally simple and serves as an
important positional baseline for long-document summarization.

This implementation relies on BaseSemanticSelector for:
    - sentence splitting,
    - Qwen3-4B tokenizer-based token counting,
    - strict token-budget enforcement,
    - latency measurement,
    - standardized SelectionResult output.

Example
-------
from lead import LeadSelector

selector = LeadSelector()
result = selector.select(document, token_budget=2048)

print(result.selected_text)
print(result.selected_tokens)
print(result.retention_ratio)
"""

from __future__ import annotations

from typing import Any, Sequence

try:
    # Package-style import:
    #   from semantic_selection.lead import LeadSelector
    from .base import BaseSemanticSelector
except ImportError:
    # Script/local-folder import:
    #   from lead import LeadSelector
    from base import BaseSemanticSelector


class LeadSelector(BaseSemanticSelector):
    """
    Select leading sentences under a target-tokenizer budget.

    Lead is a purely positional baseline:

        s_1, s_2, ..., s_n

    are considered in exactly that order. The shared BaseSemanticSelector then
    keeps as many complete sentences as possible without exceeding
    ``token_budget``.

    Notes
    -----
    - No semantic model, embedding model, or extra neural inference is used.
    - Selector overhead is therefore close to the minimum achievable by a
      sentence-level selection baseline.
    - ``preserve_order`` is effectively irrelevant for Lead because its
      priority order is already the original document order.
    """

    @property
    def name(self) -> str:
        return "lead"

    def _select_priority(
        self,
        sentences: Sequence[str],
        token_counts: Sequence[int],
        token_budget: int,
        **kwargs: Any,
    ) -> Sequence[int]:
        """
        Rank sentences solely by original document position.

        Parameters
        ----------
        sentences:
            Sentences extracted from the document.

        token_counts:
            Per-sentence target-tokenizer counts. They are not needed for
            ranking because Lead uses position only; budget enforcement is
            performed by the base class.

        token_budget:
            Maximum number of target-model tokens allowed in the selected
            context. It is handled by BaseSemanticSelector.

        Returns
        -------
        Sequence[int]
            Sentence indices in original order: [0, 1, ..., n - 1].
        """
        del token_counts, token_budget, kwargs
        return list(range(len(sentences)))