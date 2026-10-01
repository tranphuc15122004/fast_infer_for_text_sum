"""Output text heuristics to detect degenerate decoding or loops."""

from __future__ import annotations

import re


_PUNCTUATION_STRIP = re.compile(r"^\W+|\W+$")


def is_degenerate_output(
    text: str,
    *,
    ngram_size: int = 4,
    repeat_threshold: int = 4,
    min_tokens_for_check: int = 20,
) -> bool:
    """Return True if the text exhibits severe repetition patterns."""
    if not text or not isinstance(text, str):
        return False
    tokens = text.strip().split()
    if len(tokens) < min_tokens_for_check:
        return False

    # Check 1: single token repeated consecutively
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    if max_consecutive >= 8:
        return True

    # Check 2: n-gram repetition
    if len(tokens) >= ngram_size * repeat_threshold:
        ngrams = [
            tuple(tokens[i : i + ngram_size])
            for i in range(len(tokens) - ngram_size + 1)
        ]
        counts: dict[tuple[str, ...], int] = {}
        for ng in ngrams:
            counts[ng] = counts.get(ng, 0) + 1
            if counts[ng] >= repeat_threshold:
                return True

    return False
