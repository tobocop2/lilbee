"""Token utilities for the RAG query pipeline."""

from __future__ import annotations

import math
import re

_MIN_TOKEN_LEN = 1
_TOKEN_SPLIT_RE = re.compile(r"\W+")

# A single-character match counts at most this much: enough to prefer the
# chunk naming the subject (C, R, W-2), too little for a stray contraction
# splinter to outrank a distinctive longer term.
_SINGLE_CHAR_MAX_WEIGHT = 1.0


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, split on any non-alnum run."""
    return [word for word in _TOKEN_SPLIT_RE.split(text.lower()) if len(word) >= _MIN_TOKEN_LEN]


def _idf_weights(
    question_terms: set[str],
    chunk_tokens: list[set[str]],
) -> dict[str, float]:
    """Inverse Document Frequency weight per query term over the candidate chunks.

    Classical IDF per Spärck Jones (1972), "A Statistical Interpretation
    of Term Specificity and Its Application in Retrieval", Journal of
    Documentation 28:11-21. Terms that appear in every chunk collapse to
    zero weight, so corpus-specific stopwords are filtered automatically.
    Single-character terms are capped, since a rare one-letter fragment
    would otherwise outrank a distinctive longer term.
    """
    n = len(chunk_tokens)
    df: dict[str, int] = {}
    for tokens in chunk_tokens:
        for term in tokens & question_terms:
            df[term] = df.get(term, 0) + 1
    weights: dict[str, float] = {}
    for term in question_terms:
        weight = max(0.0, math.log(n / (1 + df.get(term, 0))))
        if len(term) <= _MIN_TOKEN_LEN:
            weight = min(weight, _SINGLE_CHAR_MAX_WEIGHT)
        weights[term] = weight
    return weights
