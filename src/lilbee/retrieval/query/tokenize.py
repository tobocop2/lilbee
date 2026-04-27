"""Token utilities for the RAG query pipeline."""

from __future__ import annotations

import math
import re

_MIN_TOKEN_LEN = 2
_TOKEN_SPLIT_RE = re.compile(r"\W+")


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
    """
    n = len(chunk_tokens)
    df: dict[str, int] = {}
    for tokens in chunk_tokens:
        for term in tokens & question_terms:
            df[term] = df.get(term, 0) + 1
    return {t: max(0.0, math.log(n / (1 + df.get(t, 0)))) for t in question_terms}
