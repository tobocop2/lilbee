"""Shared text tokenization helpers.

Two tokenizers live here so different callers can trade off strictness:

- ``tokenize_unique`` preserves the historical ``query.py`` behavior
  (whitespace split, case-fold, drop stopwords, drop single-char tokens,
  returns a set). Used for token-overlap checks in query expansion and
  context selection.

- ``tokenize`` is a richer variant used for term-frequency work: strips
  non-alphanumeric punctuation, drops tokens shorter than 3 characters,
  and preserves duplicates so callers can count them.
"""

from __future__ import annotations

STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "between",
        "through",
        "after",
        "before",
        "above",
        "below",
        "and",
        "or",
        "but",
        "not",
        "no",
        "if",
        "then",
        "than",
        "that",
        "this",
        "it",
        "its",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
    }
)

_MIN_TF_TOKEN_LEN = 3  # tokens shorter than this are dropped from TF counts


def tokenize_unique(text: str) -> set[str]:
    """Tokenize into a set of lowercase words, dropping stopwords and single chars.

    Matches the historical ``query._tokenize_query`` behavior so callers
    that rely on set-membership checks (token overlap, query coverage)
    keep working without change.
    """
    return {w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 1}


def tokenize(text: str) -> list[str]:
    """Tokenize into a list of lowercase words for term-frequency work.

    Splits on whitespace, strips non-alphanumeric characters, drops
    stopwords, and drops tokens shorter than three characters. Duplicates
    are preserved so callers can compute term frequency directly.
    """
    result: list[str] = []
    for raw in text.lower().split():
        word = "".join(ch for ch in raw if ch.isalnum())
        if len(word) < _MIN_TF_TOKEN_LEN or word in STOP_WORDS:
            continue
        result.append(word)
    return result
