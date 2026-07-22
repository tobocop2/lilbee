"""Detect document-structure chunks that dilute retrieval precision.

Flags two classes: tables of contents (generic), and classification-banner
cover/title pages. Both carry a document's title and section words but never
*answer* a substantive question. The detector is deliberately conservative:
it only flags chunks that are unambiguously structural, so a false negative
(some noise slips through) is preferred to a false positive (dropping real
content).
"""

from __future__ import annotations

import re

# A TOC line ends in dot leaders followed by a page number: "Geographic Trends ....... 9".
_TOC_LINE = re.compile(r"\.{3,}\s*\d{1,4}\s*$")

# Classification banners head cover/title pages and running headers alike, so the
# banner alone is not enough -- it is combined with low prose density below.
_CLASSIFICATION = re.compile(r"\b(UNCLASSIFIED|CONFIDENTIAL|SECRET|FOR OFFICIAL USE ONLY|FOUO)\b")

# A chunk needs at least this many non-empty lines before the TOC ratio means anything.
_MIN_TOC_LINES = 3
# And at least this many dot-leader lines, so a page with one stray "... 42" is not a TOC.
_MIN_TOC_HITS = 3
_TOC_RATIO = 0.30

# Cover/title-page gates: a title page is very short with essentially no prose.
# Deliberately tight -- looser gates fire on short banner-carrying body pages and
# drop content the answer needs, so a real body page's word count or its first
# full sentence must take it out of scope.
_COVER_MAX_WORDS = 60
_COVER_MAX_SENTENCES = 1
_COVER_CAPS_RATIO = 0.30


def _is_toc(nonempty: list[str]) -> bool:
    """A table of contents: several dot-leader-to-page-number lines."""
    if len(nonempty) < _MIN_TOC_LINES:
        return False
    hits = sum(1 for line in nonempty if _TOC_LINE.search(line))
    return hits >= _MIN_TOC_HITS and hits / len(nonempty) >= _TOC_RATIO


def _is_cover_page(text: str) -> bool:
    """A cover/title page: short, almost no sentences, shouting-case dominated,
    and carrying a classification banner."""
    if not _CLASSIFICATION.search(text):
        return False
    words = text.split()
    if not words or len(words) > _COVER_MAX_WORDS:
        return False
    sentences = text.count(".") + text.count("!") + text.count("?")
    if sentences > _COVER_MAX_SENTENCES:
        return False
    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    return caps / len(words) >= _COVER_CAPS_RATIO


def is_structural_chunk(text: str) -> bool:
    """True when *text* is a table of contents or a cover/title page -- a
    document-structure chunk that should not compete as an answer passage."""
    if not text or not text.strip():
        return False
    nonempty = [line for line in text.splitlines() if line.strip()]
    return _is_toc(nonempty) or _is_cover_page(text)
