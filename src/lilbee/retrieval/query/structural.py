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
_TOC_LINE = re.compile(r"\.{3,}\s*(\d{1,4})\s*$")

# Leader variants normalized to plain dots before matching: ellipsis and
# spaced dots (". . . .") are common PDF extractions of the same leaders.
_ELLIPSIS = "…"
_SPACED_DOTS = re.compile(r"\.(?:[ \t]+\.){2,}")

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
# Ratio of fully upper-case LINES, not words: acronym-dense prose ("NATO",
# "GDP") stays mixed-case at line level while cover banners are whole lines.
# Real covers mix caps org lines with title-case title lines, hence 0.4.
_COVER_CAPS_LINE_RATIO = 0.4


def _normalize_leaders(line: str) -> str:
    """Fold ellipsis and spaced-dot leaders into plain dots."""
    line = line.replace(_ELLIPSIS, "...")
    return _SPACED_DOTS.sub(lambda m: "." * (m.group().count(".")), line)


def _is_toc(nonempty: list[str]) -> bool:
    """A table of contents: several dot-leader lines with non-decreasing page numbers.

    The monotonic check separates a TOC from dot-leader data pages (price
    lists, log output), whose trailing numbers are not ordered.
    """
    if len(nonempty) < _MIN_TOC_LINES:
        return False
    pages = [
        int(m.group(1)) for line in nonempty if (m := _TOC_LINE.search(_normalize_leaders(line)))
    ]
    if len(pages) < _MIN_TOC_HITS or len(pages) / len(nonempty) < _TOC_RATIO:
        return False
    return all(a <= b for a, b in zip(pages, pages[1:]))


def _is_cover_page(text: str) -> bool:
    """A cover/title page: short, almost no sentences, banner-line dominated,
    and carrying a classification banner."""
    if not _CLASSIFICATION.search(text):
        return False
    words = text.split()
    if not words or len(words) > _COVER_MAX_WORDS:
        return False
    sentences = text.count(".") + text.count("!") + text.count("?")
    if sentences > _COVER_MAX_SENTENCES:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # The banner satisfies its own gate; the caps ratio measures the title
    # lines beyond it, so one banner cannot tip a page with real content.
    content = [line for line in lines if not _CLASSIFICATION.search(line)]
    if not content:
        return True
    caps_lines = sum(1 for line in content if len(line) > 1 and line.isupper())
    return caps_lines / len(content) >= _COVER_CAPS_LINE_RATIO


def is_structural_chunk(text: str) -> bool:
    """True when *text* is a table of contents or a cover/title page -- a
    document-structure chunk that should not compete as an answer passage."""
    if not text or not text.strip():
        return False
    nonempty = [line for line in text.splitlines() if line.strip()]
    return _is_toc(nonempty) or _is_cover_page(text)
