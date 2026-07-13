"""Query intent detection: known-item lookups and corpus aggregates.

Top-k similarity retrieval answers topical questions. Two other question
shapes reach the same pipe and fail structurally:

- A known-item lookup ("summarize survey_214.pdf") names the thing it
  wants; the answer is a document, not a ranking.
- An aggregate ("how many documents mention the observatory") is a property
  of the whole corpus; the top 20 of 500k chunks cannot count anything.

Detection here is deterministic and deliberately conservative: a missed
route degrades to topical retrieval, which handles the query the way it
always has, while a false positive would hijack a topical question. Every
pattern therefore requires an explicit structural cue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AggregateKind(Enum):
    """What a count-shaped question is asking to count."""

    TOTAL_SOURCES = "total_sources"
    TERM_MENTIONS = "term_mentions"
    DISTINCT_TYPE = "distinct_type"
    TYPE_ASSOCIATION = "type_association"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AggregateQuery:
    """A parsed aggregate question.

    ``noun`` carries the thing being counted for the typed kinds;
    ``group_noun`` the per-group dimension of an association question. Both
    are question words, resolved against the extraction schema by the caller
    (the parser stays schema-free and purely syntactic).
    """

    kind: AggregateKind
    term: str = ""
    noun: str = ""
    group_noun: str = ""


# Filename-shaped tokens: a path-ish word with a known document extension.
# No spaces: a name containing them arrives quoted and the quote pattern
# catches it; allowing spaces here would swallow leading sentence words.
_FILENAME_RE = re.compile(
    r"[\w.-][\w./-]*\.(?:pdf|md|txt|docx?|rst|html?|epub|csv|py|rs|js|ts|java|go)\b",
    re.IGNORECASE,
)

# "document 214", "doc #47", "exhibit 12b", "file 2020-03": a document noun
# followed by a short identifier.
_DOC_REF_RE = re.compile(
    r"\b(?:document|doc|file|exhibit|attachment|report)\s+#?([\w][\w.-]{0,23})\b",
    re.IGNORECASE,
)

# Quoted names: 'harbor survey 2002' / "harbor survey 2002". A quote only
# delimits when the pair matches and neither end touches a word from the
# outside, so contractions and possessives ("what's", "Alice's") never pair
# into a phantom name; double-quoted names may contain apostrophes.
_QUOTED_RE = re.compile(r"(?<!\w)\"([^\"]{2,80})\"(?!\w)|(?<!\w)'([^']{2,80})'(?!\w)")

# Generic nouns that follow a document noun in topical questions ("the
# documents mention...", "which document says..."); never identifiers.
_REF_STOPWORDS = frozenset(
    {"that", "which", "the", "this", "these", "those", "it", "them", "was", "is", "are"}
)

_HOW_MANY_RE = re.compile(
    r"^\s*(?:roughly\s+|approximately\s+|about\s+)?how\s+many\b", re.IGNORECASE
)

# "how many documents/sources/files are there/indexed": corpus totals.
_TOTAL_RE = re.compile(
    r"how\s+many\s+(?:documents|sources|files|pages|chunks)\s*"
    r"(?:are\s+(?:there|indexed|in\s+the\s+index)|do(?:es)?\s+.*\b(?:index|corpus|vault)\b.*)?[?\s]*$",
    re.IGNORECASE,
)

# "how many X is each Y associated with" / "how many X per Y": typed
# association counts over extracted entities.
_ASSOCIATION_RE = re.compile(
    r"how\s+many\s+(.+?)\s+(?:is|are)\s+each\s+(.+?)\s+"
    r"(?:associated\s+with|linked\s+to|recorded\s+(?:for|against))",
    re.IGNORECASE,
)
_PER_RE = re.compile(r"how\s+many\s+(.+?)\s+per\s+(.+?)[?.\s]*$", re.IGNORECASE)

# "how many distinct/unique X ...": typed distinct counts.
_DISTINCT_RE = re.compile(
    r"how\s+many\s+(?:distinct|unique|different)\s+(.+?)"
    r"(?:\s+(?:are|were|is|exist)\b.*)?[?.\s]*$",
    re.IGNORECASE,
)

# "how many documents mention/contain/reference X": term-mention counts.
_TERM_MENTION_RE = re.compile(
    r"how\s+many\s+(?:documents|sources|files|pages|chunks|passages)\s+"
    r"(?:mention|mentions|mentioning|contain|contains|containing|reference|references|referencing|discuss|discussing)\s+"
    r"(.+?)[?.\s]*$",
    re.IGNORECASE,
)


def document_references(question: str) -> list[str]:
    """Candidate document identifiers named in *question*, best-first.

    Filenames beat quoted names beat "document N" references; all are
    resolved against real source metadata by the caller, so a wrong
    candidate costs one lookup, not a wrong route.
    """
    candidates: list[str] = []
    for m in _FILENAME_RE.finditer(question):
        candidates.append(m.group(0).strip())
    for m in _QUOTED_RE.finditer(question):
        quoted = (m.group(1) or m.group(2)).strip()
        if quoted:
            candidates.append(quoted)
    for m in _DOC_REF_RE.finditer(question):
        ref = m.group(1).strip()
        if ref.lower() not in _REF_STOPWORDS:
            candidates.append(ref)
    seen: set[str] = set()
    unique = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-z]+")


def matches_reference(ref: str, filename: str) -> bool:
    """Whether *filename* names the document *ref* refers to, token-exactly.

    Substring search cannot resolve a bare number against zero-padded ids:
    "482" is a substring of both "...00000482" and "...00010482". Tokens
    split on non-alphanumerics are compared whole; numeric tokens compare by
    value so leading zeros don't hide the match, and a longer number sharing
    a suffix stays a non-match.
    """
    ref_token = ref.strip().lower()
    if ref_token in (filename.lower(), Path(filename).name.lower()):
        return True
    stem = Path(filename).stem.lower()
    for token in _TOKEN_SPLIT_RE.split(stem):
        if not token:
            continue
        if token == ref_token:
            return True
        if token.isdigit() and ref_token.isdigit() and int(token) == int(ref_token):
            return True
    return False


def parse_aggregate(question: str) -> AggregateQuery | None:
    """Parse a count-shaped question, or ``None`` for anything else.

    Only "how many ..." questions qualify; of those, term-mention and
    corpus-total forms are answerable against today's schema. The rest
    (counts over typed records the store does not hold) come back as
    ``UNSUPPORTED`` so the caller can decline precisely instead of feeding
    the question to top-k retrieval that structurally cannot count.
    """
    if not _HOW_MANY_RE.search(question):
        return None
    m = _ASSOCIATION_RE.search(question) or _PER_RE.search(question)
    if m:
        return AggregateQuery(
            AggregateKind.TYPE_ASSOCIATION,
            noun=m.group(1).strip(),
            group_noun=m.group(2).strip(),
        )
    m = _DISTINCT_RE.search(question)
    if m:
        return AggregateQuery(AggregateKind.DISTINCT_TYPE, noun=m.group(1).strip())
    m = _TERM_MENTION_RE.search(question)
    if m:
        term = m.group(1).strip().strip("\"'")
        # Strip a leading article so 'mention the observatory' counts 'observatory'.
        term = re.sub(r"^(?:the|a|an)\s+", "", term, flags=re.IGNORECASE)
        if term:
            return AggregateQuery(AggregateKind.TERM_MENTIONS, term=term)
    if _TOTAL_RE.search(question):
        return AggregateQuery(AggregateKind.TOTAL_SOURCES)
    return AggregateQuery(AggregateKind.UNSUPPORTED)
