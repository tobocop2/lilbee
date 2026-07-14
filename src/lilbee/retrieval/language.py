"""Language packs for query understanding.

ALL language-specific patterns for intent detection and noun matching live
here, mirroring ``cli/tui/messages.py``: parsing logic consumes the active
pack instead of hardcoding English, so supporting another language means
adding a pack, not editing parsers. English is the only shipped pack; the
accessor grows config-driven selection when a second pack exists.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryLanguage:
    """Everything language-specific the query-understanding layer consumes.

    The patterns encode question *shapes*, not vocabulary lists: a pack for
    another language supplies its own shapes (word order included), and the
    parsing logic in ``query.intent`` stays untouched.
    """

    code: str
    # "document 214", "doc #47": a document noun followed by a short identifier.
    doc_ref_pattern: re.Pattern[str]
    # Generic nouns that follow a document noun in topical questions ("the
    # documents mention...", "which document says..."); never identifiers.
    ref_stopwords: frozenset[str]
    # Count-question shapes; routing rules live in ``query.intent.parse_aggregate``.
    how_many_pattern: re.Pattern[str]
    total_pattern: re.Pattern[str]
    association_pattern: re.Pattern[str]
    per_pattern: re.Pattern[str]
    distinct_pattern: re.Pattern[str]
    term_mention_pattern: re.Pattern[str]
    # Leading article stripped from a counted term ("the observatory").
    leading_article_pattern: re.Pattern[str]
    # Known-item question shapes ("summarize X", "what is X about"): each
    # captures a candidate document title, resolved token-exactly against
    # source names by the caller. Shapes, not vocabulary: a wrong candidate
    # costs one lookup, never a wrong route.
    known_item_patterns: tuple[re.Pattern[str], ...]
    # Spelling variants of a noun phrase (singular/plural) for entity-type
    # matching; morphology is the most language-specific piece of all.
    noun_variants: Callable[[str], set[str]]


# Plural forms the suffix rules below can't produce, mapped both ways.
_EN_IRREGULAR_PLURALS = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "foot": "feet",
    "tooth": "teeth",
    "mouse": "mice",
    "goose": "geese",
}
_EN_IRREGULAR_SINGULARS = {plural: singular for singular, plural in _EN_IRREGULAR_PLURALS.items()}


def _english_noun_variants(noun: str) -> set[str]:
    """Normalized spelling variants of a noun phrase: itself plus
    singular/plural forms of its last word ("tail numbers" ~ "tail number",
    "people" ~ "person"). Over-generated junk forms match nothing; a missed
    form only fails to resolve, never resolves wrongly.
    """
    normalized = " ".join(noun.strip().lower().split())
    if not normalized:
        return set()
    head, _, last = normalized.rpartition(" ")
    prefix = head + " " if head else ""
    forms = {last}
    if last in _EN_IRREGULAR_PLURALS:
        forms.add(_EN_IRREGULAR_PLURALS[last])
    if last in _EN_IRREGULAR_SINGULARS:
        forms.add(_EN_IRREGULAR_SINGULARS[last])
    if last.endswith("ies") and len(last) > len("ies"):
        forms.add(last[:-3] + "y")
    if last.endswith("y"):
        forms.add(last[:-1] + "ies")
    if last.endswith(("ses", "xes", "zes", "ches", "shes")):
        forms.add(last[:-2])
    forms.add(last[:-1] if last.endswith("s") else last + "s")
    return {prefix + form for form in forms}


# Nouns that name the corpus's units rather than entities within them. A
# count over these is a document scan; a count over an entity noun (people,
# aircraft) needs extracted records and must NOT route to the scan, because
# the scan answers "N documents", a different question than the one asked.
_EN_CORPUS_NOUNS = (
    r"(?:documents|sources|files|pages|chunks|passages|books|novels|texts|works"
    r"|articles|papers|reports|letters|stories|entries|records|notes|emails"
    r"|posts|volumes|manuscripts)"
)
# "how many of these books ...", "how many of the stories ...".
_EN_OF_THESE = r"(?:of\s+(?:these|those|the|my|our)\s+)?"

ENGLISH = QueryLanguage(
    code="en",
    doc_ref_pattern=re.compile(
        r"\b(?:document|doc|file|exhibit|attachment|report)\s+#?([\w][\w.-]{0,23})\b",
        re.IGNORECASE,
    ),
    ref_stopwords=frozenset(
        {"that", "which", "the", "this", "these", "those", "it", "them", "was", "is", "are"}
    ),
    how_many_pattern=re.compile(
        r"^\s*(?:roughly\s+|approximately\s+|about\s+)?how\s+many\b", re.IGNORECASE
    ),
    # "how many documents/books/sources are there/indexed": corpus totals.
    total_pattern=re.compile(
        r"how\s+many\s+" + _EN_OF_THESE + _EN_CORPUS_NOUNS + r"\s*"
        r"(?:are\s+(?:there|indexed|in\s+the\s+index)|do(?:es)?\s+.*\b(?:index|corpus|vault)\b.*)?[?\s]*$",
        re.IGNORECASE,
    ),
    # "how many X is each Y associated with" / "how many X per Y": typed
    # association counts over extracted entities.
    association_pattern=re.compile(
        r"how\s+many\s+(.+?)\s+(?:is|are)\s+each\s+(.+?)\s+"
        r"(?:associated\s+with|linked\s+to|recorded\s+(?:for|against))",
        re.IGNORECASE,
    ),
    per_pattern=re.compile(r"how\s+many\s+(.+?)\s+per\s+(.+?)[?.\s]*$", re.IGNORECASE),
    # "how many distinct/unique X ...": typed distinct counts.
    distinct_pattern=re.compile(
        r"how\s+many\s+(?:distinct|unique|different)\s+(.+?)"
        r"(?:\s+(?:are|were|is|exist)\b.*)?[?.\s]*$",
        re.IGNORECASE,
    ),
    # "how many <corpus noun> mention/contain/reference X": term-mention
    # counts, including "of these/those" phrasing ("how many of these books
    # mention blood"). Entity nouns deliberately fall through (see
    # _EN_CORPUS_NOUNS).
    term_mention_pattern=re.compile(
        r"how\s+many\s+" + _EN_OF_THESE + _EN_CORPUS_NOUNS + r"\s+"
        r"(?:mention|mentions|mentioning|contain|contains|containing|reference|references|referencing|discuss|discussing)\s+"
        r"(.+?)[?.\s]*$",
        re.IGNORECASE,
    ),
    leading_article_pattern=re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE),
    known_item_patterns=(
        re.compile(r"^\s*(?:please\s+)?summari[sz]e\s+(.+?)[?.!\s]*$", re.IGNORECASE),
        re.compile(r"^\s*(?:please\s+)?describe\s+(.+?)[?.!\s]*$", re.IGNORECASE),
        re.compile(r"^\s*what\s+is\s+(.+?)\s+about[?.!\s]*$", re.IGNORECASE),
        re.compile(
            r"^\s*(?:give\s+me\s+)?(?:a\s+|an\s+)?(?:summary|overview)\s+of\s+(.+?)[?.!\s]*$",
            re.IGNORECASE,
        ),
    ),
    noun_variants=_english_noun_variants,
)

_PACKS = {ENGLISH.code: ENGLISH}


def query_language() -> QueryLanguage:
    """The active language pack.

    English is the only shipped pack; when more exist this resolves from
    the configured language instead of a constant.
    """
    return _PACKS["en"]


def noun_variants(noun: str) -> set[str]:
    """Spelling variants of *noun* under the active language pack."""
    return query_language().noun_variants(noun)
