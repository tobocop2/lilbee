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

Language-specific patterns live in ``retrieval.language`` packs; the logic
here consumes the active pack, so another language is an added pack, not an
edited parser. Only language-neutral shapes (filenames, quoting, token
splitting) are defined in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lilbee.retrieval.language import QueryLanguage, query_language


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
# Language-neutral: filenames look the same in every language.
_FILENAME_RE = re.compile(
    r"[\w.-][\w./-]*\.(?:pdf|md|txt|docx?|rst|html?|epub|csv|py|rs|js|ts|java|go)\b",
    re.IGNORECASE,
)

# Quoted names: 'harbor survey 2002' / "harbor survey 2002". A quote only
# delimits when the pair matches and neither end touches a word from the
# outside, so contractions and possessives ("what's", "Alice's") never pair
# into a phantom name; double-quoted names may contain apostrophes.
_QUOTED_RE = re.compile(r"(?<!\w)\"([^\"]{2,80})\"(?!\w)|(?<!\w)'([^']{2,80})'(?!\w)")


def document_references(question: str, lang: QueryLanguage | None = None) -> list[str]:
    """Candidate document identifiers named in *question*, best-first.

    Filenames beat quoted names beat "document N" references; all are
    resolved against real source metadata by the caller, so a wrong
    candidate costs one lookup, not a wrong route.
    """
    lang = lang or query_language()
    candidates: list[str] = []
    for m in _FILENAME_RE.finditer(question):
        candidates.append(m.group(0).strip())
    for m in _QUOTED_RE.finditer(question):
        quoted = (m.group(1) or m.group(2)).strip()
        if quoted:
            candidates.append(quoted)
    for m in lang.doc_ref_pattern.finditer(question):
        ref = m.group(1).strip()
        if ref.lower() not in lang.ref_stopwords:
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


def parse_aggregate(question: str, lang: QueryLanguage | None = None) -> AggregateQuery | None:
    """Parse a count-shaped question, or ``None`` for anything else.

    Only "how many ..." questions qualify; of those, term-mention and
    corpus-total forms are answerable against today's schema. The rest
    (counts over typed records the store does not hold) come back as
    ``UNSUPPORTED`` so the caller can decline precisely instead of feeding
    the question to top-k retrieval that structurally cannot count.
    """
    lang = lang or query_language()
    if not lang.how_many_pattern.search(question):
        return None
    m = lang.association_pattern.search(question) or lang.per_pattern.search(question)
    if m:
        return AggregateQuery(
            AggregateKind.TYPE_ASSOCIATION,
            noun=m.group(1).strip(),
            group_noun=m.group(2).strip(),
        )
    m = lang.distinct_pattern.search(question)
    if m:
        return AggregateQuery(AggregateKind.DISTINCT_TYPE, noun=m.group(1).strip())
    m = lang.term_mention_pattern.search(question)
    if m:
        term = m.group(1).strip().strip("\"'")
        # Strip a leading article so 'mention the observatory' counts 'observatory'.
        term = lang.leading_article_pattern.sub("", term)
        if term:
            return AggregateQuery(AggregateKind.TERM_MENTIONS, term=term)
    if lang.total_pattern.search(question):
        return AggregateQuery(AggregateKind.TOTAL_SOURCES)
    return AggregateQuery(AggregateKind.UNSUPPORTED)


# --- LLM-backed classification (config-gated; see Searcher.route_direct_answer) ---

# Answer budget for the classification call: one small JSON object.
INTENT_CLASSIFY_MAX_TOKENS = 96

# The classifier prompt is intentionally language-agnostic about the QUESTION
# (the model reads any language); only the label vocabulary is fixed.
INTENT_CLASSIFY_PROMPT = """Classify this question for a document-search engine.
Respond with ONLY a JSON object, no other text:
{{"kind": "...", "term": "", "noun": "", "group_noun": ""}}

kind must be exactly one of:
- "topical": an ordinary question answered by reading passages (the default)
- "total_sources": asks how many documents/files the collection holds
- "term_mentions": asks how many documents mention or contain a specific \
term; put that term in "term"
- "distinct_type": asks how many distinct entities of some type exist; put \
the type noun in "noun"
- "type_association": asks how many X each Y has; put X in "noun" and Y in \
"group_noun"

When unsure, use "topical".

Question: {question}
"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_LLM_KINDS = {
    "total_sources": AggregateKind.TOTAL_SOURCES,
    "term_mentions": AggregateKind.TERM_MENTIONS,
    "distinct_type": AggregateKind.DISTINCT_TYPE,
    "type_association": AggregateKind.TYPE_ASSOCIATION,
}


def parse_llm_aggregate(text: str) -> AggregateQuery | None:
    """Map the classifier's reply to a route, or ``None`` for no route.

    Conservative on every axis: anything malformed, unknown, "topical", or
    missing a required field yields ``None``, which sends the question to
    ordinary retrieval -- the same harmless degrade as a deterministic miss.
    ``UNSUPPORTED`` is never produced here; declining is reserved for the
    deterministic layer, whose patterns prove the question is count-shaped.
    """
    import json

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        # A brace-delimited match parses to a dict or raises; no shape check needed.
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    kind = _LLM_KINDS.get(data.get("kind", ""))
    if kind is None:
        return None
    term = str(data.get("term", "") or "").strip()
    noun = str(data.get("noun", "") or "").strip()
    group_noun = str(data.get("group_noun", "") or "").strip()
    required_ok = {
        AggregateKind.TOTAL_SOURCES: True,
        AggregateKind.TERM_MENTIONS: bool(term),
        AggregateKind.DISTINCT_TYPE: bool(noun),
        AggregateKind.TYPE_ASSOCIATION: bool(noun and group_noun),
    }[kind]
    if not required_ok:
        return None
    return AggregateQuery(kind, term=term, noun=noun, group_noun=group_noun)
