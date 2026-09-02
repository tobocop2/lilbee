"""Prompts and budgets for LLM-driven query expansion, HyDE, and history
condensation, plus the validation a condensed query must pass."""

from __future__ import annotations

import re

from lilbee.retrieval.query.tokenize import content_tokens

EXPANSION_PROMPT = (
    "Generate {count} alternative search queries for the following question. "
    "Return ONLY the queries, one per line, no numbering or explanation.\n\n"
    "Question: {question}"
)

EXPANSION_MAX_TOKENS = 200

# HyDE writes one hypothetical answer passage to embed, which is a different
# shape of output from a list of query variants. Same number today, but tuning
# either one must not move the other.
HYDE_MAX_TOKENS = 200

CONDENSE_PROMPT = (
    "Rewrite the follow-up question as one standalone search query, resolving "
    "pronouns and references using the conversation. Return ONLY the rewritten "
    "query, nothing else. If the question already stands alone, return it "
    "unchanged.\n\nConversation:\n{history}\n\nFollow-up question: {question}"
)

CONDENSE_MAX_TOKENS = 120

# Trailing history messages included in the condensation prompt (one
# ChatMessage each, so 6 is the last 3 user/assistant exchanges); older
# messages rarely change what a follow-up refers to and only add latency.
CONDENSE_HISTORY_TURNS = 6

# A rewrite restates one question, so it stays near the question's size: the
# ratio bounds a long question, the floor a two-word one, and the character
# cap stops a single unbroken paragraph.
CONDENSE_MAX_WORD_RATIO = 3
CONDENSE_MIN_WORD_ALLOWANCE = 12
CONDENSE_MAX_CHARS = 300

# A sentence that ends and continues: prose, not a query. The three-character
# prefix applies to the period alone, since no abbreviation ends in "?" or "!":
# "Dr. Smith" and "U.S. policy" are not sentence ends, "What is X? And Y?" is.
# The fullwidth terminators are the CJK equivalents, not typos, and take no
# space after them.
_SENTENCE_BREAK_RE = re.compile(r"(?:(?:\w{3,}\.|[!?])\s+|[。！？]\s*)\S")  # noqa: RUF001
# Typographic quotes included: a model wraps its rewrite in whatever quote
# style it writes prose in.
_QUOTE_CHARS = "\"'“”‘’«»"  # noqa: RUF001 -- the ambiguous glyphs are the point


def choose_retrieval_query(reply: str, question: str, context: str) -> str:
    """The condensed query from *reply*, or *question* when no line reads as a query.

    A small model answers the condense prompt with a lead-in, a refusal, or its
    reasoning as readily as with the query, and an unvalidated reply sends
    retrieval after text the user never typed. Every check is punctuation or
    vocabulary the turn already used, not a list of known lead-ins, so it holds
    for any language whose pack supplies stopwords. Falling back costs the
    pronoun resolution and nothing else.
    """
    question_terms = content_tokens(question)
    context_terms = question_terms | content_tokens(context)
    # The question anchors the rewrite; a question of pure pronouns has no
    # terms of its own, so the conversation it refers to anchors it instead.
    anchor = question_terms or context_terms
    for line in reply.splitlines():
        candidate = _strip_lead_in(_unquote(line), context_terms)
        if _is_query_shaped(candidate, question) and content_tokens(candidate) & anchor:
            return candidate
    return question


def _unquote(line: str) -> str:
    """Strip surrounding whitespace and quote characters from one reply line."""
    return line.strip().strip(_QUOTE_CHARS).strip()


def _strip_lead_in(line: str, context_terms: set[str]) -> str:
    """Drop a lead-in before the first colon when it uses no word from the turn.

    "Sure, here is the standalone query: X" loses the lead-in in any language;
    "Split Rock: the journal" keeps it, because the conversation used that name.
    """
    head, colon, tail = line.partition(":")
    if not colon or content_tokens(head) & context_terms:
        return line
    return _unquote(tail)


def _is_query_shaped(candidate: str, question: str) -> bool:
    """Whether *candidate* has the size and shape of a single search query."""
    if not candidate or len(candidate) > CONDENSE_MAX_CHARS:
        return False
    if _SENTENCE_BREAK_RE.search(candidate):
        return False
    allowance = max(CONDENSE_MIN_WORD_ALLOWANCE, CONDENSE_MAX_WORD_RATIO * len(question.split()))
    return len(candidate.split()) <= allowance
