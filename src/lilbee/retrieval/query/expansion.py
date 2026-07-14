"""Constants for LLM-driven query expansion and history condensation."""

from __future__ import annotations

EXPANSION_PROMPT = (
    "Generate {count} alternative search queries for the following question. "
    "Return ONLY the queries, one per line, no numbering or explanation.\n\n"
    "Question: {question}"
)

EXPANSION_MAX_TOKENS = 200

CONDENSE_PROMPT = (
    "Rewrite the follow-up question as one standalone search query, resolving "
    "pronouns and references using the conversation. Return ONLY the rewritten "
    "query, nothing else. If the question already stands alone, return it "
    "unchanged.\n\nConversation:\n{history}\n\nFollow-up question: {question}"
)

CONDENSE_MAX_TOKENS = 120

# History turns included in the condensation prompt; older turns rarely change
# what a follow-up refers to and only add latency.
CONDENSE_HISTORY_TURNS = 6
