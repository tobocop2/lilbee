"""Constants for LLM-driven query expansion and history condensation.

``EXPANSION_MAX_TOKENS`` additionally caps HyDE hypothetical-passage
generation in the searcher, which is a semantically different budget (a
generated answer passage to embed, not a set of query variants). Retuning
it therefore moves both knobs; splitting out a HYDE_MAX_TOKENS would
decouple them.
"""

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

# Trailing history messages included in the condensation prompt (one
# ChatMessage each, so 6 is the last 3 user/assistant exchanges); older
# messages rarely change what a follow-up refers to and only add latency.
CONDENSE_HISTORY_TURNS = 6
