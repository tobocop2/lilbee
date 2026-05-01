"""Constants for LLM-driven query expansion."""

from __future__ import annotations

_EXPANSION_PROMPT = (
    "Generate {count} alternative search queries for the following question. "
    "Return ONLY the queries, one per line, no numbering or explanation.\n\n"
    "Question: {question}"
)

_EXPANSION_MAX_TOKENS = 200
