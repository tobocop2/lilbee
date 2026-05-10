"""Constants for LLM-driven query expansion."""

from __future__ import annotations

EXPANSION_PROMPT = (
    "Generate {count} alternative search queries for the following question. "
    "Return ONLY the queries, one per line, no numbering or explanation.\n\n"
    "Question: {question}"
)

EXPANSION_MAX_TOKENS = 200
