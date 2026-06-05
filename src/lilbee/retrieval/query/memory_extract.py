"""Auto-extraction of durable memories from a chat turn.

A small LLM pass over the user's message and the assistant's answer that
proposes durable facts/preferences worth remembering. The model is asked for
a strict JSON array; parsing is defensive (a non-conforming reply yields no
memories rather than an error). Callers store the results unconfirmed so the
user reviews them before they are ever recalled.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from lilbee.data.store import MemoryKind

log = logging.getLogger(__name__)

ChatFn = Callable[..., str]

# Bounds mirror the prior-art auto-capture filter: too-short strings carry no
# durable signal, too-long ones are usually the model restating the answer.
_MIN_MEMORY_CHARS = 10
_MAX_MEMORY_CHARS = 500

_EXTRACT_SYSTEM_PROMPT = (
    "You extract durable, long-term memories about the user from a single chat turn. "
    "A memory is a stable fact about the user or their project (not a one-off question) "
    "or a standing preference for how they want help. "
    "Ignore transient details, the assistant's own content, and anything specific to "
    "just this question. "
    "Respond with ONLY a JSON array (no prose). Each element is an object "
    '{"text": "<the memory, third person>", "kind": "fact" | "preference"}. '
    "Return [] when nothing is worth remembering."
)

_EXTRACT_USER_TEMPLATE = "User said:\n{question}\n\nAssistant replied:\n{answer}"

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(frozen=True)
class ExtractedMemory:
    """A single memory proposed by the extraction pass."""

    text: str
    kind: MemoryKind


def build_extract_messages(question: str, answer: str) -> list[dict[str, str]]:
    """Build the system+user message pair for the extraction prompt."""
    return [
        {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _EXTRACT_USER_TEMPLATE.format(question=question, answer=answer),
        },
    ]


def _coerce_kind(value: object) -> MemoryKind:
    """Decode a kind string, defaulting to FACT for anything unrecognized."""
    if not isinstance(value, str):
        return MemoryKind.FACT
    try:
        return MemoryKind(value)
    except ValueError:
        return MemoryKind.FACT


def parse_extraction(raw: str) -> list[ExtractedMemory]:
    """Parse the model's reply into memories; tolerate a non-conforming reply.

    Extracts the first JSON array in *raw* (models often wrap it in prose or a
    code fence), keeps only objects with a usable-length ``text``, and decodes
    the kind. Any parse failure yields an empty list.
    """
    match = _JSON_ARRAY_RE.search(raw)
    if match is None:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []

    memories: list[ExtractedMemory] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not _MIN_MEMORY_CHARS <= len(text) <= _MAX_MEMORY_CHARS:
            continue
        memories.append(ExtractedMemory(text=text, kind=_coerce_kind(item.get("kind"))))
    return memories


def extract_memories(question: str, answer: str, chat: ChatFn) -> list[ExtractedMemory]:
    """Run the extraction pass for one turn; never raises.

    *chat* is the provider's non-streaming chat callable. A model or transport
    failure logs and yields no memories so a bad extraction never disrupts the
    chat session.
    """
    if not question.strip() or not answer.strip():
        return []
    try:
        raw = chat(build_extract_messages(question, answer), stream=False)
    except Exception:
        log.debug("Memory extraction call failed", exc_info=True)
        return []
    return parse_extraction(raw)
