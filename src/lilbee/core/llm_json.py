"""Extract the first JSON value from an LLM reply.

Models wrap JSON in prose, fences, and trailing commentary. A greedy
``\\{.*\\}`` span runs to the LAST brace in the reply, so any trailing text
containing one drops the whole value; a hand-rolled brace counter miscounts
braces inside string literals. ``raw_decode`` scanned from each opener lets the
stdlib own the parsing state, which is the only thing that gets both right.
"""

from __future__ import annotations

import json
from typing import TypeVar

_DECODER = json.JSONDecoder()


def json_reply_format() -> dict[str, str]:
    """Provider option asking for a bare JSON value rather than JSON in prose.

    llama.cpp's server and OpenAI-compatible remotes both honour it; providers
    that do not drop it rather than refusing (see the litellm adapter). Returns
    a fresh dict per call because it is handed to a third-party SDK that is free
    to mutate the request it is given. The scans below stay the fallback: a
    provider can ignore the request and a model can comply imperfectly, and
    neither should cost the caller its answer.
    """
    return {"type": "json_object"}


_JsonT = TypeVar("_JsonT", dict, list)


def _first_json_value(text: str, opener: str, expected: type[_JsonT]) -> _JsonT | None:
    """The first *expected*-typed JSON value starting at an *opener*, or None."""
    start = text.find(opener)
    while start >= 0:
        try:
            parsed, _ = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find(opener, start + 1)
            continue
        return parsed if isinstance(parsed, expected) else None
    return None


def first_json_object(text: str) -> dict | None:
    """The first JSON object in *text*, or None."""
    return _first_json_value(text, "{", dict)


def first_json_array(text: str) -> list | None:
    """The first JSON array in *text*, or None."""
    return _first_json_value(text, "[", list)
