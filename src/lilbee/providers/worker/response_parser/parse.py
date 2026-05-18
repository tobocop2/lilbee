"""Apply a response schema to model output and convert to canonical types."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from lilbee.providers.worker.transport import ToolCall

log = logging.getLogger(__name__)

_FUNCTION_KEY = "function"
_NAME_KEY = "name"
_ARGUMENTS_KEY = "arguments"
_TOOL_CALLS_KEY = "tool_calls"
_CONTENT_KEY = "content"


@dataclass(frozen=True)
class ParsedResponse:
    """Outcome of running a response schema over a model's text output."""

    content: str
    tool_calls: tuple[ToolCall, ...]


def parse_response(text: str, schema: dict[str, Any]) -> ParsedResponse:
    """Extract structured tool calls from *text* using *schema*.

    Returns a ``ParsedResponse`` with the schema-extracted ``content`` and any
    detected tool calls. On parse failure the original ``text`` is returned as
    content with no tool calls, so a malformed model output never raises out
    of this layer. ``transformers`` is imported lazily: pulling it in eagerly
    adds ~1.2s to worker subprocess boot, paid even when no chat request
    carries tools. Deferring the import to the first parse keeps cold-start
    fast and pays the cost only when tool extraction is actually needed.
    """
    from transformers.utils.chat_parsing_utils import recursive_parse

    try:
        parsed = recursive_parse(text, schema)
    except (ValueError, TypeError, ImportError) as exc:
        # ValueError / TypeError from the schema walker on a malformed model
        # output; ImportError reserved for ``x-parser-args.transform`` schemas
        # that pull in jmespath (no shipped schema uses one today; guarded for
        # forward safety so a future schema cannot break the worker). Any of
        # them falls back to returning the raw text as content rather than
        # propagating to the worker as a 500.
        log.debug("response schema parse failed: %s", exc)
        return ParsedResponse(content=text, tool_calls=())
    if not isinstance(parsed, dict):
        return ParsedResponse(content=text, tool_calls=())
    content_value = parsed.get(_CONTENT_KEY)
    content = content_value if isinstance(content_value, str) else ""
    tool_calls = _tool_calls_from_parsed(parsed.get(_TOOL_CALLS_KEY))
    return ParsedResponse(content=content, tool_calls=tool_calls)


def _tool_calls_from_parsed(raw: object) -> tuple[ToolCall, ...]:
    """Coerce the parsed ``tool_calls`` list into a tuple of ``ToolCall``."""
    if not isinstance(raw, list):
        return ()
    out: list[ToolCall] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        call = _coerce_one(entry)
        if call is not None:
            out.append(call)
    return tuple(out)


def _coerce_one(entry: dict[str, Any]) -> ToolCall | None:
    """Build a ``ToolCall`` from one parsed entry, handling flat and wrapped shapes.

    Some schemas (Mistral) emit ``{"name": ..., "arguments": ...}`` at the top
    level. Others (Qwen3, Qwen3-Coder, Gemma 4) wrap in ``{"function": {...}}``.
    Both shapes are accepted so the schema authors can pick whichever fits
    the model's output most directly.
    """
    if _FUNCTION_KEY in entry and isinstance(entry[_FUNCTION_KEY], dict):
        body = entry[_FUNCTION_KEY]
    else:
        body = entry
    name = body.get(_NAME_KEY)
    if not isinstance(name, str) or not name:
        return None
    arguments = body.get(_ARGUMENTS_KEY)
    arguments_str = _arguments_to_string(arguments)
    return ToolCall(id="", name=name, arguments=arguments_str)


def _arguments_to_string(arguments: object) -> str:
    """Render parsed arguments as the JSON string ``ToolCall.arguments`` expects."""
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, separators=(",", ":"))
    except (TypeError, ValueError):
        log.debug("could not serialise tool-call arguments: %r", arguments)
        return "{}"
