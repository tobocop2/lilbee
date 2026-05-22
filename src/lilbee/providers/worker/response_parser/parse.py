"""Apply a response schema to model output and convert to canonical types."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from lilbee.providers.families.profile import OutputFormat
from lilbee.providers.worker.response_parser.format_fallbacks import (
    bare_json_tool_calls,
    split_content_at_bare_json,
)
from lilbee.providers.worker.response_parser.schemas import ResponseSchema
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


def parse_response(
    text: str,
    schema: ResponseSchema,
    output_format: OutputFormat = OutputFormat.NATIVE,
) -> ParsedResponse:
    """Extract content and structured tool calls from *text* using *schema*.

    *output_format* drives the bare-JSON fallback: ``BARE_JSON`` skips the
    schema entirely and uses only the bare-JSON extractor; ``DUAL`` tries
    the schema first and falls back to bare-JSON when zero tool calls were
    extracted. ``NATIVE`` (default) preserves legacy behavior.
    """
    if output_format is OutputFormat.BARE_JSON:
        return _parse_bare_json(text)

    # transformers is heavy (~1.2s cold import); lazy so the cost only lands
    # the first time a chat request actually carries tools. If the utility
    # is missing on this transformers release we fall through to the raw
    # text so a chat without tool extraction still runs.
    try:
        from transformers.utils.chat_parsing_utils import recursive_parse
    except (ImportError, AttributeError) as exc:
        log.debug("transformers.recursive_parse unavailable: %s", exc)
        return ParsedResponse(content=text, tool_calls=())

    try:
        parsed = recursive_parse(text, schema)
    except (ValueError, TypeError) as exc:
        log.debug("response schema parse failed: %s", exc)
        parsed = None
    if not isinstance(parsed, dict):
        if output_format is OutputFormat.DUAL:
            return _parse_bare_json(text)
        return ParsedResponse(content=text, tool_calls=())
    content_value = parsed.get(_CONTENT_KEY)
    content = content_value if isinstance(content_value, str) else ""
    tool_calls = _tool_calls_from_parsed(parsed.get(_TOOL_CALLS_KEY))
    if not tool_calls and output_format is OutputFormat.DUAL:
        fallback = _parse_bare_json(text)
        if fallback.tool_calls:
            return fallback
    return ParsedResponse(content=content, tool_calls=tool_calls)


def _parse_bare_json(text: str) -> ParsedResponse:
    """Run the shared bare-JSON extractor (OutputFormat.BARE_JSON / DUAL fallback)."""
    tool_calls = bare_json_tool_calls(text)
    if not tool_calls:
        return ParsedResponse(content=text, tool_calls=())
    content = split_content_at_bare_json(text)
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
    """Build a ``ToolCall`` from one parsed entry; accepts flat or function-wrapped shape."""
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
