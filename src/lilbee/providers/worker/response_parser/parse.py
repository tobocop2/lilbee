"""Apply a response schema to model output and convert to canonical types."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
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

_CHATML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_HARMONY_TOOL_CALL_RE = re.compile(
    r"<\|channel\|>commentary\s+to=(?:functions\.)?(?P<name>[A-Za-z0-9_.-]+)"
    r"[^<]*?<\|message\|>(?P<args>\{.*?\})\s*<\|call\|>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedResponse:
    """Outcome of running a response schema over a model's text output."""

    content: str
    tool_calls: tuple[ToolCall, ...]


def parse_response(
    text: str,
    schema: ResponseSchema,
    output_format: OutputFormat,
) -> ParsedResponse:
    """Extract content and structured tool calls from *text* using *schema*.

    ``output_format`` selects the extractor branch: ``NATIVE`` runs the schema
    only; ``BARE_JSON`` uses :func:`bare_json_tool_calls` only; ``DUAL`` tries
    the schema, falls back to bare-JSON when it produced none; ``CHATML_TOOL_CALL``
    extracts ``<tool_call>{json}</tool_call>`` wrappers; ``HARMONY`` extracts
    ``<|channel|>commentary to=...<|message|>{json}<|call|>`` blocks.
    """
    direct = _EXTRACTORS.get(output_format)
    if direct is not None:
        return direct(text)
    return _parse_via_schema(text, schema, output_format)


def _parse_via_schema(
    text: str, schema: ResponseSchema, output_format: OutputFormat
) -> ParsedResponse:
    """Run the transformers schema engine; ``DUAL`` falls through to bare-JSON on miss."""
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
    tool_calls = bare_json_tool_calls(text)
    if not tool_calls:
        return ParsedResponse(content=text, tool_calls=())
    return ParsedResponse(content=split_content_at_bare_json(text), tool_calls=tool_calls)


_RegexMatchToCall = Callable[[re.Match[str]], ToolCall | None]
_Extractor = Callable[[str], ParsedResponse]


def _parse_with_regex(
    text: str,
    pattern: re.Pattern[str],
    to_call: _RegexMatchToCall,
) -> ParsedResponse:
    """Extract every wrapper match in *text* and return prefix + calls."""
    calls: list[ToolCall] = []
    first_start: int | None = None
    for match in pattern.finditer(text):
        call = to_call(match)
        if call is None:
            continue
        calls.append(call)
        if first_start is None:
            first_start = match.start()
    if not calls:
        return ParsedResponse(content=text, tool_calls=())
    content = text if first_start is None else text[:first_start]
    return ParsedResponse(content=content, tool_calls=tuple(calls))


def _chatml_tool_call_from_match(match: re.Match[str]) -> ToolCall | None:
    try:
        obj = json.loads(match.group("body"))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get(_NAME_KEY)
    if not isinstance(name, str) or not name:
        return None
    return ToolCall(id="", name=name, arguments=_arguments_to_string(obj.get(_ARGUMENTS_KEY)))


def _harmony_tool_call_from_match(match: re.Match[str]) -> ToolCall | None:
    name = match.group("name")
    if not name:
        return None
    arguments = match.group("args")
    if not arguments:
        return None
    return ToolCall(id="", name=name, arguments=arguments)


def _tool_calls_from_parsed(raw: object) -> tuple[ToolCall, ...]:
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
    if _FUNCTION_KEY in entry and isinstance(entry[_FUNCTION_KEY], dict):
        body = entry[_FUNCTION_KEY]
    else:
        body = entry
    name = body.get(_NAME_KEY)
    if not isinstance(name, str) or not name:
        return None
    return ToolCall(id="", name=name, arguments=_arguments_to_string(body.get(_ARGUMENTS_KEY)))


def _arguments_to_string(arguments: object) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, separators=(",", ":"))
    except (TypeError, ValueError):
        log.debug("could not serialise tool-call arguments: %r", arguments)
        return "{}"


_EXTRACTORS: dict[OutputFormat, _Extractor] = {
    OutputFormat.BARE_JSON: _parse_bare_json,
    OutputFormat.CHATML_TOOL_CALL: lambda text: _parse_with_regex(
        text, _CHATML_TOOL_CALL_RE, _chatml_tool_call_from_match
    ),
    OutputFormat.HARMONY: lambda text: _parse_with_regex(
        text, _HARMONY_TOOL_CALL_RE, _harmony_tool_call_from_match
    ),
}
