"""Shared wire-format extractors keyed by ``OutputFormat``.

The family-native schema in ``schemas/<family>.json`` describes what the
model emits when prompted via its NATIVE training prompt. But
OpenAI-compatible clients (opencode, aider, the ai-sdk family) prompt
every tool-trained model via the standard ``tools`` parameter, which
typically elicits a bare ``{"name": ..., "arguments": ...}`` shape from
the model regardless of family. The profile's ``output_format`` field
declares which fallbacks the parser engine should consult when the
family-native regex misses.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from lilbee.providers.families.profile import OutputFormat
from lilbee.providers.worker.transport import ToolCall

log = logging.getLogger(__name__)

_BARE_JSON_TOOL_CALL_RE = re.compile(
    r"(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})",
)
_BARE_JSON_BOUNDARY_RE = re.compile(r'^(.*?)(?=\{\s*"name"\s*:)', re.DOTALL)


@dataclass(frozen=True)
class FormatFallback:
    """Tool-call regex + content-boundary regex for one OutputFormat shape."""

    tool_call_regex: re.Pattern[str]
    content_boundary_regex: re.Pattern[str]


FALLBACKS: dict[OutputFormat, FormatFallback] = {
    OutputFormat.BARE_JSON: FormatFallback(
        tool_call_regex=_BARE_JSON_TOOL_CALL_RE,
        content_boundary_regex=_BARE_JSON_BOUNDARY_RE,
    ),
    OutputFormat.DUAL: FormatFallback(
        tool_call_regex=_BARE_JSON_TOOL_CALL_RE,
        content_boundary_regex=_BARE_JSON_BOUNDARY_RE,
    ),
}


def extract_bare_json_tool_calls(text: str) -> tuple[tuple[str, str], ...]:
    """Find ``{"name": ..., "arguments": ...}`` JSON blocks in *text*."""
    out: list[tuple[str, str]] = []
    for match in _BARE_JSON_TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = obj.get("arguments")
        if arguments is None:
            arguments_str = "{}"
        elif isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments, separators=(",", ":"))
        out.append((name, arguments_str))
    return tuple(out)


def bare_json_tool_calls(text: str) -> tuple[ToolCall, ...]:
    """``ToolCall`` form of :func:`extract_bare_json_tool_calls`."""
    return tuple(
        ToolCall(id="", name=name, arguments=args)
        for name, args in extract_bare_json_tool_calls(text)
    )


def split_content_at_bare_json(text: str) -> str:
    """Return the prefix of *text* before the first ``{"name": ...}`` JSON object."""
    match = _BARE_JSON_BOUNDARY_RE.match(text)
    return match.group(1) if match else text
