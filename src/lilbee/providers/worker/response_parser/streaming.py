"""Streaming wrapper that re-parses the accumulated buffer per chunk."""

from __future__ import annotations

from typing import Any

from lilbee.providers.worker.response_parser.parse import parse_response
from lilbee.providers.worker.transport import ToolCallDelta

_MARKER_OPENERS = ("<", "[")
"""First characters of every tool-call / reasoning marker any shipped schema
watches for. ``<`` for ``<tool_call>``, ``<think>``, ``<|tool_call>``;
``[`` for Mistral's ``[TOOL_CALLS]``. Over-broad on purpose: also pauses
emission briefly on benign content (math, JSON-looking prose). The pause
self-heals on the next chunk as more bytes either complete a real marker
or push the `<` / `[` past the peek window."""

_MARKER_PEEK_WINDOW = 16
"""How far back from the tail to scan for a marker-opener character.

Sized to fit the longest marker any shipped schema watches for (``<|tool_call>``
at 12 chars). The window must exceed the longest marker length minus one so a
partial marker landing at the boundary still gets caught."""


class StreamingResponseParser:
    """Re-parse the accumulated stream buffer per chunk and emit deltas.

    ``feed(text)`` is called for each text fragment from the provider. It
    returns the *new* content to forward to the client plus any tool calls
    that have completed since the last call. ``flush()`` is called at end of
    stream to release any content held by the safety margin.

    Content emissions are stable: once a character has been emitted it is
    never retracted, even if a subsequent re-parse extracts a shorter
    ``content`` because a tool-call marker just appeared in the buffer.
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema
        self._buffer = ""
        self._emitted_content_len = 0
        self._emitted_tool_call_count = 0

    def feed(self, text: str) -> tuple[str, list[ToolCallDelta]]:
        """Append *text* to the buffer and return ``(content_delta, new_tool_calls)``."""
        self._buffer += text
        return self._emit(finalize=False)

    def flush(self) -> tuple[str, list[ToolCallDelta]]:
        """Release any content held by the safety margin at end of stream."""
        return self._emit(finalize=True)

    def _emit(self, *, finalize: bool) -> tuple[str, list[ToolCallDelta]]:
        parsed = parse_response(self._buffer, self._schema)

        new_calls = parsed.tool_calls[self._emitted_tool_call_count :]
        tool_deltas = [
            ToolCallDelta(
                index=self._emitted_tool_call_count + offset,
                id=None,
                name=call.name,
                arguments_delta=call.arguments,
            )
            for offset, call in enumerate(new_calls)
        ]
        self._emitted_tool_call_count += len(new_calls)

        if finalize or parsed.tool_calls:
            safe_end = len(parsed.content)
        else:
            safe_end = _safe_emit_position(parsed.content)

        if safe_end > self._emitted_content_len:
            content_delta = parsed.content[self._emitted_content_len : safe_end]
            self._emitted_content_len = safe_end
        else:
            content_delta = ""

        return content_delta, tool_deltas


def _safe_emit_position(content: str) -> int:
    """Latest position safe to emit; anything beyond might be a partial marker.

    Scans the trailing :data:`_MARKER_PEEK_WINDOW` characters for a marker-
    opener character; if one is found, emission stops at that character so a
    later chunk can complete the marker without the partial bytes leaking out
    as text. Content with no marker-opener in the window is fully emittable.
    """
    end = len(content)
    window_start = max(0, end - _MARKER_PEEK_WINDOW)
    for i in range(window_start, end):
        if content[i] in _MARKER_OPENERS:
            return i
    return end
