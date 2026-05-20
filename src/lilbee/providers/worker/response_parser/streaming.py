"""Streaming wrapper that re-parses the accumulated buffer per chunk."""

from __future__ import annotations

from lilbee.providers.worker.response_parser.parse import parse_response
from lilbee.providers.worker.response_parser.schemas import ResponseSchema
from lilbee.providers.worker.transport import ToolCallDelta

# First characters of every marker the shipped schemas watch for. When an
# opener appears in the unemitted content, we hold from that position until
# the closing marker arrives and parse_response strips the block. Benign
# content with these characters (e.g., "5 < 10") is also held; it releases
# on the next chunk that completes whatever the parser would otherwise
# treat as a tool-call block, or on stream finalisation.
_MARKER_OPENERS = ("<", "[")


class StreamingResponseParser:
    """Re-parse the accumulated stream buffer per chunk and emit deltas."""

    def __init__(self, schema: ResponseSchema) -> None:
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

        if finalize:
            safe_end = len(parsed.content)
        else:
            safe_end = _safe_emit_position(parsed.content, self._emitted_content_len)

        if safe_end > self._emitted_content_len:
            content_delta = parsed.content[self._emitted_content_len : safe_end]
            self._emitted_content_len = safe_end
        else:
            content_delta = ""

        return content_delta, tool_deltas


def _safe_emit_position(content: str, start: int) -> int:
    """Latest position safe to emit; everything after the first opener is held.

    Anything from the first opener onward could turn into a tool-call block
    once the closing marker arrives. We release it only when ``parse_response``
    strips the now-complete block out of ``content`` (so the opener disappears)
    or on stream finalisation.
    """
    for i in range(start, len(content)):
        if content[i] in _MARKER_OPENERS:
            return i
    return len(content)
