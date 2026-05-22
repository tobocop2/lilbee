"""Streaming wrapper that re-parses the accumulated buffer per chunk."""

from __future__ import annotations

from lilbee.providers.families.profile import OutputFormat
from lilbee.providers.worker.response_parser.parse import parse_response
from lilbee.providers.worker.response_parser.schemas import ResponseSchema
from lilbee.providers.worker.transport import ToolCallDelta

# First character of every marker that could open a tool-call envelope. When
# one of these appears in the unemitted content we hold from that position
# until the closing marker arrives and parse_response strips the block, or
# the stream finalises. Benign content with these characters ("5 < 10",
# "function foo") is also held; releases on the next chunk that completes
# the would-be tool-call block.
_NATIVE_OPENERS = ("<", "[", "f", ">")
_BARE_JSON_OPENER = "{"

# Hard cap on the held buffer; protects against models that emit a marker
# opener and then never close it (the parser would otherwise hold the entire
# remaining stream and re-run the regex over it on every chunk). 64 KB is
# well over any realistic tool-call envelope.
_BUFFER_HARD_CAP = 64 * 1024


class StreamingResponseParser:
    """Re-parse the accumulated stream buffer per chunk and emit deltas."""

    def __init__(self, schema: ResponseSchema, *, output_format: OutputFormat) -> None:
        self._schema = schema
        self._output_format = output_format
        self._openers = _openers_for(output_format)
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
        # Hard-cap the held buffer so a never-closing opener can't drive
        # unbounded memory or O(n^2) regex cost.
        if not finalize and len(self._buffer) > _BUFFER_HARD_CAP:
            finalize = True
        parsed = parse_response(self._buffer, self._schema, output_format=self._output_format)

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
            safe_end = _safe_emit_position(parsed.content, self._emitted_content_len, self._openers)

        if safe_end > self._emitted_content_len:
            content_delta = parsed.content[self._emitted_content_len : safe_end]
            self._emitted_content_len = safe_end
        else:
            content_delta = ""

        return content_delta, tool_deltas


def _openers_for(output_format: OutputFormat) -> tuple[str, ...]:
    """Marker openers to hold on for *output_format*."""
    if output_format in (OutputFormat.BARE_JSON, OutputFormat.DUAL):
        return (*_NATIVE_OPENERS, _BARE_JSON_OPENER)
    return _NATIVE_OPENERS


def _safe_emit_position(content: str, start: int, openers: tuple[str, ...]) -> int:
    """Latest position safe to emit; everything after the first opener is held."""
    for i in range(start, len(content)):
        if content[i] in openers:
            return i
    return len(content)
