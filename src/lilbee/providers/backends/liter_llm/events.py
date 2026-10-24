"""Map liter-llm replies and stream chunks to lilbee value types."""

from __future__ import annotations

from typing import Any

from lilbee.providers.backends.liter_llm.client import _require_sdk
from lilbee.providers.base import TokenUsage
from lilbee.providers.sdk_backend import SdkToolCall, SdkToolCallDelta


def _tool_call(call: Any) -> SdkToolCall:
    """Build one ``SdkToolCall`` from a liter-llm tool call."""
    return SdkToolCall(
        id=str(call.id or ""),
        name=str(call.function.name or ""),
        arguments=str(call.function.arguments or ""),
    )


def _tool_call_delta(call: Any) -> SdkToolCallDelta:
    """Build one ``SdkToolCallDelta`` from a streaming tool-call slot.

    Empty-string ``name`` / ``arguments`` are normalised to ``None`` so the
    SDK stream shape matches the native worker's deltas (the dispatch's
    ``_StreamState`` gates on ``is not None``; emitting ``""`` produces a
    spurious empty ContentBlockDelta on every opener).
    """
    function = call.function
    return SdkToolCallDelta(
        index=call.index,
        id=str(call.id) if call.id else None,
        name=str(function.name) if function is not None and function.name else None,
        arguments_delta=(
            str(function.arguments) if function is not None and function.arguments else None
        ),
    )


def _text(content: Any) -> str:
    """Return a reply's text: ``text`` for a text reply, the text parts joined for a parts reply."""
    if content is None:
        return ""
    text: str | None = content.text
    if text is not None:
        return text
    if content.parts is not None:
        return "".join(part.text for part in content.parts if part.type == "text")
    return ""


def _finish_reason(reason: Any) -> str | None:
    """Return a ``FinishReason`` member as its wire string, or ``None``."""
    if reason is None:
        return None
    finish = _require_sdk().FinishReason
    wire = {
        finish.STOP: "stop",
        finish.LENGTH: "length",
        finish.TOOL_CALLS: "tool_calls",
        finish.CONTENT_FILTER: "content_filter",
        finish.FUNCTION_CALL: "function_call",
        finish.OTHER: "other",
    }
    return wire[reason]


def _usage(usage: Any) -> TokenUsage:
    """Return a ``Usage`` as lilbee's token counts; zero when the reply carries none."""
    if usage is None:
        return TokenUsage()
    return TokenUsage(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)


async def _next_chunk(stream: Any) -> Any:
    """Return the next chunk of an async chunk stream, or ``None`` when it ends."""
    try:
        return await stream.__anext__()
    except StopAsyncIteration:
        return None
