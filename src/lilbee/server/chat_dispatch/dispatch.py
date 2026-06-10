"""Canonical chat dispatch: canonical request to provider call to canonical response."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from enum import StrEnum
from typing import Any, Literal

from lilbee.app.services import get_services
from lilbee.providers.base import ChatResult, FinishReason, TokenUsage, ToolCallDelta
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalMessage,
    CanonicalResponse,
    CanonicalStreamEvent,
    CanonicalTool,
    CanonicalToolChoice,
    CanonicalUsage,
    ContentBlock,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    StopReason,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)
from lilbee.server.chat_dispatch.capability import model_supports_tools
from lilbee.server.chat_dispatch.tool_args import parse_tool_arguments

log = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when the requested model is not installed or reachable."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"Model {model!r} is not installed. Run 'lilbee model list' to see "
            f"installed models, or 'lilbee model pull {model}' to download it."
        )


class ModelDoesNotSupportToolsError(Exception):
    """Raised when the request carries tools but the model template cannot use them."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"Model {model!r} does not support tool calls. Pick a chat model "
            f"with a tool-aware chat template, or remove tools from the request."
        )


_FINISH_REASON_TO_STOP: dict[FinishReason, StopReason] = {
    FinishReason.STOP: StopReason.END_TURN,
    FinishReason.LENGTH: StopReason.MAX_TOKENS,
    FinishReason.TOOL_CALLS: StopReason.TOOL_USE,
    FinishReason.CONTENT_FILTER: StopReason.END_TURN,
}

_CanonicalChoiceMode = Literal["auto", "any", "none"]
_ProviderChoiceMode = Literal["auto", "required", "none"]

_TOOL_CHOICE_MODES: dict[_CanonicalChoiceMode, _ProviderChoiceMode] = {
    "auto": "auto",
    "any": "required",
    "none": "none",
}


class _OpenBlockKind(StrEnum):
    NONE = "none"
    TEXT = "text"
    TOOL = "tool"


def _provider_chat_kwargs(req: CanonicalChatRequest, canonical_model: str) -> dict[str, Any]:
    """Shared provider.chat keyword arguments for both stream and non-stream paths."""
    return {
        "messages": _provider_messages(req),
        "options": _provider_options(req),
        "model": canonical_model,
        "tools": _provider_tools(req.tools),
        "tool_choice": _provider_tool_choice(req.tool_choice),
    }


def _content_blocks_from_result(result: ChatResult) -> list[ContentBlock]:
    """Build canonical content blocks from a non-streaming provider result."""
    content: list[ContentBlock] = []
    if result.text:
        content.append(TextBlock(text=result.text))
    for call in result.tool_calls:
        content.append(
            ToolUseBlock(
                id=call.id or _new_call_id(),
                name=call.name,
                input=parse_tool_arguments(call.arguments),
            )
        )
    return content


def dispatch_chat(req: CanonicalChatRequest) -> CanonicalResponse:
    """Run a non-streaming chat request through the provider and return canonical output."""
    canonical_model = preflight_chat_request(req)
    result = get_services().provider.chat(**_provider_chat_kwargs(req, canonical_model))
    return CanonicalResponse(
        id=_new_message_id(),
        model=canonical_model,
        content=_content_blocks_from_result(result),
        stop_reason=_FINISH_REASON_TO_STOP.get(result.finish_reason, StopReason.END_TURN),
        usage=CanonicalUsage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        ),
    )


async def dispatch_chat_stream(
    req: CanonicalChatRequest,
) -> AsyncIterator[CanonicalStreamEvent]:
    """Stream a canonical event sequence by translating provider frames on the fly."""
    # The preflight can do blocking HTTP model discovery when its TTL lapses;
    # run it in a thread so the event loop stays responsive.
    canonical_model = await asyncio.to_thread(preflight_chat_request, req)
    stream = get_services().provider.chat(
        stream=True, **_provider_chat_kwargs(req, canonical_model)
    )
    try:
        yield MessageStart(id=_new_message_id(), model=canonical_model)
        state = _StreamState()
        async for frame in _async_iter_provider_stream(stream):
            for event in state.feed(frame):
                yield event
        for event in state.finish():
            yield event
        yield MessageStop()
    finally:
        stream.close()


async def _async_iter_provider_stream(
    stream: Iterator[str | ToolCallDelta | TokenUsage]
    | AsyncIterator[str | ToolCallDelta | TokenUsage],
) -> AsyncIterator[str | ToolCallDelta | TokenUsage]:
    """Iterate a provider chat stream without blocking the event loop.

    An async-native stream is consumed directly, without thread hops. The
    fleet and SDK providers stream via plain sync generators; iterating one
    inline on the event loop would block, so each ``next()`` runs in a
    worker thread via ``asyncio.to_thread``. ``LLMProvider.chat`` cannot
    narrow this distinction in the Protocol because the SDK backend has no
    async-native path.
    """
    if isinstance(stream, AsyncIterable):
        async for frame in stream:
            yield frame
        return
    while True:
        frame = await asyncio.to_thread(_next_or_done, stream)
        if frame is _STREAM_DONE:
            return
        yield frame


_STREAM_DONE: Any = object()
"""Sentinel returned by :func:`_next_or_done` to mean ``StopIteration``."""


def _next_or_done(
    stream: Iterator[str | ToolCallDelta | TokenUsage],
) -> str | ToolCallDelta | TokenUsage | Any:
    """Pull the next frame from *stream*; return ``_STREAM_DONE`` at exhaustion.

    Raising ``StopIteration`` inside a coroutine becomes ``RuntimeError`` per
    PEP 479; this helper converts that signal into a sentinel value the async
    caller can branch on.
    """
    try:
        return next(stream)
    except StopIteration:
        return _STREAM_DONE


class _StreamState:
    """Tracks open content blocks so deltas land in the right index."""

    def __init__(self) -> None:
        self._open: _OpenBlockKind = _OpenBlockKind.NONE
        self._index: int = -1
        self._tool_index: int | None = None
        self._stop_reason: StopReason = StopReason.END_TURN
        self._usage: TokenUsage | None = None

    def feed(self, frame: str | ToolCallDelta | TokenUsage) -> Iterator[CanonicalStreamEvent]:
        if isinstance(frame, str):
            yield from self._feed_text(frame)
        elif isinstance(frame, TokenUsage):
            # Terminator-only frame: carries token totals, no content. Stash it
            # so finish() can attach the counts to the closing MessageDelta.
            self._usage = frame
        else:
            yield from self._feed_tool(frame)

    def finish(self) -> Iterator[CanonicalStreamEvent]:
        if self._open != _OpenBlockKind.NONE:
            yield ContentBlockStop(index=self._index)
            self._open = _OpenBlockKind.NONE
        usage = (
            CanonicalUsage(
                input_tokens=self._usage.prompt_tokens,
                output_tokens=self._usage.completion_tokens,
            )
            if self._usage is not None
            else None
        )
        yield MessageDelta(stop_reason=self._stop_reason, usage=usage)

    def _feed_text(self, text: str) -> Iterator[CanonicalStreamEvent]:
        if self._open != _OpenBlockKind.TEXT:
            yield from self._close_current()
            self._index += 1
            self._open = _OpenBlockKind.TEXT
            yield ContentBlockStart(index=self._index, block=TextBlock(text=""))
        yield ContentBlockDelta(index=self._index, delta=TextDelta(text=text))

    def _feed_tool(self, frame: ToolCallDelta) -> Iterator[CanonicalStreamEvent]:
        self._stop_reason = StopReason.TOOL_USE
        is_new_call = self._open != _OpenBlockKind.TOOL or frame.index != self._tool_index
        if is_new_call:
            yield from self._close_current()
            self._index += 1
            self._open = _OpenBlockKind.TOOL
            self._tool_index = frame.index
            yield ContentBlockStart(
                index=self._index,
                block=ToolUseBlock(
                    id=frame.id or _new_call_id(),
                    name=frame.name or "",
                    input={},
                ),
            )
        if frame.arguments_delta is not None:
            yield ContentBlockDelta(
                index=self._index,
                delta=ToolUseDelta(partial_json=frame.arguments_delta),
            )

    def _close_current(self) -> Iterator[CanonicalStreamEvent]:
        if self._open != _OpenBlockKind.NONE:
            yield ContentBlockStop(index=self._index)
            self._open = _OpenBlockKind.NONE


def _resolve_canonical_model(model: str) -> str:
    """Return the canonical ref for *model*, or raise ``ModelNotFoundError``.

    Consults the cached union of native + remote + frontier refs on
    Services, so an Ollama-managed model resolves the same way a locally
    installed GGUF does. A bare ``name:tag`` matches the corresponding
    ``ollama/<name:tag>`` entry when one exists in the discovered set.
    """
    canonical = get_services().known_models.resolve(model)
    if canonical is None:
        raise ModelNotFoundError(model)
    return canonical


def _ensure_tool_capability(req: CanonicalChatRequest, model: str) -> None:
    if req.tools and not model_supports_tools(model):
        raise ModelDoesNotSupportToolsError(model)


def preflight_chat_request(req: CanonicalChatRequest) -> str:
    """Synchronously validate *req* before any streaming response starts.

    Raises ``ModelNotFoundError`` or ``ModelDoesNotSupportToolsError``
    so the route layer can return a real 4xx HTTP status instead of
    burying the failure in an SSE error frame after headers flush.
    Returns the resolved canonical model ref.
    """
    canonical = _resolve_canonical_model(req.model)
    _ensure_tool_capability(req, canonical)
    return canonical


def _provider_messages(req: CanonicalChatRequest) -> list[dict[str, Any]]:
    """Flatten canonical messages to the OpenAI-shaped wire format the provider speaks."""
    out: list[dict[str, Any]] = []
    if req.system is not None:
        out.append({"role": "system", "content": req.system})
    for msg in req.messages:
        out.extend(_translate_message(msg))
    return out


def _translate_message(msg: CanonicalMessage) -> list[dict[str, Any]]:
    text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
    tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]
    tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
    text = "".join(text_parts)

    # One ``tool`` wire-message per result block; tool_call_id pairs it back to
    # the originating ToolUseBlock. Text blocks in the same canonical message
    # follow as their own content message rather than being dropped.
    out: list[dict[str, Any]] = [
        {
            "role": "tool",
            "tool_call_id": block.tool_use_id,
            "content": _flatten_text(block.content),
        }
        for block in tool_results
    ]
    if tool_uses:
        out.append(
            {
                "role": msg.role,
                "content": text,
                "tool_calls": [
                    {
                        "id": tu.id,
                        "type": "function",
                        "function": {
                            "name": tu.name,
                            "arguments": json.dumps(tu.input),
                        },
                    }
                    for tu in tool_uses
                ],
            }
        )
    elif text or not tool_results:
        out.append({"role": msg.role, "content": text})
    return out


def _flatten_text(blocks: list[ContentBlock]) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


def _provider_tools(
    tools: list[CanonicalTool] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def _provider_tool_choice(
    choice: CanonicalToolChoice | None,
) -> str | dict[str, Any] | None:
    if choice is None:
        return None
    if choice.mode == "tool":
        return {"type": "function", "function": {"name": choice.tool_name}}
    return _TOOL_CHOICE_MODES[choice.mode]


def _provider_options(req: CanonicalChatRequest) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    if req.temperature is not None:
        out["temperature"] = req.temperature
    if req.top_p is not None:
        out["top_p"] = req.top_p
    if req.top_k is not None:
        out["top_k"] = req.top_k
    if req.max_tokens is not None:
        out["num_predict"] = req.max_tokens
    if req.stop is not None:
        out["stop"] = req.stop
    return out or None


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"
