"""Canonical chat dispatch: canonical request to provider call to canonical response."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from lilbee.app.services import get_services
from lilbee.providers.worker.transport import (
    ChatResult,
    FinishReason,
    ToolCallDelta,
)
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
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

log = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when the requested model is not present in the registry."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model {model!r} not found in registry")


class ModelDoesNotSupportToolsError(Exception):
    """Raised when the request carries tools but the model template cannot use them."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model {model!r} does not support tool calls")


_FINISH_REASON_TO_STOP: dict[FinishReason, StopReason] = {
    FinishReason.STOP: StopReason.END_TURN,
    FinishReason.LENGTH: StopReason.MAX_TOKENS,
    FinishReason.TOOL_CALLS: StopReason.TOOL_USE,
    FinishReason.CONTENT_FILTER: StopReason.END_TURN,
}

_TOOL_CHOICE_MODES: dict[str, str] = {
    "auto": "auto",
    "any": "required",
    "none": "none",
}


def dispatch_chat(req: CanonicalChatRequest) -> CanonicalResponse:
    """Run a non-streaming chat request through the provider and return canonical output."""
    _ensure_model_known(req.model)
    _ensure_tool_capability(req)

    provider = get_services().provider
    result = provider.chat(
        messages=_provider_messages(req),
        options=_provider_options(req),
        model=req.model,
        tools=_provider_tools(req.tools),
        tool_choice=_provider_tool_choice(req.tool_choice),
    )
    if isinstance(result, str):
        result = ChatResult(text=result, tool_calls=(), finish_reason=FinishReason.STOP)

    content: list[ContentBlock] = []
    if result.text:
        content.append(TextBlock(text=result.text))
    for call in result.tool_calls:
        content.append(
            ToolUseBlock(
                id=call.id or _new_call_id(),
                name=call.name,
                input=_parse_json_args(call.arguments),
            )
        )

    return CanonicalResponse(
        id=_new_message_id(),
        model=req.model,
        content=content,
        stop_reason=_FINISH_REASON_TO_STOP.get(result.finish_reason, StopReason.END_TURN),
        usage=CanonicalUsage(input_tokens=0, output_tokens=0),
    )


async def dispatch_chat_stream(
    req: CanonicalChatRequest,
) -> AsyncIterator[CanonicalStreamEvent]:
    """Stream a canonical event sequence by translating provider frames on the fly."""
    _ensure_model_known(req.model)
    _ensure_tool_capability(req)

    provider = get_services().provider
    stream = provider.chat(
        messages=_provider_messages(req),
        stream=True,
        options=_provider_options(req),
        model=req.model,
        tools=_provider_tools(req.tools),
        tool_choice=_provider_tool_choice(req.tool_choice),
    )

    # The llama-cpp pool iterator implements both Iterator and AsyncIterator;
    # the Protocol declares only the sync side because the SDK provider's
    # streaming path is a sync generator and cannot satisfy AsyncIterator.
    # Async iteration here is required so token-by-token reads do not block
    # the event loop. See ClosableIterator in providers/base.py.
    async_stream = cast(AsyncIterator[str | ToolCallDelta], stream)
    try:
        yield MessageStart(id=_new_message_id(), model=req.model)
        state = _StreamState()
        async for frame in async_stream:
            for event in state.feed(frame):
                yield event
        for event in state.finish():
            yield event
        yield MessageStop()
    finally:
        stream.close()


class _StreamState:
    """Tracks open content blocks so deltas land in the right index."""

    _NO_BLOCK = "none"
    _TEXT_BLOCK = "text"
    _TOOL_BLOCK = "tool"

    def __init__(self) -> None:
        self._open: str = self._NO_BLOCK
        self._index: int = -1
        self._tool_index: int | None = None
        self._stop_reason: StopReason = StopReason.END_TURN

    def feed(self, frame: str | ToolCallDelta) -> Iterator[CanonicalStreamEvent]:
        if isinstance(frame, str):
            yield from self._feed_text(frame)
        else:
            yield from self._feed_tool(frame)

    def finish(self) -> Iterator[CanonicalStreamEvent]:
        if self._open != self._NO_BLOCK:
            yield ContentBlockStop(index=self._index)
            self._open = self._NO_BLOCK
        yield MessageDelta(stop_reason=self._stop_reason)

    def _feed_text(self, text: str) -> Iterator[CanonicalStreamEvent]:
        if self._open != self._TEXT_BLOCK:
            yield from self._close_current()
            self._index += 1
            self._open = self._TEXT_BLOCK
            yield ContentBlockStart(index=self._index, block=TextBlock(text=""))
        yield ContentBlockDelta(index=self._index, delta=TextDelta(text=text))

    def _feed_tool(self, frame: ToolCallDelta) -> Iterator[CanonicalStreamEvent]:
        self._stop_reason = StopReason.TOOL_USE
        is_new_call = self._open != self._TOOL_BLOCK or frame.index != self._tool_index
        if is_new_call:
            yield from self._close_current()
            self._index += 1
            self._open = self._TOOL_BLOCK
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
        if self._open != self._NO_BLOCK:
            yield ContentBlockStop(index=self._index)
            self._open = self._NO_BLOCK


def _ensure_model_known(model: str) -> None:
    registry = get_services().registry
    refs = {m.ref for m in registry.list_installed()}
    if model not in refs:
        raise ModelNotFoundError(model)


def _ensure_tool_capability(req: CanonicalChatRequest) -> None:
    if req.tools and not model_supports_tools(req.model):
        raise ModelDoesNotSupportToolsError(req.model)


def _provider_messages(req: CanonicalChatRequest) -> list[dict[str, Any]]:
    """Flatten canonical messages to the OpenAI-shaped wire format the provider speaks."""
    out: list[dict[str, Any]] = []
    if req.system is not None:
        out.append({"role": "system", "content": req.system})
    for msg in req.messages:
        out.extend(_translate_message(msg))
    return out


def _translate_message(msg: Any) -> list[dict[str, Any]]:
    text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
    tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]
    tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]

    if tool_results:
        # One ``tool`` wire-message per result block; tool_call_id pairs
        # it back to the originating ToolUseBlock.
        return [
            {
                "role": "tool",
                "tool_call_id": block.tool_use_id,
                "content": _flatten_text(block.content),
            }
            for block in tool_results
        ]

    text = "".join(text_parts)
    if tool_uses:
        return [
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
        ]
    return [{"role": msg.role, "content": text}]


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


def _parse_json_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"
