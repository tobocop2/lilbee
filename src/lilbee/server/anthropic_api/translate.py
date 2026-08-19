"""Translation between Anthropic Messages models and the canonical types."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal

from lilbee.core.config.enums import ReasoningMode
from lilbee.retrieval.reasoning import StreamToken, TagParser, split_reasoning
from lilbee.server.anthropic_api.models import (
    _THINKING_DISABLED,
    AnthropicEventType,
    AnthropicMessage,
    AnthropicThinking,
    AnthropicTool,
    AnthropicToolChoice,
    AnthropicUsage,
    ContentBlockParam,
    ImageBlockParam,
    MessagesRequest,
    MessagesResponse,
    SystemTextBlock,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    UnknownBlockParam,
)
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
)

_IMAGE_CONTENT_UNSUPPORTED = (
    "Image content is not supported by /v1/messages yet. Send a text-only request."
)
_TOOL_CHOICE_NAME_REQUIRED = 'tool_choice type "tool" requires a name.'


class _BlockKind(StrEnum):
    """Kind of the mapper's open output block."""

    THINKING = "thinking"
    TEXT = "text"
    TOOL = "tool"


_ANTHROPIC_CHOICE_MODES: dict[str, str] = {
    "auto": "auto",
    "any": "any",
    "none": "none",
}


def resolve_reasoning_mode(
    thinking: AnthropicThinking | None, *, default: ReasoningMode
) -> ReasoningMode:
    """Pick the reasoning mode for one call from the request and the setting.

    Thinking is opt-in per request, as on the Anthropic API: a body with no
    ``thinking`` gets none, whatever the setting presents it as. The setting
    says how to present thinking a request asked for, and ``off`` refuses it
    outright, so a request can only tighten.
    """
    if default is ReasoningMode.OFF:
        return ReasoningMode.OFF
    if thinking is None or thinking.type == _THINKING_DISABLED:
        return ReasoningMode.OFF
    return default


def messages_to_canonical_request(
    request: MessagesRequest, *, mode: ReasoningMode = ReasoningMode.SEPARATE
) -> CanonicalChatRequest:
    """Translate a validated ``MessagesRequest`` to the canonical request."""
    messages: list[CanonicalMessage] = []
    for msg in request.messages:
        messages.extend(_canonical_messages_for(msg))
    return CanonicalChatRequest(
        model=request.model,
        messages=messages,
        system=_system_text(request.system),
        tools=_tools_from_request(request.tools),
        tool_choice=_tool_choice_from_request(request.tool_choice),
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        stop=list(request.stop_sequences) if request.stop_sequences else None,
        stream=request.stream,
        # OFF asks the template to skip thinking; the other modes only change
        # presentation, so the template default stands.
        think=False if mode is ReasoningMode.OFF else None,
    )


def _system_text(system: str | list[SystemTextBlock] | None) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    joined = "\n\n".join(block.text for block in system)
    return joined or None


def _canonical_messages_for(msg: AnthropicMessage) -> list[CanonicalMessage]:
    """Fan one Anthropic message out to canonical messages.

    Tool results become their own ``role: "tool"`` messages, emitted before
    the user's text so the provider sees results adjacent to the calls they
    answer. Unknown blocks (replayed thinking) are dropped. A mid-conversation
    ``system`` message becomes a system-reminder user turn -- Anthropic's own
    documented degradation for models without the operator channel, and it
    keeps the canonical layer's role set unchanged.
    """
    if msg.role == "system":
        text = _message_text(msg)
        if not text:
            return []
        return [
            CanonicalMessage.from_string(
                role="user", text=f"<system-reminder>\n{text}\n</system-reminder>"
            )
        ]
    if isinstance(msg.content, str):
        if not msg.content:
            return []
        return [CanonicalMessage.from_string(role=msg.role, text=msg.content)]
    return _block_messages(msg.role, msg.content)


def _block_messages(
    role: Literal["user", "assistant"], content: list[ContentBlockParam]
) -> list[CanonicalMessage]:
    """Canonical messages for a block-form user or assistant message."""
    tool_messages: list[CanonicalMessage] = []
    blocks: list[ContentBlock] = []
    for block in content:
        if isinstance(block, TextBlockParam):
            blocks.append(TextBlock(text=block.text))
        elif isinstance(block, ToolUseBlockParam):
            blocks.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
        elif isinstance(block, ToolResultBlockParam):
            tool_messages.append(_tool_result_message(block))
        elif isinstance(block, ImageBlockParam):
            raise ValueError(_IMAGE_CONTENT_UNSUPPORTED)
        elif isinstance(block, UnknownBlockParam):
            continue

    out = tool_messages
    if blocks:
        out = [*tool_messages, CanonicalMessage(role=role, content=blocks)]
    return out


def _tool_result_message(block: ToolResultBlockParam) -> CanonicalMessage:
    return CanonicalMessage(
        role="tool",
        content=[
            ToolResultBlock(
                tool_use_id=block.tool_use_id,
                content=_tool_result_content(block),
                is_error=block.is_error,
            )
        ],
    )


def _message_text(msg: AnthropicMessage) -> str:
    """The concatenated text of a message, ignoring non-text blocks."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.text for b in msg.content if isinstance(b, TextBlockParam))


def _tool_result_content(block: ToolResultBlockParam) -> list[ContentBlock]:
    if block.content is None:
        return []
    if isinstance(block.content, str):
        return [TextBlock(text=block.content)]
    parts: list[ContentBlock] = []
    for part in block.content:
        if isinstance(part, TextBlockParam):
            parts.append(TextBlock(text=part.text))
        elif isinstance(part, ImageBlockParam):
            raise ValueError(_IMAGE_CONTENT_UNSUPPORTED)
        # UnknownBlockParam: dropped
    return parts


def _tools_from_request(tools: list[AnthropicTool] | None) -> list[CanonicalTool] | None:
    if not tools:
        return None
    return [
        CanonicalTool(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.input_schema,
        )
        for tool in tools
    ]


def _tool_choice_from_request(
    choice: AnthropicToolChoice | None,
) -> CanonicalToolChoice | None:
    if choice is None:
        return None
    if choice.type == "tool":
        if not choice.name:
            raise ValueError(_TOOL_CHOICE_NAME_REQUIRED)
        return CanonicalToolChoice(mode="tool", tool_name=choice.name)
    mode = _ANTHROPIC_CHOICE_MODES[choice.type]
    return CanonicalToolChoice(mode=mode)  # type: ignore[arg-type]


def canonical_to_messages_response(
    resp: CanonicalResponse, *, response_id: str, mode: ReasoningMode = ReasoningMode.SEPARATE
) -> MessagesResponse:
    """Translate a canonical chat response to the Anthropic message shape.

    lilbee carries a reasoning model's thinking inline as ``<think>...</think>``;
    SEPARATE reports it as a leading ``thinking`` block so clients render a
    clean answer. INLINE folds it into the answer text with the markers
    stripped, for clients that never render thinking blocks. OFF drops it: the
    caller asked for no thinking, and a template that ignores the request still
    thinks, so the block would contradict the answer the caller asked for.
    """
    text_parts = [b.text for b in resp.content if isinstance(b, TextBlock)]
    reasoning, answer = split_reasoning("".join(text_parts))
    if mode is ReasoningMode.INLINE and reasoning:
        answer = f"{reasoning}\n\n{answer}" if answer else reasoning
    if mode is not ReasoningMode.SEPARATE:
        reasoning = ""
    content: list[dict[str, Any]] = []
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})
    tool_uses = [b for b in resp.content if isinstance(b, ToolUseBlock)]
    if answer or not (reasoning or tool_uses):
        content.append({"type": "text", "text": answer})
    content.extend(
        {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input} for b in tool_uses
    )
    return MessagesResponse(
        id=response_id,
        model=resp.model,
        content=content,
        stop_reason=str(resp.stop_reason),
        usage=AnthropicUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        ),
    )


class _AnthropicStreamMapper:
    """Per-stream state for the canonical-to-Anthropic event converter.

    lilbee streams reasoning inline as ``<think>`` text, and Anthropic's wire
    format wants thinking and answer text in separate indexed blocks; the
    mapper re-blocks the stream, closing the open block whenever the token
    kind (thinking / text / tool_use) changes.

    INLINE routes reasoning into the text block instead, and OFF drops it: a
    parser built with ``show=False`` reports reasoning tokens with empty
    content, which never opens a block or emits a delta.
    """

    def __init__(self, *, mode: ReasoningMode = ReasoningMode.SEPARATE) -> None:
        self._reasoning = TagParser(show=mode is not ReasoningMode.OFF)
        self._inline = mode is ReasoningMode.INLINE
        self._next_index = 0
        self._open: _BlockKind | None = None

    def _close_open(self) -> list[tuple[AnthropicEventType, dict[str, Any]]]:
        if self._open is None:
            return []
        index = self._next_index - 1
        self._open = None
        return [
            (AnthropicEventType.CONTENT_BLOCK_STOP, {"type": "content_block_stop", "index": index})
        ]

    def _ensure_block(self, kind: _BlockKind) -> list[tuple[AnthropicEventType, dict[str, Any]]]:
        if self._open == kind:
            return []
        events = self._close_open()
        shell = (
            {"type": "thinking", "thinking": ""}
            if kind is _BlockKind.THINKING
            else {"type": "text", "text": ""}
        )
        events.append(
            (
                AnthropicEventType.CONTENT_BLOCK_START,
                {
                    "type": "content_block_start",
                    "index": self._next_index,
                    "content_block": shell,
                },
            )
        )
        self._open = kind
        self._next_index += 1
        return events

    def _text_events(
        self, tokens: list[StreamToken]
    ) -> list[tuple[AnthropicEventType, dict[str, Any]]]:
        events: list[tuple[AnthropicEventType, dict[str, Any]]] = []
        for token in tokens:
            if not token.content:
                continue
            thinking = token.is_reasoning and not self._inline
            kind = _BlockKind.THINKING if thinking else _BlockKind.TEXT
            events.extend(self._ensure_block(kind))
            index = self._next_index - 1
            delta = (
                {"type": "thinking_delta", "thinking": token.content}
                if thinking
                else {"type": "text_delta", "text": token.content}
            )
            events.append(
                (
                    AnthropicEventType.CONTENT_BLOCK_DELTA,
                    {"type": "content_block_delta", "index": index, "delta": delta},
                )
            )
        return events

    def block_start(
        self, event: ContentBlockStart
    ) -> list[tuple[AnthropicEventType, dict[str, Any]]]:
        if isinstance(event.block, ToolUseBlock):
            events = self._close_open()
            index = self._next_index
            self._next_index += 1
            self._open = _BlockKind.TOOL
            events.append(
                (
                    AnthropicEventType.CONTENT_BLOCK_START,
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": event.block.id,
                            "name": event.block.name,
                            "input": {},
                        },
                    },
                )
            )
            if event.block.input:
                # A provider that announces a whole call up front carries the
                # parsed input on the start block; forward it as one delta so
                # SDK accumulation still sees arguments.
                events.append(
                    (
                        AnthropicEventType.CONTENT_BLOCK_DELTA,
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(event.block.input),
                            },
                        },
                    )
                )
            return events
        # Text blocks open lazily on the first delta so an empty block never
        # emits a start/stop pair.
        return []

    def block_delta(
        self, event: ContentBlockDelta
    ) -> list[tuple[AnthropicEventType, dict[str, Any]]]:
        if isinstance(event.delta, TextDelta):
            return self._text_events(self._reasoning.feed(event.delta.text))
        if self._open is not _BlockKind.TOOL:
            # A tool delta for a block that never started is a provider quirk,
            # not a stream error; dropping beats crashing the stream.
            return []
        return [
            (
                AnthropicEventType.CONTENT_BLOCK_DELTA,
                {
                    "type": "content_block_delta",
                    "index": self._next_index - 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": event.delta.partial_json,
                    },
                },
            )
        ]

    def block_stop(self) -> list[tuple[AnthropicEventType, dict[str, Any]]]:
        remaining = self._reasoning.flush()
        events = self._text_events([remaining] if remaining else [])
        events.extend(self._close_open())
        return events


async def canonical_stream_to_anthropic_events(
    events: AsyncIterator[CanonicalStreamEvent],
    *,
    model: str,
    response_id: str,
    mode: ReasoningMode = ReasoningMode.SEPARATE,
) -> AsyncIterator[tuple[AnthropicEventType, dict[str, Any]]]:
    """Turn canonical stream events into Anthropic SSE ``(type, payload)`` pairs."""
    mapper = _AnthropicStreamMapper(mode=mode)
    yield (
        AnthropicEventType.MESSAGE_START,
        {
            "type": "message_start",
            "message": {
                "id": response_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    async for event in events:
        if isinstance(event, ContentBlockStart):
            for out in mapper.block_start(event):
                yield out
        elif isinstance(event, ContentBlockDelta):
            for out in mapper.block_delta(event):
                yield out
        elif isinstance(event, ContentBlockStop):
            for out in mapper.block_stop():
                yield out
        elif isinstance(event, MessageDelta):
            usage = event.usage or CanonicalUsage(input_tokens=0, output_tokens=0)
            yield (
                AnthropicEventType.MESSAGE_DELTA,
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": str(event.stop_reason or StopReason.END_TURN),
                        "stop_sequence": None,
                    },
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                    },
                },
            )
        elif isinstance(event, MessageStart | MessageStop):
            # The Anthropic message_start is emitted eagerly above (it needs no
            # canonical data), and message_stop follows the loop so it stays
            # last even if the provider never sends one.
            continue
    yield (AnthropicEventType.MESSAGE_STOP, {"type": "message_stop"})
