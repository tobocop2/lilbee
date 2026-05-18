"""Translation between OpenAI chat-completions models and the canonical types."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Literal

from lilbee.server.chat_completions_api.models import (
    CompletionsImageContent,
    CompletionsImageUrl,
    CompletionsMessage,
    CompletionsNamedToolChoice,
    CompletionsRequest,
    CompletionsResponse,
    CompletionsResponseChoice,
    CompletionsResponseMessage,
    CompletionsResponseToolCall,
    CompletionsResponseToolCallFunction,
    CompletionsStreamChoice,
    CompletionsStreamChunk,
    CompletionsStreamDelta,
    CompletionsStreamToolCall,
    CompletionsStreamToolCallFunction,
    CompletionsTextContent,
    CompletionsTool,
    CompletionsUsage,
    FinishReason,
    ToolChoiceMode,
)
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalMessage,
    CanonicalResponse,
    CanonicalStreamEvent,
    CanonicalTool,
    CanonicalToolChoice,
    ContentBlock,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    ImageBlock,
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
from lilbee.server.chat_dispatch.tool_args import parse_tool_arguments

_TOOL_CHOICE_MODES: dict[ToolChoiceMode, Literal["auto", "any", "none"]] = {
    ToolChoiceMode.AUTO: "auto",
    ToolChoiceMode.NONE: "none",
    ToolChoiceMode.REQUIRED: "any",
}

_STOP_REASON_TO_FINISH: dict[StopReason, FinishReason] = {
    StopReason.END_TURN: FinishReason.STOP,
    StopReason.MAX_TOKENS: FinishReason.LENGTH,
    StopReason.STOP_SEQUENCE: FinishReason.STOP,
    StopReason.TOOL_USE: FinishReason.TOOL_CALLS,
    StopReason.ERROR: FinishReason.STOP,
}


def completions_to_canonical_request(request: CompletionsRequest) -> CanonicalChatRequest:
    """Translate a validated ``CompletionsRequest`` to the canonical request."""
    system_parts: list[str] = []
    messages: list[CanonicalMessage] = []
    for msg in request.messages:
        if msg.role == "system":
            system_parts.append(_system_text(msg))
            continue
        messages.append(_message_from_request(msg))

    return CanonicalChatRequest(
        model=request.model,
        messages=messages,
        system="\n\n".join(system_parts) if system_parts else None,
        tools=_tools_from_request(request.tools),
        tool_choice=_tool_choice_from_request(request.tool_choice),
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        stop=_stop_from_request(request.stop),
        stream=request.stream,
    )


def canonical_to_completions_response(
    resp: CanonicalResponse, *, response_id: str
) -> CompletionsResponse:
    """Translate a canonical chat response to the OpenAI ``chat.completion`` model."""
    text_parts = [b.text for b in resp.content if isinstance(b, TextBlock)]
    tool_calls = [_response_tool_call(b) for b in resp.content if isinstance(b, ToolUseBlock)]
    joined = "".join(text_parts)
    content: str | None = joined if joined or not tool_calls else None

    total = resp.usage.input_tokens + resp.usage.output_tokens
    return CompletionsResponse(
        id=response_id,
        created=int(time.time()),
        model=resp.model,
        choices=[
            CompletionsResponseChoice(
                index=0,
                message=CompletionsResponseMessage(
                    content=content,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=_STOP_REASON_TO_FINISH[resp.stop_reason],
            )
        ],
        usage=CompletionsUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=total,
        ),
    )


class _StreamMapper:
    """Per-stream state for the canonical-to-OpenAI chunk converter."""

    def __init__(self) -> None:
        self._role_emitted = False
        self._tool_index_for_block: dict[int, int] = {}
        self._next_tool_index = 0

    def block_start(self, event: ContentBlockStart) -> CompletionsStreamDelta | None:
        if isinstance(event.block, TextBlock):
            if self._role_emitted:
                return None
            self._role_emitted = True
            return CompletionsStreamDelta(role="assistant")
        if isinstance(event.block, ToolUseBlock):
            tool_index = self._next_tool_index
            self._tool_index_for_block[event.index] = tool_index
            self._next_tool_index += 1
            return CompletionsStreamDelta(
                tool_calls=[_tool_call_open(tool_index, event.block.id, event.block.name)],
            )
        return None

    def block_delta(self, event: ContentBlockDelta) -> CompletionsStreamDelta | None:
        if isinstance(event.delta, TextDelta):
            return CompletionsStreamDelta(content=event.delta.text)
        if isinstance(event.delta, ToolUseDelta):
            tool_index = self._tool_index_for_block[event.index]
            return CompletionsStreamDelta(
                tool_calls=[_tool_call_args(tool_index, event.delta.partial_json)],
            )
        return None


def _tool_call_open(index: int, call_id: str, name: str) -> CompletionsStreamToolCall:
    return CompletionsStreamToolCall(
        index=index,
        id=call_id,
        type="function",
        function=CompletionsStreamToolCallFunction(name=name, arguments=""),
    )


def _tool_call_args(index: int, partial_json: str) -> CompletionsStreamToolCall:
    return CompletionsStreamToolCall(
        index=index,
        function=CompletionsStreamToolCallFunction(arguments=partial_json),
    )


def _finish_reason_for(event: MessageDelta) -> FinishReason:
    if event.stop_reason is None:
        return FinishReason.STOP
    return _STOP_REASON_TO_FINISH[event.stop_reason]


async def canonical_stream_to_completions_chunks(
    events: AsyncIterator[CanonicalStreamEvent],
    *,
    model: str,
    response_id: str,
) -> AsyncIterator[CompletionsStreamChunk]:
    """Turn canonical stream events into ``CompletionsStreamChunk`` instances."""
    mapper = _StreamMapper()
    async for event in events:
        if isinstance(event, ContentBlockStart):
            delta = mapper.block_start(event)
            if delta is not None:
                yield _chunk(model, response_id, delta)
        elif isinstance(event, ContentBlockDelta):
            delta = mapper.block_delta(event)
            if delta is not None:
                yield _chunk(model, response_id, delta)
        elif isinstance(event, MessageDelta):
            yield _chunk(
                model,
                response_id,
                CompletionsStreamDelta(),
                finish_reason=_finish_reason_for(event),
            )
        elif isinstance(event, MessageStart | ContentBlockStop | MessageStop):
            # OpenAI's wire format has no equivalent for these canonical events:
            # MessageStart carries metadata we already encoded in the chunk header,
            # ContentBlockStop is implicit (the next block opens), and MessageStop
            # is replaced by the final chunk's finish_reason. Explicit branch so
            # a new event type added later forces a translation decision.
            continue


def _chunk(
    model: str,
    response_id: str,
    delta: CompletionsStreamDelta,
    *,
    finish_reason: FinishReason | None = None,
) -> CompletionsStreamChunk:
    return CompletionsStreamChunk(
        id=response_id,
        created=int(time.time()),
        model=model,
        choices=[CompletionsStreamChoice(index=0, delta=delta, finish_reason=finish_reason)],
    )


def _system_text(msg: CompletionsMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return "".join(
            part.text for part in msg.content if isinstance(part, CompletionsTextContent)
        )
    return ""


def _message_from_request(msg: CompletionsMessage) -> CanonicalMessage:
    role = msg.role
    if role == "system":
        raise ValueError("system messages should be extracted by the caller")
    if role == "tool":
        return CanonicalMessage(
            role="tool",
            content=[
                ToolResultBlock(
                    tool_use_id=msg.tool_call_id or "",
                    content=_tool_result_content(msg.content),
                )
            ],
        )

    blocks: list[ContentBlock] = list(_content_blocks(msg.content))
    for call in msg.tool_calls or []:
        blocks.append(
            ToolUseBlock(
                id=call.id,
                name=call.function.name,
                input=parse_tool_arguments(call.function.arguments),
            )
        )
    return CanonicalMessage(role=role, content=blocks)


def _content_blocks(content: str | list | None) -> list[ContentBlock]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [TextBlock(text=content)]
    blocks: list[ContentBlock] = []
    for part in content:
        if isinstance(part, CompletionsTextContent):
            blocks.append(TextBlock(text=part.text))
        elif isinstance(part, CompletionsImageContent):
            blocks.append(_image_from_request(part.image_url))
    return blocks


def _tool_result_content(content: str | list | None) -> list[ContentBlock]:
    if isinstance(content, str):
        return [TextBlock(text=content)]
    if isinstance(content, list):
        return _content_blocks(content)
    return [TextBlock(text="" if content is None else str(content))]


def _image_from_request(image_url: CompletionsImageUrl) -> ImageBlock:
    url = image_url.url
    if url.startswith("data:"):
        header, _, b64 = url.partition(",")
        media_type = header.removeprefix("data:").partition(";")[0] or "image/png"
        return ImageBlock(media_type=media_type, data=base64.b64decode(b64))
    return ImageBlock(media_type="image/url", data=url.encode())


def _response_tool_call(block: ToolUseBlock) -> CompletionsResponseToolCall:
    return CompletionsResponseToolCall(
        id=block.id,
        function=CompletionsResponseToolCallFunction(
            name=block.name, arguments=json.dumps(block.input)
        ),
    )


def _tools_from_request(tools: list[CompletionsTool] | None) -> list[CanonicalTool] | None:
    if not tools:
        return None
    return [
        CanonicalTool(
            name=tool.function.name,
            description=tool.function.description or "",
            input_schema=tool.function.parameters,
        )
        for tool in tools
    ]


def _tool_choice_from_request(
    choice: ToolChoiceMode | CompletionsNamedToolChoice | None,
) -> CanonicalToolChoice | None:
    if choice is None:
        return None
    if isinstance(choice, ToolChoiceMode):
        return CanonicalToolChoice(mode=_TOOL_CHOICE_MODES[choice])
    return CanonicalToolChoice(mode="tool", tool_name=choice.function.name)


def _stop_from_request(stop: str | list[str] | None) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)
