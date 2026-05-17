"""Translation between Anthropic Messages wire shape and lilbee canonical."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any, Literal

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
    StopReason,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)

_STOP_REASON_TO_MESSAGES: dict[StopReason, str] = {
    StopReason.END_TURN: "end_turn",
    StopReason.MAX_TOKENS: "max_tokens",
    StopReason.STOP_SEQUENCE: "stop_sequence",
    StopReason.TOOL_USE: "tool_use",
    StopReason.ERROR: "end_turn",
}

_TOOL_CHOICE_MODES: frozenset[str] = frozenset({"auto", "any", "none", "tool"})

_MessageRole = Literal["user", "assistant", "tool"]


def messages_to_canonical_request(payload: dict[str, Any]) -> CanonicalChatRequest:
    """Translate an Anthropic ``/v1/messages`` request body to canonical form."""
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Request missing required 'model' field")
    messages_raw = payload.get("messages")
    if not isinstance(messages_raw, list):
        raise ValueError("Request missing required 'messages' field")
    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int):
        raise ValueError("Request missing required 'max_tokens' field")

    return CanonicalChatRequest(
        model=model,
        messages=[_translate_message(m) for m in messages_raw],
        system=payload.get("system"),
        tools=_translate_tools(payload.get("tools")),
        tool_choice=_translate_tool_choice(payload.get("tool_choice")),
        temperature=payload.get("temperature"),
        top_p=payload.get("top_p"),
        top_k=payload.get("top_k"),
        max_tokens=max_tokens,
        stop=payload.get("stop_sequences"),
        stream=bool(payload.get("stream", False)),
    )


def canonical_to_messages_response(resp: CanonicalResponse) -> dict[str, Any]:
    """Translate a canonical chat response to an Anthropic message object."""
    return {
        "id": resp.id,
        "type": "message",
        "role": "assistant",
        "model": resp.model,
        "content": [_block_to_wire(b) for b in resp.content],
        "stop_reason": _STOP_REASON_TO_MESSAGES[resp.stop_reason],
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
    }


async def canonical_stream_to_messages_events(
    events: AsyncIterator[CanonicalStreamEvent],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield ``(event_name, payload)`` pairs in Anthropic streaming shape."""
    async for event in events:
        if isinstance(event, MessageStart):
            yield (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": event.id,
                        "type": "message",
                        "role": "assistant",
                        "model": event.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        elif isinstance(event, ContentBlockStart):
            yield (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": event.index,
                    "content_block": _block_to_wire(event.block),
                },
            )
        elif isinstance(event, ContentBlockDelta):
            yield (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": event.index,
                    "delta": _delta_to_wire(event.delta),
                },
            )
        elif isinstance(event, ContentBlockStop):
            yield (
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": event.index,
                },
            )
        elif isinstance(event, MessageDelta):
            stop = (
                _STOP_REASON_TO_MESSAGES[event.stop_reason]
                if event.stop_reason is not None
                else None
            )
            usage = event.usage
            yield (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop, "stop_sequence": None},
                    "usage": {
                        "input_tokens": usage.input_tokens if usage else 0,
                        "output_tokens": usage.output_tokens if usage else 0,
                    },
                },
            )
        else:
            # MessageStop is the only remaining variant in the union.
            yield "message_stop", {"type": "message_stop"}


def _translate_message(raw: dict[str, Any]) -> CanonicalMessage:
    role: _MessageRole = raw["role"]
    content = raw["content"]
    if isinstance(content, str):
        blocks: list[ContentBlock] = [TextBlock(text=content)]
    else:
        blocks = [_translate_block(b) for b in content]
    return CanonicalMessage(role=role, content=blocks)


def _translate_block(raw: dict[str, Any]) -> ContentBlock:
    btype = raw.get("type")
    if btype == "text":
        return TextBlock(text=raw["text"])
    if btype == "image":
        source = raw["source"]
        if source.get("type") != "base64":
            raise ValueError("Unsupported image source type; only 'base64' is accepted")
        return ImageBlock(
            media_type=source["media_type"],
            data=base64.b64decode(source["data"]),
        )
    if btype == "tool_use":
        return ToolUseBlock(
            id=raw["id"],
            name=raw["name"],
            input=raw.get("input", {}),
        )
    if btype == "tool_result":
        result_content = raw.get("content", "")
        if isinstance(result_content, str):
            nested: list[ContentBlock] = [TextBlock(text=result_content)]
        else:
            nested = [_translate_block(b) for b in result_content]
        return ToolResultBlock(
            tool_use_id=raw["tool_use_id"],
            content=nested,
            is_error=bool(raw.get("is_error", False)),
        )
    raise ValueError(f"Unknown content block type: {btype!r}")


def _translate_tools(
    raw: list[dict[str, Any]] | None,
) -> list[CanonicalTool] | None:
    if not raw:
        return None
    return [
        CanonicalTool(
            name=t["name"],
            description=t.get("description", ""),
            input_schema=t.get("input_schema", {}),
        )
        for t in raw
    ]


def _translate_tool_choice(
    raw: dict[str, Any] | None,
) -> CanonicalToolChoice | None:
    if raw is None:
        return None
    mode = raw.get("type")
    if mode not in _TOOL_CHOICE_MODES:
        raise ValueError(f"Unknown tool_choice type: {mode!r}")
    if mode == "tool":
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice {'type': 'tool'} requires a 'name'")
        return CanonicalToolChoice(mode="tool", tool_name=name)
    return CanonicalToolChoice(mode=mode)  # type: ignore[arg-type]


def _block_to_wire(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.media_type,
                "data": base64.b64encode(block.data).decode("ascii"),
            },
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    # ToolResultBlock is the only remaining variant.
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": [_block_to_wire(b) for b in block.content],
        "is_error": block.is_error,
    }


def _delta_to_wire(delta: TextDelta | ToolUseDelta) -> dict[str, Any]:
    if isinstance(delta, TextDelta):
        return {"type": "text_delta", "text": delta.text}
    return {"type": "input_json_delta", "partial_json": delta.partial_json}
