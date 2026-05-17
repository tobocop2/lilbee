"""Translation between the chat-completions wire shape and the canonical types."""

from __future__ import annotations

import base64
import json
import time
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
    ImageBlock,
    MessageDelta,
    StopReason,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)

_OPENAI_TOOL_CHOICE_MODES: dict[str, Literal["auto", "any", "none"]] = {
    "auto": "auto",
    "none": "none",
    "required": "any",
}

_STOP_REASON_TO_OPENAI_FINISH: dict[StopReason, str] = {
    StopReason.END_TURN: "stop",
    StopReason.MAX_TOKENS: "length",
    StopReason.STOP_SEQUENCE: "stop",
    StopReason.TOOL_USE: "tool_calls",
    StopReason.ERROR: "stop",
}

_VALID_ROLES = frozenset({"user", "assistant", "tool", "system"})


def completions_to_canonical_request(payload: dict[str, Any]) -> CanonicalChatRequest:
    """Translate a ``/v1/chat/completions`` JSON payload to a canonical request."""
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Field 'model' is required.")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Field 'messages' is required and must be non-empty.")

    system_parts: list[str] = []
    messages: list[CanonicalMessage] = []
    for raw in raw_messages:
        role = raw.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"Unknown message role: {role!r}")
        if role == "system":
            system_parts.append(_extract_system_text(raw))
            continue
        messages.append(_message_from_openai(raw))

    tools = _tools_from_openai(payload.get("tools"))
    tool_choice = _tool_choice_from_openai(payload.get("tool_choice"))
    stop = _stop_from_openai(payload.get("stop"))

    return CanonicalChatRequest(
        model=model,
        messages=messages,
        system="\n\n".join(system_parts) if system_parts else None,
        tools=tools,
        tool_choice=tool_choice,
        temperature=payload.get("temperature"),
        top_p=payload.get("top_p"),
        top_k=payload.get("top_k"),
        max_tokens=payload.get("max_tokens"),
        stop=stop,
        stream=bool(payload.get("stream", False)),
    )


def canonical_to_completions_response(
    resp: CanonicalResponse, *, response_id: str | None = None
) -> dict[str, Any]:
    """Translate a canonical chat response to an OpenAI ``chat.completion`` object."""
    message: dict[str, Any] = {"role": "assistant"}
    text_parts = [b.text for b in resp.content if isinstance(b, TextBlock)]
    tool_calls = [_openai_tool_call(b) for b in resp.content if isinstance(b, ToolUseBlock)]
    if tool_calls:
        message["content"] = "".join(text_parts) if text_parts else None
        message["tool_calls"] = tool_calls
    else:
        message["content"] = "".join(text_parts)

    total = resp.usage.input_tokens + resp.usage.output_tokens
    return {
        "id": response_id if response_id is not None else resp.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _STOP_REASON_TO_OPENAI_FINISH[resp.stop_reason],
            }
        ],
        "usage": {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": total,
        },
    }


async def canonical_stream_to_completions_chunks(
    events: AsyncIterator[CanonicalStreamEvent],
    *,
    model: str,
    response_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """Turn canonical stream events into a sequence of OpenAI chunk dicts."""
    role_emitted = False
    tool_index_for_block: dict[int, int] = {}
    next_tool_index = 0

    async for event in events:
        if isinstance(event, ContentBlockStart):
            if isinstance(event.block, TextBlock):
                if not role_emitted:
                    yield _chunk(model, response_id, {"role": "assistant"})
                    role_emitted = True
            elif isinstance(event.block, ToolUseBlock):
                tool_index = next_tool_index
                tool_index_for_block[event.index] = tool_index
                next_tool_index += 1
                yield _chunk(
                    model,
                    response_id,
                    {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": event.block.id,
                                "type": "function",
                                "function": {
                                    "name": event.block.name,
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                )
        elif isinstance(event, ContentBlockDelta):
            if isinstance(event.delta, TextDelta):
                yield _chunk(model, response_id, {"content": event.delta.text})
            elif isinstance(event.delta, ToolUseDelta):
                tool_index = tool_index_for_block[event.index]
                yield _chunk(
                    model,
                    response_id,
                    {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "function": {"arguments": event.delta.partial_json},
                            }
                        ]
                    },
                )
        elif isinstance(event, MessageDelta):
            finish = (
                _STOP_REASON_TO_OPENAI_FINISH[event.stop_reason]
                if event.stop_reason is not None
                else "stop"
            )
            yield _chunk(model, response_id, {}, finish_reason=finish)


def _chunk(
    model: str,
    response_id: str,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }


def _extract_system_text(raw: dict[str, Any]) -> str:
    content = raw.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    raise ValueError(f"Unsupported system content shape: {type(content).__name__}")


def _message_from_openai(raw: dict[str, Any]) -> CanonicalMessage:
    role = raw["role"]
    if role == "tool":
        return CanonicalMessage(
            role="tool",
            content=[
                ToolResultBlock(
                    tool_use_id=raw.get("tool_call_id", ""),
                    content=_tool_result_content(raw.get("content")),
                )
            ],
        )

    blocks: list[ContentBlock] = []
    text_blocks = _content_blocks(raw.get("content"))
    blocks.extend(text_blocks)
    for call in raw.get("tool_calls") or []:
        blocks.append(_tool_use_from_openai(call))
    return CanonicalMessage(role=role, content=blocks)


def _content_blocks(content: Any) -> list[ContentBlock]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [TextBlock(text=content)]
    if isinstance(content, list):
        out: list[ContentBlock] = []
        for part in content:
            kind = part.get("type")
            if kind == "text":
                out.append(TextBlock(text=part.get("text", "")))
            elif kind == "image_url":
                out.append(_image_from_openai(part.get("image_url", {})))
            else:
                raise ValueError(f"Unsupported content block type: {kind!r}")
        return out
    raise ValueError(f"Unsupported content shape: {type(content).__name__}")


def _tool_result_content(content: Any) -> list[ContentBlock]:
    if isinstance(content, str):
        return [TextBlock(text=content)]
    if isinstance(content, list):
        return _content_blocks(content)
    return [TextBlock(text="" if content is None else str(content))]


def _image_from_openai(image_url: dict[str, Any]) -> ImageBlock:
    url = image_url.get("url", "")
    if url.startswith("data:"):
        header, _, b64 = url.partition(",")
        media_type = header.removeprefix("data:").partition(";")[0] or "image/png"
        return ImageBlock(media_type=media_type, data=base64.b64decode(b64))
    return ImageBlock(media_type="image/url", data=url.encode())


def _tool_use_from_openai(call: dict[str, Any]) -> ToolUseBlock:
    function = call.get("function", {})
    name = function.get("name", "")
    raw_args = function.get("arguments", "")
    try:
        parsed = json.loads(raw_args) if raw_args else {}
    except (TypeError, ValueError):
        parsed = {"_raw": raw_args}
    if not isinstance(parsed, dict):
        parsed = {"_raw": raw_args}
    return ToolUseBlock(id=call.get("id", ""), name=name, input=parsed)


def _openai_tool_call(block: ToolUseBlock) -> dict[str, Any]:
    return {
        "id": block.id,
        "type": "function",
        "function": {
            "name": block.name,
            "arguments": json.dumps(block.input),
        },
    }


def _tools_from_openai(raw: Any) -> list[CanonicalTool] | None:
    if not raw:
        return None
    tools: list[CanonicalTool] = []
    for entry in raw:
        function = entry.get("function", {})
        tools.append(
            CanonicalTool(
                name=function.get("name", ""),
                description=function.get("description", ""),
                input_schema=function.get("parameters", {}),
            )
        )
    return tools


def _tool_choice_from_openai(raw: Any) -> CanonicalToolChoice | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        mode = _OPENAI_TOOL_CHOICE_MODES.get(raw)
        if mode is None:
            raise ValueError(f"Unknown tool_choice mode: {raw!r}")
        return CanonicalToolChoice(mode=mode)
    if isinstance(raw, dict):
        function = raw.get("function", {})
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice function dict requires a non-empty 'name'.")
        return CanonicalToolChoice(mode="tool", tool_name=name)
    raise ValueError(f"Unsupported tool_choice shape: {type(raw).__name__}")


def _stop_from_openai(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    raise ValueError(f"Unsupported stop shape: {type(raw).__name__}")
