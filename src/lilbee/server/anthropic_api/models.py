"""Pydantic models for the Anthropic Messages API wire shapes."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AnthropicEventType(StrEnum):
    """SSE event vocabulary of the Anthropic Messages stream."""

    MESSAGE_START = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    PING = "ping"
    ERROR = "error"


class _AnthropicModel(BaseModel):
    """Base for request models: unknown fields parse and are ignored.

    Anthropic clients send fields this surface does not act on (``thinking``,
    ``metadata``, ``cache_control``, ``output_config``, ``betas``). Rejecting
    them with a 400 hard-fails Claude Code, so they are tolerated instead.
    """

    model_config = ConfigDict(extra="allow")


class SystemTextBlock(_AnthropicModel):
    """One text block of a block-form ``system`` prompt."""

    type: Literal["text"]
    text: str


class TextBlockParam(_AnthropicModel):
    """Text content block inside a request message."""

    type: Literal["text"]
    text: str


class ToolUseBlockParam(_AnthropicModel):
    """Assistant-side tool invocation replayed in the conversation."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ImageBlockParam(_AnthropicModel):
    """Image content block; parsed so the translator can reject it clearly."""

    type: Literal["image"]
    source: dict[str, Any] = Field(default_factory=dict)


class UnknownBlockParam(_AnthropicModel):
    """Catch-all for block types this surface ignores (``thinking``, ...).

    Claude Code replays ``thinking``/``redacted_thinking`` blocks from earlier
    assistant turns; failing validation on them would break every follow-up
    turn, so they parse here and the translator drops them.
    """

    type: str


class ToolResultBlockParam(_AnthropicModel):
    """Caller-supplied result for a prior tool_use, paired by id."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: (
        str
        | list[
            Annotated[
                TextBlockParam | ImageBlockParam | UnknownBlockParam,
                Field(union_mode="left_to_right"),
            ]
        ]
        | None
    ) = None
    is_error: bool = False


ContentBlockParam = Annotated[
    TextBlockParam | ToolUseBlockParam | ToolResultBlockParam | ImageBlockParam | UnknownBlockParam,
    # Left-to-right keeps dispatch deterministic: each known type matches its
    # literal or fails fast, and anything new lands on the catch-all.
    Field(union_mode="left_to_right"),
]


class AnthropicMessage(_AnthropicModel):
    """One entry in the request ``messages`` list.

    ``system`` is Anthropic's mid-conversation operator channel; Claude Code
    sends it routinely (mode switches, injected context), so rejecting it
    breaks every session after the first such turn.
    """

    role: Literal["user", "assistant", "system"]
    content: str | list[ContentBlockParam]


class AnthropicTool(_AnthropicModel):
    """Tool definition; server-tool entries parse with an empty schema."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolChoice(_AnthropicModel):
    """Tool-choice selector; ``name`` accompanies ``type == "tool"``."""

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None


class MessagesRequest(_AnthropicModel):
    """The ``POST /v1/messages`` request body."""

    model: str
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[SystemTextBlock] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False


class AnthropicUsage(BaseModel):
    """Token counts in the Anthropic response shape."""

    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    """The non-streaming ``/v1/messages`` response body.

    ``content`` blocks vary in shape (``thinking``/``text``/``tool_use``), so
    they stay plain dicts; ``stop_sequence`` is emitted as an explicit null
    because Anthropic SDK clients expect the field present.
    """

    id: str
    model: str
    content: list[dict[str, Any]]
    stop_reason: str
    usage: AnthropicUsage
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    stop_sequence: str | None = None
