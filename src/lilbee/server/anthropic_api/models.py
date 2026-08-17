"""Pydantic models for the Anthropic Messages API wire shapes."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_THINKING_TYPES = frozenset({"enabled", "disabled"})

MIN_THINKING_BUDGET_TOKENS = 1024
"""Anthropic's documented minimum for ``thinking.budget_tokens``."""


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

    Anthropic clients send fields this surface does not act on (``metadata``,
    ``cache_control``, ``output_config``, ``betas``). Rejecting them with a 400
    hard-fails Claude Code, so they are tolerated instead.
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


class AnthropicThinking(_AnthropicModel):
    """The ``thinking`` parameter: whether the model may reason on this call.

    ``budget_tokens`` tightens the reasoning cap for this call, at roughly four
    characters per token. It may only tighten: a budget above the configured
    cap leaves the configured cap in place.

    The ``1024`` floor is Anthropic's own documented minimum, and it is what
    keeps the tightening rule true. The cap reads ``0`` as unlimited, so a
    budget that resolved to zero characters would turn a per-request limit into
    no limit at all.
    """

    type: Literal["enabled", "disabled"]
    budget_tokens: int | None = Field(default=None, ge=MIN_THINKING_BUDGET_TOKENS)


class MessagesRequest(_AnthropicModel):
    """The ``POST /v1/messages`` request body.

    ``thinking`` picks the reasoning mode for this call, overriding the
    ``messages_reasoning`` setting.
    """

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
    thinking: AnthropicThinking | None = None

    @field_validator("thinking", mode="before")
    @classmethod
    def _known_thinking_shapes_only(cls, value: object) -> object:
        # Anthropic defines enabled/disabled today. A shape this surface does
        # not know falls back to the setting instead of failing the request,
        # because a 400 here stops the agent mid-session.
        if value is None or (isinstance(value, dict) and value.get("type") in _THINKING_TYPES):
            return value
        return None


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
