"""Protocol-neutral chat request, response, and stream-event types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


class StopReason(StrEnum):
    """Why a canonical chat response ended."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"


@dataclass(frozen=True)
class TextBlock:
    """Plain-text content block."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    """Assistant-emitted tool invocation with parsed JSON arguments."""

    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True)
class ToolResultBlock:
    """Caller-supplied tool result paired to a prior ToolUseBlock by id."""

    tool_use_id: str
    content: list[ContentBlock]
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True)
class CanonicalMessage:
    """One chat turn; content is always a typed-block list."""

    role: Literal["user", "assistant", "tool"]
    content: list[ContentBlock]

    @classmethod
    def from_string(
        cls,
        *,
        role: Literal["user", "assistant", "tool"],
        text: str,
    ) -> CanonicalMessage:
        """Build a single-text-block message from a raw string."""
        return cls(role=role, content=[TextBlock(text=text)])


@dataclass(frozen=True)
class CanonicalTool:
    """Tool definition (JSON-Schema input shape)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class CanonicalToolChoice:
    """Tool-choice mode; ``tool_name`` is required only when ``mode == "tool"``."""

    mode: Literal["auto", "any", "none", "tool"]
    tool_name: str | None = None


@dataclass(frozen=True)
class CanonicalChatRequest:
    """Canonical chat request consumed by the dispatch layer."""

    model: str
    messages: list[CanonicalMessage]
    system: str | None = None
    tools: list[CanonicalTool] | None = None
    tool_choice: CanonicalToolChoice | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    stream: bool = False


@dataclass(frozen=True)
class CanonicalUsage:
    """Token-count summary for one chat response."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CanonicalResponse:
    """Canonical non-streaming chat response."""

    id: str
    model: str
    content: list[ContentBlock]
    stop_reason: StopReason
    usage: CanonicalUsage


@dataclass(frozen=True)
class MessageStart:
    """Stream prelude carrying the message id and model ref."""

    id: str
    model: str


@dataclass(frozen=True)
class ContentBlockStart:
    """Opens a fresh content block at ``index`` with an initial shell."""

    index: int
    block: ContentBlock


@dataclass(frozen=True)
class TextDelta:
    """One text-token delta within an open text block."""

    text: str


@dataclass(frozen=True)
class ToolUseDelta:
    """Accumulating JSON fragment within an open tool-use block."""

    partial_json: str


@dataclass(frozen=True)
class ContentBlockDelta:
    """Delta payload routed to the content block at ``index``."""

    index: int
    delta: TextDelta | ToolUseDelta


@dataclass(frozen=True)
class ContentBlockStop:
    """Closes the content block at ``index``."""

    index: int


@dataclass(frozen=True)
class MessageDelta:
    """Trailing metadata; either field may carry a value, never both required."""

    stop_reason: StopReason | None = None
    usage: CanonicalUsage | None = None


@dataclass(frozen=True)
class MessageStop:
    """Stream terminator."""


CanonicalStreamEvent = (
    MessageStart
    | ContentBlockStart
    | ContentBlockDelta
    | ContentBlockStop
    | MessageDelta
    | MessageStop
)
