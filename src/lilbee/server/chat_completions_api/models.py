"""Pydantic models for the OpenAI Chat Completions wire shapes."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


class ToolChoiceMode(StrEnum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class CompletionsTextContent(BaseModel):
    """Text part of a multi-part message content."""

    type: Literal["text"]
    text: str


class CompletionsImageUrl(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class CompletionsImageContent(BaseModel):
    """Image part of a multi-part message content."""

    type: Literal["image_url"]
    image_url: CompletionsImageUrl


CompletionsMessageContentPart = Annotated[
    CompletionsTextContent | CompletionsImageContent,
    Field(discriminator="type"),
]


class CompletionsToolCallFunction(BaseModel):
    name: str
    arguments: str = "{}"


class CompletionsToolCall(BaseModel):
    """Assistant-side tool_call entry inside a request message."""

    id: str
    type: Literal["function"] = "function"
    function: CompletionsToolCallFunction


class CompletionsMessage(BaseModel):
    """One entry in the request ``messages`` list."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[CompletionsMessageContentPart] | None = None
    name: str | None = None
    tool_calls: list[CompletionsToolCall] | None = None
    tool_call_id: str | None = None


class CompletionsFunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class CompletionsTool(BaseModel):
    type: Literal["function"] = "function"
    function: CompletionsFunctionDef


class CompletionsToolChoiceFunction(BaseModel):
    name: str


class CompletionsNamedToolChoice(BaseModel):
    """Explicit ``{type: "function", function: {name: ...}}`` tool_choice."""

    type: Literal["function"]
    function: CompletionsToolChoiceFunction


class CompletionsRequest(BaseModel):
    """Top-level ``POST /v1/chat/completions`` request body."""

    model: str = Field(min_length=1)
    messages: list[CompletionsMessage] = Field(min_length=1)
    tools: list[CompletionsTool] | None = None
    tool_choice: ToolChoiceMode | CompletionsNamedToolChoice | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False


class CompletionsResponseToolCallFunction(BaseModel):
    name: str
    arguments: str


class CompletionsResponseToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: CompletionsResponseToolCallFunction


class CompletionsResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[CompletionsResponseToolCall] | None = None


class CompletionsResponseChoice(BaseModel):
    index: int = 0
    message: CompletionsResponseMessage
    finish_reason: FinishReason


class CompletionsUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionsResponse(BaseModel):
    """Non-streaming ``/v1/chat/completions`` response body."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionsResponseChoice]
    usage: CompletionsUsage


class CompletionsStreamToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class CompletionsStreamToolCall(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: CompletionsStreamToolCallFunction | None = None


class CompletionsStreamDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[CompletionsStreamToolCall] | None = None


class CompletionsStreamChoice(BaseModel):
    index: int = 0
    delta: CompletionsStreamDelta
    finish_reason: FinishReason | None = None


class CompletionsStreamChunk(BaseModel):
    """Single SSE frame from streaming ``/v1/chat/completions``."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[CompletionsStreamChoice]


class ModelEntry(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "lilbee"
    created: int


class ModelsListResponse(BaseModel):
    """``GET /v1/models`` response envelope."""

    object: Literal["list"] = "list"
    data: list[ModelEntry]
