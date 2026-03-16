"""Request and response models for the lilbee HTTP API.

Typed pydantic models so Litestar's OpenAPI schema has field-level detail.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AskRequest(BaseModel):
    """Request body for /api/ask."""

    question: str
    top_k: int = 0
    options: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    """Request body for /api/chat."""

    question: str
    history: list[ChatMessage] = []
    top_k: int = 0
    options: dict[str, Any] | None = None


class SyncRequest(BaseModel):
    """Request body for /api/sync."""

    force_vision: bool = False


class AddRequest(BaseModel):
    """Request body for /api/add."""

    paths: list[str]
    force: bool = False
    vision_model: str = ""


class SetModelRequest(BaseModel):
    """Request body for /api/models/chat and /api/models/vision."""

    model: str


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    content: str


class CleanedChunk(BaseModel):
    """A search result chunk with vector stripped and distance renamed."""

    source: str
    content_type: str
    chunk: str
    distance: float
    page_start: int = 0
    page_end: int = 0
    line_start: int = 0
    line_end: int = 0
    chunk_index: int = 0


class HealthResponse(BaseModel):
    """Response for /api/health."""

    status: str
    version: str


class AskResponse(BaseModel):
    """Response for /api/ask and /api/chat."""

    answer: str
    sources: list[CleanedChunk]


class SetModelResponse(BaseModel):
    """Response for /api/models/chat and /api/models/vision."""

    model: str
