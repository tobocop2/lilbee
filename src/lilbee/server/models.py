"""Request and response models for the lilbee HTTP API.

Typed pydantic models so Litestar's OpenAPI schema has field-level detail.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for /api/ask."""

    question: str
    top_k: int = Field(default=0, le=100)
    options: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    """Request body for /api/chat."""

    question: str
    history: list[ChatMessage] = []
    top_k: int = Field(default=0, le=100)
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

    role: Literal["user", "assistant"]
    content: str


class CleanedChunk(BaseModel):
    """A search result chunk with vector stripped and distance renamed."""

    source: str
    content_type: str
    chunk: str
    distance: float | None = None
    relevance_score: float | None = None
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


class CrawlRequest(BaseModel):
    """Request body for /api/crawl."""

    url: str
    depth: int = Field(default=0, le=10)
    max_pages: int = Field(default=50, le=1000)


class WikiPageSummary(BaseModel):
    """Summary of a wiki page for list endpoints."""

    slug: str
    title: str = ""
    page_type: str = "unknown"
    source_count: int = 0
    created_at: str = ""


class WikiCitation(BaseModel):
    """A citation linking a wiki claim to a source location."""

    citation_key: str
    claim_type: str = "fact"
    source_filename: str = ""
    page_start: int = 0
    page_end: int = 0
    line_start: int = 0
    line_end: int = 0
    excerpt: str = ""


class LintIssue(BaseModel):
    """A single lint finding for a wiki page."""

    wiki_source: str
    citation_key: str = ""
    status: str = "valid"
    message: str = ""
