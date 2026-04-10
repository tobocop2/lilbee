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


class StatusSourceInfo(BaseModel):
    """A single indexed source in a status response."""

    filename: str
    file_hash: str
    chunk_count: int
    ingested_at: str


class StatusConfigInfo(BaseModel):
    """Configuration section of a status response."""

    documents_dir: str
    data_dir: str
    chat_model: str
    embedding_model: str
    vision_model: str | None = None


class StatusResponse(BaseModel):
    """Response for GET /api/status."""

    command: str = "status"
    config: StatusConfigInfo
    sources: list[StatusSourceInfo]
    total_chunks: int


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


class ConfigUpdateResponse(BaseModel):
    """Response for PATCH /api/config."""

    updated: list[str]
    reindex_required: bool


class CrawlRequest(BaseModel):
    """Request body for /api/crawl."""

    url: str
    depth: int = Field(default=0, le=10)
    max_pages: int = Field(default=50, le=1000)


class DocumentInfo(BaseModel):
    """A single indexed document in a list response."""

    filename: str
    chunk_count: int = 0
    ingested_at: str = ""


class DocumentListResponse(BaseModel):
    """Response for GET /api/documents."""

    documents: list[DocumentInfo]
    total: int
    limit: int
    offset: int


class DocumentRemoveResponse(BaseModel):
    """Response for POST /api/documents/remove."""

    removed: list[str]
    not_found: list[str]


class ConfigResponse(BaseModel):
    """Response for GET /api/config."""

    model_config = {"extra": "allow"}


class ModelsShowResponse(BaseModel):
    """Response for POST /api/models/show."""

    model_config = {"extra": "allow"}


class CatalogEntryResponse(BaseModel):
    """A single model in the catalog browser."""

    name: str
    display_name: str
    size_gb: float
    min_ram_gb: float
    description: str
    quality_tier: str
    installed: bool
    source: str


class ModelsCatalogResponse(BaseModel):
    """Response for GET /api/models/catalog."""

    total: int
    limit: int
    offset: int
    models: list[CatalogEntryResponse]


class InstalledModelEntry(BaseModel):
    """A single installed model."""

    name: str
    source: str


class ModelsInstalledResponse(BaseModel):
    """Response for GET /api/models/installed."""

    models: list[InstalledModelEntry]


class ModelsDeleteResponse(BaseModel):
    """Response for DELETE /api/models/{model}."""

    deleted: bool
    model: str
    freed_gb: float


class ExternalModelsResponse(BaseModel):
    """Response for GET /api/models/external."""

    models: list[str]
    error: str | None = None


class SyncSummary(BaseModel):
    """Embedded sync result within an add-files response."""

    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    unchanged: int = 0
    failed: list[str] = []


class AddSummary(BaseModel):
    """Summary returned by the add-files handler."""

    copied: list[str]
    skipped: list[str]
    errors: list[str]
    sync: SyncSummary | None = None


class WikiPageSummary(BaseModel):
    """Summary of a wiki page for list endpoints."""

    slug: str
    title: str = ""
    page_type: str = "unknown"
    source_count: int = 0
    created_at: str = ""


class WikiCitationRecord(BaseModel):
    """A citation record from the store, used in reverse lookup responses."""

    wiki_source: str = ""
    wiki_chunk_index: int = 0
    citation_key: str = ""
    claim_type: str = "fact"
    source_filename: str = ""
    source_hash: str = ""
    page_start: int = 0
    page_end: int = 0
    line_start: int = 0
    line_end: int = 0
    excerpt: str = ""
    created_at: str = ""


class WikiPageDetail(BaseModel):
    """Full content of a single wiki page."""

    slug: str
    title: str = ""
    content: str = ""


class WikiCitationsResult(BaseModel):
    """Citations attached to a single wiki page."""

    slug: str
    citations: list[WikiCitationRecord] = []


class WikiLintIssueItem(BaseModel):
    """A single lint finding on a wiki page."""

    wiki_source: str = ""
    issue_type: str = ""
    severity: str = ""
    message: str = ""


class WikiLintResult(BaseModel):
    """Result of a full wiki lint run."""

    issues: list[WikiLintIssueItem] = []
    errors: int = 0
    warnings: int = 0


class WikiPruneRecordResponse(BaseModel):
    """A single pruning action."""

    wiki_source: str
    action: str
    reason: str


class WikiPruneResult(BaseModel):
    """Result of wiki pruning."""

    records: list[WikiPruneRecordResponse] = []
    archived: int = 0
    flagged: int = 0
