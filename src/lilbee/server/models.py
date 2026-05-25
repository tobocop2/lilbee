"""Request and response models for the lilbee HTTP API.

Typed pydantic models so Litestar's OpenAPI schema has field-level detail.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from lilbee.catalog.types import ModelCompat, ModelSource, ModelTask
from lilbee.data.store import SearchScope
from lilbee.runtime.hardware import FitLevel, SizeVariantInfo

_VALID_CHUNK_TYPES = frozenset({SearchScope.RAW.value, SearchScope.WIKI.value})


def _validate_chunk_type(value: str | None) -> str | None:
    """Reject unknown ``chunk_type`` values at the HTTP boundary.

    Matches the CLI/MCP behaviour: only ``"raw"`` or ``"wiki"`` filter the
    pool; everything else (including ``None`` and the UI-side ``"both"``)
    means no filter.
    """
    if value is None or value == SearchScope.BOTH.value:
        return None
    if value not in _VALID_CHUNK_TYPES:
        raise ValueError(
            f"chunk_type must be one of 'raw', 'wiki', 'both', or omitted; got {value!r}"
        )
    return value


class AskRequest(BaseModel):
    """Request body for /api/ask."""

    question: str
    top_k: int = Field(default=0, le=100)
    options: dict[str, Any] | None = None
    chunk_type: str | None = None

    @field_validator("chunk_type")
    @classmethod
    def _check_chunk_type(cls, v: str | None) -> str | None:
        return _validate_chunk_type(v)


class ChatRequest(BaseModel):
    """Request body for /api/chat."""

    question: str
    history: list[ChatMessage] = []
    top_k: int = Field(default=0, le=100)
    options: dict[str, Any] | None = None
    chunk_type: str | None = None

    @field_validator("chunk_type")
    @classmethod
    def _check_chunk_type(cls, v: str | None) -> str | None:
        return _validate_chunk_type(v)


class SyncRequest(BaseModel):
    """Request body for /api/sync.

    ``force_rebuild`` triggers a full drop-and-reingest equivalent to ``lilbee rebuild``.
    Use it to recover from an embedding-model switch (when the store refuses search
    or ingest because ``cfg.embedding_model`` no longer matches the persisted vectors).
    ``retry_skipped`` is the lighter recovery: it clears the markers for files that
    failed a previous sync (Tesseract timeout, decode failure, no usable text) so this
    sync attempts them again, without dropping the existing store. The default is an
    incremental sync.
    """

    enable_ocr: bool | None = None
    force_rebuild: bool = False
    retry_skipped: bool = False


class AddRequest(BaseModel):
    """Request body for /api/add."""

    paths: list[str]
    force: bool = False
    enable_ocr: bool | None = None
    ocr_timeout: float | None = None


class SetModelRequest(BaseModel):
    """Request body for /api/models/chat."""

    model: str


class SourceContentResponse(BaseModel):
    """JSON body for ``GET /api/source`` (``raw=0``); empty ``markdown`` for binary types."""

    markdown: str
    content_type: str
    title: str | None = None


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
    # Vault-relative path when ``cfg.vault_base`` is set and the source file
    # lives inside the vault. Absent when the server is running headless or
    # the source isn't resolvable as a vault file. Clients use this to open
    # the source in a native editor instead of fetching ``/api/source``.
    vault_path: str | None = None


class StatusSourceInfo(BaseModel):
    """A single indexed source in a status response."""

    filename: str
    file_hash: str
    chunk_count: int
    ingested_at: str


class StatusConfigInfo(BaseModel):
    """Configuration section of a status response.

    Exposes all four role-bound model fields so plugins/TUI can show
    what's active per role without a second round trip.
    """

    documents_dir: str
    data_dir: str
    chat_model: str
    embedding_model: str
    vision_model: str = ""
    reranker_model: str = ""
    enable_ocr: bool | None = None


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
    """Response for PUT /api/models/{chat|embedding|vision|reranker}.

    ``reindex_required`` is ``True`` only when the new embedding model differs from
    the model that built the persisted vector store. The chat, vision, and reranker
    handlers always return ``False`` because their changes do not invalidate stored
    vectors. Mirrors the ``reindex_required`` flag on ``ConfigUpdateResponse``.
    """

    model: str
    reindex_required: bool = False


class ConfigUpdateResponse(BaseModel):
    """Response for PATCH /api/config."""

    updated: list[str]
    reindex_required: bool


class CrawlRequest(BaseModel):
    """Request body for /api/crawl.

    depth: null / omitted = whole-site unbounded recursion. 0 = single URL
    only. Positive int = max depth. max_pages: null / omitted = no cap.
    Positive int = explicit page cap.
    """

    url: str
    depth: int | None = Field(default=None, ge=0)
    max_pages: int | None = Field(default=None, ge=1)


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
    has_more: bool = False


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
    """A single model in the catalog browser.

    ``fit`` and ``size_variants`` carry server-computed hardware-fit
    data so clients (TUI, plugin) can render fit chips and size strips
    without probing local memory themselves. ``fit`` is ``None`` when
    the row's footprint cannot be assessed against host memory (e.g.
    a future cloud-only entry whose weights live off-host).
    """

    hf_repo: str
    gguf_filename: str
    task: ModelTask
    display_name: str
    param_count: str
    size_gb: float
    min_ram_gb: float
    description: str
    quality_tier: str
    featured: bool
    downloads: int
    installed: bool
    source: ModelSource
    fit: FitLevel | None = None
    size_variants: list[SizeVariantInfo] = []
    architecture: str = ""
    compat: ModelCompat = ModelCompat.UNKNOWN


class ModelsCatalogResponse(BaseModel):
    """Response for GET /api/models/catalog."""

    total: int
    limit: int
    offset: int
    models: list[CatalogEntryResponse]
    has_more: bool = False


class InstalledModelEntry(BaseModel):
    """A single installed model."""

    name: str
    source: ModelSource


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
    skipped: list[str] = []
    truncated: int = 0


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


class WikiBuildResult(BaseModel):
    """Result of a full wiki build/update."""

    paths: list[str] = []
    entities: int = 0
    count: int = 0


class WikiStatusResult(BaseModel):
    """Wiki layer status counters."""

    wiki_enabled: bool
    summaries: int = 0
    drafts: int = 0
    pages: int = 0
    lint_errors: int = 0
    lint_warnings: int = 0


class WikiSynthesizeResult(BaseModel):
    """Result of generating synthesis pages for cross-source concept clusters."""

    paths: list[str] = []
    count: int = 0


class DraftInfoResponse(BaseModel):
    """Metadata about a single wiki draft, mirroring ``DraftInfo.to_dict()``.

    ``pending_kind`` distinguishes drift drafts (``None``) from
    batched-generation markers (``"parse"``, ``"collision"``).
    """

    slug: str
    path: str
    drift_ratio: float | None = None
    faithfulness_score: float | None = None
    bad_title: bool = False
    published_path: str | None = None
    published_exists: bool = False
    mtime: float = 0.0
    pending_kind: str | None = None


class WikiDraftDiffResponse(BaseModel):
    """Unified diff of a draft against its published counterpart."""

    slug: str
    diff: str


class WikiDraftAcceptResponse(BaseModel):
    """Outcome of accepting a draft: where it landed and how many chunks reindexed.

    ``slug`` is the slug where the content was published.
    ``requested_slug`` is the slug the client asked to accept. The two
    differ for PENDING-COLLISION drafts, where the request slug carries
    a ``-collision-<hash>`` suffix that is stripped on publish.
    """

    slug: str
    requested_slug: str
    moved_to: str
    reindexed_chunks: int


class WikiDraftRejectResponse(BaseModel):
    """Outcome of rejecting a draft."""

    slug: str
