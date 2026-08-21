"""Event type enums and Pydantic models for the progress protocol."""

from enum import StrEnum

from pydantic import BaseModel


class EventType(StrEnum):
    """Progress event types emitted during sync/ingest."""

    FILE_START = "file_start"
    FILE_DONE = "file_done"
    BATCH_PROGRESS = "batch_progress"
    DONE = "done"
    EMBED = "embed"
    EXTRACT = "extract"
    CRAWL_START = "crawl_start"
    CRAWL_PAGE = "crawl_page"
    CRAWL_PAGE_FAILED = "crawl_page_failed"
    CRAWL_DONE = "crawl_done"
    SETUP_START = "setup_start"
    SETUP_PROGRESS = "setup_progress"
    SETUP_DONE = "setup_done"
    WIKI_PHASE = "wiki_phase"
    WIKI_PAGE = "wiki_page"


class SseEvent(StrEnum):
    """SSE event names used in the HTTP streaming protocol."""

    TOKEN = "token"  # noqa: S105 -- SSE event name, not a credential
    REASONING = "reasoning"
    SOURCES = "sources"
    ERROR = "error"
    DONE = "done"
    PROGRESS = "progress"
    HEARTBEAT = "heartbeat"
    ALREADY_INGESTING = "already_ingesting"
    WARMING = "warming"
    WARM = "warm"
    COMPACTING = "compacting"
    COMPACTION = "compaction"
    MEMORY_EXTRACTED = "memory_extracted"
    GPU_STATS = "gpu_stats"


class SseErrorCode(StrEnum):
    """Stable ``code`` values on SSE error events for clients to branch on."""

    MODEL_TOO_LARGE = "model_too_large"
    MODEL_NOT_INSTALLED = "model_not_installed"
    INDEX_EMBEDDER_MISMATCH = "index_embedder_mismatch"


class FileStartEvent(BaseModel):
    """Emitted when a file begins ingestion."""

    file: str
    total_files: int
    current_file: int


class FileDoneEvent(BaseModel):
    """Emitted when a file finishes ingestion (success or error)."""

    file: str
    status: str
    chunks: int


class BatchStatus(StrEnum):
    """Status values for BatchProgressEvent.status."""

    INGESTED = "ingested"
    SKIPPED = "skipped"
    FAILED = "failed"
    RASTERIZING = "rasterizing"


class BatchProgressEvent(BaseModel):
    """Emitted after each file completes during batch ingestion."""

    file: str
    status: BatchStatus
    current: int
    total: int


class ExtractEvent(BaseModel):
    """Emitted with page-level extraction progress.

    OCR fires one event per page as xberg processes it, as a running count
    with ``total_pages == 0`` (the total is unknown mid-extraction). Extraction
    then fires once per file with ``page == total_pages`` so subscribers see
    "extracted N pages" before the embed phase ticks.
    """

    file: str
    page: int
    total_pages: int


class EmbedEvent(BaseModel):
    """Emitted per batch during embedding."""

    file: str
    chunk: int
    total_chunks: int


class CrawlStartEvent(BaseModel):
    """Emitted when a crawl operation begins."""

    url: str
    depth: int


# Sentinel used in CrawlPageEvent.total when the crawl's final page count is
# not yet known (BFS streaming, page N emitted before N+1 is discovered).
# Consumers (plugin, TUI, CLI) treat total <= 0 as indeterminate progress.
CRAWL_TOTAL_UNKNOWN = -1


class CrawlPageEvent(BaseModel):
    """Emitted per page during crawling."""

    url: str
    current: int
    total: int


class CrawlPageFailedEvent(BaseModel):
    """Emitted when a crawled page yields nothing to save, with the reason why."""

    url: str
    reason: str


class CrawlDoneEvent(BaseModel):
    """Emitted when a crawl operation completes."""

    pages_crawled: int
    files_written: int


class SyncDoneEvent(BaseModel):
    """Emitted when the sync operation completes."""

    added: int
    updated: int
    removed: int
    failed: int
    skipped: int = 0
    relocated: int = 0


class SetupStartEvent(BaseModel):
    """Emitted when a setup/bootstrap operation begins."""

    component: str
    size_estimate_bytes: int | None = None


class SetupProgressEvent(BaseModel):
    """Emitted periodically during a setup/bootstrap operation."""

    component: str
    downloaded_bytes: int
    total_bytes: int | None = None
    detail: str = ""


class SetupDoneEvent(BaseModel):
    """Emitted when a setup/bootstrap operation completes."""

    component: str
    success: bool
    error: str | None = None


class WikiPhase(StrEnum):
    """Stage of a wiki build or synthesis run."""

    EXTRACT = "extract"
    GENERATE = "generate"
    INDEX = "index"


class WikiPhaseEvent(BaseModel):
    """Emitted when a wiki run enters a new phase.

    ``total`` is the number of units the phase will process (sources for a
    build, clusters for synthesis), 0 where the phase has no unit count.
    """

    phase: WikiPhase
    total: int = 0


class WikiPageEvent(BaseModel):
    """Emitted after each source (build) or cluster (synthesis) is written."""

    label: str
    pages: int
    current: int
    total: int
