"""Document extraction: one kreuzberg pass that natively extracts text and OCRs
scanned pages/images through the registered backend; chunk + embed the result."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.data.chunk import build_chunking_config, chunk_text
from lilbee.data.ingest.types import (
    IMAGE_CONTENT_TYPE,
    MARKDOWN_OUTPUT,
    PDF_CONTENT_TYPE,
    ChunkRecord,
    ExtractMode,
    OcrBackendName,
)
from lilbee.data.ingest.vision_ocr_backend import backend_options_for, ocr_request
from lilbee.data.store import ChunkType, PageTextRecord
from lilbee.runtime.cpu import cpu_quota
from lilbee.runtime.progress import (
    DetailedProgressCallback,
    EventType,
    ExtractEvent,
    noop_callback,
)

if TYPE_CHECKING:
    from kreuzberg import ExtractionConfig, OcrConfig

    # extract_* return the pyo3 result (attribute access), not the public
    # ExtractionResult TypedDict (kreuzberg-7ih).
    from kreuzberg._kreuzberg import ExtractionResult

log = logging.getLogger(__name__)


def content_type_to_mode(content_type: str) -> ExtractMode:
    """Map a content_type to the extraction mode (paginated for PDFs and images)."""
    if content_type in (PDF_CONTENT_TYPE, IMAGE_CONTENT_TYPE):
        return ExtractMode.PAGINATED
    return ExtractMode.MARKDOWN


def _page_text_record(source: str, page: int, text: str, content_type: str) -> PageTextRecord:
    """Build one per-page text row for the export dataset."""
    return PageTextRecord(source=source, page=page, text=text, content_type=content_type)


_ocr_enable_override: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "lilbee_ocr_enable_override", default=None
)
_ocr_timeout_override: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "lilbee_ocr_timeout_override", default=None
)


def _effective_enable_ocr() -> bool | None:
    """``cfg.enable_ocr`` unless a per-request OCR override is active.

    The override is a ContextVar, not a global cfg mutation, so concurrent
    ingests on the shared HTTP daemon each see their own setting.
    """
    override = _ocr_enable_override.get()
    return cfg.enable_ocr if override is None else override


def _effective_ocr_timeout() -> float:
    """``cfg.ocr_timeout`` unless a per-request OCR timeout override is active."""
    override = _ocr_timeout_override.get()
    return cfg.ocr_timeout if override is None else override


@contextmanager
def ocr_override(
    enable_ocr: bool | None = None, ocr_timeout: float | None = None
) -> Generator[None, None, None]:
    """Scope per-request OCR settings without mutating the global cfg.

    A ``None`` argument leaves that setting at its cfg default. Each override is
    isolated to the entering context, so overlapping ingests never clobber one
    another's OCR config.
    """
    tokens: list[tuple[contextvars.ContextVar[Any], contextvars.Token[Any]]] = []
    try:
        if enable_ocr is not None:
            tokens.append((_ocr_enable_override, _ocr_enable_override.set(enable_ocr)))
        if ocr_timeout is not None:
            tokens.append((_ocr_timeout_override, _ocr_timeout_override.set(ocr_timeout)))
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _ocr_config(ocr_token: str | None) -> OcrConfig:
    """Pick the OCR backend for this extraction.

    Mirrors the prior fallback policy: OCR off when ``enable_ocr`` is False; lilbee's
    vision backend when a vision model is configured; otherwise kreuzberg's tesseract.
    kreuzberg auto-OCRs only the pages that lack a text layer.
    """
    from kreuzberg import OcrConfig

    if _effective_enable_ocr() is False:
        return OcrConfig(enabled=False)
    if cfg.vision_model:
        options = backend_options_for(ocr_token) if ocr_token else None
        return OcrConfig(backend=OcrBackendName.LILBEE_VISION, backend_options=options)
    return OcrConfig(backend=OcrBackendName.TESSERACT)


def extraction_config(mode: ExtractMode, *, ocr_token: str | None = None) -> ExtractionConfig:
    """Build ExtractionConfig for the given extraction mode."""
    from kreuzberg import ExtractionConfig, PageConfig

    chunking = build_chunking_config()
    ocr = _ocr_config(ocr_token)
    # Bound batch extraction to the CPU budget so kreuzberg and the pipeline
    # semaphore stop competing for cores.
    max_concurrent = cpu_quota()
    if mode is ExtractMode.PAGINATED:
        return ExtractionConfig(
            chunking=chunking,
            pages=PageConfig(extract_pages=True, insert_page_markers=False),
            ocr=ocr,
            max_concurrent_extractions=max_concurrent,
        )
    return ExtractionConfig(
        chunking=chunking,
        output_format=MARKDOWN_OUTPUT,
        ocr=ocr,
        max_concurrent_extractions=max_concurrent,
    )


def _chunk_pages(page_texts: Sequence[tuple[int, str]]) -> list[tuple[int, str]]:
    """Chunk each page's text. Semantic chunking is off: a single page rarely spans
    multiple topics, so the semantic round-trip is not worth it."""
    return [
        (page_num, chunk)
        for page_num, text in page_texts
        for chunk in chunk_text(text, use_semantic=False)
    ]


async def chunk_and_embed_pages(
    page_texts: Sequence[tuple[int, str]],
    source_name: str,
    content_type: str,
    on_progress: DetailedProgressCallback,
) -> list[ChunkRecord]:
    """Chunk per-page text and embed every chunk. Used by the dataset import path."""
    if not page_texts:
        return []

    # chunk_text runs kreuzberg's synchronous extractor; offload it so a long
    # document does not stall sibling files sharing this event loop.
    all_chunks = await asyncio.to_thread(_chunk_pages, page_texts)
    if not all_chunks:
        return []
    texts = [c for _, c in all_chunks]
    vectors = await asyncio.to_thread(
        get_services().embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )
    return [
        ChunkRecord(
            source=source_name,
            content_type=content_type,
            chunk_type=ChunkType.RAW,
            page_start=page_num,
            page_end=page_num,
            line_start=0,
            line_end=0,
            chunk=text,
            chunk_index=i,
            vector=vec,
        )
        for i, ((page_num, text), vec) in enumerate(zip(all_chunks, vectors, strict=True))
    ]


def _capture_result_page_texts(
    result: ExtractionResult,
    source_name: str,
    content_type: str,
    page_texts_out: list[PageTextRecord] | None,
) -> None:
    """Append an extraction's page texts to the export accumulator.

    Paginated documents yield one row per ``result.pages`` entry; others have no
    page split, so the full ``result.content`` is recorded as page 0.
    """
    if page_texts_out is None:
        return
    if result.pages:
        page_texts_out.extend(
            _page_text_record(source_name, page.page_number, page.content, content_type)
            for page in result.pages
        )
    elif result.content.strip():
        page_texts_out.append(_page_text_record(source_name, 0, result.content, content_type))


def _warn_empty_ocr(source_name: str, media: str) -> None:
    """Warn that extraction yielded no text and point to the vision-model remedy."""
    log.warning(
        "Skipped %s: text extraction produced no usable text. "
        "For better results on %s, configure a vision model "
        "via PUT /api/models/vision or set LILBEE_ENABLE_OCR=true.",
        source_name,
        media,
    )


async def ingest_document(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Extract, chunk, and embed a document in a single kreuzberg pass.

    kreuzberg extracts native text and, where a page has none, OCRs it through the
    registered backend (lilbee's vision model, or tesseract). Per-page OCR progress
    is streamed as a running count via ``ocr_request``. ``quiet`` is accepted for
    pipeline call compatibility.
    """
    del quiet
    from kreuzberg import extract_file

    page_seen = 0

    def _tick() -> None:
        nonlocal page_seen
        page_seen += 1
        on_progress(
            EventType.EXTRACT,
            ExtractEvent(file=source_name, page=page_seen, total_pages=0),
        )

    with ocr_request(on_page=_tick, timeout=_effective_ocr_timeout()) as token:
        config = extraction_config(content_type_to_mode(content_type), ocr_token=token)
        # Async keeps the OCR page loop off this event loop's thread.
        result = await extract_file(str(path), config=config)

    if not result.chunks:
        if content_type in (PDF_CONTENT_TYPE, IMAGE_CONTENT_TYPE):
            _warn_empty_ocr(source_name, "scanned documents")
        return []

    _capture_result_page_texts(result, source_name, content_type, page_texts_out)

    # One EXTRACT event per file so subscribers (chat /add, /sync, CLI Rich
    # progress) show "extracted N pages" before the embed phase; result.pages is
    # the canonical page list, falling back to the chunk count for non-paginated docs.
    page_count = len(result.pages or []) or len(result.chunks or [])
    on_progress(
        EventType.EXTRACT,
        ExtractEvent(file=source_name, page=page_count, total_pages=page_count),
    )

    texts = [chunk.content for chunk in result.chunks]
    vectors = await asyncio.to_thread(
        get_services().embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )
    return [
        ChunkRecord(
            source=source_name,
            content_type=content_type,
            chunk_type=ChunkType.RAW,
            page_start=chunk.metadata.first_page or 0,
            page_end=chunk.metadata.last_page or 0,
            line_start=0,
            line_end=0,
            chunk=text,
            chunk_index=chunk.metadata.chunk_index,
            vector=vec,
        )
        for chunk, text, vec in zip(result.chunks, texts, vectors, strict=True)
    ]


async def ingest_markdown(
    path: Path,
    source_name: str,
    on_progress: DetailedProgressCallback = noop_callback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Chunk a markdown file with heading context prepended to each chunk.
    Each chunk gets the heading hierarchy path (e.g. "# Setup > ## Install")
    prepended for better retrieval context. When ``page_texts_out`` is given,
    the full text is appended as page 0 for export.
    """
    raw_text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
    if not raw_text.strip():
        return []

    # chunk_text runs kreuzberg's synchronous extractor; offload it so a large
    # markdown doc does not stall sibling files sharing this event loop.
    texts = await asyncio.to_thread(
        chunk_text, raw_text, mime_type="text/markdown", heading_context=True
    )
    if not texts:
        return []

    if page_texts_out is not None:
        page_texts_out.append(_page_text_record(source_name, 0, raw_text, "text"))

    vectors = await asyncio.to_thread(
        get_services().embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )
    return [
        ChunkRecord(
            source=source_name,
            content_type="text",
            chunk_type=ChunkType.RAW,
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            chunk=t,
            chunk_index=idx,
            vector=vec,
        )
        for idx, (t, vec) in enumerate(zip(texts, vectors, strict=True))
    ]
