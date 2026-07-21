"""Document extraction: one xberg pass that natively extracts text and OCRs
scanned pages/images through the registered backend; chunk + embed the result."""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.app.services import get_services
from lilbee.core.config import active_config
from lilbee.data.chunk import build_chunking_config, chunk_text
from lilbee.data.ingest.offload import to_ingest_thread
from lilbee.data.ingest.title import derive_title, source_meta_from_extraction
from lilbee.data.ingest.trace import ExtractionTrace, trace_extraction, trace_log
from lilbee.data.ingest.types import (
    IMAGE_CONTENT_TYPE,
    MARKDOWN_OUTPUT,
    PDF_CONTENT_TYPE,
    ChunkRecord,
    ExtractMode,
    OcrBackendName,
)
from lilbee.data.ingest.vision_ocr_backend import backend_options_for, ocr_request
from lilbee.data.store import ChunkType, PageTextRecord, SourceMeta
from lilbee.runtime.progress import (
    DetailedProgressCallback,
    EventType,
    ExtractEvent,
    noop_callback,
)

if TYPE_CHECKING:
    from xberg import ExtractedDocument, ExtractionConfig, OcrConfig, PdfConfig, Table

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
    return active_config().enable_ocr if override is None else override


def _effective_ocr_timeout() -> float:
    """``cfg.ocr_timeout`` unless a per-request OCR timeout override is active."""
    override = _ocr_timeout_override.get()
    return active_config().ocr_timeout if override is None else override


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
    vision backend when a vision model is configured; otherwise xberg's tesseract.
    xberg auto-OCRs only the pages that lack a text layer.
    """
    from xberg import OcrConfig

    config = active_config()
    if _effective_enable_ocr() is False:
        return OcrConfig(enabled=False)
    if config.vision_model:
        options = backend_options_for(ocr_token) if ocr_token else None
        return OcrConfig(
            backend=OcrBackendName.LILBEE_VISION,
            backend_options=options,
        )
    # xberg requires a non-empty language list (4.x defaulted to English;
    # 5.x errors on an empty one). cfg.ocr_language is validated non-empty.
    return OcrConfig(backend=OcrBackendName.TESSERACT, language=list(config.ocr_language))


def _ocr_force_requested() -> bool:
    """Whether LILBEE_OCR_FORCE opts every page into vision OCR.

    Born-digital PDFs carry a clean text layer that xberg extracts natively and,
    since rc25, short-circuits past the OCR backend entirely -- so a re-OCR run
    never touches the vision model. This lever sets ExtractionConfig.force_ocr,
    which overrides that short-circuit and OCRs every page. A targeted-reingest
    lever, not a default: normal runs keep native-first extraction."""
    import os

    return os.environ.get("LILBEE_OCR_FORCE", "").strip().lower() in {"1", "true", "yes"}


# Fraction of page height treated as the running header/footer band when
# layout detection is on. Conservative: strips only the outermost 5%.
_TOP_MARGIN_FRACTION = 0.05
_BOTTOM_MARGIN_FRACTION = 0.05


def _pdf_options() -> PdfConfig | None:
    """PdfConfig for the enabled opt-in ingest features, or None when all are off.

    Table extraction turns on structure recognition; layout detection orders
    text by detected reading order and strips the header/footer margin bands.
    """
    config = active_config()
    if not (config.table_extraction or config.layout_detection):
        return None
    from xberg import PdfConfig

    kwargs: dict[str, Any] = {}
    if config.table_extraction:
        kwargs["extract_tables"] = True
    if config.layout_detection:
        kwargs.update(
            reading_order=True,
            top_margin_fraction=_TOP_MARGIN_FRACTION,
            bottom_margin_fraction=_BOTTOM_MARGIN_FRACTION,
        )
    return PdfConfig(**kwargs)


def _layout_options() -> dict[str, Any]:
    """ExtractionConfig kwargs enabling layout detection, empty when off."""
    if not active_config().layout_detection:
        return {}
    from xberg import LayoutDetectionConfig

    return {"layout": LayoutDetectionConfig(), "use_layout_for_markdown": True}


def extraction_config(mode: ExtractMode, *, ocr_token: str | None = None) -> ExtractionConfig:
    """Build ExtractionConfig for the given extraction mode."""
    from xberg import ExtractionConfig, PageConfig

    # Files are extracted one per call, so xberg parallelizes OCR across the
    # pages of each document internally; cross-file concurrency is the pipeline's
    # semaphore. (max_concurrent_extractions only bounds multi-file batch calls,
    # which lilbee never makes, so it is intentionally not set here.)
    chunking = build_chunking_config()
    ocr = _ocr_config(ocr_token)
    # force_ocr defeats xberg's text-layer short-circuit so every page reaches the
    # vision backend; scoped to the vision path, since it is a GPU re-OCR lever.
    force_ocr = _ocr_force_requested() and ocr.backend == OcrBackendName.LILBEE_VISION
    if mode is ExtractMode.PAGINATED:
        return ExtractionConfig(
            chunking=chunking,
            pages=PageConfig(extract_pages=True, insert_page_markers=False),
            ocr=ocr,
            force_ocr=force_ocr,
            pdf_options=_pdf_options(),
            **_layout_options(),
        )
    return ExtractionConfig(
        chunking=chunking,
        output_format=MARKDOWN_OUTPUT,
        ocr=ocr,
        force_ocr=force_ocr,
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

    # chunk_text runs xberg's synchronous extractor; offload it so a long
    # document does not stall sibling files sharing this event loop.
    all_chunks = await to_ingest_thread(_chunk_pages, page_texts)
    if not all_chunks:
        return []
    texts = [c for _, c in all_chunks]
    vectors = await to_ingest_thread(
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
    doc: ExtractedDocument,
    source_name: str,
    content_type: str,
    page_texts_out: list[PageTextRecord] | None,
) -> None:
    """Append an extraction's page texts to the export accumulator.

    Paginated documents yield one row per ``doc.pages`` entry; others have no
    page split, so the full ``doc.content`` is recorded as page 0.
    """
    if page_texts_out is None:
        return
    if doc.pages:
        page_texts_out.extend(
            _page_text_record(source_name, page.page_number, page.content, content_type)
            for page in doc.pages
        )
    elif doc.content.strip():
        page_texts_out.append(_page_text_record(source_name, 0, doc.content, content_type))


def _document_tables(doc: ExtractedDocument) -> list[Table]:
    """The result tables to index as dedicated chunks, when table extraction is on.

    Each table becomes its own chunk carrying xberg's markdown serialization and
    page metadata. The table's flattened text also stays inside the content
    chunks: stripping it there would tear holes in reading-order prose and the
    page-text export, so both are indexed deliberately. The dedicated table
    chunk adds the structured serialization for targeted retrieval.
    """
    if not active_config().table_extraction:
        return []
    return [t for t in (doc.tables or []) if t.markdown and t.markdown.strip()]


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
) -> tuple[list[ChunkRecord], SourceMeta]:
    """Extract, chunk, and embed a document in a single xberg pass.

    xberg extracts native text and, where a page has none, OCRs it through the
    registered backend (lilbee's vision model, or tesseract). Per-page OCR progress
    is streamed as a running count via ``ocr_request``. ``quiet`` is accepted for
    pipeline call compatibility.

    The returned metadata carries the document's title/authors/date where xberg
    reports them, and the filename stem otherwise. Images are always the latter:
    the image extractor files EXIF under ``metadata.format.image.exif`` and never
    populates ``metadata.title``.
    """
    del quiet
    from lilbee.data.xberg_extract import aextract_document

    page_seen = 0

    def _tick() -> None:
        nonlocal page_seen
        page_seen += 1
        on_progress(
            EventType.EXTRACT,
            ExtractEvent(file=source_name, page=page_seen, total_pages=0),
        )

    trace_log.debug("extract-start source=%r type=%s", source_name, content_type)
    started = time.perf_counter()
    with ocr_request(on_page=_tick, timeout=_effective_ocr_timeout()) as token:
        config = extraction_config(content_type_to_mode(content_type), ocr_token=token)
        # xberg's extract is async; awaiting it keeps the OCR page loop off this thread.
        doc = await aextract_document(path.read_bytes(), filename=path.name, config=config)
    elapsed = time.perf_counter() - started

    # Captured before the empty-result return: a scan's PDF metadata (title,
    # authors) survives even when the document yields no text.
    meta = source_meta_from_extraction(doc.metadata, source_name)

    # One trace line per xberg extraction (filename, timing, counts, OCR pages),
    # plus a dedicated vision line for scanned files -- the diagnostics an xberg
    # author needs. Emitted for empty results too: a slow file that yields nothing
    # is exactly the case worth surfacing.
    trace_extraction(
        ExtractionTrace(
            source=source_name,
            content_type=content_type,
            elapsed_s=elapsed,
            page_count=len(doc.pages or []) or len(doc.chunks or []),
            chunk_count=len(doc.chunks or []),
            ocr_pages=page_seen,
            vision_configured=bool(active_config().vision_model),
        )
    )

    tables = _document_tables(doc)
    if not doc.chunks and not tables:
        if content_type in (PDF_CONTENT_TYPE, IMAGE_CONTENT_TYPE):
            _warn_empty_ocr(source_name, "scanned documents")
        return [], meta

    _capture_result_page_texts(doc, source_name, content_type, page_texts_out)

    # One EXTRACT event per file so subscribers (chat /add, /sync, CLI Rich
    # progress) show "extracted N pages" before the embed phase; doc.pages is
    # the canonical page list, falling back to the chunk count for non-paginated docs.
    page_count = len(doc.pages or []) or len(doc.chunks or [])
    on_progress(
        EventType.EXTRACT,
        ExtractEvent(file=source_name, page=page_count, total_pages=page_count),
    )

    # Content chunks and table serializations share one embed batch; the vector
    # list is split back apart below by position.
    texts = [chunk.content for chunk in doc.chunks or []]
    table_texts = [table.markdown for table in tables]
    vectors = await to_ingest_thread(
        get_services().embedder.embed_batch,
        texts + table_texts,
        source=source_name,
        on_progress=on_progress,
    )
    records = [
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
        for chunk, text, vec in zip(doc.chunks or [], texts, vectors[: len(texts)], strict=True)
    ]
    # Table chunk indices continue after the content chunks so a source's
    # (source, chunk_index) pairs stay unique.
    records.extend(
        ChunkRecord(
            source=source_name,
            content_type=content_type,
            chunk_type=ChunkType.TABLE,
            page_start=table.page_number,
            page_end=table.page_number,
            line_start=0,
            line_end=0,
            chunk=text,
            chunk_index=len(texts) + i,
            vector=vec,
        )
        for i, (table, text, vec) in enumerate(
            zip(tables, table_texts, vectors[len(texts) :], strict=True)
        )
    )
    return records, meta


def _markdown_h1(text: str) -> str | None:
    """The document's leading ``# Heading``, the best title a note carries.

    Only a top-level ATX heading counts; ``##`` and deeper are sections, not the
    document title. None when the note opens without one.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
        return None
    return None


async def ingest_markdown(
    path: Path,
    source_name: str,
    on_progress: DetailedProgressCallback = noop_callback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> tuple[list[ChunkRecord], SourceMeta]:
    """Chunk a markdown file with heading context prepended to each chunk.

    Each chunk gets the heading hierarchy path (e.g. "# Setup > ## Install")
    prepended for better retrieval context. When ``page_texts_out`` is given,
    the full text is appended as page 0 for export. The returned metadata's
    title is the note's leading ``# Heading`` when it has one, else the stem.
    """
    raw_text = await to_ingest_thread(path.read_text, encoding="utf-8", errors="replace")
    meta = SourceMeta(title=derive_title(source_name, _markdown_h1(raw_text)))
    if not raw_text.strip():
        return [], meta

    # chunk_text runs xberg's synchronous extractor; offload it so a large
    # markdown doc does not stall sibling files sharing this event loop.
    texts = await to_ingest_thread(
        chunk_text, raw_text, mime_type="text/markdown", heading_context=True
    )
    if not texts:
        return [], meta

    if page_texts_out is not None:
        page_texts_out.append(_page_text_record(source_name, 0, raw_text, "text"))

    vectors = await to_ingest_thread(
        get_services().embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )
    records = [
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
    return records, meta
