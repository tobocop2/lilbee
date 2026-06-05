"""Document extraction: kreuzberg config, OCR fallback, markdown/document chunking."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kreuzberg import ExtractionConfig, ExtractionResult

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.data.chunk import build_chunking_config, chunk_text
from lilbee.data.ingest.types import (
    MARKDOWN_OUTPUT,
    MIN_MEANINGFUL_CHARS,
    PDF_CONTENT_TYPE,
    TESSERACT_BACKEND,
    ChunkRecord,
    ExtractMode,
)
from lilbee.data.store import ChunkType, PageTextRecord
from lilbee.runtime.cpu import cpu_quota
from lilbee.runtime.progress import (
    DetailedProgressCallback,
    EventType,
    ExtractEvent,
    noop_callback,
)

log = logging.getLogger(__name__)


def _has_meaningful_text(result: ExtractionResult) -> bool:
    """True when extraction yielded real text; content is primary, chunks the fallback."""
    if len(result.content.strip()) > MIN_MEANINGFUL_CHARS:
        return True
    return sum(len(c.content.strip()) for c in result.chunks or []) > MIN_MEANINGFUL_CHARS


def content_type_to_mode(content_type: str) -> ExtractMode:
    """Map a content_type to the extraction mode."""
    return ExtractMode.PAGINATED if content_type == PDF_CONTENT_TYPE else ExtractMode.MARKDOWN


def _page_text_record(source: str, page: int, text: str, content_type: str) -> PageTextRecord:
    """Build one per-page text row for the export dataset."""
    return PageTextRecord(source=source, page=page, text=text, content_type=content_type)


def extraction_config(mode: ExtractMode) -> ExtractionConfig:
    """Build ExtractionConfig for the given extraction mode."""
    from kreuzberg import ConcurrencyConfig, ExtractionConfig, OcrConfig, PageConfig

    chunking = build_chunking_config()
    pages = PageConfig(extract_pages=True, insert_page_markers=False)
    ocr = OcrConfig(backend=TESSERACT_BACKEND)
    # Bound kreuzberg's internal pool to the same CPU budget as the
    # pipeline semaphore so the two stop competing for cores.
    concurrency = ConcurrencyConfig(max_threads=cpu_quota())
    builders: dict[ExtractMode, Callable[[], ExtractionConfig]] = {
        ExtractMode.MARKDOWN: lambda: ExtractionConfig(
            chunking=chunking,
            output_format=MARKDOWN_OUTPUT,
            concurrency=concurrency,
        ),
        ExtractMode.PAGINATED: lambda: ExtractionConfig(
            chunking=chunking,
            pages=pages,
            concurrency=concurrency,
        ),
        ExtractMode.PAGINATED_OCR: lambda: ExtractionConfig(
            chunking=chunking,
            pages=pages,
            ocr=ocr,
            concurrency=concurrency,
        ),
    }
    return builders[mode]()


def _should_run_ocr() -> bool:
    """Decide whether to attempt vision-based OCR on scanned PDFs.

    Uses ``cfg.enable_ocr`` and ``cfg.vision_model``:
    True = force on (requires ``cfg.vision_model`` to be set for a real
    vision run; otherwise the caller falls back to Tesseract).
    False = force off.
    None = auto-detect: run vision OCR when ``cfg.vision_model`` is set.
    """
    if cfg.enable_ocr is True:
        return True
    if cfg.enable_ocr is False:
        return False
    return bool(cfg.vision_model)


def _record_page_texts(
    page_texts: Sequence[tuple[int, str]],
    source_name: str,
    content_type: str,
    page_texts_out: list[PageTextRecord] | None,
) -> None:
    """Append OCR page texts to the export accumulator when one is supplied."""
    if page_texts_out is None:
        return
    page_texts_out.extend(
        _page_text_record(source_name, page, text, content_type) for page, text in page_texts
    )


async def _vision_ocr_fallback(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    on_progress: DetailedProgressCallback,
    quiet: bool,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Vision OCR via the persistent worker pool, chunk + embed the pages.

    Pool routing is what amortises the multi-second vision-Llama load
    cost across PDFs. Tesseract has no shared model state and is run
    inline by ``_tesseract_ocr_fallback``; this helper is vision-only.
    """
    try:
        page_texts = await asyncio.to_thread(
            get_services().provider.pdf_ocr,
            path,
            backend="vision",
            model=cfg.vision_model,
            per_page_timeout_s=cfg.ocr_timeout,
            quiet=quiet,
            on_progress=on_progress,
        )
    except Exception:
        log.warning("OCR via vision backend failed for %s.", source_name, exc_info=True)
        return []
    _record_page_texts(page_texts, source_name, content_type, page_texts_out)
    return await chunk_and_embed_pages(page_texts, source_name, content_type, on_progress)


def _run_tesseract_sync(path: Path) -> Any:
    """Run kreuzberg Tesseract OCR with the worker's stderr redirected to /dev/null.

    Tesseract writes "Line cannot be recognized!!", "Image too small to
    scale!!", and "Detected N diacritics" directly to fd 2 from inside libc.
    Without the redirect those lines flood the TUI log file (1 000+ entries
    per scanned PDF) and can bleed into the TUI itself. We hold the
    suppression for just the duration of the extraction call so other
    threads' stderr writes still go through.
    """
    from kreuzberg import extract_file_sync

    from lilbee.providers.llama_cpp.log_dispatch import stderr_suppressed

    with stderr_suppressed():
        return extract_file_sync(str(path), config=extraction_config(ExtractMode.PAGINATED_OCR))


async def _tesseract_ocr_fallback(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    on_progress: DetailedProgressCallback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Tesseract OCR via ``asyncio.to_thread`` (no model load = no pool).

    ``cfg.tesseract_timeout`` caps the whole-document extract; 0 means
    unlimited. Failures (including timeout) log a warning and return an
    empty list so the caller can skip the file.
    """
    coro = asyncio.to_thread(_run_tesseract_sync, path)
    try:
        if cfg.tesseract_timeout > 0:
            result = await asyncio.wait_for(coro, timeout=cfg.tesseract_timeout)
        else:
            result = await coro
    except TimeoutError:
        log.warning(
            "Tesseract OCR exceeded %.0fs timeout on %s; skipping.",
            cfg.tesseract_timeout,
            source_name,
        )
        return []
    except Exception:
        log.warning("OCR via tesseract backend failed for %s.", source_name, exc_info=True)
        return []

    by_page: dict[int, list[str]] = {}
    for chunk in result.chunks or []:
        page = int(chunk.metadata.get("first_page") or 1)
        by_page.setdefault(page, []).append(chunk.content)
    page_texts = [(page, "\n".join(by_page[page])) for page in sorted(by_page)]
    _record_page_texts(page_texts, source_name, content_type, page_texts_out)
    return await chunk_and_embed_pages(page_texts, source_name, content_type, on_progress)


async def chunk_and_embed_pages(
    page_texts: Sequence[tuple[int, str]],
    source_name: str,
    content_type: str,
    on_progress: DetailedProgressCallback,
) -> list[ChunkRecord]:
    """Chunk per-page text and embed every chunk. Shared by OCR ingest and import."""
    if not page_texts:
        return []

    # Single OCR page rarely spans multiple topics; skip the semantic round-trip.
    all_chunks = [
        (page_num, chunk)
        for page_num, text in page_texts
        for chunk in chunk_text(text, use_semantic=False)
    ]
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
    """Append a normal extraction's page texts to the export accumulator.

    Paginated PDFs yield one row per ``result.pages`` entry; other documents
    have no page split, so the full ``result.content`` is recorded as page 0.
    """
    if page_texts_out is None:
        return
    if result.pages:
        page_texts_out.extend(
            _page_text_record(source_name, page["page_number"], page["content"], content_type)
            for page in result.pages
        )
    elif result.content.strip():
        page_texts_out.append(_page_text_record(source_name, 0, result.content, content_type))


async def _handle_scanned_pdf_fallback(
    path: Path,
    source_name: str,
    content_type: str,
    result: ExtractionResult,
    *,
    quiet: bool,
    on_progress: DetailedProgressCallback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Route a scanned PDF through the configured OCR backend.

    Vision OCR (pool-routed) runs when ``_should_run_ocr()`` is True
    and a vision model is configured; otherwise the file falls back to
    inline Tesseract. Both paths return chunk records; an empty list
    means OCR found no usable text and the caller should skip the
    file.
    """
    del result  # Both backends re-extract; the kreuzberg result is not reused.
    use_ocr = _should_run_ocr()
    if use_ocr and cfg.vision_model:
        log.info(
            "Scanned PDF: using vision OCR for %s (model=%s)",
            source_name,
            cfg.vision_model,
        )
        return await _vision_ocr_fallback(
            path,
            source_name,
            content_type,
            on_progress=on_progress,
            quiet=quiet,
            page_texts_out=page_texts_out,
        )

    log.info("Scanned PDF: falling back to Tesseract OCR for %s", source_name)
    chunks = await _tesseract_ocr_fallback(
        path,
        source_name,
        content_type,
        on_progress=on_progress,
        page_texts_out=page_texts_out,
    )
    if not chunks:
        log.warning(
            "Skipped %s: text extraction produced no usable text. "
            "For better results on scanned PDFs, configure a vision model "
            "via PUT /api/models/vision or set LILBEE_ENABLE_OCR=true.",
            source_name,
        )
    return chunks


async def ingest_document(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Extract and chunk a document, embed, return records.

    Vision OCR is controlled by ``cfg.enable_ocr`` (see ``_should_run_ocr``).
    When ``page_texts_out`` is given, per-page text is appended for export.
    """
    from kreuzberg import extract_file_sync

    config = extraction_config(content_type_to_mode(content_type))
    result = await asyncio.to_thread(extract_file_sync, str(path), config=config)

    if content_type == PDF_CONTENT_TYPE and not _has_meaningful_text(result):
        return await _handle_scanned_pdf_fallback(
            path,
            source_name,
            content_type,
            result,
            quiet=quiet,
            on_progress=on_progress,
            page_texts_out=page_texts_out,
        )

    if not result.chunks:
        return []

    _capture_result_page_texts(result, source_name, content_type, page_texts_out)

    # Fire one EXTRACT event per file so subscribers (chat /add, /sync,
    # CLI Rich progress) can show "extracted N pages" before the embed
    # phase starts; otherwise a 44MB PDF sits at file-level 0% for
    # minutes. get_page_count is the canonical PDF page count; for
    # non-paginated formats we fall back to the chunk count.
    page_count = result.get_page_count() or len(result.chunks)
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
            page_start=chunk.metadata.get("first_page") or 0,
            page_end=chunk.metadata.get("last_page") or 0,
            line_start=0,
            line_end=0,
            chunk=text,
            chunk_index=chunk.metadata.get("chunk_index", idx),
            vector=vec,
        )
        for idx, (chunk, text, vec) in enumerate(zip(result.chunks, texts, vectors, strict=True))
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

    texts = chunk_text(raw_text, mime_type="text/markdown", heading_context=True)
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
