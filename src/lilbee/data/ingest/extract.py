"""Document extraction: kreuzberg config, OCR fallback, markdown/document chunking."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kreuzberg import ExtractionConfig, ExtractionResult

from lilbee.data.chunk import build_chunking_config, chunk_text
from lilbee.core.config import cfg
from lilbee.core.services import get_services
from lilbee.data.ingest.types import (
    _MARKDOWN_OUTPUT,
    _MIN_MEANINGFUL_CHARS,
    _PDF_CONTENT_TYPE,
    _TESSERACT_BACKEND,
    ChunkRecord,
    ExtractMode,
)
from lilbee.runtime.progress import DetailedProgressCallback, noop_callback
from lilbee.data.store import CHUNK_TYPE_RAW
from lilbee.vision import extract_pdf_vision

log = logging.getLogger(__name__)


def _has_meaningful_text(result: Any) -> bool:
    """Check if extraction produced meaningful text."""
    chunks = getattr(result, "chunks", None)
    if chunks:
        total = sum(len(c.content.strip()) for c in chunks)
        return total > _MIN_MEANINGFUL_CHARS
    return False


def content_type_to_mode(content_type: str) -> ExtractMode:
    """Map a content_type to the extraction mode."""
    return ExtractMode.PAGINATED if content_type == _PDF_CONTENT_TYPE else ExtractMode.MARKDOWN


def extraction_config(mode: ExtractMode) -> ExtractionConfig:
    """Build ExtractionConfig for the given extraction mode."""
    from kreuzberg import ExtractionConfig, OcrConfig, PageConfig

    chunking = build_chunking_config()
    pages = PageConfig(extract_pages=True, insert_page_markers=False)
    ocr = OcrConfig(backend=_TESSERACT_BACKEND)
    builders: dict[ExtractMode, Callable[[], ExtractionConfig]] = {
        ExtractMode.MARKDOWN: lambda: ExtractionConfig(
            chunking=chunking,
            output_format=_MARKDOWN_OUTPUT,
        ),
        ExtractMode.PAGINATED: lambda: ExtractionConfig(
            chunking=chunking,
            pages=pages,
        ),
        ExtractMode.PAGINATED_OCR: lambda: ExtractionConfig(
            chunking=chunking,
            pages=pages,
            ocr=ocr,
        ),
    }
    return builders[mode]()


@contextlib.contextmanager
def suppress_fd_stderr() -> Generator[None, None, None]:
    """Suppress stderr at the file-descriptor level.
    Catches subprocess output (e.g. Tesseract's "Detected N diacritics")
    that ``contextlib.redirect_stderr`` cannot intercept.
    """
    old_stderr = os.dup(2)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 2)
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
    finally:
        os.close(old_stderr)


async def _try_tesseract_ocr(
    path: Path, source_name: str, fallback: ExtractionResult
) -> ExtractionResult:
    """Attempt Tesseract OCR on a scanned PDF. Returns the OCR result or *fallback* on failure.

    Wraps extraction in ``asyncio.wait_for(cfg.tesseract_timeout)`` so a
    huge scanned document can't monopolize an ingest worker for many
    minutes (which the caller perceives as a UI lockup). The timeout is
    configurable via ``LILBEE_TESSERACT_TIMEOUT``; 0 disables the cap.
    """
    try:
        from kreuzberg import extract_file

        log.info("PDF text extraction empty, trying Tesseract OCR: %s", source_name)
        with suppress_fd_stderr():
            coro = extract_file(str(path), config=extraction_config(ExtractMode.PAGINATED_OCR))
            if cfg.tesseract_timeout > 0:
                return await asyncio.wait_for(coro, timeout=cfg.tesseract_timeout)
            return await coro
    except TimeoutError:
        log.warning(
            "Tesseract OCR exceeded %.0fs timeout on %s; skipping.",
            cfg.tesseract_timeout,
            source_name,
        )
        return fallback
    except Exception:
        log.debug("Tesseract OCR unavailable or failed for %s, skipping", source_name)
        return fallback


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


async def _vision_fallback(
    path: Path,
    source_name: str,
    content_type: str,
    on_progress: DetailedProgressCallback = noop_callback,
    *,
    quiet: bool = False,
) -> list[ChunkRecord]:
    """OCR a scanned PDF via the configured vision model, chunk, and embed.

    Uses ``cfg.vision_model`` unconditionally. The chat model is never
    loaded as a vision backend. If ``cfg.vision_model`` is empty this
    returns an empty list; callers should fall back to Tesseract via
    ``_handle_scanned_pdf_fallback``.
    """
    if not cfg.vision_model:
        return []
    try:
        page_texts = await asyncio.to_thread(
            extract_pdf_vision,
            path,
            cfg.vision_model,
            quiet=quiet,
            timeout=cfg.ocr_timeout,
            on_progress=on_progress,
        )
    except Exception:
        log.warning(
            "Vision OCR failed for %s using vision model %s.",
            source_name,
            cfg.vision_model,
            exc_info=True,
        )
        return []
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
            chunk_type=CHUNK_TYPE_RAW,
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


async def _handle_scanned_pdf_fallback(
    path: Path,
    source_name: str,
    content_type: str,
    result: ExtractionResult,
    *,
    quiet: bool,
    on_progress: DetailedProgressCallback,
) -> list[ChunkRecord] | ExtractionResult:
    """Handle scanned PDF fallback chain: Tesseract OCR then vision model.

    Returns chunk records if a fallback produced final results, or an
    updated ExtractionResult when Tesseract OCR succeeded (so the
    caller can proceed with normal chunking/embedding).

    When vision OCR is available (``_should_run_ocr()`` True) we go
    straight to it. Tesseract is only attempted when vision isn't an
    option at all. Running a huge scanned PDF through vision *and then*
    through Tesseract would double the wall-clock cost for no reason,
    and Tesseract on a 50+ MB document otherwise feels like a TUI
    lockup to the user.
    """
    use_ocr = _should_run_ocr()

    if use_ocr and cfg.vision_model:
        log.info(
            "Scanned PDF: using vision OCR for %s (model=%s)",
            source_name,
            cfg.vision_model,
        )
        return await _vision_fallback(path, source_name, content_type, on_progress, quiet=quiet)

    result = await _try_tesseract_ocr(path, source_name, result)

    if not _has_meaningful_text(result):
        log.warning(
            "Skipped %s: text extraction produced no usable text. "
            "For better results on scanned PDFs, configure a vision model "
            "via PUT /api/models/vision or set LILBEE_ENABLE_OCR=true.",
            source_name,
        )
        return []

    log.info(
        "Scanned PDF detected: extracted with Tesseract OCR: %s. "
        "For structured markdown output (tables, headings), "
        "configure a vision model via PUT /api/models/vision.",
        source_name,
    )
    return result


async def ingest_document(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[ChunkRecord]:
    """Extract and chunk a document, embed, return records.

    Vision OCR is controlled by ``cfg.enable_ocr`` (see ``_should_run_ocr``).
    """
    from kreuzberg import extract_file

    config = extraction_config(content_type_to_mode(content_type))
    result = await extract_file(str(path), config=config)

    if content_type == _PDF_CONTENT_TYPE and not _has_meaningful_text(result):
        fallback = await _handle_scanned_pdf_fallback(
            path,
            source_name,
            content_type,
            result,
            quiet=quiet,
            on_progress=on_progress,
        )
        if isinstance(fallback, list):
            return fallback
        # Tesseract OCR succeeded: use the updated ExtractionResult
        result = fallback

    if not result.chunks:
        return []

    texts = [chunk.content for chunk in result.chunks]
    vectors = await asyncio.to_thread(
        get_services().embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )

    return [
        ChunkRecord(
            source=source_name,
            content_type=content_type,
            chunk_type=CHUNK_TYPE_RAW,
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
) -> list[ChunkRecord]:
    """Chunk a markdown file with heading context prepended to each chunk.
    Each chunk gets the heading hierarchy path (e.g. "# Setup > ## Install")
    prepended for better retrieval context.
    """
    raw_text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
    if not raw_text.strip():
        return []

    texts = chunk_text(raw_text, mime_type="text/markdown", heading_context=True)
    if not texts:
        return []

    vectors = await asyncio.to_thread(
        get_services().embedder.embed_batch, texts, source=source_name, on_progress=on_progress
    )
    return [
        ChunkRecord(
            source=source_name,
            content_type="text",
            chunk_type=CHUNK_TYPE_RAW,
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
