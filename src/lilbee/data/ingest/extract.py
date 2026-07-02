"""Document extraction: kreuzberg config, OCR fallback, markdown/document chunking."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageSequence

if TYPE_CHECKING:
    from kreuzberg import ExtractionConfig, ExtractionResult

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.data.chunk import build_chunking_config, chunk_text
from lilbee.data.ingest.discovery import file_hash
from lilbee.data.ingest.ocr_cache import load_ocr_pages, ocr_cache_key, store_ocr_pages
from lilbee.data.ingest.types import (
    IMAGE_CONTENT_TYPE,
    MARKDOWN_OUTPUT,
    MIN_MEANINGFUL_CHARS,
    PDF_CONTENT_TYPE,
    TESSERACT_BACKEND,
    ChunkRecord,
    ExtractMode,
)
from lilbee.data.store import ChunkType, PageTextRecord
from lilbee.runtime.cancellation import TaskCancelledError
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


_ocr_enable_override: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "lilbee_ocr_enable_override", default=None
)
_ocr_timeout_override: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "lilbee_ocr_timeout_override", default=None
)


def _effective_enable_ocr() -> bool | None:
    """``cfg.enable_ocr`` unless a per-request OCR override is active.

    The override is a ContextVar, not a global cfg mutation, so concurrent
    ingests on the shared HTTP daemon each see their own setting. The override
    also propagates into ``asyncio.to_thread`` workers (``to_thread`` copies the
    calling task's context), which is how the timeout reaches the image-OCR call.
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
    isolated to the entering context (and any ``to_thread`` work it spawns), so
    overlapping ingests never clobber one another's OCR config.
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


def _should_run_ocr() -> bool:
    """Decide whether to attempt vision-based OCR on scanned PDFs.

    Uses the effective OCR setting (``cfg.enable_ocr`` or a per-request
    override) and ``cfg.vision_model``:
    True = force on (requires ``cfg.vision_model`` to be set for a real
    vision run; otherwise the caller falls back to Tesseract).
    False = force off.
    None = auto-detect: run vision OCR when ``cfg.vision_model`` is set.
    """
    enable_ocr = _effective_enable_ocr()
    if enable_ocr is True:
        return True
    if enable_ocr is False:
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


async def _vision_ocr_cached(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    ocr_fn: Callable[[], Awaitable[list[tuple[int, str]]]],
    on_progress: DetailedProgressCallback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Cache-wrapped vision OCR: reuse stored pages, else run *ocr_fn*, then chunk + embed.

    Pool routing amortises the multi-second vision-Llama load across files.
    *ocr_fn* returns the OCR'd pages as ``(page_number, text)`` tuples (a PDF page
    loop or a single image). Output is cached by file content + model, so a retry
    after a downstream failure (chunk/embed/store) reuses the pages, not re-OCR-ing.
    """
    # The per-page timeout bounds completeness: a page that exhausts the budget
    # yields empty text. Key on it so raising the timeout re-OCRs the file rather
    # than serving the earlier, partially-empty cached result for the same content.
    key = ocr_cache_key(
        file_hash(path),
        backend="vision",
        model=cfg.vision_model,
        extra=f"{cfg.vision_ocr_max_tokens}:{_effective_ocr_timeout()}",
    )
    cached = load_ocr_pages(key)
    if cached is not None:
        _record_page_texts(cached, source_name, content_type, page_texts_out)
        return await chunk_and_embed_pages(cached, source_name, content_type, on_progress)
    try:
        pages = await ocr_fn()
    except (asyncio.CancelledError, TaskCancelledError):
        # A user cancel (SIGINT / TUI cancel) raised cooperatively through the
        # per-page on_progress callback must abort the file, not be logged as an
        # OCR failure and swallowed.
        raise
    except Exception:
        # A vision-backend failure (dead replica, exhausted failover, transport
        # error) is a per-file ingest FAILURE, not a "document had no text".
        # Returning [] would classify the file as empty and skip-mark it under
        # its current hash, silently dropping it from search until
        # retry_skipped; raising routes it through the pipeline's failed path
        # with the real reason.
        log.warning("OCR via vision backend failed for %s.", source_name, exc_info=True)
        raise
    store_ocr_pages(key, pages)
    _record_page_texts(pages, source_name, content_type, page_texts_out)
    return await chunk_and_embed_pages(pages, source_name, content_type, on_progress)


async def _vision_ocr_fallback(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    on_progress: DetailedProgressCallback,
    quiet: bool,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Vision OCR a scanned PDF: rasterize + OCR each page through the worker pool."""

    async def _ocr() -> list[tuple[int, str]]:
        pages = await asyncio.to_thread(
            get_services().provider.pdf_ocr,
            path,
            backend="vision",
            model=cfg.vision_model,
            per_page_timeout_s=_effective_ocr_timeout(),
            quiet=quiet,
            on_progress=on_progress,
        )
        return [(p.page, p.text) for p in pages]

    return await _vision_ocr_cached(
        path,
        source_name,
        content_type,
        ocr_fn=_ocr,
        on_progress=on_progress,
        page_texts_out=page_texts_out,
    )


async def _vision_image_ocr(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    on_progress: DetailedProgressCallback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """Vision OCR an image through the worker pool, one page per frame.

    A multi-frame TIFF/GIF yields one OCR page per frame instead of silently
    dropping every frame after the first.
    """

    async def _ocr() -> list[tuple[int, str]]:
        page_pngs = await asyncio.to_thread(_image_page_pngs, path)
        pages: list[tuple[int, str]] = []
        for page_num, png in enumerate(page_pngs, start=1):
            text = await asyncio.to_thread(_ocr_image_png, png)
            pages.append((page_num, text))
        return pages

    return await _vision_ocr_cached(
        path,
        source_name,
        content_type,
        ocr_fn=_ocr,
        on_progress=on_progress,
        page_texts_out=page_texts_out,
    )


def _ocr_image_png(png: bytes) -> str:
    """OCR one rendered image page through the vision server."""
    return get_services().provider.vision_ocr(
        png, cfg.vision_model, timeout=_effective_ocr_timeout()
    )


def _image_page_pngs(path: Path) -> list[bytes]:
    """Each frame of a (possibly multi-frame) image, re-encoded as PNG.

    A single-frame image yields one entry; a multipage TIFF/GIF yields one per
    frame so every page reaches the projector instead of just the first.
    """
    pages: list[bytes] = []
    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            buf = BytesIO()
            frame.convert("RGB").save(buf, format="PNG")
            pages.append(buf.getvalue())
    return pages


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

    from lilbee.core.system import stderr_suppressed

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
    empty list so the caller can skip the file. OCR output is cached by file
    content so a downstream failure doesn't force a re-OCR on retry.
    """
    key = ocr_cache_key(file_hash(path), backend=TESSERACT_BACKEND, model=TESSERACT_BACKEND)
    page_texts = load_ocr_pages(key)
    if page_texts is None:
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
        store_ocr_pages(key, page_texts)
    _record_page_texts(page_texts, source_name, content_type, page_texts_out)
    return await chunk_and_embed_pages(page_texts, source_name, content_type, on_progress)


def _chunk_pages(page_texts: Sequence[tuple[int, str]]) -> list[tuple[int, str]]:
    """Chunk each OCR page's text. Semantic chunking is off: a single OCR page
    rarely spans multiple topics, so the semantic round-trip is not worth it."""
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
    """Chunk per-page text and embed every chunk. Shared by OCR ingest and import."""
    if not page_texts:
        return []

    # chunk_text runs kreuzberg's synchronous extractor; offload it so a long OCR
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


def _warn_empty_ocr(source_name: str, media: str) -> None:
    """Warn that OCR yielded no text and point to the vision-model remedy."""
    log.warning(
        "Skipped %s: text extraction produced no usable text. "
        "For better results on %s, configure a vision model "
        "via PUT /api/models/vision or set LILBEE_ENABLE_OCR=true.",
        source_name,
        media,
    )


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
    if _effective_enable_ocr() is False:
        # OCR explicitly disabled: skip entirely (vision and Tesseract) rather
        # than paying the full Tesseract cost the config says is turned off.
        log.info("OCR disabled; skipping scanned-PDF OCR for %s", source_name)
        return []
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
        _warn_empty_ocr(source_name, "scanned PDFs")
    return chunks


async def _handle_image(
    path: Path,
    source_name: str,
    content_type: str,
    *,
    on_progress: DetailedProgressCallback,
    page_texts_out: list[PageTextRecord] | None = None,
) -> list[ChunkRecord]:
    """OCR an image: vision OCR when a vision model is configured, else Tesseract.

    An image has no text layer to extract first, so it routes straight to OCR --
    the same downstream call a PDF page hits after it is rasterized to an image.
    """
    if _effective_enable_ocr() is False:
        # OCR explicitly disabled: an image has no text layer, so skip it rather
        # than paying the full Tesseract cost the config says is turned off.
        log.info("OCR disabled; skipping image OCR for %s", source_name)
        return []
    if _should_run_ocr() and cfg.vision_model:
        log.info("Image: using vision OCR for %s (model=%s)", source_name, cfg.vision_model)
        return await _vision_image_ocr(
            path, source_name, content_type, on_progress=on_progress, page_texts_out=page_texts_out
        )

    log.info("Image: falling back to Tesseract OCR for %s", source_name)
    chunks = await _tesseract_ocr_fallback(
        path, source_name, content_type, on_progress=on_progress, page_texts_out=page_texts_out
    )
    if not chunks:
        _warn_empty_ocr(source_name, "images")
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
    # An image carries no text layer; route it straight to OCR (vision or Tesseract)
    # instead of a no-op kreuzberg markdown extract that yields nothing for a scan.
    if content_type == IMAGE_CONTENT_TYPE:
        return await _handle_image(
            path, source_name, content_type, on_progress=on_progress, page_texts_out=page_texts_out
        )

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
