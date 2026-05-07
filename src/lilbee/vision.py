"""Vision model OCR extraction for scanned PDFs.

Rasterizes PDF pages to PNG, sends each to a local vision model
via the configured LLM provider, and concatenates the extracted text.
"""

import concurrent.futures
import contextlib
import logging
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, NamedTuple

from lilbee.core.config import cfg
from lilbee.core.services import get_services
from lilbee.runtime.progress import (
    DetailedProgressCallback,
    EventType,
    ExtractEvent,
    noop_callback,
    shared_progress,
)

log = logging.getLogger(__name__)


class PageText(NamedTuple):
    """Extracted text for a single PDF page."""

    page: int
    text: str


class PdfOcrChunk(NamedTuple):
    """One streaming PDF-OCR worker frame: page index, total pages, page text."""

    page: int
    total: int
    text: str


OCR_PROMPT = (
    "Extract ALL text from this page as clean markdown. "
    "Preserve table structure using markdown table syntax. "
    "Include all rows, columns, headers, and page text exactly as shown."
)

_RASTER_DPI = 150


class _SharedTask:
    """Updates the batch task's description with per-page vision progress."""

    def __init__(self, progress: Any, batch_task: Any, name: str, total: int) -> None:
        self._progress = progress
        self._batch_task = batch_task
        self._name = name
        self._total = total
        self._current = 0

    def __enter__(self) -> "_SharedTask":
        self._progress.update(
            self._batch_task, description=f"Vision OCR {self._name} (0/{self._total})"
        )
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # batch loop updates the description after each file completes

    def advance(self, _task_id: Any) -> None:
        self._current += 1
        self._progress.update(
            self._batch_task,
            description=f"Vision OCR {self._name} ({self._current}/{self._total})",
        )


def pdf_page_count(path: Path) -> int:
    """Return the number of pages in a PDF without rasterizing."""
    from kreuzberg import PdfPageIterator  # lazy: heavy dependency

    it = PdfPageIterator(str(path), dpi=_RASTER_DPI)
    return len(it)


def rasterize_pdf(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield (0-based index, PNG bytes) for each page of a PDF."""
    from kreuzberg import PdfPageIterator  # lazy: heavy dependency

    with PdfPageIterator(str(path), dpi=_RASTER_DPI) as pages:
        yield from pages


def _png_to_data_url(png_bytes: bytes) -> str:
    """Convert raw PNG bytes to a base64 data URL for OpenAI-compatible messages."""
    import base64

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_vision_messages(prompt: str, png_bytes: bytes) -> list[dict]:
    """Build OpenAI-compatible messages with image content for vision models.
    Uses the multipart content format expected by llama.cpp's mtmd
    pipeline.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _png_to_data_url(png_bytes)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def extract_page_text(png_bytes: bytes, model: str, *, timeout: float | None = None) -> str | None:
    """OCR one page; returns ``None`` on provider failure so the caller skips it."""
    try:
        return get_services().provider.vision_ocr(png_bytes, model, OCR_PROMPT, timeout=timeout)
    except Exception as exc:
        log.warning("Vision OCR: page skipped (%s: %s)", type(exc).__name__, exc)
        log.debug("Vision OCR traceback for model %s", model, exc_info=True)
        return None


def _make_progress(name: str, total: int, quiet: bool) -> tuple[AbstractContextManager[Any], Any]:
    """Return (context_manager, task_id | None) for optional Rich progress."""
    if quiet:
        return contextlib.nullcontext(), None

    parent = shared_progress.get(None)
    if parent is not None:
        progress, batch_task = parent
        return _SharedTask(progress, batch_task, name, total), batch_task

    from rich.console import Console
    from rich.progress import (  # lazy: heavy dependency
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=Console(file=sys.__stderr__ or sys.stderr),
    )
    task = progress.add_task(f"Vision OCR {name}", total=total)
    return progress, task


def _record_page(
    fut: concurrent.futures.Future[tuple[int, str | None]],
    extracted: dict[int, str | None],
    on_progress: DetailedProgressCallback,
    progress_ctx: AbstractContextManager[Any],
    progress_task: Any,
    path: Path,
    total: int,
) -> None:
    """Drain one completed page future into ``extracted`` and fire progress."""
    i, text = fut.result()
    extracted[i] = text
    log.info(
        "Vision OCR completed page %d/%d of %s (%d chars)",
        i + 1,
        total,
        path.name,
        len(text) if text else 0,
    )
    on_progress(
        EventType.EXTRACT,
        ExtractEvent(file=path.name, page=i + 1, total_pages=total),
    )
    if progress_task is not None:
        progress_ctx.advance(progress_task)  # type: ignore[attr-defined]


def _collate_extracted_pages(
    extracted: dict[int, str | None],
) -> tuple[list[PageText], int]:
    """Sort extracted pages by index, returning ``(pages, failed_count)``."""
    result: list[PageText] = []
    failed = 0
    for i in sorted(extracted):
        text = extracted[i]
        if text is None:
            failed += 1
        elif text.strip():
            result.append(PageText(i + 1, text))
    return result, failed


def _report_vision_failures(failed: int, total: int, extracted: int, quiet: bool) -> None:
    """Log + print a yellow warning when any pages failed OCR."""
    if not failed:
        return
    log.warning("Vision OCR: %d/%d pages failed", failed, total)
    if not quiet:
        from rich.console import Console

        Console(stderr=True).print(
            f"[yellow]Vision OCR: {failed}/{total} pages failed, "
            f"{extracted}/{total} extracted[/yellow]"
        )


def extract_pdf_vision(
    path: Path,
    model: str,
    *,
    quiet: bool = False,
    timeout: float | None = None,
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[PageText]:
    """Extract text from a PDF using vision model OCR.
    Returns a list of (1-based page number, text) tuples for pages that
    produced non-empty text. Fires ``extract`` progress events per page.
    """
    total = pdf_page_count(path)
    if total == 0:
        return []

    concurrency = max(1, cfg.vision_concurrency)
    extracted: dict[int, str | None] = {}
    progress_ctx, progress_task = _make_progress(path.name, total, quiet)

    def _extract(i: int, png: bytes) -> tuple[int, str | None]:
        log.debug("Vision OCR page %d/%d with %s", i + 1, total, model)
        return i, extract_page_text(png, model, timeout=timeout)

    # Inflight futures are bounded to roughly ``concurrency * 2`` so raster
    # bytes never accumulate beyond what the pool can drain. Fully eager
    # submission would buffer every page's PNG in memory before any OCR
    # returned.
    max_inflight = concurrency * 2
    pages_iter = iter(rasterize_pdf(path))
    with progress_ctx, concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        inflight: set[concurrent.futures.Future[tuple[int, str | None]]] = set()
        for i, png in pages_iter:
            if len(inflight) >= max_inflight:
                done, inflight = concurrent.futures.wait(
                    inflight, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    _record_page(
                        fut, extracted, on_progress, progress_ctx, progress_task, path, total
                    )
            inflight.add(pool.submit(_extract, i, png))
        for fut in concurrent.futures.as_completed(inflight):
            _record_page(fut, extracted, on_progress, progress_ctx, progress_task, path, total)

    result, failed = _collate_extracted_pages(extracted)
    _report_vision_failures(failed, total, extracted=len(result), quiet=quiet)
    return result
