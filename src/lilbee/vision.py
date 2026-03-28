"""Vision model OCR extraction for scanned PDFs.

Rasterizes PDF pages to PNG, sends each to a local vision model
via the configured LLM provider, and concatenates the extracted text.
"""

import contextlib
import logging
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

from lilbee.progress import DetailedProgressCallback, EventType, noop_callback, shared_progress

log = logging.getLogger(__name__)

_OCR_PROMPT = (
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


def extract_page_text(png_bytes: bytes, model: str, *, timeout: float | None = None) -> str | None:
    """Send a page image to a vision model and return extracted text."""
    try:
        from lilbee.services import get_services

        provider = get_services().provider
        messages = [{"role": "user", "content": _OCR_PROMPT, "images": [png_bytes]}]
        return cast(str, provider.chat(messages, stream=False, model=model))
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


def extract_pdf_vision(
    path: Path,
    model: str,
    *,
    quiet: bool = False,
    timeout: float | None = None,
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[tuple[int, str]]:
    """Extract text from a PDF using vision model OCR.

    Returns a list of (1-based page number, text) tuples for pages that
    produced non-empty text. Fires ``extract`` progress events per page.
    """
    total = pdf_page_count(path)
    if total == 0:
        return []

    result: list[tuple[int, str]] = []
    failed = 0
    progress_ctx, progress_task = _make_progress(path.name, total, quiet)

    with progress_ctx:
        for i, png in rasterize_pdf(path):
            on_progress(
                EventType.EXTRACT,
                {"file": path.name, "page": i + 1, "total_pages": total},
            )
            log.debug("Vision OCR page %d/%d with %s", i + 1, total, model)
            text = extract_page_text(png, model, timeout=timeout)
            if text is None:
                failed += 1
            elif text.strip():
                result.append((i + 1, text))
            if progress_task is not None:
                progress_ctx.advance(progress_task)  # type: ignore[attr-defined]

    if failed:
        log.warning("Vision OCR: %d/%d pages failed", failed, total)
        if not quiet:
            from rich.console import Console

            Console(stderr=True).print(
                f"[yellow]Vision OCR: {failed}/{total} pages failed, "
                f"{len(result)}/{total} extracted[/yellow]"
            )

    return result
