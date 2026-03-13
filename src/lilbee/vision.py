"""Vision model OCR extraction for scanned PDFs.

Rasterizes PDF pages to PNG, sends each to a local vision model
via Ollama, and concatenates the extracted text.
"""

import contextlib
import io
import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_OCR_PROMPT = (
    "Extract ALL text from this page as clean markdown. "
    "Preserve table structure using markdown table syntax. "
    "Include all rows, columns, headers, and page text exactly as shown."
)

_RASTER_DPI = 150


def pdf_page_count(path: Path) -> int:
    """Return the number of pages in a PDF without rasterizing."""
    import pypdfium2 as pdfium  # lazy: heavy dependency

    pdf = pdfium.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def rasterize_pdf(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield (0-based index, PNG bytes) for each page of a PDF."""
    import pypdfium2 as pdfium  # lazy: heavy dependency

    pdf = pdfium.PdfDocument(path)
    try:
        scale = _RASTER_DPI / 72
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            page.close()
            yield (i, buf.getvalue())
    finally:
        pdf.close()


def extract_page_text(png_bytes: bytes, model: str, *, timeout: float | None = None) -> str | None:
    """Send a page image to a vision model and return extracted text."""
    import ollama  # lazy: heavy dependency

    try:
        messages = [{"role": "user", "content": _OCR_PROMPT, "images": [png_bytes]}]
        if timeout is not None:
            client = ollama.Client(timeout=timeout)
            response = client.chat(model=model, messages=messages)
        else:
            response = ollama.chat(model=model, messages=messages)
        return str(response.message.content or "")
    except Exception as exc:
        log.warning("Vision OCR: page skipped (%s: %s)", type(exc).__name__, exc)
        log.debug("Vision OCR traceback for model %s", model, exc_info=True)
        return None


def _make_progress(name: str, total: int, quiet: bool) -> tuple[AbstractContextManager[Any], Any]:
    """Return (context_manager, task_id | None) for optional Rich progress."""
    if quiet:
        return contextlib.nullcontext(), None
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
    )
    task = progress.add_task(f"Vision OCR {name}", total=total)
    return progress, task


def extract_pdf_vision(
    path: Path, model: str, *, quiet: bool = False, timeout: float | None = None
) -> list[tuple[int, str]]:
    """Extract text from a PDF using vision model OCR.

    Returns a list of (1-based page number, text) tuples for pages that
    produced non-empty text.
    """
    total = pdf_page_count(path)
    if total == 0:
        return []

    result: list[tuple[int, str]] = []
    failed = 0
    progress_ctx, progress_task = _make_progress(path.name, total, quiet)

    with progress_ctx:
        for i, png in rasterize_pdf(path):
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
