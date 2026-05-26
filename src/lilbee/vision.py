"""Helpers for PDF rasterisation and vision-model OCR.

Multi-page vision OCR runs through ``FleetProvider.pdf_ocr``, which rasterises
each page and sends it to the vision server; this module hosts the small helpers
(page count, rasterisation, prompt + chat-message construction, and the shared
:class:`PageText` / :class:`PdfOcrChunk` types) that the provider and its callers
share.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

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

    Uses the multipart content format expected by llama.cpp's mtmd pipeline.
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
