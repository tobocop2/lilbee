"""Vision model OCR extraction for scanned PDFs.

Rasterizes PDF pages to PNG, sends each to a local vision model
via Ollama, and concatenates the extracted text.
"""

import io
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_OCR_PROMPT = (
    "Extract ALL text from this page as clean markdown. "
    "Preserve table structure using markdown table syntax. "
    "Include all rows, columns, headers, and page text exactly as shown."
)

_RASTER_DPI = 150


def rasterize_pdf(path: Path) -> list[bytes]:
    """Rasterize each page of a PDF to PNG bytes."""
    import pypdfium2 as pdfium  # lazy: heavy dependency

    pages: list[bytes] = []
    pdf = pdfium.PdfDocument(path)
    try:
        scale = _RASTER_DPI / 72
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            pages.append(buf.getvalue())
            page.close()
    finally:
        pdf.close()
    return pages


def extract_page_text(png_bytes: bytes, model: str) -> str:
    """Send a page image to a vision model and return extracted text."""
    import ollama  # lazy: heavy dependency

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": _OCR_PROMPT, "images": [png_bytes]}],
        )
        return str(response.message.content or "")
    except Exception:
        log.warning("Vision OCR failed for a page with model %s", model, exc_info=True)
        return ""


def extract_pdf_vision(path: Path, model: str) -> list[tuple[int, str]]:
    """Extract text from a PDF using vision model OCR.

    Returns a list of (1-based page number, text) tuples for pages that
    produced non-empty text.
    """
    pages = rasterize_pdf(path)
    if not pages:
        return []
    result: list[tuple[int, str]] = []
    for i, png in enumerate(pages):
        log.debug("Vision OCR page %d/%d with %s", i + 1, len(pages), model)
        text = extract_page_text(png, model)
        if text.strip():
            result.append((i + 1, text))
    return result
