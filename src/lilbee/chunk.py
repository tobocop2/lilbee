"""Text chunking with optional heading-aware splitting."""

from __future__ import annotations

from lilbee.config import cfg

# Approximate characters-per-token ratio used to convert token counts to char counts.
CHARS_PER_TOKEN = 4


def chunk_text(
    text: str,
    *,
    mime_type: str = "text/plain",
    heading_context: bool = False,
) -> list[str]:
    """Split text into chunks.
    Args:
        text: The text to chunk.
        mime_type: MIME type hint for chunker selection.
        heading_context: If True, prepend heading hierarchy to each chunk.

    Returns:
        List of chunk strings. Empty if text is empty.
    """
    if not text or not text.strip():
        return []

    from kreuzberg import ChunkingConfig, ExtractionConfig, extract_bytes_sync

    max_chars = cfg.chunk_size * CHARS_PER_TOKEN
    max_overlap = min(cfg.chunk_overlap * CHARS_PER_TOKEN, max_chars // 2)

    config = ExtractionConfig(
        chunking=ChunkingConfig(
            max_chars=max_chars,
            max_overlap=max_overlap,
            chunker_type="markdown" if heading_context else None,
            prepend_heading_context=heading_context,  # type: ignore[call-arg]
        )
    )
    result = extract_bytes_sync(text.encode("utf-8"), mime_type, config=config)
    if result.chunks:
        if heading_context:
            return [_dedup_heading(c.content) for c in result.chunks]
        return [c.content for c in result.chunks]
    return []


def _dedup_heading(text: str) -> str:
    """Remove duplicate heading caused by kreuzberg's prepend_heading_context.

    kreuzberg prepends a heading breadcrumb (e.g. ``# Top > ## Sub``) followed
    by a blank line. When the chunk body starts with the same heading that ends
    the breadcrumb, the heading appears twice. This strips the duplicate.
    """
    parts = text.split("\n\n", 2)
    if len(parts) < 2:
        return text
    ctx = parts[0]
    last_seg = ctx.rsplit(" > ", 1)[-1]
    if parts[1].strip() == last_seg.strip():
        rest = parts[2] if len(parts) > 2 else ""
        return ctx + ("\n\n" + rest if rest else "")
    return text
