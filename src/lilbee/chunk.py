"""Text chunking with optional heading-aware splitting."""

from __future__ import annotations

from lilbee.config import cfg


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

    max_chars = cfg.chunk_size * 4
    max_overlap = min(cfg.chunk_overlap * 4, max_chars // 2)

    config = ExtractionConfig(
        chunking=ChunkingConfig(
            max_chars=max_chars,
            max_overlap=max_overlap,
            chunker_type="markdown" if heading_context else None,  # type: ignore[call-arg]
            prepend_heading_context=heading_context,  # type: ignore[call-arg]
        )
    )
    result = extract_bytes_sync(text.encode("utf-8"), mime_type, config=config)
    if result.chunks:
        return [c.content for c in result.chunks]
    return []
