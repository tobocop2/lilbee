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

    Precedence for chunker selection:
    ``heading_context=True`` always wins so the markdown chunker preserves
    heading hierarchy for wiki paths. Otherwise if ``cfg.semantic_chunking``
    is true, the topic-aware semantic chunker is used. Otherwise the default
    chunker runs with a fixed character budget.

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

    if heading_context:
        chunking = ChunkingConfig(
            max_chars=max_chars,
            max_overlap=max_overlap,
            chunker_type="markdown",
            prepend_heading_context=True,  # type: ignore[call-arg]
        )
    elif cfg.semantic_chunking:
        chunking = ChunkingConfig(
            chunker_type="semantic",
            topic_threshold=cfg.topic_threshold,
            max_overlap=max_overlap,
        )
    else:
        chunking = ChunkingConfig(max_chars=max_chars, max_overlap=max_overlap)

    config = ExtractionConfig(chunking=chunking)
    result = extract_bytes_sync(text.encode("utf-8"), mime_type, config=config)
    if result.chunks:
        return [c.content for c in result.chunks]
    return []
