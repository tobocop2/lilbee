"""Text chunking with optional heading-aware and topic-aware splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.core.config import cfg

if TYPE_CHECKING:
    from kreuzberg import ChunkingConfig

CHARS_PER_TOKEN = 4

_SEMANTIC_CHUNKER = "semantic"
_MARKDOWN_CHUNKER = "markdown"
# Kreuzberg silently falls back to a non-semantic path when embedding is None.
_SEMANTIC_EMBEDDING_PRESET = "fast"


def build_chunking_config(*, use_semantic: bool = True) -> ChunkingConfig:
    """Build a kreuzberg ChunkingConfig from the current cfg."""
    from kreuzberg import ChunkingConfig, EmbeddingConfig, EmbeddingModelType

    max_chars = cfg.chunk_size * CHARS_PER_TOKEN
    max_overlap = min(cfg.chunk_overlap * CHARS_PER_TOKEN, max_chars // 2)

    if use_semantic and cfg.semantic_chunking:
        return ChunkingConfig(
            chunker_type=_SEMANTIC_CHUNKER,
            embedding=EmbeddingConfig(
                model=EmbeddingModelType.preset(_SEMANTIC_EMBEDDING_PRESET),
                show_download_progress=True,
            ),
            topic_threshold=cfg.topic_threshold,
            max_chars=max_chars,
            max_overlap=max_overlap,
        )
    return ChunkingConfig(max_chars=max_chars, max_overlap=max_overlap)


def chunk_text(
    text: str,
    *,
    mime_type: str = "text/plain",
    heading_context: bool = False,
    use_semantic: bool = True,
) -> list[str]:
    """Split text into chunks; heading_context wins over use_semantic wins over char-budget."""
    if not text or not text.strip():
        return []

    from kreuzberg import ChunkingConfig, ExtractionConfig, extract_bytes_sync

    if heading_context:
        max_chars = cfg.chunk_size * CHARS_PER_TOKEN
        max_overlap = min(cfg.chunk_overlap * CHARS_PER_TOKEN, max_chars // 2)
        chunking = ChunkingConfig(
            max_chars=max_chars,
            max_overlap=max_overlap,
            chunker_type=_MARKDOWN_CHUNKER,
            prepend_heading_context=True,  # type: ignore[call-arg]
        )
    else:
        chunking = build_chunking_config(use_semantic=use_semantic)

    config = ExtractionConfig(chunking=chunking)
    result = extract_bytes_sync(text.encode("utf-8"), mime_type, config=config)
    if result.chunks:
        return [c.content for c in result.chunks]
    return []
