"""Text chunking with optional heading-aware and topic-aware splitting."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from lilbee.core.config import active_config

if TYPE_CHECKING:
    from kreuzberg import ChunkingConfig

CHARS_PER_TOKEN = 4

_SEMANTIC_CHUNKER = "semantic"
_MARKDOWN_CHUNKER = "markdown"
# Kreuzberg silently falls back to a non-semantic path when embedding is None.
_SEMANTIC_EMBEDDING_PRESET = "fast"
_DISABLE_PROGRESS_ENV = "HF_HUB_DISABLE_PROGRESS_BARS"


def _char_budget() -> tuple[int, int]:
    """Return (max_chars, max_overlap) in characters from the token-based cfg."""
    config = active_config()
    max_chars = config.chunk_size * CHARS_PER_TOKEN
    max_overlap = min(config.chunk_overlap * CHARS_PER_TOKEN, max_chars // 2)
    return max_chars, max_overlap


def _show_download_progress() -> bool:
    """Honor lilbee's global progress-bar suppression (set for quiet/JSON modes).

    lilbee defaults ``HF_HUB_DISABLE_PROGRESS_BARS`` on in ``__init__``; mirroring
    it here keeps the embedding-model download silent instead of hardcoding a bar
    that would corrupt JSON output.
    """
    return os.environ.get(_DISABLE_PROGRESS_ENV, "0").lower() not in ("1", "true")


def build_chunking_config(*, use_semantic: bool = True) -> ChunkingConfig:
    """Build a kreuzberg ChunkingConfig from the current cfg."""
    from kreuzberg import ChunkingConfig, EmbeddingConfig, EmbeddingModelType

    max_chars, max_overlap = _char_budget()

    config = active_config()
    if use_semantic and config.semantic_chunking:
        return ChunkingConfig(
            chunker_type=_SEMANTIC_CHUNKER,
            embedding=EmbeddingConfig(
                model=EmbeddingModelType.preset(_SEMANTIC_EMBEDDING_PRESET),
                show_download_progress=_show_download_progress(),
            ),
            topic_threshold=config.topic_threshold,
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
        max_chars, max_overlap = _char_budget()
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
