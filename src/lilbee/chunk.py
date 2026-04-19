"""Text chunking with optional heading-aware and topic-aware splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.config import cfg

if TYPE_CHECKING:
    from kreuzberg import ChunkingConfig

# Approximate characters-per-token ratio used to convert token counts to char counts.
CHARS_PER_TOKEN = 4

# Kreuzberg chunker_type identifiers (external API boundary).
_SEMANTIC_CHUNKER = "semantic"
_MARKDOWN_CHUNKER = "markdown"

# Kreuzberg embedding preset used by the semantic chunker. ``"fast"`` is the
# lightest ONNX option and is sufficient for topic-boundary detection.
# Kreuzberg maintains this model in its own cache, separate from lilbee's
# chunk-to-vector embedder. The double-embedding cost is accepted for now;
# see bb-kau6 for the upstream ask to accept an injected embedder.
_SEMANTIC_EMBEDDING_PRESET = "fast"


def build_chunking_config(*, use_semantic: bool = True) -> ChunkingConfig:
    """Build a kreuzberg ``ChunkingConfig`` from the current ``cfg``.

    When ``use_semantic`` and ``cfg.semantic_chunking`` are both true the
    topic-aware semantic chunker is selected. An ``EmbeddingConfig`` is
    attached so kreuzberg actually runs the semantic path; without one it
    silently falls back to a non-semantic heuristic.

    The semantic chunker honors ``cfg.chunk_size`` (via ``max_chars``) and
    ``cfg.topic_threshold`` only when an embedding is provided.
    """
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
    """Split text into chunks.

    Precedence for chunker selection:
    ``heading_context=True`` always wins so the markdown chunker preserves
    heading hierarchy for wiki paths. Otherwise ``use_semantic`` combined
    with ``cfg.semantic_chunking`` selects the topic-aware chunker. Otherwise
    the default chunker runs with a fixed character budget.

    Args:
        text: The text to chunk.
        mime_type: MIME type hint for chunker selection.
        heading_context: If True, prepend heading hierarchy to each chunk.
        use_semantic: If False, bypass the semantic chunker even when
            ``cfg.semantic_chunking`` is true. Used by callers that chunk
            already-segmented text (e.g. per-page vision OCR) where
            topic-based splitting is wasteful.

    Returns:
        List of chunk strings. Empty if text is empty.
    """
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
