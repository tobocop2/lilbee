"""Text chunking with optional heading-aware and topic-aware splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.core.config import active_config

if TYPE_CHECKING:
    from xberg import ChunkingConfig, EmbeddingConfig

CHARS_PER_TOKEN = 4

_SEMANTIC_CHUNKER = "semantic"
_MARKDOWN_CHUNKER = "markdown"


def _char_budget() -> tuple[int, int]:
    """Return (max_chars, max_overlap) in characters from the token-based cfg."""
    config = active_config()
    max_chars = config.chunk_size * CHARS_PER_TOKEN
    max_overlap = min(config.chunk_overlap * CHARS_PER_TOKEN, max_chars // 2)
    return max_chars, max_overlap


def _semantic_embedding_config() -> EmbeddingConfig:
    """EmbeddingConfig for semantic chunking. Boundary-detection embeddings route to
    lilbee's embedder, registered as xberg's plugin backend in
    ``app.services.sync_embedding_backend``, so the model that vectorizes chunks for
    retrieval is the one that decides where they split."""
    from xberg import EmbeddingConfig, EmbeddingModelType

    # Lazy: importing ingest.types at module scope cycles back through chunk.py.
    from lilbee.data.ingest.types import EmbeddingBackendName

    model = EmbeddingModelType.plugin(EmbeddingBackendName.LILBEE)
    return EmbeddingConfig(model=model)


def build_chunking_config(*, use_semantic: bool = True) -> ChunkingConfig:
    """Build an xberg ChunkingConfig from the current cfg."""
    from xberg import ChunkingConfig

    max_chars, max_overlap = _char_budget()

    config = active_config()
    if use_semantic and config.semantic_chunking:
        return ChunkingConfig(
            chunker_type=_SEMANTIC_CHUNKER,
            embedding=_semantic_embedding_config(),
            topic_threshold=config.topic_threshold,
            max_characters=max_chars,
            overlap=max_overlap,
        )
    return ChunkingConfig(max_characters=max_chars, overlap=max_overlap)


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

    from xberg import ChunkingConfig, ExtractionConfig

    from lilbee.data.xberg_extract import extract_document

    if heading_context:
        max_chars, max_overlap = _char_budget()
        chunking = ChunkingConfig(
            max_characters=max_chars,
            overlap=max_overlap,
            chunker_type=_MARKDOWN_CHUNKER,
            prepend_heading_context=True,
        )
    else:
        chunking = build_chunking_config(use_semantic=use_semantic)

    config = ExtractionConfig(chunking=chunking)
    doc = extract_document(text.encode("utf-8"), mime_type, config=config)
    if doc.chunks:
        return [c.content for c in doc.chunks]
    return []
