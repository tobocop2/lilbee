"""Text chunking with optional heading-aware and topic-aware splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.core.config import cfg

if TYPE_CHECKING:
    from xberg import ChunkingConfig, EmbeddingConfig

CHARS_PER_TOKEN = 4

_SEMANTIC_CHUNKER = "semantic"
_MARKDOWN_CHUNKER = "markdown"


def _char_budget() -> tuple[int, int]:
    """Return (max_chars, max_overlap) in characters from the token-based cfg."""
    max_chars = cfg.chunk_size * CHARS_PER_TOKEN
    max_overlap = min(cfg.chunk_overlap * CHARS_PER_TOKEN, max_chars // 2)
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
    # xberg's public EmbeddingConfig still types `model` as the legacy
    # str|int|LlmConfig alias, not the EmbeddingModelType class its own .plugin()
    # returns; the constructor accepts the class instance at runtime.
    return EmbeddingConfig(model=model)  # type: ignore[arg-type]


def build_chunking_config(*, use_semantic: bool = True) -> ChunkingConfig:
    """Build an xberg ChunkingConfig from the current cfg."""
    from xberg import ChunkingConfig

    max_chars, max_overlap = _char_budget()

    if use_semantic and cfg.semantic_chunking:
        return ChunkingConfig(
            chunker_type=_SEMANTIC_CHUNKER,
            embedding=_semantic_embedding_config(),
            topic_threshold=cfg.topic_threshold,
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

    from xberg import ChunkingConfig, ExtractionConfig, extract_bytes_sync

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
    result = extract_bytes_sync(text.encode("utf-8"), mime_type, config=config)
    if result.chunks:
        return [c.content for c in result.chunks]
    return []
