"""Text chunking with optional heading-aware and topic-aware splitting."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from lilbee.core.config import active_config
from lilbee.data.types import EmbeddingBackendName, TokenizerBackendName

if TYPE_CHECKING:
    from xberg import ChunkingConfig, ChunkSizing, EmbeddingConfig, TableChunkingMode

# Char->token ratio for English.
CHARS_PER_TOKEN = 4

_SEMANTIC_CHUNKER = "semantic"
_MARKDOWN_CHUNKER = "markdown"


def _char_budget() -> tuple[int, int]:
    """Return (max_chars, max_overlap) in characters from the token-based cfg."""
    config = active_config()
    max_chars = config.chunk_size * CHARS_PER_TOKEN
    max_overlap = min(config.chunk_overlap * CHARS_PER_TOKEN, max_chars // 2)
    return max_chars, max_overlap


def _size_params() -> tuple[int, int, ChunkSizing | None]:
    """Return (max, overlap, sizing) for the plain and heading chunkers.

    With ``cfg.token_sizing`` on, the budget is a raw token count and ``sizing``
    routes to lilbee's registered tokenizer backend, so ``chunk_size`` is a real
    token ceiling. Otherwise the character heuristic with no sizing (xberg's default
    character sizer). The semantic chunker does not use this -- it sizes by
    characters and ignores ChunkSizing."""
    config = active_config()
    if config.token_sizing:
        from xberg import ChunkSizing

        overlap = min(config.chunk_overlap, config.chunk_size // 2)
        return (
            config.chunk_size,
            overlap,
            ChunkSizing(type="tokenizer", model=TokenizerBackendName.LILBEE),
        )
    max_chars, max_overlap = _char_budget()
    return max_chars, max_overlap, None


def _semantic_embedding_config() -> EmbeddingConfig:
    """EmbeddingConfig for semantic chunking. Boundary-detection embeddings route to
    lilbee's embedder, registered as xberg's plugin backend in
    ``lilbee.data.extract.backends.registry`` (embedding binding), so the model that
    vectorizes chunks for retrieval is the one that decides where they split."""
    from xberg import EmbeddingConfig, EmbeddingModelType

    model = EmbeddingModelType.plugin(EmbeddingBackendName.LILBEE)
    return EmbeddingConfig(model=model)


def _table_chunking() -> TableChunkingMode | None:
    """Header-repeating table splits when table extraction is on, else None for
    xberg's default.

    REPEAT_HEADER carries the header row into every piece of a long table, so
    no chunk holds headerless rows.
    """
    config = active_config()
    if not config.table_extraction:
        return None
    from xberg import TableChunkingMode

    return TableChunkingMode.REPEAT_HEADER


def build_chunking_config(*, use_semantic: bool = True) -> ChunkingConfig:
    """Build an xberg ChunkingConfig from the current cfg."""
    from xberg import ChunkingConfig

    config = active_config()
    if use_semantic and config.semantic_chunking:
        # The semantic chunker sizes by characters and ignores ChunkSizing, so it
        # stays on the character budget regardless of cfg.token_sizing.
        max_chars, max_overlap = _char_budget()
        chunking = ChunkingConfig(
            chunker_type=_SEMANTIC_CHUNKER,
            embedding=_semantic_embedding_config(),
            topic_threshold=config.topic_threshold,
            max_characters=max_chars,
            overlap=max_overlap,
        )
    else:
        max_size, max_overlap, sizing = _size_params()
        chunking = ChunkingConfig(
            max_characters=max_size,
            overlap=max_overlap,
            sizing=sizing,
        )
    # table_chunking has no "unset" value on xberg's frozen ChunkingConfig, so the
    # field is left at its default rather than overwritten with None.
    mode = _table_chunking()
    return chunking if mode is None else replace(chunking, table_chunking=mode)


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

    from .xberg import extract_document

    if heading_context:
        max_size, max_overlap, sizing = _size_params()
        chunking = ChunkingConfig(
            max_characters=max_size,
            overlap=max_overlap,
            sizing=sizing,
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


class ChunkLimitError(Exception):
    """One file produced more chunks than ``cfg.max_chunks_per_file`` allows."""

    def __init__(self, count: int, limit: int) -> None:
        super().__init__(
            f"{count} chunks exceed the per-file limit of {limit}; "
            f"raise max_chunks_per_file (0 = no limit), then retry skipped files"
        )
        self.count = count
        self.limit = limit


def enforce_chunk_limit(count: int) -> None:
    """Refuse a file whose *count* chunks exceed the per-file limit; a limit of 0 accepts any."""
    limit = active_config().max_chunks_per_file
    if limit and count > limit:
        raise ChunkLimitError(count, limit)
