"""Text chunking with optional heading-aware and topic-aware splitting."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from lilbee.core.config import active_config

if TYPE_CHECKING:
    from xberg import (
        ChunkingConfig,
        ChunkSizing,
        EmbeddingConfig,
        HeadingContext,
        TableChunkingMode,
    )

# Char->token ratio for English.
CHARS_PER_TOKEN = 4

_SEMANTIC_CHUNKER = "semantic"
_MARKDOWN_CHUNKER = "markdown"
# Markdown heading path rendered into a chunk: "# Setup > ## Install".
_HEADING_MARK = "#"
_BREADCRUMB_SEPARATOR = " > "


def _embed_token_cap() -> int | None:
    """Tokens the embedder truncates one input to; None when it has no fixed cap."""
    # circular: chunk -> app.services -> retrieval.embedder -> chunk via CHARS_PER_TOKEN
    from lilbee.app.services import get_services

    return get_services().provider.embed_token_cap()


def count_embed_tokens(text: str) -> int:
    """Tokens in *text* under the embedder's tokenizer, or a character estimate without one."""
    from lilbee.app.services import get_services

    from .backends.tokenizer import LilbeeTokenizerBackend

    return LilbeeTokenizerBackend(count_fn=get_services().provider.count_tokens).count_tokens(text)


def token_budget() -> int:
    """Tokens one chunk may hold: ``chunk_size``, cut to the embedder's cap when it has one."""
    cap = _embed_token_cap()
    size = active_config().chunk_size
    return size if cap is None else min(size, cap)


def _char_budget() -> tuple[int, int]:
    """Return (max_chars, max_overlap) in characters from the token-based cfg."""
    config = active_config()
    max_chars = config.chunk_size * CHARS_PER_TOKEN
    max_overlap = min(config.chunk_overlap * CHARS_PER_TOKEN, max_chars // 2)
    return max_chars, max_overlap


def _tokenizer_sizing() -> ChunkSizing:
    """xberg sizing through the tokenizer registered for the current provider."""
    from xberg import ChunkSizing

    from lilbee.app.services import get_services

    from .backends.tokenizer import TOKENIZER_SIZING_TYPE, provider_tokenizers

    return ChunkSizing(
        type=TOKENIZER_SIZING_TYPE, model=provider_tokenizers.name_for(get_services().provider)
    )


def _size_params(max_tokens: int | None = None) -> tuple[int, int, ChunkSizing]:
    """Return (max, overlap, sizing) in embedder tokens for the plain and heading chunkers.

    The semantic chunker does not use this: it sizes by characters and ignores ChunkSizing,
    so its chunks go through :func:`fit_token_budget` instead.
    """
    if max_tokens is None:
        max_tokens = token_budget()
    overlap = min(active_config().chunk_overlap, max_tokens // 2)
    return max_tokens, overlap, _tokenizer_sizing()


def _semantic_embedding_config() -> EmbeddingConfig:
    """EmbeddingConfig for semantic chunking. Boundary-detection embeddings route to
    the current provider's embedder, registered as xberg's plugin backend in
    ``lilbee.data.extract.backends.embedding``, so the model that vectorizes
    chunks for retrieval is the one that decides where they split."""
    from xberg import EmbeddingConfig, EmbeddingModelType

    from lilbee.app.services import get_services

    from .backends.embedding import provider_embedders

    model = EmbeddingModelType.plugin(provider_embedders.name_for(get_services().provider))
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
        # stays on the character budget; fit_token_budget then cuts its chunks to tokens.
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

    from xberg import ChunkingConfig

    if heading_context:
        budget = token_budget()
        max_size, max_overlap, sizing = _size_params(budget)
        chunking = ChunkingConfig(
            max_characters=max_size,
            overlap=max_overlap,
            sizing=sizing,
            chunker_type=_MARKDOWN_CHUNKER,
        )
        chunks: list[str] = []
        for c in _extract_chunks(text, mime_type, chunking):
            chunks.extend(_fit_breadcrumb(c.content, c.metadata.heading_context, budget))
        return chunks
    contents = [
        c.content
        for c in _extract_chunks(text, mime_type, build_chunking_config(use_semantic=use_semantic))
    ]
    if not (use_semantic and active_config().semantic_chunking):
        return contents
    budget = token_budget()
    return [piece for content in contents for piece in fit_token_budget(content, budget)]


def fit_token_budget(content: str, budget: int) -> list[str]:
    """*content* whole when the embedder counts it within *budget* tokens, else re-split to it.

    The semantic chunker sizes by characters, and a byte-level tokenizer makes more than one
    token from a character, so its chunks are counted with the embedder's own tokenizer.
    """
    if count_embed_tokens(content) <= budget:
        return [content]
    return _split_to_tokens(content, "text/plain", budget)


def fit_table_budget(markdown: str, budget: int) -> list[str]:
    """A markdown table whole when it fits *budget* tokens, else cut at its rows.

    Each piece repeats the header row and the separator row. The header is counted
    in the budget, and a header that takes more than half of it is not repeated,
    so the rows keep their room.
    """
    if count_embed_tokens(markdown) <= budget:
        return [markdown]
    lines = markdown.splitlines()
    header = "\n".join(lines[:2]) + "\n"
    room = budget - count_embed_tokens(header)
    if room < budget // 2:
        return _split_to_tokens(markdown, "text/plain", budget)
    # The rows are cut as plain text: xberg's markdown chunker parses the table, so its
    # pieces do not keep the table's own text.
    rows = "\n".join(lines[2:])
    return [header + piece for piece in _split_to_tokens(rows, "text/plain", room)]


def _split_to_tokens(content: str, mime_type: str, budget: int) -> list[str]:
    """*content* cut into pieces of at most *budget* tokens of the embedder."""
    from xberg import ChunkingConfig

    max_size, overlap, sizing = _size_params(budget)
    pieces = _extract_chunks(
        content,
        mime_type,
        ChunkingConfig(max_characters=max_size, overlap=overlap, sizing=sizing),
    )
    return [p.content for p in pieces]


def _extract_chunks(text: str, mime_type: str, chunking: ChunkingConfig) -> list[Any]:
    """The chunks xberg cuts from *text* under *chunking*."""
    from xberg import ExtractionConfig

    from .xberg import extract_document

    doc = extract_document(
        text.encode("utf-8"), mime_type, config=ExtractionConfig(chunking=chunking)
    )
    return list(doc.chunks or [])


def _breadcrumb(context: HeadingContext | None) -> str:
    """The heading path of *context*, e.g. ``# Setup > ## Install``, or empty without one.

    xberg's ``render_heading_breadcrumb`` is Rust-only; the Python binding exposes
    the headings as the chunk's ``metadata.heading_context``.
    """
    if context is None or not context.headings:
        return ""
    return _BREADCRUMB_SEPARATOR.join(
        f"{_HEADING_MARK * h.level} {h.text}" for h in context.headings
    )


def _fit_breadcrumb(content: str, context: HeadingContext | None, budget: int) -> list[str]:
    """Chunks of *content* that stay within *budget* tokens with the heading path prefixed.

    A chunk that fits with its prefix is returned whole. One that overflows is re-split to
    the room the prefix leaves. A path that takes more than half the budget is dropped, so
    the body keeps its room.
    """
    breadcrumb = _breadcrumb(context)
    if not breadcrumb:
        return [content]
    prefix = f"{breadcrumb}\n\n"
    if count_embed_tokens(prefix + content) <= budget:
        return [prefix + content]
    room = budget - count_embed_tokens(prefix)
    if room < budget // 2:
        return [content]
    return [prefix + piece for piece in _split_to_tokens(content, "text/markdown", room)]


class ChunkLimitError(Exception):
    """One file produced more chunks than ``cfg.max_chunks_per_file`` allows."""

    def __init__(self, count: int, limit: int, member: str | None = None) -> None:
        prefix = f"{member}: " if member else ""
        super().__init__(
            f"{prefix}{count} chunks exceed the per-file limit of {limit}; "
            f"raise max_chunks_per_file (0 = no limit), then retry skipped files"
        )
        self.count = count
        self.member = member
        self.limit = limit


def enforce_chunk_limit(count: int) -> None:
    """Refuse a file whose *count* chunks exceed the per-file limit; a limit of 0 accepts any."""
    limit = active_config().max_chunks_per_file
    if limit and count > limit:
        raise ChunkLimitError(count, limit)
