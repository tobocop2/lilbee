"""Text chunking with optional heading-aware and topic-aware splitting."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from lilbee.core.config import cfg

if TYPE_CHECKING:
    from xberg import ChunkingConfig, EmbeddingConfig

    from lilbee.providers.base import EmbeddingEndpoint

CHARS_PER_TOKEN = 4

_SEMANTIC_CHUNKER = "semantic"
_MARKDOWN_CHUNKER = "markdown"
# Xberg silently falls back to a non-semantic path when embedding is None.
_SEMANTIC_EMBEDDING_PRESET = "fast"
_DISABLE_PROGRESS_ENV = "HF_HUB_DISABLE_PROGRESS_BARS"


def _char_budget() -> tuple[int, int]:
    """Return (max_chars, max_overlap) in characters from the token-based cfg."""
    max_chars = cfg.chunk_size * CHARS_PER_TOKEN
    max_overlap = min(cfg.chunk_overlap * CHARS_PER_TOKEN, max_chars // 2)
    return max_chars, max_overlap


def _show_download_progress() -> bool:
    """Honor lilbee's global progress-bar suppression (set for quiet/JSON modes).

    lilbee defaults ``HF_HUB_DISABLE_PROGRESS_BARS`` on in ``__init__``; mirroring
    it here keeps the embedding-model download silent instead of hardcoding a bar
    that would corrupt JSON output.
    """
    return os.environ.get(_DISABLE_PROGRESS_ENV, "0").lower() not in ("1", "true")


def _semantic_embedding_endpoint() -> EmbeddingEndpoint | None:
    """lilbee's own embeddings endpoint for semantic-chunk boundary detection, so
    xberg reuses the fleet instead of downloading a preset. None -> use the preset
    (no fleet endpoint, e.g. a remote provider)."""
    from lilbee.app.services import get_services

    return get_services().provider.embedding_endpoint()


def _semantic_embedding_config() -> EmbeddingConfig:
    """EmbeddingConfig for semantic chunking: route boundary-detection embeddings
    at lilbee's fleet via the Llm variant when available, else xberg's 'fast' preset.
    """
    from xberg import EmbeddingConfig, EmbeddingModelType

    # The public xberg.LlmConfig is a distinct class from the one EmbeddingModelType.llm()
    # accepts; the constructor's isinstance check only passes the internal type (alef-m07).
    from xberg._xberg import LlmConfig

    endpoint = _semantic_embedding_endpoint()
    # xberg's .pyi omits the per-variant constructors alef #147 added at runtime;
    # .llm()/.preset() work but aren't declared in the stub yet (alef-m07).
    if endpoint is not None:
        # endpoint.model is the bare id the endpoint routes by (no provider prefix);
        # endpoint.base_url already carries the /v1 the client appends /embeddings to.
        model = EmbeddingModelType.llm(  # type: ignore[attr-defined]  # style-check: allow-smell
            LlmConfig(
                model=endpoint.model,
                base_url=endpoint.base_url,
                api_key=endpoint.api_key,
            )
        )
    else:
        model = EmbeddingModelType.preset(  # type: ignore[attr-defined]  # style-check: allow-smell
            _SEMANTIC_EMBEDDING_PRESET
        )
    return EmbeddingConfig(model=model, show_download_progress=_show_download_progress())


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
