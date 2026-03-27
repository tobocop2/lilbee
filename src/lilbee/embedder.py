"""Thin wrapper around LLM provider embeddings API."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from lilbee.config import cfg
from lilbee.progress import DetailedProgressCallback, EventType, noop_callback
from lilbee.providers import get_provider

if TYPE_CHECKING:
    from lilbee.config import Config
    from lilbee.providers.base import LLMProvider

log = logging.getLogger(__name__)

MAX_BATCH_CHARS = 6000


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to *max_chars*."""
    if len(text) <= max_chars:
        return text
    log.debug("Truncating chunk from %d to %d chars for embedding", len(text), max_chars)
    return text[:max_chars]


def truncate(text: str) -> str:
    """Truncate text to stay within the embedding model's context window."""
    return _truncate(text, cfg.max_embed_chars)


def _validate_vector(vector: list[float], expected_dim: int) -> None:
    """Validate embedding vector dimension and values."""
    if len(vector) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(vector)}"
        )
    for i, v in enumerate(vector):
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Embedding contains invalid value at index {i}: {v}")


def validate_vector(vector: list[float]) -> None:
    """Validate embedding vector dimension and values (uses global cfg)."""
    _validate_vector(vector, cfg.embedding_dim)


def validate_model() -> None:
    """Ensure the configured embedding model is available, pulling if needed."""
    from lilbee.model_manager import get_model_manager

    try:
        if not get_model_manager().is_installed(cfg.embedding_model):
            log.info("Pulling embedding model '%s'...", cfg.embedding_model)
            get_provider().pull_model(cfg.embedding_model, on_progress=lambda _: None)
    except (ConnectionError, OSError) as exc:
        raise RuntimeError(
            f"Cannot connect to embedding backend: {exc}. Is the server running?"
        ) from exc


def validate_model_di(*, config: Config, provider: LLMProvider) -> None:
    """Ensure the configured embedding model is available (DI version)."""
    from lilbee.model_manager import get_model_manager

    try:
        if not get_model_manager().is_installed(config.embedding_model):
            log.info("Pulling embedding model '%s'...", config.embedding_model)
            provider.pull_model(config.embedding_model, on_progress=lambda _: None)
    except (ConnectionError, OSError) as exc:
        raise RuntimeError(
            f"Cannot connect to embedding backend: {exc}. Is the server running?"
        ) from exc


def embed_di(text: str, *, provider: LLMProvider, config: Config) -> list[float]:
    """Embed a single text string with explicit provider and config."""
    vectors = provider.embed([_truncate(text, config.max_embed_chars)])
    result: list[float] = vectors[0]
    _validate_vector(result, config.embedding_dim)
    return result


def embed(text: str) -> list[float]:
    """Embed a single text string, return vector."""
    provider = get_provider()
    vectors = provider.embed([truncate(text)])
    result: list[float] = vectors[0]
    validate_vector(result)
    return result


def embed_batch_di(
    texts: list[str],
    *,
    provider: LLMProvider,
    config: Config,
    source: str = "",
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[list[float]]:
    """Embed multiple texts with explicit provider and config."""
    if not texts:
        return []
    total_chunks = len(texts)
    max_chars = config.max_embed_chars
    vectors: list[list[float]] = []
    batch: list[str] = []
    batch_chars = 0
    for text in texts:
        truncated = _truncate(text, max_chars)
        chunk_len = len(truncated)
        if batch and batch_chars + chunk_len > MAX_BATCH_CHARS:
            vectors.extend(provider.embed(batch))
            on_progress(
                EventType.EMBED,
                {"file": source, "chunk": len(vectors), "total_chunks": total_chunks},
            )
            batch = []
            batch_chars = 0
        batch.append(truncated)
        batch_chars += chunk_len
    if batch:
        vectors.extend(provider.embed(batch))
        on_progress(
            EventType.EMBED,
            {"file": source, "chunk": len(vectors), "total_chunks": total_chunks},
        )
    for vec in vectors:
        _validate_vector(vec, config.embedding_dim)
    return vectors


def embed_batch(
    texts: list[str],
    *,
    source: str = "",
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[list[float]]:
    """Embed multiple texts with adaptive batching, return list of vectors.

    Fires ``embed`` progress events per batch when *on_progress* is provided.
    """
    if not texts:
        return []
    total_chunks = len(texts)
    provider = get_provider()
    vectors: list[list[float]] = []
    batch: list[str] = []
    batch_chars = 0
    for text in texts:
        truncated = truncate(text)
        chunk_len = len(truncated)
        if batch and batch_chars + chunk_len > MAX_BATCH_CHARS:
            vectors.extend(provider.embed(batch))
            on_progress(
                EventType.EMBED,
                {"file": source, "chunk": len(vectors), "total_chunks": total_chunks},
            )
            batch = []
            batch_chars = 0
        batch.append(truncated)
        batch_chars += chunk_len
    if batch:
        vectors.extend(provider.embed(batch))
        on_progress(
            EventType.EMBED,
            {"file": source, "chunk": len(vectors), "total_chunks": total_chunks},
        )
    for vec in vectors:
        validate_vector(vec)
    return vectors
