"""Thin wrapper around LLM provider embeddings API."""

import logging
import math

from lilbee.config import cfg
from lilbee.progress import DetailedProgressCallback, EventType, noop_callback
from lilbee.providers import get_provider

log = logging.getLogger(__name__)

MAX_BATCH_CHARS = 6000


def truncate(text: str) -> str:
    """Truncate text to stay within the embedding model's context window."""
    if len(text) <= cfg.max_embed_chars:
        return text
    log.debug("Truncating chunk from %d to %d chars for embedding", len(text), cfg.max_embed_chars)
    return text[: cfg.max_embed_chars]


def validate_vector(vector: list[float]) -> None:
    """Validate embedding vector dimension and values."""
    if len(vector) != cfg.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {cfg.embedding_dim}, got {len(vector)}"
        )
    for i, v in enumerate(vector):
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Embedding contains invalid value at index {i}: {v}")


def validate_model() -> None:
    """Ensure the configured embedding model is available, pulling if needed."""
    from lilbee.model_manager import get_model_manager

    try:
        if not get_model_manager().is_installed(cfg.embedding_model):
            log.info("Pulling embedding model '%s'...", cfg.embedding_model)
            get_provider().pull_model(cfg.embedding_model, on_progress=lambda _: None)
    except (ConnectionError, OSError) as exc:
        raise RuntimeError(f"Cannot connect to Ollama: {exc}. Is Ollama running?") from exc


def embed(text: str) -> list[float]:
    """Embed a single text string, return vector."""
    provider = get_provider()
    vectors = provider.embed([truncate(text)])
    result: list[float] = vectors[0]
    validate_vector(result)
    return result


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
