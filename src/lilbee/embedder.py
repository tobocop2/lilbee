"""Thin wrapper around Ollama embeddings API."""

import logging
import math
import time
from typing import Any, cast

import ollama

from lilbee.config import EMBEDDING_DIM, EMBEDDING_MODEL, MAX_EMBED_CHARS

log = logging.getLogger(__name__)

# nomic-embed-text has 8192 token context but uses a BERT tokenizer that counts
# whitespace-heavy text (tables, formatted code) much more expensively than tiktoken.
# Worst-case table text (87% special chars) fails at 2345 chars; 2000 gives ~15% margin.
_MAX_EMBED_CHARS = MAX_EMBED_CHARS

_MAX_BATCH_CHARS = 6000


def _call_with_retry(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Retry fn up to 3 times with exponential backoff on connection errors."""
    delays = [1, 2, 4]
    last_err: Exception | None = None
    for attempt, delay in enumerate(delays):
        try:
            return fn(*args, **kwargs)  # type: ignore[operator]
        except (ConnectionError, OSError) as exc:
            last_err = exc
            log.warning("Ollama call failed (attempt %d/3): %s", attempt + 1, exc)
            time.sleep(delay)
    raise last_err  # type: ignore[misc]


def _truncate(text: str) -> str:
    """Truncate text to stay within the embedding model's context window."""
    if len(text) <= _MAX_EMBED_CHARS:
        return text
    log.debug("Truncating chunk from %d to %d chars for embedding", len(text), _MAX_EMBED_CHARS)
    return text[:_MAX_EMBED_CHARS]


def _validate_vector(vector: list[float]) -> None:
    """Validate embedding vector dimension and values."""
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(vector)}"
        )
    for i, v in enumerate(vector):
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Embedding contains invalid value at index {i}: {v}")


def validate_model() -> None:
    """Ensure the configured embedding model is available, pulling if needed."""
    try:
        models = ollama.list()
        names = {m.model for m in models.models if m.model}
        # Also match without :latest tag
        base_names = {n.split(":")[0] for n in names}
        if EMBEDDING_MODEL not in names and EMBEDDING_MODEL not in base_names:
            log.info("Pulling embedding model '%s' from Ollama...", EMBEDDING_MODEL)
            ollama.pull(EMBEDDING_MODEL)
    except (ConnectionError, OSError) as exc:
        raise RuntimeError(f"Cannot connect to Ollama: {exc}. Is Ollama running?") from exc


def embed(text: str) -> list[float]:
    """Embed a single text string, return vector."""
    response = _call_with_retry(ollama.embed, model=EMBEDDING_MODEL, input=_truncate(text))
    result = cast(list[float], response["embeddings"][0])
    _validate_vector(result)
    return result


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts with adaptive batching, return list of vectors."""
    if not texts:
        return []
    vectors: list[list[float]] = []
    batch: list[str] = []
    batch_chars = 0
    for text in texts:
        truncated = _truncate(text)
        chunk_len = len(truncated)
        if batch and batch_chars + chunk_len > _MAX_BATCH_CHARS:
            response = _call_with_retry(ollama.embed, model=EMBEDDING_MODEL, input=batch)
            vectors.extend(cast(list[list[float]], response["embeddings"]))
            batch = []
            batch_chars = 0
        batch.append(truncated)
        batch_chars += chunk_len
    if batch:
        response = _call_with_retry(ollama.embed, model=EMBEDDING_MODEL, input=batch)
        vectors.extend(cast(list[list[float]], response["embeddings"]))
    for vec in vectors:
        _validate_vector(vec)
    return vectors
