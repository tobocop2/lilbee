"""Thin wrapper around Ollama embeddings API."""

import logging
from typing import cast

import ollama

from lilbee.config import EMBEDDING_MODEL

log = logging.getLogger(__name__)

# nomic-embed-text has 8192 token context but uses a BERT tokenizer that counts
# whitespace-heavy text (tables, formatted code) much more expensively than tiktoken.
# Worst-case table text (87% special chars) fails at 2345 chars; 2000 gives ~15% margin.
_MAX_EMBED_CHARS = 2000

_MAX_BATCH_CHARS = 6000


def _truncate(text: str) -> str:
    """Truncate text to stay within the embedding model's context window."""
    if len(text) <= _MAX_EMBED_CHARS:
        return text
    log.debug("Truncating chunk from %d to %d chars for embedding", len(text), _MAX_EMBED_CHARS)
    return text[:_MAX_EMBED_CHARS]


def embed(text: str) -> list[float]:
    """Embed a single text string, return vector."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=_truncate(text))
    return cast(list[float], response["embeddings"][0])


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
            response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
            vectors.extend(cast(list[list[float]], response["embeddings"]))
            batch = []
            batch_chars = 0
        batch.append(truncated)
        batch_chars += chunk_len
    if batch:
        response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(cast(list[list[float]], response["embeddings"]))
    return vectors
