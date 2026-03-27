"""Thin wrapper around LLM provider embeddings API."""

import logging
import math

from lilbee.config import Config, cfg
from lilbee.progress import DetailedProgressCallback, EventType, noop_callback
from lilbee.providers import get_provider
from lilbee.providers.base import LLMProvider

log = logging.getLogger(__name__)

MAX_BATCH_CHARS = 6000


class Embedder:
    """Embedding wrapper — truncates, batches, and validates vectors."""

    def __init__(self, config: Config, provider: LLMProvider) -> None:
        self._config = config
        self._provider = provider

    def truncate(self, text: str) -> str:
        """Truncate text to stay within the embedding model's context window."""
        if len(text) <= self._config.max_embed_chars:
            return text
        log.debug(
            "Truncating chunk from %d to %d chars for embedding",
            len(text),
            self._config.max_embed_chars,
        )
        return text[: self._config.max_embed_chars]

    def validate_vector(self, vector: list[float]) -> None:
        """Validate embedding vector dimension and values."""
        if len(vector) != self._config.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._config.embedding_dim}, "
                f"got {len(vector)}"
            )
        for i, v in enumerate(vector):
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"Embedding contains invalid value at index {i}: {v}")

    def validate_model(self) -> None:
        """Ensure the configured embedding model is available, pulling if needed."""
        from lilbee.model_manager import get_model_manager

        try:
            if not get_model_manager().is_installed(self._config.embedding_model):
                log.info("Pulling embedding model '%s'...", self._config.embedding_model)
                self._provider.pull_model(self._config.embedding_model, on_progress=lambda _: None)
        except (ConnectionError, OSError) as exc:
            raise RuntimeError(
                f"Cannot connect to embedding backend: {exc}. Is the server running?"
            ) from exc

    def embed(self, text: str) -> list[float]:
        """Embed a single text string, return vector."""
        vectors = self._provider.embed([self.truncate(text)])
        result: list[float] = vectors[0]
        self.validate_vector(result)
        return result

    def embed_batch(
        self,
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
        vectors: list[list[float]] = []
        batch: list[str] = []
        batch_chars = 0
        for text in texts:
            truncated = self.truncate(text)
            chunk_len = len(truncated)
            if batch and batch_chars + chunk_len > MAX_BATCH_CHARS:
                vectors.extend(self._provider.embed(batch))
                on_progress(
                    EventType.EMBED,
                    {"file": source, "chunk": len(vectors), "total_chunks": total_chunks},
                )
                batch = []
                batch_chars = 0
            batch.append(truncated)
            batch_chars += chunk_len
        if batch:
            vectors.extend(self._provider.embed(batch))
            on_progress(
                EventType.EMBED,
                {"file": source, "chunk": len(vectors), "total_chunks": total_chunks},
            )
        for vec in vectors:
            self.validate_vector(vec)
        return vectors


# ---------------------------------------------------------------------------
# Backwards-compatible module-level API (delegates to a default Embedder)
# ---------------------------------------------------------------------------

_default_embedder: Embedder | None = None


def _get_default_embedder() -> Embedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder(cfg, get_provider())
    return _default_embedder


def _reset_default_embedder() -> None:
    """Reset the cached default embedder (for tests that mutate cfg)."""
    global _default_embedder
    _default_embedder = None


def truncate(text: str) -> str:
    return _get_default_embedder().truncate(text)


def validate_vector(vector: list[float]) -> None:
    _get_default_embedder().validate_vector(vector)


def validate_model() -> None:
    _get_default_embedder().validate_model()


def embed(text: str) -> list[float]:
    return _get_default_embedder().embed(text)


def embed_batch(
    texts: list[str],
    *,
    source: str = "",
    on_progress: DetailedProgressCallback = noop_callback,
) -> list[list[float]]:
    return _get_default_embedder().embed_batch(texts, source=source, on_progress=on_progress)
