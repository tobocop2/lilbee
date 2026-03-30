"""Thin wrapper around LLM provider embeddings API."""

import logging
import math

from lilbee.config import Config
from lilbee.progress import DetailedProgressCallback, EmbedEvent, EventType, noop_callback
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

    def validate_model(self) -> bool:
        """Check if the configured embedding model is available. No side effects."""
        return self.embedding_available()

    def embedding_available(self) -> bool:
        """Return True if the embedding model can be resolved by the active provider."""
        model = self._config.embedding_model
        if not model:
            return False
        try:
            available = self._provider.list_models()
            model_base = model.split(":")[0].lower()
            return any(model_base in m.lower() for m in available)
        except Exception:
            log.debug("embedding_available check failed", exc_info=True)
            return False

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
                    EmbedEvent(file=source, chunk=len(vectors), total_chunks=total_chunks),
                )
                batch = []
                batch_chars = 0
            batch.append(truncated)
            batch_chars += chunk_len
        if batch:
            vectors.extend(self._provider.embed(batch))
            on_progress(
                EventType.EMBED,
                EmbedEvent(file=source, chunk=len(vectors), total_chunks=total_chunks),
            )
        for vec in vectors:
            self.validate_vector(vec)
        return vectors
