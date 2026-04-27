"""Thin wrapper around LLM provider embeddings API."""

import logging
import math

from lilbee.core.config import Config
from lilbee.progress import DetailedProgressCallback, EmbedEvent, EventType, noop_callback
from lilbee.providers.base import LLMProvider
from lilbee.providers.model_ref import ProviderModelRef, parse_model_ref

log = logging.getLogger(__name__)

MAX_BATCH_CHARS = 6000


def _name_base(ref: ProviderModelRef) -> str:
    return ref.name.split(":")[0].lower().replace(" ", "-")


def _remote_sees_model(ref: ProviderModelRef, provider: LLMProvider) -> bool:
    try:
        available = provider.list_models()
    except Exception:
        log.debug("provider list_models failed during availability check", exc_info=True)
        return False
    base = _name_base(ref)
    return any(base in m.lower().replace(" ", "-") for m in available)


def _native_has_model(model: str) -> bool:
    from lilbee.providers.llama_cpp_provider import resolve_model_path

    try:
        resolve_model_path(model)
    except Exception:
        return False
    return True


def is_model_available(model: str, provider: LLMProvider) -> bool:
    """Return True if *model* resolves via *provider* or the native registry.

    Remote-prefixed refs (``ollama/`` and API providers) skip the native
    probe since they resolve through the SDK backend at call time.
    """
    if not model:
        return False
    ref = parse_model_ref(model)
    if _remote_sees_model(ref, provider):
        return True
    if ref.is_remote:
        return False
    return _native_has_model(model)


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
        """Return True if the embedding model can be resolved.

        Checks the provider model list and the native registry path
        resolution. Returns True if either finds the model.
        """
        return is_model_available(self._config.embedding_model, self._provider)

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
