"""Thin wrapper around LLM provider embeddings API."""

import logging
import threading

import numpy as np

from lilbee.core.config import Config
from lilbee.core.vectors import Vector
from lilbee.data.extract.chunk import CHARS_PER_TOKEN
from lilbee.providers.base import LLMProvider
from lilbee.providers.model_ref import ProviderModelRef, parse_model_ref
from lilbee.retrieval.embedding_profiles import EmbeddingProfile, resolve_embedding_profile
from lilbee.runtime.progress import DetailedProgressCallback, EmbedEvent, EventType, noop_callback

log = logging.getLogger(__name__)


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


def is_model_installed(model: str) -> bool:
    """True if *model* resolves to a file on this machine.

    Provider-free by design: paint paths need an installed check that cannot
    build the services container (a cold build eager-starts the worker pool).
    """
    from lilbee.providers.engine_params import resolve_model_path

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
    return is_model_installed(model)


class Embedder:
    """Embedding wrapper: truncates, batches, validates vectors, and counts truncations."""

    def __init__(self, config: Config, provider: LLMProvider) -> None:
        self._config = config
        self._provider = provider
        self._truncated_total = 0
        self._truncated_lock = threading.Lock()

    @property
    def embed_char_budget(self) -> int:
        """Effective char limit, never below the chunker's max chunk size.

        ``max_embed_chars`` guards the embed model's context; clamping it up to
        ``chunk_size * CHARS_PER_TOKEN`` stops a finished full-budget chunk from
        silently losing its tail to a limit set below what the chunker emits.
        """
        return max(self._config.max_embed_chars, self._config.chunk_size * CHARS_PER_TOKEN)

    @property
    def batch_char_budget(self) -> int:
        """Per-request char cap: a full packed batch of maximum-size chunks.

        The target sequences-per-request is ``embed_batch_sequences`` (tunable to
        keep a multi-GPU fleet's batching slots full); the engine still re-splits
        to its physical batch, so it is an upper bound, not a guarantee.
        """
        return self._config.embed_batch_sequences * self._config.chunk_size * CHARS_PER_TOKEN

    @property
    def truncated_total(self) -> int:
        """Cumulative count of chunks truncated since process start (thread-safe read)."""
        with self._truncated_lock:
            return self._truncated_total

    def truncate(self, text: str, reserved: int = 0) -> str:
        """Truncate text to the embed char budget, counting any truncation.

        *reserved* is charged against the budget so an instruction prefix
        prepended afterwards still leaves the sent text inside the guard.
        """
        budget = max(1, self.embed_char_budget - reserved)
        if len(text) <= budget:
            return text
        log.debug("Truncating chunk from %d to %d chars for embedding", len(text), budget)
        with self._truncated_lock:
            self._truncated_total += 1
        return text[:budget]

    def validate_vector(self, vector: Vector) -> None:
        """Validate embedding vector dimension and values."""
        if len(vector) != self._config.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._config.embedding_dim}, "
                f"got {len(vector)}"
            )
        arr = np.asarray(vector, dtype=np.float64)
        bad = np.where(~np.isfinite(arr))[0]
        if bad.size:
            i = int(bad[0])
            raise ValueError(f"Embedding contains invalid value at index {i}: {vector[i]}")

    def validate_model(self) -> bool:
        """Availability gate for the startup and ingest paths: warns when it fails.

        Same probe as :meth:`embedding_available`, but this is the entry-point
        check whose whole job is to not fail silently -- a missing model here
        means every chunk of the run ahead will fail to embed, and the operator
        needs to hear that once, up front, rather than as a per-file error much
        later. Callers that can genuinely degrade (search falling back to
        keyword) ask :meth:`embedding_available` and stay quiet.
        """
        if self.embedding_available():
            return True
        if not self._config.embedding_model:
            # Unconfigured is a state, not a failure: nothing was chosen, so
            # there is nothing to warn about. One INFO line for the curious.
            log.info("No embedding model configured; embedding is off until one is set.")
            return False
        log.warning(
            "Embedding model %r is not available; embedding will fail. "
            "Pull it or set a different embedding_model.",
            self._config.embedding_model,
        )
        return False

    def embedding_available(self) -> bool:
        """Return True if the embedding model can be resolved.

        Checks the provider model list and the native registry path
        resolution. Returns True if either finds the model.
        """
        return is_model_available(self._config.embedding_model, self._provider)

    def _profile(self) -> EmbeddingProfile:
        """Instruction profile for the configured embedder (symmetric if unrecognized)."""
        return resolve_embedding_profile(self._config.embedding_model)

    def embed(self, text: str) -> Vector:
        """Embed a single document string, return vector."""
        return self._embed_with([text], self._profile().doc_prefix)[0]

    def embed_query(self, text: str) -> Vector:
        """Embed a single query string, applying the model's query instruction if any."""
        return self._embed_with([text], self._profile().query_instruction)[0]

    def embed_batch(
        self,
        texts: list[str],
        *,
        source: str = "",
        on_progress: DetailedProgressCallback = noop_callback,
    ) -> list[Vector]:
        """Embed document texts with adaptive batching, return list of vectors."""
        return self._embed_with(
            texts, self._profile().doc_prefix, source=source, on_progress=on_progress
        )

    def embed_query_batch(
        self,
        texts: list[str],
        *,
        source: str = "",
        on_progress: DetailedProgressCallback = noop_callback,
    ) -> list[Vector]:
        """Embed query texts (query instruction applied), adaptive batching."""
        return self._embed_with(
            texts, self._profile().query_instruction, source=source, on_progress=on_progress
        )

    def _embed_with(
        self,
        texts: list[str],
        prefix: str,
        *,
        source: str = "",
        on_progress: DetailedProgressCallback = noop_callback,
    ) -> list[Vector]:
        """Adaptive-batched embed of *texts*, each prefixed (query/document instruction).

        Fires ``embed`` progress events per batch when *on_progress* is provided.
        """
        if not texts:
            return []
        total_chunks = len(texts)
        max_batch_chars = self.batch_char_budget
        vectors: list[Vector] = []
        batch: list[str] = []
        batch_chars = 0
        for text in texts:
            # The prefix is charged against the budget: it is part of what the
            # embed model receives, so counting only the text would ship
            # budget+len(prefix) chars and lose the tail the clamp protects.
            truncated = prefix + self.truncate(text, reserved=len(prefix))
            chunk_len = len(truncated)
            if batch and batch_chars + chunk_len > max_batch_chars:
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
