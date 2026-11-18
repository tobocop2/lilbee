"""How an ingest failure is recorded against the file that raised it."""

from __future__ import annotations

from lilbee.providers.base import EmbeddingOverflowError

OVERFLOW_REASON = (
    "A piece of this file is longer than the embedding model's context window. "
    "Set a smaller chunk_size or use a model with a larger window; "
    "the next sync then tries the file again."
)


def error_reason(error: BaseException) -> str:
    """The reason recorded for a failed file: ``Type: message``, or ingest words for an overflow."""
    if isinstance(error, EmbeddingOverflowError):
        return OVERFLOW_REASON
    return f"{type(error).__name__}: {error}"


def log_reason(error: BaseException) -> str:
    """The reason logged for a failed file: the error text, or the overflow reason."""
    return error_reason(error) if isinstance(error, EmbeddingOverflowError) else str(error)
