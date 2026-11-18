"""How an ingest failure is recorded against the file that raised it."""

from __future__ import annotations

from lilbee.providers.base import ProviderError

_OVERFLOW_REASON = (
    "A piece of this file is longer than the embedding model's context window. "
    "Set a smaller chunk_size, then run 'lilbee sync --retry-skipped'"
)


def error_reason(error: BaseException) -> str:
    """The reason recorded for a failed file: ``Type: message``, or ingest words for an overflow."""
    if isinstance(error, ProviderError) and error.repeats_on_retry:
        return _OVERFLOW_REASON
    return f"{type(error).__name__}: {error}"
