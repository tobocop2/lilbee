"""How an ingest failure is recorded against the file that raised it."""

from __future__ import annotations


def error_reason(error: BaseException) -> str:
    """The ``Type: message`` reason recorded for a failed file."""
    return f"{type(error).__name__}: {error}"
