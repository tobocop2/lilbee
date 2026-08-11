"""Shared rendering of Litestar body-validation failures for wire envelopes."""

from __future__ import annotations

from litestar.exceptions import ValidationException


def format_validation(exc: ValidationException) -> str:
    """Render a litestar/pydantic ValidationException as one user-facing string.

    Litestar wraps pydantic errors as ``{"key": "field_name", "message": "..."}``
    entries on ``exc.extra``; they flatten into a semicolon-joined string so the
    error envelope carries the same field names a client expects to see.
    """
    items: list[dict[str, str]] = exc.extra if isinstance(exc.extra, list) else []
    parts = [f"{err.get('key') or ''}: {err.get('message', '')}".lstrip(": ") for err in items]
    return "; ".join(parts) if parts else str(exc.detail)
