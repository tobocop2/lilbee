"""Shared rendering of Litestar body-validation failures for wire envelopes."""

from __future__ import annotations

import re

from litestar.exceptions import ValidationException

_BODY_KIND_PATTERN = re.compile(r"^Input should be a valid dictionary or instance of \w+$")
BODY_KIND_MESSAGE = "The request body must be a JSON object."


def plain_validation(exc: ValidationException) -> ValidationException:
    """Replace the message that names an internal model class with a plain one.

    A body that is not an object makes pydantic say "a valid dictionary or instance of
    <ModelName>", which leaks the name of a class a client never sees.
    """
    if isinstance(exc.extra, list):
        exc.extra = [
            {**item, "message": BODY_KIND_MESSAGE}
            if isinstance(item, dict) and _BODY_KIND_PATTERN.match(str(item.get("message", "")))
            else item
            for item in exc.extra
        ]
    return exc


def format_validation(exc: ValidationException) -> str:
    """Render a litestar/pydantic ValidationException as one user-facing string.

    Litestar wraps pydantic errors as ``{"key": "field_name", "message": "..."}``
    entries on ``exc.extra``; they flatten into a semicolon-joined string so the
    error envelope carries the same field names a client expects to see.
    """
    items: list[dict[str, str]] = exc.extra if isinstance(exc.extra, list) else []
    parts = [f"{err.get('key') or ''}: {err.get('message', '')}".lstrip(": ") for err in items]
    return "; ".join(parts) if parts else str(exc.detail)
