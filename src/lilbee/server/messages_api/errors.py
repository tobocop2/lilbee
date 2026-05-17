"""Anthropic-shape error envelope and HTTP-status mapping for ``/v1/messages``."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class MessagesErrorType(StrEnum):
    """Anthropic Messages API error-type tags."""

    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    PERMISSION = "permission_error"
    NOT_FOUND = "not_found_error"
    RATE_LIMIT = "rate_limit_error"
    API = "api_error"
    OVERLOADED = "overloaded_error"


_STATUS_BY_TYPE: dict[MessagesErrorType, int] = {
    MessagesErrorType.INVALID_REQUEST: 400,
    MessagesErrorType.AUTHENTICATION: 401,
    MessagesErrorType.PERMISSION: 403,
    MessagesErrorType.NOT_FOUND: 404,
    MessagesErrorType.RATE_LIMIT: 429,
    MessagesErrorType.API: 500,
    MessagesErrorType.OVERLOADED: 529,
}


def messages_error_body(error_type: MessagesErrorType, message: str) -> dict[str, Any]:
    """Build the Anthropic error envelope: ``{"type": "error", "error": {...}}``."""
    return {
        "type": "error",
        "error": {"type": error_type.value, "message": message},
    }


def status_for_error_type(error_type: MessagesErrorType) -> int:
    """Return the HTTP status code Anthropic uses for *error_type*."""
    return _STATUS_BY_TYPE[error_type]
