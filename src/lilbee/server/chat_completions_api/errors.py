"""OpenAI-shaped error envelope for the chat-completions surface.

OpenAI client SDKs branch on ``error.type`` and ``error.code``. Adding a
new code means adding a :class:`CompletionsErrorCode` member and an entry
in :data:`COMPLETIONS_ERROR_TYPES`; the type system then forces every
call site to use the enum.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class CompletionsErrorCode(StrEnum):
    """Stable error-code vocabulary for the chat-completions surface."""

    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_DOES_NOT_SUPPORT_TOOLS = "model_does_not_support_tools"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    INVALID_API_KEY = "invalid_api_key"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INTERNAL_ERROR = "internal_error"


COMPLETIONS_ERROR_TYPES: dict[CompletionsErrorCode, str] = {
    CompletionsErrorCode.INVALID_REQUEST: "invalid_request_error",
    CompletionsErrorCode.MODEL_NOT_FOUND: "invalid_request_error",
    CompletionsErrorCode.MODEL_DOES_NOT_SUPPORT_TOOLS: "invalid_request_error",
    CompletionsErrorCode.CONTEXT_LENGTH_EXCEEDED: "invalid_request_error",
    CompletionsErrorCode.INVALID_API_KEY: "authentication_error",
    CompletionsErrorCode.RATE_LIMIT_EXCEEDED: "rate_limit_error",
    CompletionsErrorCode.INTERNAL_ERROR: "api_error",
}


def completions_error_body(code: CompletionsErrorCode, message: str) -> dict[str, Any]:
    """Build the JSON body for an OpenAI-shaped error response."""
    return {
        "error": {
            "message": message,
            "type": COMPLETIONS_ERROR_TYPES[code],
            "code": str(code),
        }
    }
