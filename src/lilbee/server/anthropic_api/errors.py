"""Anthropic error envelope over the shared provider-error classification."""

from __future__ import annotations

from typing import Any

from lilbee.server.chat_completions_api.errors import CompletionsErrorCode

_FALLBACK_ERROR_TYPE = "invalid_request_error"

# The status/code classification is shared with the completions surface
# (classify_provider_error); only the envelope vocabulary differs.
_ANTHROPIC_ERROR_TYPES: dict[CompletionsErrorCode, str] = {
    CompletionsErrorCode.INVALID_REQUEST: "invalid_request_error",
    CompletionsErrorCode.MODEL_NOT_FOUND: "not_found_error",
    CompletionsErrorCode.MODEL_DOES_NOT_SUPPORT_TOOLS: "invalid_request_error",
    CompletionsErrorCode.CONTEXT_LENGTH_EXCEEDED: "invalid_request_error",
    CompletionsErrorCode.INVALID_API_KEY: "authentication_error",
    CompletionsErrorCode.RATE_LIMIT_EXCEEDED: "rate_limit_error",
    CompletionsErrorCode.INTERNAL_ERROR: "api_error",
}


def anthropic_error_type(code: CompletionsErrorCode) -> str:
    """The Anthropic ``error.type`` for a shared error code."""
    # .get, not a subscript: a new enum member must degrade to a handled 4xx
    # envelope, not surface as a 500.
    return _ANTHROPIC_ERROR_TYPES.get(code, _FALLBACK_ERROR_TYPE)


def anthropic_error_body(error_type: str, message: str) -> dict[str, Any]:
    """Build the Anthropic JSON error envelope."""
    return {"type": "error", "error": {"type": error_type, "message": message}}
