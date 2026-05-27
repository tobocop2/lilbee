"""Error envelope and code/type taxonomy for the chat-completions surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.server.chat_dispatch.dispatch import (
    ModelDoesNotSupportToolsError,
    ModelNotFoundError,
)


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
    """Build the JSON body for an error response."""
    return {
        "error": {
            "message": message,
            "type": COMPLETIONS_ERROR_TYPES[code],
            "code": str(code),
        }
    }


@dataclass(frozen=True)
class ClassifiedError:
    """A typed provider/dispatch failure mapped to its client-facing shape."""

    http_status: int
    code: CompletionsErrorCode
    message: str


_PROVIDER_KIND_CLASSIFICATIONS: dict[ProviderErrorKind, tuple[int, CompletionsErrorCode]] = {
    ProviderErrorKind.CONTEXT_OVERFLOW: (400, CompletionsErrorCode.CONTEXT_LENGTH_EXCEEDED),
    ProviderErrorKind.NOT_FOUND: (404, CompletionsErrorCode.MODEL_NOT_FOUND),
}


def classify_provider_error(exc: BaseException) -> ClassifiedError | None:
    """Map a typed dispatch/provider failure to ``(status, code, message)``, or None.

    Returns None for any exception that isn't a recognized typed dispatch error
    or a ProviderError with a client-mappable kind; callers apply their own
    generic fallback (internal_error 500 / service-unavailable 503).
    """
    if isinstance(exc, ModelNotFoundError):
        return ClassifiedError(404, CompletionsErrorCode.MODEL_NOT_FOUND, str(exc))
    if isinstance(exc, ModelDoesNotSupportToolsError):
        return ClassifiedError(400, CompletionsErrorCode.MODEL_DOES_NOT_SUPPORT_TOOLS, str(exc))
    if isinstance(exc, ProviderError):
        mapped = _PROVIDER_KIND_CLASSIFICATIONS.get(exc.kind)
        if mapped is not None:
            status, code = mapped
            return ClassifiedError(status, code, str(exc))
    return None
