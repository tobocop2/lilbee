"""Map liter-llm failures to ``ProviderError`` by exception type."""

from __future__ import annotations

from lilbee.providers.base import ProviderError, ProviderErrorKind

_PROVIDER_NAME = "remote"

# User-facing message per recognised error kind. Each names the problem against
# {model} and makes clear the cause sits with the user's provider account or
# network, not with lilbee. UNKNOWN has no entry and falls back to the raw error.
_KIND_MESSAGES: dict[ProviderErrorKind, str] = {
    ProviderErrorKind.RATE_LIMIT: (
        "{model} is rate-limited or out of quota. That's a limit on your provider "
        "API key, not a lilbee problem. Check your plan and billing with the "
        "provider, or pick a different model."
    ),
    ProviderErrorKind.AUTH: (
        "{model} rejected your API key. Check that the key is set correctly and has "
        "access to this model. That's between your key and the provider, not a lilbee problem."
    ),
    ProviderErrorKind.NOT_FOUND: (
        "The provider doesn't offer {model} on your account. "
        "Pick a different model or check the name."
    ),
    ProviderErrorKind.CONTEXT_OVERFLOW: (
        "This conversation is too long for {model}'s context window. "
        "Start a new chat or pick a model with a larger context."
    ),
    ProviderErrorKind.BAD_REQUEST: (
        "The provider rejected the request for {model}. Check the model name and your settings."
    ),
    ProviderErrorKind.CONNECTION: (
        "Couldn't reach the provider for {model}, or it timed out. Check your "
        "connection and base URL, then try again or pick a different model."
    ),
    ProviderErrorKind.SERVER: (
        "The provider for {model} is unavailable right now. That's on the provider's "
        "side, not a lilbee problem. Try again shortly or pick a different model."
    ),
}

# liter-llm exception class name -> error kind. Every HTTP status maps to one
# typed class in liter-llm, and the classes are flat (each subclasses
# ``LiterLlmError`` directly), so this one table covers all providers. Names, not
# classes, so the table stays importable without the native module.
_ERROR_KINDS: dict[str, ProviderErrorKind] = {
    "AuthenticationError": ProviderErrorKind.AUTH,
    "NotFoundError": ProviderErrorKind.NOT_FOUND,
    "RateLimitedError": ProviderErrorKind.RATE_LIMIT,
    "ProviderQuotaExceededError": ProviderErrorKind.RATE_LIMIT,
    "ContextWindowExceededError": ProviderErrorKind.CONTEXT_OVERFLOW,
    "BadRequestError": ProviderErrorKind.BAD_REQUEST,
    "ContentPolicyError": ProviderErrorKind.BAD_REQUEST,
    "EndpointNotSupportedError": ProviderErrorKind.BAD_REQUEST,
    "LiterLlmTimeoutError": ProviderErrorKind.CONNECTION,
    "NetworkError": ProviderErrorKind.CONNECTION,
    "StreamingError": ProviderErrorKind.CONNECTION,
    "ServiceUnavailableError": ProviderErrorKind.SERVER,
    "ServerError": ProviderErrorKind.SERVER,
}

# Operation labels prefixed onto the fallback message for an unrecognised error.
_CHAT_FAILED = "Chat failed"
_EMBED_FAILED = "Embedding failed"
_RERANK_FAILED = "Rerank failed"


def _classify_error(exc: BaseException) -> ProviderErrorKind:
    """Map a liter-llm exception to a ``ProviderErrorKind`` by type, never by message."""
    return _ERROR_KINDS.get(type(exc).__name__, ProviderErrorKind.UNKNOWN)


def _provider_error(fallback: str, exc: Exception, model: str) -> ProviderError:
    """Wrap a liter-llm failure as a ``ProviderError`` classified by type.

    Recognised kinds get a blob-free, user-facing message; unrecognised ones
    keep the raw ``{fallback}: {exc}`` shape so nothing is lost when debugging.
    """
    kind = _classify_error(exc)
    template = _KIND_MESSAGES.get(kind)
    message = template.format(model=model) if template is not None else f"{fallback}: {exc}"
    return ProviderError(message, provider=_PROVIDER_NAME, kind=kind)
