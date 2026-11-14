"""Map liter-llm failures to ``ProviderError`` by exception type."""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.providers.base import ProviderError, ProviderErrorKind, is_prompt_length_overflow
from lilbee.providers.model_ref import ProviderModelRef
from lilbee.providers.roles import WorkerRole

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
# ``LiterLlmError`` directly), so this one table covers all providers. Seven classes
# are not in the table and classify as UNKNOWN (BudgetExceededError, HookRejectedError,
# IdempotencyConflictError, IdempotencyInFlightError, InternalError,
# InvalidHeaderError, SerializationError). Names, not
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
    "OutboundForbiddenError": ProviderErrorKind.BAD_REQUEST,
}

# Messages for single exception types that need their own wording. The SDK refuses a
# preset's connection to a blocked network with ``OutboundForbiddenError``.
_TYPE_MESSAGES: dict[str, str] = {
    "OutboundForbiddenError": (
        "The server for {model} resolves to an address that a server preset cannot use. "
        "Fix the URL of the preset or pick a different model."
    ),
}

# A server preset answers 404 when the server has no route at the base URL for the call.
_PRESET_NOT_FOUND = (
    "The server for {model} answered 404 to the {role} request at {url}. "
    "It does not serve that route there. "
    "Fix the base URL of the preset, or the route prefix of the server."
)

# Operation labels prefixed onto the fallback message for an unrecognised error.
_CHAT_FAILED = "Chat failed"
_EMBED_FAILED = "Embedding failed"
_RERANK_FAILED = "Rerank failed"


def _classify_error(exc: BaseException) -> ProviderErrorKind:
    """Map a liter-llm exception to a ``ProviderErrorKind`` by type.

    One bad request is read by message: a server that refuses a prompt past its
    window answers a plain 400 with no overflow type.
    """
    kind = _ERROR_KINDS.get(type(exc).__name__, ProviderErrorKind.UNKNOWN)
    if kind is ProviderErrorKind.BAD_REQUEST and is_prompt_length_overflow(str(exc)):
        return ProviderErrorKind.CONTEXT_OVERFLOW
    return kind


@dataclass(frozen=True)
class PresetCall:
    """The server preset a call goes to: the base URL the user typed and the role of the call."""

    base_url: str
    role: WorkerRole


def _preset_call(
    ref: ProviderModelRef, api_base: str | None, role: WorkerRole
) -> PresetCall | None:
    """The preset call of *ref*, or None for a hosted or local-server ref."""
    if not ref.is_preset or not api_base:
        return None
    return PresetCall(api_base, role)


def _provider_error(
    fallback: str, exc: Exception, model: str, *, preset: PresetCall | None = None
) -> ProviderError:
    """Wrap a liter-llm failure as a ``ProviderError`` classified by type.

    Recognised kinds get a blob-free, user-facing message; unrecognised ones
    keep the raw ``{fallback}: {exc}`` shape so nothing is lost when debugging.
    """
    kind = _classify_error(exc)
    if kind is ProviderErrorKind.NOT_FOUND and preset is not None:
        message = _PRESET_NOT_FOUND.format(model=model, role=preset.role, url=preset.base_url)
        return ProviderError(message, provider=_PROVIDER_NAME, kind=kind)
    template = _TYPE_MESSAGES.get(type(exc).__name__) or _KIND_MESSAGES.get(kind)
    message = template.format(model=model) if template is not None else f"{fallback}: {exc}"
    return ProviderError(message, provider=_PROVIDER_NAME, kind=kind)
