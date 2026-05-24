"""litellm failures are classified into ``ProviderErrorKind`` by exception type.

litellm is an optional extra, so a fake module mirroring its exception hierarchy
is injected. The classifier only does MRO/isinstance checks, so the real litellm
classes aren't required to verify the mapping.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from unittest import mock

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.litellm_sdk import _KIND_MESSAGES, _classify_litellm_error, _provider_error


def test_every_actionable_kind_has_a_user_message() -> None:
    # Adding a kind without a message would silently fall back to the raw error.
    for kind in ProviderErrorKind:
        if kind is ProviderErrorKind.UNKNOWN:
            assert kind not in _KIND_MESSAGES
        else:
            assert kind in _KIND_MESSAGES, f"{kind} has no user-facing message"
            assert "{model}" in _KIND_MESSAGES[kind]


def _fake_litellm() -> types.ModuleType:
    ns = types.ModuleType("litellm")
    ns.APIError = type("APIError", (Exception,), {})
    ns.AuthenticationError = type("AuthenticationError", (ns.APIError,), {})
    ns.PermissionDeniedError = type("PermissionDeniedError", (ns.APIError,), {})
    ns.NotFoundError = type("NotFoundError", (ns.APIError,), {})
    ns.RateLimitError = type("RateLimitError", (ns.APIError,), {})
    ns.BadRequestError = type("BadRequestError", (ns.APIError,), {})
    # Mirrors litellm: ContextWindowExceededError subclasses BadRequestError.
    ns.ContextWindowExceededError = type("ContextWindowExceededError", (ns.BadRequestError,), {})
    ns.Timeout = type("Timeout", (ns.APIError,), {})
    ns.APIConnectionError = type("APIConnectionError", (ns.APIError,), {})
    ns.ServiceUnavailableError = type("ServiceUnavailableError", (ns.APIError,), {})
    ns.InternalServerError = type("InternalServerError", (ns.APIError,), {})
    return ns


@pytest.fixture
def fake_litellm() -> Iterator[types.ModuleType]:
    ns = _fake_litellm()
    with mock.patch.dict(sys.modules, {"litellm": ns}):
        yield ns


@pytest.mark.parametrize(
    ("exc_name", "expected"),
    [
        ("AuthenticationError", ProviderErrorKind.AUTH),
        ("PermissionDeniedError", ProviderErrorKind.AUTH),
        ("NotFoundError", ProviderErrorKind.NOT_FOUND),
        ("RateLimitError", ProviderErrorKind.RATE_LIMIT),
        ("ContextWindowExceededError", ProviderErrorKind.CONTEXT_OVERFLOW),
        ("BadRequestError", ProviderErrorKind.BAD_REQUEST),
        ("Timeout", ProviderErrorKind.CONNECTION),
        ("APIConnectionError", ProviderErrorKind.CONNECTION),
        ("ServiceUnavailableError", ProviderErrorKind.SERVER),
        ("InternalServerError", ProviderErrorKind.SERVER),
    ],
)
def test_classifies_each_exception_type(
    fake_litellm: types.ModuleType, exc_name: str, expected: ProviderErrorKind
) -> None:
    exc = getattr(fake_litellm, exc_name)("boom")
    assert _classify_litellm_error(exc) == expected


def test_context_overflow_beats_bad_request_base(fake_litellm: types.ModuleType) -> None:
    # ContextWindowExceededError subclasses BadRequestError; the MRO walk must
    # return the most specific kind, not the base.
    exc = fake_litellm.ContextWindowExceededError("too long")
    assert _classify_litellm_error(exc) == ProviderErrorKind.CONTEXT_OVERFLOW


def test_unrecognized_exception_is_unknown(fake_litellm: types.ModuleType) -> None:
    assert _classify_litellm_error(RuntimeError("???")) == ProviderErrorKind.UNKNOWN


def test_mid_stream_fallback_unwraps_to_root_cause(fake_litellm: types.ModuleType) -> None:
    # litellm wraps the real cause (a 429) inside a ServiceUnavailableError-shaped
    # fallback. Classifying the wrapper alone would mislabel it SERVER; the cause
    # chain must surface the underlying rate limit instead.
    inner = fake_litellm.RateLimitError("quota exceeded")
    outer = fake_litellm.ServiceUnavailableError("fallbacks exhausted")
    outer.original_exception = inner  # type: ignore[attr-defined]
    assert _classify_litellm_error(outer) == ProviderErrorKind.RATE_LIMIT


def test_provider_error_recognized_kind_drops_raw_blob(fake_litellm: types.ModuleType) -> None:
    # The whole point of the fix: the raw SDK blob must not reach the user.
    blob = 'b\'{"error": {"code": 429, ...huge JSON dump...}}\''
    err = _provider_error("Chat failed", fake_litellm.RateLimitError(blob), "gemini/gemini-3.1-pro")
    assert err.kind == ProviderErrorKind.RATE_LIMIT
    assert err.provider == "litellm"
    assert "gemini-3.1-pro" in str(err)
    assert "quota" in str(err).lower() or "rate" in str(err).lower()
    assert "lilbee" in str(err).lower()
    assert "huge JSON dump" not in str(err)


def test_provider_error_unknown_keeps_raw_fallback(fake_litellm: types.ModuleType) -> None:
    err = _provider_error("Embedding failed", RuntimeError("nope"), "openai/text-embedding-3-small")
    assert err.kind == ProviderErrorKind.UNKNOWN
    assert str(err) == "Embedding failed: nope"


def test_provider_error_is_a_provider_error(fake_litellm: types.ModuleType) -> None:
    err = _provider_error(
        "Rerank failed", fake_litellm.AuthenticationError("bad key"), "cohere/rerank"
    )
    assert isinstance(err, ProviderError)
    assert err.kind == ProviderErrorKind.AUTH


def test_complete_surfaces_classified_message_end_to_end(fake_litellm: types.ModuleType) -> None:
    # The reported bug: a Gemini 429 wrapped in a mid-stream fallback, driven
    # all the way through backend.complete. The user must see the clean
    # rate-limit message, never the raw blob.
    from lilbee.providers.litellm_sdk import LitellmSdkBackend
    from lilbee.providers.model_ref import parse_model_ref
    from lilbee.providers.sdk_backend import CompletionRequest

    inner = fake_litellm.RateLimitError('b\'{"error": {"code": 429, ...blob...}}\'')
    outer = fake_litellm.ServiceUnavailableError("fallbacks exhausted")
    outer.original_exception = inner  # type: ignore[attr-defined]
    fake_litellm.completion = mock.MagicMock(side_effect=outer)

    backend = LitellmSdkBackend()
    req = CompletionRequest(
        ref=parse_model_ref("gemini/gemini-3.1-pro"),
        messages=[{"role": "user", "content": "test"}],
    )
    with pytest.raises(ProviderError) as caught:
        backend.complete(req)
    assert caught.value.kind == ProviderErrorKind.RATE_LIMIT
    assert "gemini-3.1-pro" in str(caught.value)
    assert "blob" not in str(caught.value)
