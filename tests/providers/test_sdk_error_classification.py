"""liter-llm failures are classified into ``ProviderErrorKind`` by exception type.

liter-llm is an optional extra, so a stand-in module with its exception classes
is injected. The classifier only reads the class name, so the real classes are
not required to verify the table.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.remote_sdk import (
    _ERROR_KINDS,
    _KIND_MESSAGES,
    RemoteSdkBackend,
    _classify_error,
    _provider_error,
)
from lilbee.providers.sdk_backend import CompletionRequest
from tests import _fake_sdk as fakes

# The kinds this backend can actually raise. Kinds that only a local engine can
# produce are excluded: a message for one here is unreachable text that reads as
# coverage, and the fleet surfaces those in its own words.
_SDK_KINDS = frozenset(
    {
        ProviderErrorKind.AUTH,
        ProviderErrorKind.NOT_FOUND,
        ProviderErrorKind.RATE_LIMIT,
        ProviderErrorKind.CONTEXT_OVERFLOW,
        ProviderErrorKind.BAD_REQUEST,
        ProviderErrorKind.CONNECTION,
        ProviderErrorKind.SERVER,
    }
)


def test_every_actionable_kind_has_a_user_message() -> None:
    # Adding a kind without a message would silently fall back to the raw error.
    for kind in _SDK_KINDS:
        assert kind in _KIND_MESSAGES, f"{kind} has no user-facing message"
        assert "{model}" in _KIND_MESSAGES[kind]
    assert ProviderErrorKind.UNKNOWN not in _KIND_MESSAGES


def test_the_table_carries_nothing_this_backend_cannot_raise() -> None:
    assert set(_KIND_MESSAGES) == set(_SDK_KINDS)


def test_every_table_entry_names_a_class_liter_llm_defines() -> None:
    assert _ERROR_KINDS
    assert set(_ERROR_KINDS) <= set(fakes.ERROR_NAMES)


@pytest.fixture
def sdk() -> Iterator[fakes.FakeSdk]:
    with fakes.installed(fakes.FakeSdk()) as fake:
        yield fake


@pytest.mark.parametrize(
    ("exc_name", "expected"),
    [
        ("AuthenticationError", ProviderErrorKind.AUTH),
        ("NotFoundError", ProviderErrorKind.NOT_FOUND),
        ("RateLimitedError", ProviderErrorKind.RATE_LIMIT),
        ("ProviderQuotaExceededError", ProviderErrorKind.RATE_LIMIT),
        ("ContextWindowExceededError", ProviderErrorKind.CONTEXT_OVERFLOW),
        ("BadRequestError", ProviderErrorKind.BAD_REQUEST),
        ("ContentPolicyError", ProviderErrorKind.BAD_REQUEST),
        ("EndpointNotSupportedError", ProviderErrorKind.BAD_REQUEST),
        ("LiterLlmTimeoutError", ProviderErrorKind.CONNECTION),
        ("NetworkError", ProviderErrorKind.CONNECTION),
        ("StreamingError", ProviderErrorKind.CONNECTION),
        ("ServiceUnavailableError", ProviderErrorKind.SERVER),
        ("ServerError", ProviderErrorKind.SERVER),
        ("InternalError", ProviderErrorKind.UNKNOWN),
    ],
)
def test_classifies_each_exception_type(
    sdk: fakes.FakeSdk, exc_name: str, expected: ProviderErrorKind
) -> None:
    assert _classify_error(sdk.error(exc_name)) == expected


def test_unrecognized_exception_is_unknown() -> None:
    assert _classify_error(RuntimeError("???")) == ProviderErrorKind.UNKNOWN


def test_provider_error_recognized_kind_drops_raw_blob(sdk: fakes.FakeSdk) -> None:
    # The raw SDK blob must not reach the user.
    blob = 'b\'{"error": {"code": 429, ...huge JSON dump...}}\''
    err = _provider_error(
        "Chat failed", sdk.error("RateLimitedError", blob), "gemini/gemini-3.1-pro"
    )
    assert err.kind == ProviderErrorKind.RATE_LIMIT
    assert err.provider == "remote"
    assert "gemini-3.1-pro" in str(err)
    assert "quota" in str(err).lower() or "rate" in str(err).lower()
    assert "lilbee" in str(err).lower()
    assert "huge JSON dump" not in str(err)


def test_provider_error_unknown_keeps_raw_fallback() -> None:
    err = _provider_error("Embedding failed", RuntimeError("nope"), "openai/text-embedding-3-small")
    assert err.kind == ProviderErrorKind.UNKNOWN
    assert str(err) == "Embedding failed: nope"


def test_provider_error_is_a_provider_error(sdk: fakes.FakeSdk) -> None:
    err = _provider_error(
        "Rerank failed", sdk.error("AuthenticationError", "bad key"), "cohere/rerank"
    )
    assert isinstance(err, ProviderError)
    assert err.kind == ProviderErrorKind.AUTH


def test_complete_surfaces_classified_message_end_to_end(sdk: fakes.FakeSdk) -> None:
    # A Gemini quota error driven all the way through backend.complete. The user
    # must see the clean rate-limit message, never the raw blob.
    sdk.chat_error = sdk.error(
        "ProviderQuotaExceededError", 'b\'{"error": {"code": 429, ...blob...}}\''
    )
    req = CompletionRequest(
        ref=parse_model_ref("gemini/gemini-3.1-pro"),
        messages=[{"role": "user", "content": "test"}],
    )
    with pytest.raises(ProviderError) as caught:
        RemoteSdkBackend().complete(req)
    assert caught.value.kind == ProviderErrorKind.RATE_LIMIT
    assert "gemini-3.1-pro" in str(caught.value)
    assert "blob" not in str(caught.value)
