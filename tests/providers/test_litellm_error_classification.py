"""Tests that litellm exceptions are classified into ProviderErrorKind by type.

litellm is an optional extra, so a fake module mirroring litellm's exception
hierarchy is injected; the classifier only does isinstance/MRO checks, so real
litellm classes are not required to verify the mapping.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from lilbee.providers.base import ProviderErrorKind
from lilbee.providers.litellm_sdk import _classify_litellm_error, _provider_error


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
def fake_litellm() -> types.ModuleType:
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


def test_provider_error_carries_kind_and_hint(fake_litellm: types.ModuleType) -> None:
    err = _provider_error("Chat", fake_litellm.AuthenticationError("bad key"))
    assert err.kind == ProviderErrorKind.AUTH
    assert err.provider == "litellm"
    assert "Chat failed: bad key" in str(err)
    assert err.hint and err.hint in str(err)
