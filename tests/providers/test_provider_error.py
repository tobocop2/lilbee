"""ProviderError carries a provider-agnostic ``kind`` for callers to branch on."""

from __future__ import annotations

from lilbee.providers.base import ProviderError, ProviderErrorKind


def test_default_kind_is_unknown() -> None:
    err = ProviderError("boom")
    assert err.kind == ProviderErrorKind.UNKNOWN
    assert err.provider == ""
    assert str(err) == "boom"


def test_kind_and_provider_are_carried() -> None:
    err = ProviderError("rate limited", provider="litellm", kind=ProviderErrorKind.RATE_LIMIT)
    assert err.kind == ProviderErrorKind.RATE_LIMIT
    assert err.provider == "litellm"
    # The message is unchanged; kind is metadata, not appended to str().
    assert str(err) == "rate limited"
