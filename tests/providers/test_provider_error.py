"""Tests for ProviderError kinds and the actionable hint surfaced via str()."""

from __future__ import annotations

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind

_ACTIONABLE = [k for k in ProviderErrorKind if k != ProviderErrorKind.UNKNOWN]


def test_default_kind_is_unknown_with_no_hint() -> None:
    err = ProviderError("boom")
    assert err.kind == ProviderErrorKind.UNKNOWN
    assert err.hint == ""
    assert str(err) == "boom"
    assert err.message == "boom"


@pytest.mark.parametrize("kind", _ACTIONABLE)
def test_actionable_kinds_append_a_hint_to_str(kind: ProviderErrorKind) -> None:
    err = ProviderError("boom", provider="litellm", kind=kind)
    assert err.hint  # every actionable kind has guidance
    assert str(err) == f"boom {err.hint}"
    assert err.provider == "litellm"


def test_raw_message_is_preserved_separately_from_str() -> None:
    err = ProviderError("raw message", kind=ProviderErrorKind.AUTH)
    assert err.message == "raw message"
    assert str(err) != err.message  # str() carries the extra hint
    assert err.message in str(err)
