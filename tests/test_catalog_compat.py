"""Unit tests for catalog.compat: classify(), SUPPORTED_ARCHS, UnsupportedArchError."""

from __future__ import annotations

import pytest

from gguf import MODEL_ARCH_NAMES

from lilbee.catalog.compat import (
    SUPPORTED_ARCHS,
    UnsupportedArchError,
    classify,
)
from lilbee.catalog.types import ModelCompat


def test_supported_archs_includes_llama() -> None:
    assert "llama" in SUPPORTED_ARCHS


def test_supported_archs_is_frozenset() -> None:
    assert isinstance(SUPPORTED_ARCHS, frozenset)


def test_supported_archs_nonempty() -> None:
    assert len(SUPPORTED_ARCHS) > 50


def test_classify_known_supported() -> None:
    assert classify("llama") is ModelCompat.SUPPORTED


def test_classify_gemma4_supported() -> None:
    # The pinned engine serves gemma4, but the bundled gguf package predates it
    # (MODEL_ARCH_NAMES stops at gemma3n), so it must be allowlisted explicitly.
    assert "gemma4" not in MODEL_ARCH_NAMES.values()
    assert classify("gemma4") is ModelCompat.SUPPORTED


def test_classify_unknown_string_is_unsupported() -> None:
    assert classify("this-arch-will-never-exist") is ModelCompat.UNSUPPORTED


def test_classify_empty_is_unknown() -> None:
    assert classify("") is ModelCompat.UNKNOWN


def test_unsupported_arch_error_carries_fields() -> None:
    err = UnsupportedArchError("acme/foo-GGUF", "kimi_k2")
    assert err.ref == "acme/foo-GGUF"
    assert err.architecture == "kimi_k2"
    assert "kimi_k2" in str(err)
    assert "acme/foo-GGUF" in str(err)


def test_unsupported_arch_error_is_exception() -> None:
    with pytest.raises(UnsupportedArchError):
        raise UnsupportedArchError("a", "b")
