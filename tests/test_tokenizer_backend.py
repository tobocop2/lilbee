"""Tests for lilbee's xberg tokenizer backend (token-budgeted chunk sizing)."""

from __future__ import annotations

import pytest

from lilbee.data.extract.backends.tokenizer import LilbeeTokenizerBackend, _estimate_tokens
from lilbee.data.types import TokenizerBackendName


def test_name_is_lilbee():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 1)
    assert backend.name() == TokenizerBackendName.LILBEE


def test_initialize_and_shutdown_are_noops():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 1)
    assert backend.initialize() is None
    assert backend.shutdown() is None


def test_exact_count_is_returned():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 7)
    assert backend.count_tokens("hello world") == 7


def test_empty_text_is_zero_without_calling_count_fn():
    calls: list[str] = []

    def count_fn(text: str) -> int:
        calls.append(text)
        return 5

    backend = LilbeeTokenizerBackend(count_fn=count_fn)
    assert backend.count_tokens("") == 0
    assert calls == []


def test_not_implemented_falls_back_to_char_estimate():
    """SDK embedders expose no local tokenizer, so the count degrades to a guess."""

    def boom(_text: str) -> int:
        raise NotImplementedError("SDK backends have no local tokenizer")

    backend = LilbeeTokenizerBackend(count_fn=boom)
    text = "x" * 9
    assert backend.count_tokens(text) == _estimate_tokens(text)


def test_unexpected_error_raises_instead_of_estimating():
    """A count that fails after the embed-grade wait raises to the caller."""

    def boom(_text: str) -> int:
        raise RuntimeError("embedder unreachable")

    backend = LilbeeTokenizerBackend(count_fn=boom)
    with pytest.raises(RuntimeError, match="embedder unreachable"):
        backend.count_tokens("x" * 9)


def test_zero_count_for_non_empty_text_falls_back_to_estimate():
    """xberg rejects a zero count for non-empty text; degrade to the estimate."""
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 0)
    text = "x" * 9
    assert backend.count_tokens(text) == _estimate_tokens(text)


def test_estimate_is_conservative_and_at_least_one():
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("abc") == 1
    assert _estimate_tokens("a" * 9) == 3
