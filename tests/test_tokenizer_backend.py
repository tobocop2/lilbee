"""Tests for lilbee's xberg tokenizer backend (token-budgeted chunk sizing)."""

from __future__ import annotations

import pytest

from lilbee.data.extract.backends.tokenizer import (
    _XBERG_PROBE_TEXT,
    LilbeeTokenizerBackend,
    _estimate_tokens,
)
from lilbee.data.types import TokenizerBackendName


def test_name_is_lilbee():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 1)
    assert backend.name() == TokenizerBackendName.LILBEE


def test_initialize_and_shutdown_are_noops():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 1)
    assert backend.initialize() is None
    assert backend.shutdown() is None


def test_a_probe_answer_taken_by_another_thread_still_returns_a_count():
    """The answer goes to one reader; a reader that finds it gone counts, and never gets None."""

    class _TakenAfterFirstRead(LilbeeTokenizerBackend):
        reads = 0

        @property
        def _probe_count(self) -> int | None:
            self.reads += 1
            return 5 if self.reads == 1 else None

        @_probe_count.setter
        def _probe_count(self, value: int | None) -> None: ...

    backend = _TakenAfterFirstRead(count_fn=lambda _t: 9)
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 5
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 9


def test_exact_count_is_returned():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 7)
    assert backend.count_tokens("hello world") == 7


def test_probe_raises_a_count_failure():
    def count_fn(_text: str) -> int:
        raise RuntimeError("engine stopped")

    backend = LilbeeTokenizerBackend(count_fn=count_fn)
    with pytest.raises(RuntimeError, match="engine stopped"):
        backend.probe()


def test_probe_answers_only_the_probe_text_without_the_provider():
    counts = iter([3, 9])
    calls: list[str] = []

    def count_fn(text: str) -> int:
        calls.append(text)
        return next(counts)

    backend = LilbeeTokenizerBackend(count_fn=count_fn)
    backend.probe()
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 3
    assert backend.count_tokens("hello world") == 9
    assert calls == [_XBERG_PROBE_TEXT, "hello world"]


def test_probe_answer_is_used_once_then_the_provider_counts_again():
    counts = iter([3, 7])
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: next(counts))
    backend.probe()
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 3
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 7


def test_probe_text_routes_to_the_provider_before_a_probe():
    backend = LilbeeTokenizerBackend(count_fn=lambda _t: 4)
    assert backend.count_tokens(_XBERG_PROBE_TEXT) == 4


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
