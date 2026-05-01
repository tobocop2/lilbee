"""Tests for the process-wide abort flag wired into llama_cpp.Llama."""

from __future__ import annotations

import pytest

from lilbee.providers.llama_cpp.abort_signal import (
    abort_callback,
    clear_abort,
    is_abort_set,
    request_abort,
)


@pytest.fixture(autouse=True)
def _reset_abort_flag() -> None:
    """Each test starts and ends with a cleared flag."""
    clear_abort()
    yield
    clear_abort()


def test_request_abort_sets_flag() -> None:
    """request_abort() flips the flag to True."""
    assert is_abort_set() is False
    request_abort()
    assert is_abort_set() is True


def test_clear_abort_resets_flag() -> None:
    """clear_abort() flips the flag back to False."""
    request_abort()
    assert is_abort_set() is True
    clear_abort()
    assert is_abort_set() is False


def test_abort_callback_returns_flag_state() -> None:
    """abort_callback() mirrors is_abort_set() for ggml's poll loop."""
    assert abort_callback() is False
    request_abort()
    assert abort_callback() is True
    clear_abort()
    assert abort_callback() is False


def test_abort_callback_signature() -> None:
    """ggml passes a user_data pointer; abort_callback accepts and ignores it."""
    assert abort_callback(None) is False
    assert abort_callback(object()) is False
    request_abort()
    assert abort_callback(None) is True
    assert abort_callback("anything") is True
