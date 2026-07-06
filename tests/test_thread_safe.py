"""Tests for the thread-safe call_from_thread wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.cli.tui.thread_safe import call_from_thread


def test_call_from_thread_forwards_to_app():
    """Normal case: delegates to node.app.call_from_thread."""
    node = MagicMock()
    fn = MagicMock()
    call_from_thread(node, fn, 1, 2, key="val")
    node.app.call_from_thread.assert_called_once_with(fn, 1, 2, key="val")


def test_call_from_thread_swallows_shutdown_error():
    """When the app is shutting down, the call is silently dropped."""
    node = MagicMock()
    node.app.call_from_thread.side_effect = OSError("[Errno 9] Bad file descriptor")
    fn = MagicMock()
    call_from_thread(node, fn, "arg")  # should not raise


def test_call_from_thread_swallows_app_not_running():
    """RuntimeError (incl. NoActiveAppError) during shutdown is dropped, not raised."""
    node = MagicMock()
    node.app.call_from_thread.side_effect = RuntimeError("App is not running")
    call_from_thread(node, MagicMock(), "arg")  # should not raise


def test_call_from_thread_propagates_genuine_callback_error():
    """A real bug in the callback (e.g. KeyError) must surface, not be swallowed."""
    node = MagicMock()
    node.app.call_from_thread.side_effect = KeyError("missing")
    with pytest.raises(KeyError):
        call_from_thread(node, MagicMock(), "arg")
