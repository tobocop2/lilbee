"""Tests for the thread-safe call_from_thread wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

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
