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


def test_call_from_thread_drops_when_the_node_has_no_reachable_app():
    """A worker outliving its screen must not raise out of the thread.

    Textual's ``node.app`` reads a contextvar that is unset in a plain thread and
    then walks ``node._parent``, which raises AttributeError on a node whose
    MessagePump state is gone. CI saw exactly that as "'ChatScreen' object has no
    attribute '_MessagePump__parent'", escaping a daemon thread and failing the
    run as a PytestUnhandledThreadExceptionWarning on both Linux and Windows.
    """
    class _NodeWithoutApp:
        """A real class, not a MagicMock: a mock's ``app`` resolves to a child
        mock whatever the type says, so the guarded branch is never reached and
        the test passes without exercising anything."""

        @property
        def app(self) -> object:
            raise AttributeError("'ChatScreen' object has no attribute '_MessagePump__parent'")

    fn = MagicMock()
    call_from_thread(_NodeWithoutApp(), fn, "arg")  # must not raise
    fn.assert_not_called()


def test_call_from_thread_propagates_an_attribute_error_raised_inside_the_callback():
    """The drop must not extend to AttributeError from *fn* itself.

    This is why resolving the app is guarded separately instead of adding
    AttributeError to the except around the call: a typo or a missing attribute
    inside the callback is a real bug and has to surface. Widening that except
    is the obvious one-line fix and it silently breaks this promise.
    """
    node = MagicMock()
    node.app.call_from_thread.side_effect = AttributeError("real bug inside fn")
    with pytest.raises(AttributeError, match="real bug inside fn"):
        call_from_thread(node, MagicMock(), "arg")
