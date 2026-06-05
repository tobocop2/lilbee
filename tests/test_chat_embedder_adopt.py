"""Tests for the TUI embedder-adopt flow.

When a query hits a downloaded index built with a different embedder, the
chat screen offers to adopt that embedder (same dim) instead of failing with
a generic stream error, and explains the rebuild path when the dims differ.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, PropertyMock, patch

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.data.store import EmbeddingModelMismatchError


def _mismatch(*, dims_match: bool = True) -> EmbeddingModelMismatchError:
    return EmbeddingModelMismatchError(
        persisted_model="orgA/repoA/built.gguf",
        persisted_dim=768,
        current_model="orgB/repoB/configured.gguf",
        current_dim=768 if dims_match else 384,
    )


def _bare_screen() -> ChatScreen:
    screen = ChatScreen.__new__(ChatScreen)
    screen.notify = MagicMock()  # type: ignore[method-assign]
    return screen


@contextmanager
def _fake_app():
    """Patch the read-only ``app`` property so detached screens can push modals."""
    app = MagicMock()
    with patch.object(ChatScreen, "app", new_callable=PropertyMock, return_value=app):
        yield app


class TestStreamResponseDispatch:
    """The stream body routes an embedder mismatch to the adopt prompt and any
    other error to the generic stream-error message."""

    def _screen(self):
        screen = _bare_screen()
        screen._history = [{"role": "user", "content": "q"}]
        screen._history_lock = MagicMock()
        screen._history_lock.__enter__ = MagicMock(return_value=None)
        screen._history_lock.__exit__ = MagicMock(return_value=False)
        screen._on_embedding_mismatch = MagicMock()  # type: ignore[method-assign]
        screen._finalize_stream = MagicMock()  # type: ignore[method-assign]
        return screen

    def test_mismatch_routes_to_adopt_prompt(self):
        screen = self._screen()
        widget = MagicMock()
        services = MagicMock()
        services.searcher.ask_stream.side_effect = _mismatch()
        with (
            patch("lilbee.cli.tui.screens.chat.get_services", return_value=services),
            patch(
                "lilbee.cli.tui.screens.chat.call_from_thread",
                side_effect=lambda _node, fn, *a, **k: fn(*a, **k),
            ),
        ):
            screen._do_stream_response("q", widget, None)
        screen._on_embedding_mismatch.assert_called_once()
        widget.append_content.assert_not_called()

    def test_other_error_shows_stream_error(self):
        screen = self._screen()
        widget = MagicMock()
        services = MagicMock()
        services.searcher.ask_stream.side_effect = RuntimeError("kaboom")
        with (
            patch("lilbee.cli.tui.screens.chat.get_services", return_value=services),
            patch(
                "lilbee.cli.tui.screens.chat.call_from_thread",
                side_effect=lambda _node, fn, *a, **k: fn(*a, **k),
            ),
        ):
            screen._do_stream_response("q", widget, None)
        screen._on_embedding_mismatch.assert_not_called()
        assert any("kaboom" in str(c.args) for c in widget.append_content.call_args_list)


class TestOnEmbeddingMismatch:
    def test_adoptable_pushes_confirm_and_notes(self):
        screen = _bare_screen()
        widget = MagicMock()
        with _fake_app() as app:
            screen._on_embedding_mismatch(_mismatch(), "what is X?", widget)
        widget.append_content.assert_called_once()
        assert "orgA/repoA/built.gguf" in widget.append_content.call_args.args[0]
        app.push_screen.assert_called_once()

    def test_dim_incompatible_explains_rebuild_no_prompt(self):
        screen = _bare_screen()
        widget = MagicMock()
        with _fake_app() as app:
            screen._on_embedding_mismatch(_mismatch(dims_match=False), "q", widget)
        assert "768" in widget.append_content.call_args.args[0]
        app.push_screen.assert_not_called()


class TestOnAdoptConfirm:
    def test_cancel_notifies_and_does_not_retry(self):
        screen = _bare_screen()
        screen._adopt_and_retry = MagicMock()  # type: ignore[method-assign]
        screen._on_adopt_confirm(False, "orgA/repoA/built.gguf", "q")
        screen.notify.assert_called_once_with(msg.EMBED_ADOPT_CANCELLED)
        screen._adopt_and_retry.assert_not_called()

    def test_confirm_kicks_off_adopt_and_retry(self):
        screen = _bare_screen()
        screen._adopt_and_retry = MagicMock()  # type: ignore[method-assign]
        screen._on_adopt_confirm(True, "orgA/repoA/built.gguf", "q")
        screen._adopt_and_retry.assert_called_once_with("orgA/repoA/built.gguf", "q")


class TestDoAdoptAndRetry:
    def test_success_adopts_then_resends(self):
        screen = _bare_screen()
        screen._send_message = MagicMock()  # type: ignore[method-assign]
        with (
            patch("lilbee.app.models.adopt_embedder") as adopt,
            patch(
                "lilbee.cli.tui.screens.chat.call_from_thread",
                side_effect=lambda _node, fn, *a, **k: fn(*a, **k),
            ),
        ):
            screen._do_adopt_and_retry("orgA/repoA/built.gguf", "q")
        adopt.assert_called_once_with("orgA/repoA/built.gguf")
        screen._send_message.assert_called_once_with("q")

    def test_failure_notifies_and_does_not_resend(self):
        screen = _bare_screen()
        screen._send_message = MagicMock()  # type: ignore[method-assign]
        with (
            patch("lilbee.app.models.adopt_embedder", side_effect=RuntimeError("boom")),
            patch(
                "lilbee.cli.tui.screens.chat.call_from_thread",
                side_effect=lambda _node, fn, *a, **k: fn(*a, **k),
            ),
        ):
            screen._do_adopt_and_retry("orgA/repoA/built.gguf", "q")
        screen._send_message.assert_not_called()
        assert any("boom" in str(c.args) for c in screen.notify.call_args_list)
