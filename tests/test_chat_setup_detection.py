"""Tests for ChatScreen._needs_setup: wizard shows on fresh data dirs."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.core.config import cfg


@pytest.fixture
def isolated_data_dir(tmp_path):
    """Point cfg at a per-test data directory and restore the full snapshot."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    try:
        yield tmp_path
    finally:
        for field_name in type(snapshot).model_fields:
            setattr(cfg, field_name, getattr(snapshot, field_name))


def _make_screen() -> ChatScreen:
    return ChatScreen.__new__(ChatScreen)


def test_needs_setup_true_when_lancedb_dir_missing(isolated_data_dir):
    """Fresh data dir must trigger the wizard even when models resolve globally."""
    assert not cfg.lancedb_dir.exists()
    with mock.patch(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        return_value="/some/resolved/path",
    ) as resolve:
        assert _make_screen()._needs_setup() is True
        resolve.assert_not_called()


def test_push_setup_wizard_no_op_when_unmounted():
    """``_push_setup_wizard`` short-circuits when the screen is no longer mounted.

    The setup-check worker may race a view switch: by the time the worker
    finishes, the chat screen is already gone. Ensure the push is gated
    on ``is_mounted`` so we don't push onto whatever screen is now on top.
    """
    screen = ChatScreen.__new__(ChatScreen)
    ChatScreen.is_mounted = property(lambda self: False)
    try:
        # If the guard breaks, this would AttributeError on ``self.app``.
        screen._push_setup_wizard()
    finally:
        del ChatScreen.is_mounted


def test_needs_setup_false_when_initialized_and_models_resolve(isolated_data_dir):
    """Initialized data dir plus resolvable models skips the wizard."""
    cfg.lancedb_dir.mkdir(parents=True)
    with mock.patch(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        return_value="/some/resolved/path",
    ):
        assert _make_screen()._needs_setup() is False


def test_needs_setup_true_when_initialized_but_model_missing(isolated_data_dir):
    """Unresolvable chat/embedding model still triggers the wizard."""
    from lilbee.providers.base import ProviderError

    cfg.lancedb_dir.mkdir(parents=True)
    with mock.patch(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        side_effect=ProviderError("no such model", provider="llama-cpp"),
    ):
        assert _make_screen()._needs_setup() is True


def test_needs_setup_skips_native_probe_for_remote_prefixed_models(isolated_data_dir):
    """ollama/ and API-prefixed models bypass the llama-cpp registry check.

    The native resolver only knows about GGUFs. A remote ref resolves at
    call time via litellm, so asking resolve_model_path about it would
    always raise and wrongly trigger the setup wizard.
    """
    cfg.lancedb_dir.mkdir(parents=True)
    cfg.chat_model = "ollama/qwen3:0.6b"
    cfg.embedding_model = "ollama/nomic-embed-text:v1.5"
    with mock.patch(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
    ) as resolve:
        assert _make_screen()._needs_setup() is False
        resolve.assert_not_called()


def test_needs_setup_true_when_lancedb_path_is_a_file(isolated_data_dir):
    """A stray file at the lancedb path is not a real data directory."""
    cfg.lancedb_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.write_text("not a directory")
    assert cfg.lancedb_dir.exists()
    assert not cfg.lancedb_dir.is_dir()
    with mock.patch(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        return_value="/some/resolved/path",
    ):
        assert _make_screen()._needs_setup() is True


@pytest.fixture
def mock_services():
    from lilbee.app.services import set_services

    svc = mock.MagicMock()
    svc.provider.list_models.return_value = []
    svc.searcher._embedder.embedding_available.return_value = True
    set_services(svc)
    try:
        yield svc
    finally:
        set_services(None)


async def test_chat_screen_cached_across_navigation(isolated_data_dir, mock_services):
    """Navigating away from Chat and back reuses the same instance.
    ChatScreen is installed via install_screen, so on_mount (and therefore
    _needs_setup) runs only on first mount, not on every revisit."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.chat import ChatScreen

    cfg.lancedb_dir.mkdir(parents=True)

    with (
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._needs_setup",
            return_value=False,
        ),
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=True,
        ),
    ):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            chat = app.screen
            assert isinstance(chat, ChatScreen)

            app.switch_view("Catalog")
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)

            app.switch_view("Chat")
            await pilot.pause()
            assert app.screen is chat  # same instance, not a new one
