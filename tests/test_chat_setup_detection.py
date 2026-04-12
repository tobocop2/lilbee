"""Tests for ChatScreen._needs_setup: wizard shows on fresh data dirs."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.config import cfg


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
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        return_value="/some/resolved/path",
    ) as resolve:
        assert _make_screen()._needs_setup() is True
        resolve.assert_not_called()


def test_needs_setup_false_when_initialized_and_models_resolve(isolated_data_dir):
    """Initialized data dir plus resolvable models skips the wizard."""
    cfg.lancedb_dir.mkdir(parents=True)
    with mock.patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        return_value="/some/resolved/path",
    ):
        assert _make_screen()._needs_setup() is False


def test_needs_setup_true_when_initialized_but_model_missing(isolated_data_dir):
    """Unresolvable chat/embedding model still triggers the wizard."""
    from lilbee.providers.base import ProviderError

    cfg.lancedb_dir.mkdir(parents=True)
    with mock.patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        side_effect=ProviderError("no such model", provider="llama-cpp"),
    ):
        assert _make_screen()._needs_setup() is True


def test_needs_setup_true_when_lancedb_path_is_a_file(isolated_data_dir):
    """A stray file at the lancedb path is not a real data directory."""
    cfg.lancedb_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.write_text("not a directory")
    assert cfg.lancedb_dir.exists()
    assert not cfg.lancedb_dir.is_dir()
    with mock.patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        return_value="/some/resolved/path",
    ):
        assert _make_screen()._needs_setup() is True


@pytest.fixture
def mock_services():
    from lilbee.services import set_services

    svc = mock.MagicMock()
    svc.provider.list_models.return_value = []
    svc.searcher._embedder.embedding_available.return_value = True
    set_services(svc)
    try:
        yield svc
    finally:
        set_services(None)


async def test_wizard_not_repushed_after_skip_and_nav(isolated_data_dir, mock_services):
    """After the wizard is skipped, navigating to Catalog and back must not
    re-push it. Regression test for the setup-wizard-flapping bug."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.screens.setup import SetupWizard

    cfg.lancedb_dir.mkdir(parents=True)
    cfg.models_dir = isolated_data_dir / "models"
    cfg.models_dir.mkdir()
    cfg.chat_model = "ghost:latest"
    cfg.embedding_model = "ghost-embed:latest"

    # _needs_setup returns True on first mount (models unresolvable), then we
    # simulate flip-flop by making it True again on the re-entry mount. The
    # app-level flag must still prevent the wizard from being re-pushed.
    with mock.patch(
        "lilbee.cli.tui.screens.chat.ChatScreen._needs_setup",
        return_value=True,
    ):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # First mount: wizard is pushed.
            assert isinstance(app.screen, SetupWizard)
            # User skips the wizard.
            app.screen.dismiss("skipped")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)
            assert app.setup_handled is True

            # Navigate to Catalog and back to Chat. _needs_setup still says
            # True but the flag must prevent the wizard from being re-pushed.
            app.switch_view("Catalog")
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)
            app.switch_view("Chat")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)


async def test_wizard_not_repushed_after_completion(isolated_data_dir, mock_services):
    """After the wizard is completed, navigating away and back must not
    re-push it either."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.screens.setup import SetupWizard

    cfg.lancedb_dir.mkdir(parents=True)
    cfg.models_dir = isolated_data_dir / "models"
    cfg.models_dir.mkdir()
    cfg.chat_model = "ghost:latest"
    cfg.embedding_model = "ghost-embed:latest"

    with mock.patch(
        "lilbee.cli.tui.screens.chat.ChatScreen._needs_setup",
        return_value=True,
    ):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, SetupWizard)
            app.screen.dismiss("completed")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

            app.switch_view("Catalog")
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)
            app.switch_view("Chat")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)


async def test_first_mount_no_wizard_marks_handled(isolated_data_dir, mock_services):
    """When the first chat mount decides setup isn't needed, the app flag must
    be set so any later flakiness in _needs_setup can't resurrect the wizard."""
    from lilbee.cli.tui.app import LilbeeApp
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
            assert isinstance(app.screen, ChatScreen)
            assert app.setup_handled is True
