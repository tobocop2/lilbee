"""ViewTabs model pill + setup wizard exit affordances.

Bug 2 (model pill): the active chat model is shown on every screen via
ViewTabs, not just on chat where ModelBar lives.
Bug 3 (setup exit): the wizard mounts a Footer so the Esc->Done binding
is visible, and the hint text changes when models are already installed.
"""

from __future__ import annotations

from unittest import mock

import pytest
from textual.app import App
from textual.widgets import Footer

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.setup import SetupWizard
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = "qwen3:8b"
    cfg.embedding_model = "nomic-embed-text"
    cfg.subprocess_embed = False
    cfg.wiki = False
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


@pytest.fixture(autouse=True)
def _mock_services():
    from lilbee.services import set_services

    mock_svc = mock.MagicMock()
    mock_svc.provider.list_models.return_value = []
    mock_svc.searcher._embedder.embedding_available.return_value = True
    set_services(mock_svc)
    try:
        yield mock_svc
    finally:
        set_services(None)


@pytest.fixture()
def _patch_chat_setup():
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
        yield


# ---------------------------------------------------------------------------
# Bug 2: ViewTabs model pill
# ---------------------------------------------------------------------------


def _rendered_tab_text(tabs: ViewTabs) -> str:
    """Pull the rendered text out of the inner Static, regardless of styling."""
    from textual.widgets import Static

    inner = tabs.query_one("#view-tabs-content", Static)
    return str(inner.render())


async def test_view_tabs_renders_active_chat_model_on_chat(_patch_chat_setup) -> None:
    """The model pill renders on the chat screen alongside the page indicator."""
    cfg.chat_model = "qwen3:8b"
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        tabs = app.screen.query_one(ViewTabs)
        text = _rendered_tab_text(tabs)
        assert "qwen3:8b" in text


async def test_view_tabs_renders_active_chat_model_on_settings(_patch_chat_setup) -> None:
    """The model pill follows the user to non-chat screens — the bug 2 fix."""
    cfg.chat_model = "llama3:8b"
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        tabs = app.screen.query_one(ViewTabs)
        text = _rendered_tab_text(tabs)
        assert "llama3:8b" in text


async def test_view_tabs_refreshes_model_pill_on_settings_change(_patch_chat_setup) -> None:
    """When chat_model is updated and the signal fires, ViewTabs re-renders."""
    cfg.chat_model = "qwen3:8b"
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tabs = app.screen.query_one(ViewTabs)
        assert "qwen3:8b" in _rendered_tab_text(tabs)

        cfg.chat_model = "gemma3:1b"
        app.settings_changed_signal.publish(("chat_model", "gemma3:1b"))
        await pilot.pause()

        assert "gemma3:1b" in _rendered_tab_text(tabs)
        assert "qwen3:8b" not in _rendered_tab_text(tabs)


async def test_view_tabs_ignores_non_model_settings_changes(_patch_chat_setup) -> None:
    """Sampling-param signal payloads do not cause spurious refreshes."""
    cfg.chat_model = "qwen3:8b"
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tabs = app.screen.query_one(ViewTabs)
        before = _rendered_tab_text(tabs)

        # Change cfg out of band but only publish an unrelated key — pill must not flip.
        cfg.chat_model = "should-not-show"
        app.settings_changed_signal.publish(("temperature", 0.5))
        await pilot.pause()

        # The unrelated payload must NOT trigger a refresh; the pill stays stale on
        # purpose (the only path that swaps the pill is a chat_model signal).
        assert _rendered_tab_text(tabs) == before


# ---------------------------------------------------------------------------
# Bug 3: setup wizard exit affordance
# ---------------------------------------------------------------------------


class _SetupHostApp(App[None]):
    """Minimal host app that pushes only the SetupWizard, no chat screen."""

    CSS = ""

    def on_mount(self) -> None:
        self.push_screen(SetupWizard())


async def test_setup_wizard_mounts_footer_so_done_keybinding_is_visible() -> None:
    """The wizard composes a Footer so the Esc->Done hint surfaces to the user."""
    with mock.patch(
        "lilbee.cli.tui.screens.setup._scan_installed_models",
        return_value=([], []),
    ):
        app = _SetupHostApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, SetupWizard)
            footers = list(app.screen.query(Footer))
            assert len(footers) == 1, "Setup wizard must mount exactly one Footer"


async def test_setup_wizard_mounts_view_tabs() -> None:
    """The wizard composes ViewTabs so the page strip matches every other screen."""
    with mock.patch(
        "lilbee.cli.tui.screens.setup._scan_installed_models",
        return_value=([], []),
    ):
        app = _SetupHostApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            tabs = list(app.screen.query(ViewTabs))
            assert len(tabs) == 1


async def test_setup_wizard_hint_when_no_models_installed() -> None:
    """Default first-run hint tells the user to press Enter to install."""
    with mock.patch(
        "lilbee.cli.tui.screens.setup._scan_installed_models",
        return_value=([], []),
    ):
        app = _SetupHostApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            label = app.screen.query_one("#setup-enter-hint")
            assert msg.SETUP_ENTER_HINT in str(label.render())


async def test_setup_wizard_hint_when_models_already_installed() -> None:
    """Models present -> hint shifts to 'Esc to return' so the wizard reads as a review."""
    with mock.patch(
        "lilbee.cli.tui.screens.setup._scan_installed_models",
        return_value=(["qwen3:8b"], ["nomic-embed-text"]),
    ):
        app = _SetupHostApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            label = app.screen.query_one("#setup-enter-hint")
            assert msg.SETUP_RETURN_HINT in str(label.render())


# ---------------------------------------------------------------------------
# Theme persistence + visible keybinding
# ---------------------------------------------------------------------------


def test_theme_keybinding_is_visible_in_app_bindings() -> None:
    """ctrl+t must surface in the Footer so users can discover theme cycling."""
    bindings = [b for b in LilbeeApp.BINDINGS if getattr(b, "key", None) == "ctrl+t"]
    assert len(bindings) == 1
    assert bindings[0].show is True
    assert bindings[0].action == "cycle_theme"


async def test_cycle_theme_persists_to_config(_patch_chat_setup) -> None:
    """Pressing the cycle binding writes the new theme name to config.toml."""
    from lilbee import settings

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        before = app.theme

        app.action_cycle_theme()
        await pilot.pause()

        assert app.theme != before
        assert cfg.theme == app.theme
        # Round-trip through the on-disk store
        assert settings.get(cfg.data_root, "theme") == app.theme


async def test_set_theme_persists_to_config(_patch_chat_setup) -> None:
    """The /theme command path also persists across sessions."""
    from lilbee import settings

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        app.set_theme("dracula")
        await pilot.pause()

        assert app.theme == "dracula"
        assert cfg.theme == "dracula"
        assert settings.get(cfg.data_root, "theme") == "dracula"


async def test_app_restores_persisted_theme_on_startup(_patch_chat_setup) -> None:
    """A theme written to cfg before mount is the one the app opens with."""
    cfg.theme = "nord"

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.theme == "nord"


async def test_app_falls_back_when_persisted_theme_invalid(_patch_chat_setup) -> None:
    """Garbage in cfg.theme doesn't brick the TUI — fall back to the default."""
    cfg.theme = "not-a-real-theme"

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Falls back to gruvbox (the module-level default), not the bad value.
        assert app.theme != "not-a-real-theme"
