"""ViewTabs model pill, theme persistence."""

from __future__ import annotations

from unittest import mock

import pytest

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.catalog import display_label_for_ref
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import await_chat

_ALT_CHAT_REF = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
_TEST_LOCAL_LABEL = display_label_for_ref(TEST_LOCAL_REF)
_ALT_CHAT_LABEL = display_label_for_ref(_ALT_CHAT_REF)


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
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
    from lilbee.app.services import set_services

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
            "lilbee.cli.tui.app.chat_ready",
            return_value=True,
        ),
        mock.patch(
            "lilbee.cli.tui.app.embedding_ready",
            return_value=True,
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
    """Render the visible tab strip text: per-tab Labels + trailing Static."""
    from textual.widgets import Static

    from lilbee.cli.tui.widgets.status_bar import ViewTab

    parts = [str(tab.render()) for tab in tabs.query(ViewTab)]
    parts.append(str(tabs.query_one("#view-tabs-trailing", Static).render()))
    return " ".join(parts)


async def test_view_tabs_hides_model_pill_on_chat(_patch_chat_setup) -> None:
    """ModelBar already shows the active chat model on chat, so the ViewTabs
    pill would just duplicate it; it must hide there."""
    cfg.chat_model = TEST_LOCAL_REF
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        tabs = app.screen.query_one(ViewTabs)
        text = _rendered_tab_text(tabs)
        assert _TEST_LOCAL_LABEL not in text


async def test_view_tabs_renders_active_chat_model_on_settings(_patch_chat_setup) -> None:
    """The model pill follows the user to non-chat screens; the bug 2 fix."""
    cfg.chat_model = TEST_LOCAL_REF
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        tabs = app.screen.query_one(ViewTabs)
        text = _rendered_tab_text(tabs)
        assert _TEST_LOCAL_LABEL in text


async def test_view_tabs_refreshes_model_pill_on_settings_change(_patch_chat_setup) -> None:
    """When chat_model is updated and the signal fires, the ViewTabs pill
    on a non-chat screen re-renders to the new value."""
    cfg.chat_model = TEST_LOCAL_REF
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()
        await pilot.pause()

        tabs = app.screen.query_one(ViewTabs)
        assert _TEST_LOCAL_LABEL in _rendered_tab_text(tabs)

        cfg.chat_model = _ALT_CHAT_REF
        app.settings_changed_signal.publish(("chat_model", _ALT_CHAT_REF))
        await pilot.pause()

        assert _ALT_CHAT_LABEL in _rendered_tab_text(tabs)
        assert _TEST_LOCAL_LABEL not in _rendered_tab_text(tabs)


async def test_view_tabs_ignores_non_model_settings_changes(_patch_chat_setup) -> None:
    """Sampling-param signal payloads do not cause spurious refreshes.

    Asserted on a non-chat screen so the pill actually renders; on chat
    the pill is hidden regardless, so a regression would slip through.
    """
    cfg.chat_model = TEST_LOCAL_REF
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()
        await pilot.pause()

        tabs = app.screen.query_one(ViewTabs)
        before = _rendered_tab_text(tabs)

        # Change cfg out of band but only publish an unrelated key; pill must not flip.
        cfg.chat_model = _ALT_CHAT_REF
        app.settings_changed_signal.publish(("temperature", 0.5))
        await pilot.pause()

        # The unrelated payload must NOT trigger a refresh; the pill stays stale on
        # purpose (the only path that swaps the pill is a chat_model signal).
        assert _rendered_tab_text(tabs) == before


# ---------------------------------------------------------------------------
# Theme persistence + visible keybinding
# ---------------------------------------------------------------------------


def test_theme_keybinding_stays_discoverable_off_the_footer() -> None:
    """ctrl+t must stay discoverable, now via F1 rather than a footer cell.

    The original guarantee was "surfaces in the Footer so users can discover
    theme cycling" (#186). The footer is now the keys that move between views
    plus each screen's own verb, so theme cycling moved to the F1 key panel.
    The discoverability requirement is unchanged and asserted here on its new
    surface: the panel lists every non-system binding regardless of ``show``.
    """
    bindings = [b for b in LilbeeApp.BINDINGS if getattr(b, "key", None) == "ctrl+t"]
    assert len(bindings) == 1
    assert bindings[0].action == "cycle_theme"
    assert bindings[0].show is False, "the footer row is for navigation"
    assert bindings[0].system is False, "system bindings are hidden from the F1 panel too"


async def test_cycle_theme_persists_to_config(_patch_chat_setup) -> None:
    """Pressing the cycle binding writes the new theme name to config.toml."""
    from lilbee.core import settings

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
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
    from lilbee.core import settings

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
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
        await await_chat(app, pilot)
        await pilot.pause()
        assert app.theme == "nord"


async def test_app_falls_back_when_persisted_theme_invalid(_patch_chat_setup) -> None:
    """Garbage in cfg.theme doesn't brick the TUI; fall back to the default."""
    cfg.theme = "not-a-real-theme"

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        # Falls back to the module-level default, not the bad value.
        from lilbee.cli.tui.app import _DEFAULT_THEME

        assert app.theme == _DEFAULT_THEME


async def test_sync_theme_index_handles_non_dark_theme(_patch_chat_setup) -> None:
    """A theme set outside DARK_THEMES must not raise; the cycle index falls
    back to 0 instead of propagating the ValueError from list.index."""
    from lilbee.app.themes import DARK_THEMES

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        # Pick any Textual theme that is intentionally NOT in DARK_THEMES.
        non_dark = next(
            (t for t in app.available_themes if t not in DARK_THEMES),
            None,
        )
        assert non_dark is not None, "Textual must ship at least one non-DARK theme"
        app.theme = non_dark
        app._sync_theme_index_to_current()
        assert app._theme_index == 0
