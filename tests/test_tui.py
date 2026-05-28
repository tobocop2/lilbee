"""Tests for the Textual TUI."""

from __future__ import annotations

from unittest import mock

import pytest
from textual.binding import Binding

from conftest import (
    TEST_LOCAL_REF,
)
from conftest import (
    make_test_catalog_model as _make_model,
)
from lilbee.catalog import CatalogResult
from lilbee.cli.tui.screens.catalog_utils import catalog_to_row, remote_to_row
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.chat_model = TEST_LOCAL_REF
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    """Patch out embedding model checks and model scanning so ChatScreen mounts cleanly."""
    with (
        mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=False,
        ),
        mock.patch(
            "lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount",
        ),
    ):
        yield


_EMPTY_CATALOG = CatalogResult(total=0, limit=50, offset=0, models=[])


class TestRunTui:
    @mock.patch("lilbee.cli.tui.app.LilbeeApp.run")
    def test_run_tui_launches_app(self, mock_run: mock.MagicMock) -> None:
        from lilbee.cli.tui import run_tui

        run_tui()
        mock_run.assert_called_once()

    @mock.patch("lilbee.cli.tui.app.LilbeeApp.run")
    def test_run_tui_forwards_initial_view(self, mock_run: mock.MagicMock) -> None:
        from lilbee.cli.tui import run_tui

        with mock.patch("lilbee.cli.tui.app.LilbeeApp.__init__", return_value=None) as init:
            run_tui(initial_view="Catalog")
        init.assert_called_once_with(initial_view="Catalog")

    @pytest.mark.asyncio
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_initial_view_switches_to_catalog(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp(initial_view="Catalog")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.active_view == "Catalog"


class TestUserMessage:
    def test_creates_with_text(self) -> None:
        msg = UserMessage("hello world")
        assert msg is not None
        assert "user-message" in msg.classes

    def test_renders_speaker_label(self) -> None:
        """UserMessage should have compose() that yields speaker label and content."""
        msg = UserMessage("hello world")
        children = list(msg.compose())
        assert len(children) == 2


class TestAssistantMessage:
    def test_compose_yields_widgets(self) -> None:
        msg = AssistantMessage()
        children = list(msg.compose())
        # Compose-time children are speaker label, content, citation. The
        # ThinkingHeader and reasoning Collapsible are mounted lazily.
        assert len(children) == 3

    def test_append_content(self) -> None:
        msg = AssistantMessage()
        list(msg.compose())
        msg._content_parts.append("test")
        assert "test" in msg._content_parts

    def test_append_reasoning(self) -> None:
        msg = AssistantMessage()
        list(msg.compose())
        msg._reasoning_parts.append("thinking")
        assert "thinking" in msg._reasoning_parts

    def test_finish_with_sources(self) -> None:
        msg = AssistantMessage()
        list(msg.compose())
        msg.finish(["doc.pdf:42"])
        assert msg._finished


class TestTaskBarUnit:
    def test_queue_enqueue_returns_id(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        task_id = q.enqueue(lambda: None, "Test", "sync")
        assert isinstance(task_id, str)
        assert len(task_id) == 8

    def test_update_task_does_not_deadlock_reentrant_subscriber(self) -> None:
        """Regression: a subscriber that reads the queue inside its callback
        must not deadlock on the non-reentrant lock held by `update_task`.
        """
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        task_id = q.enqueue(lambda: None, "Test", "sync")
        q.advance()

        observed: list[int] = []

        def _on_change() -> None:
            # Re-enter the queue from inside the callback (TaskBar does this
            # via `displayable_tasks` during `_refresh_display`). If `_notify`
            # fired while still holding the lock, this would hang forever.
            task = q.get_task(task_id)
            if task is not None:
                observed.append(task.progress)

        q.subscribe(_on_change)
        q.update_task(task_id, 42, "halfway")
        assert observed == [42]


class TestRemoteClassification:
    @mock.patch("httpx.get")
    def test_classifies_models(self, mock_get: mock.MagicMock) -> None:
        from lilbee.modelhub.model_manager import classify_remote_models

        mock_get.return_value = mock.MagicMock(
            status_code=200,
            json=lambda: {
                "models": [
                    {
                        "name": "nomic-embed-text:latest",
                        "details": {"family": "nomic-bert", "parameter_size": "137M"},
                    },
                    {"name": "qwen3:8b", "details": {"family": "qwen3", "parameter_size": "8.2B"}},
                    {
                        "name": "llava:latest",
                        "details": {"family": "llava", "parameter_size": "7B"},
                    },
                ]
            },
        )
        mock_get.return_value.raise_for_status = lambda: None
        result = classify_remote_models()
        by_task = {m.task: m.name for m in result}
        assert by_task["embedding"] == "nomic-embed-text:latest"
        assert by_task["chat"] == "qwen3:8b"
        assert by_task["vision"] == "llava:latest"


class TestCatalogToRow:
    def test_stores_catalog_model(self) -> None:
        m = _make_model("Qwen3 8B", featured=True)
        row = catalog_to_row(m, installed=False)
        assert row.catalog_model is m

    def test_featured_flag_set(self) -> None:
        m = _make_model("TestModel", task="chat", size_gb=5.0, featured=True)
        row = catalog_to_row(m, installed=False)
        assert row.featured is True

    def test_installed_flag_set(self) -> None:
        m = _make_model("TestModel", task="chat", size_gb=5.0)
        row = catalog_to_row(m, installed=True)
        assert row.installed is True


class TestChatScreenAsync:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_app_launches(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.title.startswith("lilbee")

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_chat_input_exists(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            assert inp is not None

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_quit_keybinding(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            with mock.patch.object(app, "exit") as mock_exit:
                await pilot.press("ctrl+q")
                await pilot.pause()
                mock_exit.assert_called()

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_help_panel(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Escape out of the chat Input so ? routes to the binding instead
            # of being typed as a character.
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("question_mark")
            await pilot.pause()
            assert app.screen.query("HelpPanel")

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_catalog_push(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.switch_view("Catalog")
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_slash_help(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.screen_stack) > 1

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_slash_unknown_notifies(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.value = "/badcommand"
            with mock.patch.object(app.screen, "notify") as mock_notify:
                await pilot.press("enter")
                await pilot.pause()
                mock_notify.assert_called()
                assert "Unknown command" in mock_notify.call_args[0][0]

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_slash_model_changes_model(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            new_ref = "ollama/new-model:latest"
            inp.value = f"/model {new_ref}"
            await pilot.press("enter")
            await pilot.pause()
            assert cfg.chat_model == new_ref

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_slash_set_changes_setting(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.value = "/set top_k 10"
            await pilot.press("enter")
            await pilot.pause()
            assert cfg.top_k == 10

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_slash_set_invalid_key(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.value = "/set nonexistent 42"
            with mock.patch.object(app.screen, "notify") as mock_notify:
                await pilot.press("enter")
                await pilot.pause()
                mock_notify.assert_called()
                assert "Unknown setting" in mock_notify.call_args[0][0]

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_empty_input_ignored(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.value = ""
            with mock.patch.object(app.screen, "_send_message") as mock_send:
                await pilot.press("enter")
                await pilot.pause()
                mock_send.assert_not_called()


class TestCatalogScreenAsync:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_catalog_shows_featured(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(CatalogScreen())
            await pilot.pause()

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_catalog_quit(self, mock_catalog: mock.MagicMock) -> None:
        """`q` dismisses the catalog (Escape no longer dismisses; see
        action_dismiss_filter).
        """
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            catalog = CatalogScreen()
            app.push_screen(catalog)
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            # Catalog should be gone, chat screen visible
            assert not isinstance(app.screen, CatalogScreen)

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_escape_from_filter_unfocuses_instead_of_leaving(
        self, mock_catalog: mock.MagicMock
    ) -> None:
        """first Escape while filter is focused should move focus to
        the list or grid, not leave the screen."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from textual.widgets import Input

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            catalog = CatalogScreen()
            app.push_screen(catalog)
            for _ in range(8):
                await pilot.pause()
            # Focus the filter input explicitly to match the scenario.
            catalog.query_one("#catalog-search", Input).focus()
            await pilot.pause()
            assert isinstance(catalog.focused, Input)
            catalog.action_go_back()
            await pilot.pause()
            # Still on the catalog screen; focus no longer on the Input.
            assert isinstance(app.screen, CatalogScreen)
            assert not isinstance(catalog.focused, Input)

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_escape_from_filter_in_list_view_focuses_list(
        self, mock_catalog: mock.MagicMock
    ) -> None:
        """in list view, Escape from filter should focus the list."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from textual.widgets import Input

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            catalog = CatalogScreen()
            app.push_screen(catalog)
            await pilot.pause()
            catalog.action_toggle_view()  # switch to list view
            await pilot.pause()
            catalog.query_one("#catalog-search", Input).focus()
            await pilot.pause()
            assert isinstance(catalog.focused, Input)
            catalog.action_go_back()
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)
            assert not isinstance(catalog.focused, Input)

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_sort_cycle_visits_all_four_columns(self, mock_catalog: mock.MagicMock) -> None:
        """cycle must visit Name -> Downloads -> Size -> Params -> Name."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            catalog = CatalogScreen()
            app.push_screen(catalog)
            await pilot.pause()
            # Pin active tab to Chat: the 6-tab shell's TabbedContent can
            # briefly land on the first-defined pane (Discover) under
            # run_test before the call_after_refresh setter runs. Sorting
            # is gated on a task tab being active.
            catalog._active_tab_id_cache = "chat"
            # Switch to list view so sort is available.
            catalog._grid_view = False
            catalog._sort_column = "Name"
            observed = []
            for _ in range(5):
                catalog.action_cycle_sort()
                await pilot.pause()
                observed.append(catalog._sort_column)
            assert observed == ["Downloads", "Size", "Params", "Name", "Downloads"]


class TestSettingsScreenAsync:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_settings_shows_table(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.settings import SettingsScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(SettingsScreen())
            await pilot.pause()
            groups = app.screen.query("TabPane")
            assert len(groups) > 0


class TestStatusScreenAsync:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_status_screen(
        self,
        mock_catalog: mock.MagicMock,
    ) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        mock_svc = mock.MagicMock()
        mock_svc.store.get_sources.return_value = []
        from lilbee.app.services import set_services
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.status import StatusScreen

        set_services(mock_svc)
        try:
            app = LilbeeApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                app.push_screen(StatusScreen())
                await pilot.pause()
                info = app.screen.query_one("#config-info")
                assert info is not None
        finally:
            set_services(None)


class TestCLIIntegration:
    def test_chat_non_tty_exits_with_error(self) -> None:
        """Non-TTY exits with error since TUI requires terminal."""
        from typer.testing import CliRunner

        from lilbee.cli.app import app

        runner = CliRunner()
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 1

    @mock.patch("lilbee.cli.tui.run_tui")
    @mock.patch("sys.stdout")
    @mock.patch("sys.stdin")
    def test_chat_tty_uses_tui(
        self,
        mock_stdin: mock.MagicMock,
        mock_stdout: mock.MagicMock,
        mock_run_tui: mock.MagicMock,
    ) -> None:
        """TTY environment launches TUI."""
        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = True
        from lilbee.cli.commands.search_chat import chat

        with mock.patch("lilbee.cli.commands.search_chat.apply_overrides"):
            chat(
                data_dir=None,
                model=None,
                use_global=False,
                temperature=None,
                top_p=None,
                top_k_sampling=None,
                repeat_penalty=None,
                num_ctx=None,
                seed=None,
            )
        mock_run_tui.assert_called_once_with()


class TestThemes:
    def test_dark_themes_available(self) -> None:
        from lilbee.app.themes import DARK_THEMES

        assert "monokai" in DARK_THEMES
        assert "dracula" in DARK_THEMES

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_cycle_theme(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            before = app.theme
            app.action_cycle_theme()
            assert app.theme != before  # cycled to the next theme

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_set_theme(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.set_theme("dracula")
            assert app.theme == "dracula"

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_dismiss_help_if_open_skips_when_no_panel(
        self, mock_catalog: mock.MagicMock
    ) -> None:
        """Esc raises SkipAction when the HelpPanel is not mounted, so screens
        keep receiving Esc as before."""
        from textual.actions import SkipAction

        from lilbee.cli.tui.app import LilbeeApp

        mock_catalog.return_value = _EMPTY_CATALOG
        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            with pytest.raises(SkipAction):
                app.action_dismiss_help_if_open()

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_dismiss_help_if_open_hides_when_panel_mounted(
        self, mock_catalog: mock.MagicMock
    ) -> None:
        """Esc dismisses the panel and does NOT raise when the panel is open."""
        from lilbee.cli.tui.app import LilbeeApp

        mock_catalog.return_value = _EMPTY_CATALOG
        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_show_help_panel()
            await pilot.pause()
            assert len(app.screen.query("HelpPanel")) == 1
            app.action_dismiss_help_if_open()
            await pilot.pause()
            assert len(app.screen.query("HelpPanel")) == 0

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_set_setting_theme_applies_live(self, mock_catalog: mock.MagicMock) -> None:
        """Settings → theme dropdown must update app.theme, not just cfg.theme.

        Regression: bb-akqw. Without this, picking a theme in Settings
        only persisted to disk; the visual theme stayed the same until
        next launch.
        """
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            target = "dracula" if app.theme != "dracula" else "gruvbox"
            app.set_setting("theme", target)
            assert app.theme == target

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_set_setting_deletes_key_for_none(self, mock_catalog: mock.MagicMock) -> None:
        """Nullable settings cleared to None drop the TOML key.

        Pydantic-settings can't coerce a persisted "" back into int|None,
        so persisting None as missing avoids a stale-config crash on the
        next startup.
        """
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            with (
                mock.patch("lilbee.app.settings.persistent_settings.update_values") as mock_update,
                mock.patch("lilbee.app.settings.persistent_settings.delete_values") as mock_delete,
            ):
                app.set_setting("seed", None)
            mock_update.assert_not_called()
            mock_delete.assert_called_once()
            assert mock_delete.call_args.args[1] == ["seed"]

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_set_setting_stringifies_list_for_toml(
        self, mock_catalog: mock.MagicMock
    ) -> None:
        """List settings must serialize as newline-joined for TOML."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            with mock.patch("lilbee.app.settings.persistent_settings.update_values") as mock_update:
                app.set_setting("crawl_exclude_patterns", ["foo", "bar"])
            mock_update.assert_called_once()
            persisted = mock_update.call_args.args[1]
            assert persisted.get("crawl_exclude_patterns") == "foo\nbar"

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_set_invalid_theme_noop(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            original = app.theme
            app.set_theme("nonexistent_theme_xyz")
            assert app.theme == original


class TestDetectRemoteEmbeddings:
    @mock.patch("httpx.get")
    def test_detects_bert_family(self, mock_get: mock.MagicMock) -> None:
        from lilbee.modelhub.model_manager import detect_remote_embedding_models

        mock_get.return_value = mock.MagicMock(
            status_code=200,
            json=lambda: {
                "models": [
                    {"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert"}},
                    {"name": "qwen3:8b", "details": {"family": "qwen3"}},
                ]
            },
        )
        mock_get.return_value.raise_for_status = lambda: None
        result = detect_remote_embedding_models()
        assert result == ["nomic-embed-text:latest"]

    @mock.patch("httpx.get", side_effect=Exception("connection refused"))
    def test_returns_empty_on_error(self, mock_get: mock.MagicMock) -> None:
        from lilbee.modelhub.model_manager import detect_remote_embedding_models

        assert detect_remote_embedding_models() == []


class TestSetupWizard:
    def test_creates(self) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        wizard = SetupWizard()
        assert wizard._selected_chat is None
        assert wizard._selected_embed is None

    async def test_first_chat_grid_focused_on_mount(self) -> None:
        """On mount, the first chat-model GridSelect must have keyboard focus.

        Regression guard for bb-rqrv: on a fresh launch the wizard's
        GridSelect widgets were focus-less, so arrow keys / Tab / Enter
        never reached them. Users could not pick a model without the mouse.
        """
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.setup import SetupWizard
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.push_screen(SetupWizard())
            await pilot.pause()
            focused = app.focused
            assert isinstance(focused, GridSelect), (
                f"expected GridSelect to have focus on mount, got {type(focused).__name__}"
            )

    async def test_single_tab_escapes_chat_grid(self) -> None:
        """A single Tab from the chat grid must move focus OUT of the grid.

        Regression guard for bb-q9gl root cause: GridSelect's default
        ``action_tab_next`` cycled highlight within the grid before
        escaping, so users who pressed Tab after selecting a card found
        their selection silently changed as the highlight wandered through
        other cards. Tab must not be a within-grid navigator.
        """
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.setup import SetupWizard
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            wizard = SetupWizard()
            await app.push_screen(wizard)
            await pilot.pause()
            assert isinstance(app.focused, GridSelect), "test precondition"
            before = app.focused
            await pilot.press("tab")
            await pilot.pause()
            assert app.focused is not before, (
                "Tab on focused GridSelect must leave the grid; "
                f"stayed on {type(app.focused).__name__}"
            )


class TestCanonicalModelsDir:
    def test_returns_platform_path(self) -> None:
        from lilbee.core.system import canonical_models_dir

        result = canonical_models_dir()
        assert result.name == "models"
        assert "lilbee" in str(result)


class TestRemoteToRow:
    def test_creates(self) -> None:
        from lilbee.modelhub.model_manager import RemoteModel

        rm = RemoteModel(name="mistral:latest", task="chat", family="llama", parameter_size="7.2B")
        row = remote_to_row(rm)
        assert row.remote_model.name == "mistral:latest"
        assert row.installed is True


class TestSlashSuggester:
    async def test_suggests_commands(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        result = await s.get_suggestion("/he")
        assert result == "/help"

    async def test_suggests_nothing_for_empty(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        result = await s.get_suggestion("")
        assert result is None

    async def test_suggests_nothing_for_plain_text(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        result = await s.get_suggestion("hello world")
        assert result is None

    async def test_suggests_set_params(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        result = await s.get_suggestion("/set temp")
        assert result is not None
        assert "temperature" in result

    async def test_suggests_theme_names(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        result = await s.get_suggestion("/theme dra")
        assert result is not None
        assert "dracula" in result

    async def test_no_suggestion_for_exact_match(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        result = await s.get_suggestion("/help")
        assert result is None


class TestContextAwareQuit:
    """Test that action_quit cancels tasks/stream before quitting."""

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_quit_cancels_active_task(self, mock_catalog: mock.MagicMock) -> None:
        """Ctrl+C cancels active TaskBar task when one exists."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            task_bar = app.task_bar
            task_bar.add_task("Test download", "download")
            task_bar.queue.advance()
            await app.action_quit()
            await pilot.pause()
            # Task should have been cancelled, app still running
            assert app.is_running

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_quit_cancels_streaming(self, mock_catalog: mock.MagicMock) -> None:
        """Ctrl+C cancels streaming when active."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            screen.streaming = True
            await app.action_quit()
            await pilot.pause()
            assert not screen.streaming
            assert app.is_running

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_quit_exits_when_idle(self, mock_catalog: mock.MagicMock) -> None:
        """Ctrl+C quits when nothing is active."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.action_quit()
            await pilot.pause()
            # App should have exited
            assert not app.is_running


class TestMinimalFooter:
    """Test that each screen shows only minimal footer keys."""

    def _visible_bindings(self, bindings: list) -> list[str]:
        """Extract descriptions of bindings where show=True."""
        return [b.description for b in bindings if b.show]

    def test_app_bindings_minimal(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        visible = self._visible_bindings(LilbeeApp.BINDINGS)
        assert any("help" in d.lower() for d in visible)
        assert any("quit" in d.lower() or "cancel" in d.lower() for d in visible)
        # Per-view nav (Catalog/Status/Settings) is via [/], not direct
        # Footer keybindings, so those view names must NOT appear.
        assert not any(d == "Catalog" for d in visible)
        assert not any(d == "Status" for d in visible)
        assert not any(d == "Settings" for d in visible)
        # Theme cycling IS exposed in the Footer (Ctrl+T) so users can
        # discover it without digging through docs.
        assert any(d == "Theme" for d in visible)

    def test_chat_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        visible = self._visible_bindings(ChatScreen.BINDINGS)
        assert any("command" in d.lower() for d in visible)
        # `/` says "Slash commands" (not the bare, uninformative "Commands")
        # so the footer tells the user what the key actually does. The
        # adjacent "Complete" hint (Tab) covers tab-completion.
        assert any(d == "Slash commands" for d in visible)
        # F2 is intentionally visible so the searchable slash-command list
        # has a discoverable keybinding in the footer alongside / -- labeled
        # "All commands" so it reads distinctly and is never "Catalog" (that
        # word belongs to /models, the model catalog).
        assert any(d == "All commands" for d in visible)
        assert not any(d == "Catalog" for d in visible)
        # Footer shows the small discoverable set: slash commands, Tab
        # completion, the dual-purpose Esc dispatch, Models, F2 all-commands.
        # Hidden helpers (history, scope cycle, other F-keys) stay show=False.
        assert len(visible) <= 6

    def test_catalog_numeric_tab_bindings(self) -> None:
        """1-6 jump to the corresponding tab in the 6-tab catalog shell.

        Earlier versions of the catalog reused numeric keys for sort
        cycling and explicitly removed them. The 6-tab redesign restores
        them with priority=True so they jump to Discover/Chat/Embed/
        Vision/Rerank/Library directly. show=False keeps the footer tidy.
        """
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        keys = {b.key for b in CatalogScreen.BINDINGS if isinstance(b, Binding)}
        for k in ("1", "2", "3", "4", "5", "6"):
            assert k in keys

    def test_catalog_bindings_minimal(self) -> None:
        """Catalog footer shows only the always-needed actions.

        Earlier the catalog row included Delete / Info / Next tab / Prev
        tab so every action was visible at all times. That made the row
        overflow on narrow terminals and truncate the rightmost global
        binding mid-word (`^t Theme` -> `^t The`). Delete and Info are
        still bound (and tab cycling still works on > / <); they're just
        not in the always-on footer row. F1 / F2 surface them on demand.
        """
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        visible = self._visible_bindings(CatalogScreen.BINDINGS)
        assert any("Back" in d for d in visible)
        assert any("Search" in d for d in visible)
        assert any("Grid/List" in d for d in visible)
        assert not any("Delete" in d for d in visible)
        assert not any(d == "Info" for d in visible)
        assert not any("tab" in d.lower() for d in visible)
        assert len(visible) <= 4

    def test_catalog_delete_bindings_cover_d_backspace_x(self) -> None:
        """D, Backspace, and the legacy X all delete an installed model.

        Backspace is the natural reach on every keyboard; D is the
        documented hotkey; X stays as a hidden alias for muscle memory.
        """
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        delete_keys = {
            b.key
            for b in CatalogScreen.BINDINGS
            if isinstance(b, Binding) and b.action == "delete_model"
        }
        assert delete_keys == {"d", "backspace", "x"}

    def test_catalog_delete_binding_hidden_from_footer(self) -> None:
        """The D Delete binding is bound but stays out of the footer row.

        Footer was overflowing on narrow terminals; Delete is still
        reachable via D / Backspace / X and surfaces in F2's command
        palette. Keeping it hidden frees space for the global view-nav
        `[ ] Navigate` binding to fit without truncation.
        """
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        d_binding = next(
            b
            for b in CatalogScreen.BINDINGS
            if isinstance(b, Binding) and b.key == "d" and b.action == "delete_model"
        )
        assert d_binding.show is False

    def test_status_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.status import StatusScreen

        visible = self._visible_bindings(StatusScreen.BINDINGS)
        assert any("Back" in d for d in visible)
        assert len(visible) <= 3

    def test_settings_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        visible = self._visible_bindings(SettingsScreen.BINDINGS)
        assert any("Back" in d for d in visible)
        # Search binding was removed when the settings filter was dropped.
        assert not any("Search" in d for d in visible)
        # 4 baseline (Back, Next field, Prev field, Reset all) + 2 visible
        # tab-cycling bindings (Next tab / Prev tab) shared with Catalog.
        assert len(visible) <= 6


class TestNavBindings:
    """Verify [/] nav bindings exist in app BINDINGS (number keys removed)."""

    def test_nav_bindings_exist(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        keys = {b.key for b in LilbeeApp.BINDINGS if isinstance(b, Binding)}
        assert "left_square_bracket" in keys
        assert "right_square_bracket" in keys

    def test_number_keys_removed(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        keys = {b.key for b in LilbeeApp.BINDINGS if isinstance(b, Binding)}
        # f4 IS bound (toggle_lilbee_path). Number keys + ctrl+n/s/e stay free.
        for k in ("1", "2", "3", "4", "f2", "f3", "ctrl+n", "ctrl+s", "ctrl+e"):
            assert k not in keys


class TestNoRichConsoleInTui:
    """B2: Verify the /add implementation doesn't pull Rich Console into the TUI."""

    def test_chat_add_uses_copy_files_not_copy_paths(self) -> None:
        import inspect

        from lilbee.cli.tui.screens.chat import ChatScreen

        source = inspect.getsource(ChatScreen._do_add)
        assert "from lilbee.cli.app import console" not in source
        assert "copy_paths" not in source
        assert "copy_files" in source


class TestLoginCommand:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("webbrowser.open")
    async def test_login_no_token_opens_browser(
        self, mock_wb: mock.MagicMock, mock_catalog: mock.MagicMock
    ) -> None:
        """'/login' with no token opens HF tokens page in browser."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.value = "/login"
            await pilot.press("enter")
            await pilot.pause()
            mock_wb.assert_called_once_with("https://huggingface.co/settings/tokens")


class TestAppSignals:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families", return_value=[])
    async def test_settings_changed_signal_exists(
        self,
        _fam: mock.MagicMock,
        _cat: mock.MagicMock,
    ) -> None:
        _cat.return_value = CatalogResult(total=0, limit=25, offset=0, models=[])
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert hasattr(app, "settings_changed_signal")

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families", return_value=[])
    async def test_signal_subscribe_and_publish(
        self,
        _fam: mock.MagicMock,
        _cat: mock.MagicMock,
    ) -> None:
        _cat.return_value = CatalogResult(total=0, limit=25, offset=0, models=[])
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            received: list[tuple[str, object]] = []
            app.settings_changed_signal.subscribe(app, lambda val: received.append(val))
            app.settings_changed_signal.publish(("chat_model", "new-model"))
            await pilot.pause()
            assert len(received) == 1
            assert received[0] == ("chat_model", "new-model")


class TestSyncHint:
    """Cover the launch-time sync detection + TaskBar hint surface."""

    @pytest.mark.asyncio
    async def test_app_on_mount_runs_detect_not_sync(self) -> None:
        """App mount kicks detection (not sync) and writes the count to the controller."""
        from lilbee.cli.tui.app import LilbeeApp

        with (
            mock.patch("lilbee.data.ingest.detect_pending", return_value=3) as detect,
            mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._run_sync") as run_sync,
        ):
            app = LilbeeApp()
            async with app.run_test() as pilot:
                # detect runs on a daemon thread; pause until the count lands.
                for _ in range(50):
                    if app.task_bar.pending_sync_count == 3:
                        break
                    await pilot.pause()
            detect.assert_called()
            run_sync.assert_not_called()
            assert app.task_bar.pending_sync_count == 3

    @pytest.mark.asyncio
    async def test_app_on_mount_no_pending_files_hint_hidden(self) -> None:
        """When the vault is in sync, the TaskBar stays hidden."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        with mock.patch("lilbee.data.ingest.detect_pending", return_value=0):
            app = LilbeeApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                bar = app.screen.query_one(TaskBar)
                bar._refresh_display()
                assert app.task_bar.pending_sync_count == 0
                assert bar.display is False

    @pytest.mark.asyncio
    async def test_taskbar_renders_pending_sync_hint(self) -> None:
        """Controller count > 0 + idle queue renders the hint copy."""
        from textual.widgets import Label

        from lilbee.cli.tui import messages as tui_msg
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        with mock.patch("lilbee.data.ingest.detect_pending", return_value=0):
            app = LilbeeApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                app.task_bar.set_pending_sync(3)
                bar = app.screen.query_one(TaskBar)
                bar._refresh_display()
                await pilot.pause()
                assert bar.display is True
                label = bar.query_one("#task-status-label", Label)
                rendered = str(label._Static__content)  # type: ignore[attr-defined]
                expected = tui_msg.TASKBAR_SYNC_PENDING_PLURAL.format(count=3)
                # Strip rich markup so the assertion ignores [b]/[/b] noise.
                assert "3 docs to sync" in rendered
                assert "S to sync" in rendered
                assert expected.replace("[b]", "").replace("[/b]", "") in (
                    rendered.replace("[b]", "").replace("[/b]", "")
                )

    @pytest.mark.asyncio
    async def test_shift_s_triggers_sync_and_clears_hint(self) -> None:
        """Pressing S from any screen routes to ChatScreen._run_sync and clears the hint."""
        from lilbee.cli.tui.app import LilbeeApp

        with (
            mock.patch("lilbee.data.ingest.detect_pending", return_value=2),
            mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._run_sync") as run_sync,
        ):
            app = LilbeeApp()
            async with app.run_test() as pilot:
                # Wait until detection has populated the count.
                for _ in range(50):
                    if app.task_bar.pending_sync_count == 2:
                        break
                    await pilot.pause()
                # ChatScreen starts in INSERT mode with the input focused;
                # escape drops to NORMAL where the global S binding fires.
                await pilot.press("escape")
                await pilot.press("S")
                await pilot.pause()
                run_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_setup_runs_detect(self) -> None:
        """Setup-wizard completion re-runs detection (not sync) on the chat screen."""
        from lilbee.cli.tui.app import LilbeeApp

        with mock.patch("lilbee.data.ingest.detect_pending", return_value=0):
            app = LilbeeApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                with mock.patch.object(app.task_bar, "start_detect_pending") as start:
                    app.screen._on_setup_complete("done")
                    start.assert_called_once()

    @pytest.mark.asyncio
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_action_run_sync_from_non_chat_screen_switches_view(
        self, mock_catalog: mock.MagicMock
    ) -> None:
        """When invoked from Catalog, action_run_sync routes back to Chat."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        with (
            mock.patch("lilbee.data.ingest.detect_pending", return_value=0),
            mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._run_sync") as run_sync,
        ):
            app = LilbeeApp(initial_view="Catalog")
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app.active_view == "Catalog"
                app.action_run_sync()
                for _ in range(50):
                    if run_sync.called:
                        break
                    await pilot.pause()
                run_sync.assert_called_once()

    @pytest.mark.asyncio
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_action_run_sync_no_op_when_chat_screen_unregistered(
        self, mock_catalog: mock.MagicMock
    ) -> None:
        """Defensive: action_run_sync returns cleanly if the chat screen lookup fails."""
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        with mock.patch("lilbee.data.ingest.detect_pending", return_value=0):
            app = LilbeeApp(initial_view="Catalog")
            async with app.run_test() as pilot:
                await pilot.pause()
                with mock.patch.object(app, "get_screen", side_effect=KeyError("chat")):
                    # Should return without raising or attempting a switch.
                    app.action_run_sync()

    @pytest.mark.asyncio
    async def test_run_sync_clears_hint_and_restarts_detect_after_completion(self) -> None:
        """Starting a sync clears the hint; the worker restarts detection at the end."""
        import threading as _threading

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.task_queue import TaskType

        observed: list[int] = []
        gate = _threading.Event()

        def _do_sync_target(self_screen, reporter, *, force_rebuild=False):
            observed.append(self_screen.app.task_bar.pending_sync_count)
            gate.set()

        with (
            mock.patch("lilbee.data.ingest.detect_pending", return_value=4),
            mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._do_sync", _do_sync_target),
        ):
            app = LilbeeApp()
            async with app.run_test() as pilot:
                for _ in range(50):
                    if app.task_bar.pending_sync_count == 4:
                        break
                    await pilot.pause()
                start_calls: list[None] = []
                original = app.task_bar.start_detect_pending

                def _track() -> None:
                    start_calls.append(None)
                    original()

                app.task_bar.start_detect_pending = _track  # type: ignore[method-assign]

                app.screen._run_sync()
                gate.wait(timeout=5)
                for _ in range(50):
                    queue = app.task_bar.queue
                    if not queue.active_tasks and not [
                        t for t in queue.queued_tasks if t.task_type == TaskType.SYNC.value
                    ]:
                        break
                    await pilot.pause()
                assert observed == [0]
                assert len(start_calls) >= 1
