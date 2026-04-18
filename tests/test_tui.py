"""Tests for the Textual TUI."""

from __future__ import annotations

from unittest import mock

import pytest
from textual.binding import Binding

from conftest import make_test_catalog_model as _make_model
from lilbee.catalog import CatalogResult
from lilbee.cli.tui.screens.catalog_utils import catalog_to_row, remote_to_row
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.chat_model = "test-model"
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
            "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
            return_value=([], []),
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
        init.assert_called_once_with(auto_sync=False, initial_view="Catalog")

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
        assert len(children) == 4  # speaker label, reasoning, markdown, citation

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
        from lilbee.model_manager import classify_remote_models

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
            app.action_push_help()
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
            inp.value = "/model new-model"
            await pilot.press("enter")
            await pilot.pause()
            assert cfg.chat_model == "new-model:latest"

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
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            catalog = CatalogScreen()
            app.push_screen(catalog)
            await pilot.pause()
            catalog.action_go_back()
            await pilot.pause()
            # Catalog should be gone, chat screen visible
            assert not isinstance(app.screen, CatalogScreen)


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
            groups = app.screen.query(".setting-group")
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
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.status import StatusScreen
        from lilbee.services import set_services

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
        from lilbee.cli.commands import chat

        with mock.patch("lilbee.cli.commands.apply_overrides"):
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
        mock_run_tui.assert_called_once_with(auto_sync=True)


class TestThemes:
    def test_dark_themes_available(self) -> None:
        from lilbee.cli.tui.app import DARK_THEMES

        assert "monokai" in DARK_THEMES
        assert "dracula" in DARK_THEMES

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_cycle_theme(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_cycle_theme()
            assert app.theme != "gruvbox"  # cycled to next

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
    async def test_set_invalid_theme_noop(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.set_theme("nonexistent_theme_xyz")
            # Should still be on default theme (monokai)
            assert app.theme == "gruvbox"


class TestDetectRemoteEmbeddings:
    @mock.patch("httpx.get")
    def test_detects_bert_family(self, mock_get: mock.MagicMock) -> None:
        from lilbee.model_manager import detect_remote_embedding_models

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
        from lilbee.model_manager import detect_remote_embedding_models

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

    async def test_install_button_fires_on_keyboard_enter(self) -> None:
        """Pressing Enter on the focused Install & Go button must trigger install.

        Regression guard: after picking both models, focusing
        #setup-action and pressing Enter must run ``_on_install`` and
        dismiss the wizard. Downloads run in the background under
        ``TaskBarController`` so the wizard doesn't linger.
        """
        from textual.widgets import Button

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            wizard = SetupWizard()
            await app.push_screen(wizard)
            await pilot.pause()
            install_btn = wizard.query_one("#setup-action", Button)
            assert install_btn.disabled is False, "install button should be enabled after preselect"
            install_btn.focus()
            await pilot.pause()
            # Block real downloads so the test doesn't hit the network.
            with mock.patch.object(app.task_bar, "start_download", return_value="tid"):
                await pilot.press("enter")
                await pilot.pause()
            assert not isinstance(app.screen, SetupWizard), (
                "Enter on focused Install & Go must dismiss the wizard"
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
            wizard._run_downloads = lambda: None  # type: ignore[method-assign]
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

    async def test_tab_reaches_install_button_from_chat_grid(self) -> None:
        """Tab cycling from the focused chat grid must reach #setup-action.

        Regression guard for bb-q9gl root cause: in the live TUI, Tab from
        the chat grid moves into the grid's own highlight cycle rather than
        to the next focusable widget, so users never reach the action
        buttons. The fix should let Tab escape the grid once the current
        card is already picked (selected).
        """
        from textual.widgets import Button

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.setup import SetupWizard
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            wizard = SetupWizard()
            wizard._run_downloads = lambda: None  # type: ignore[method-assign]
            await app.push_screen(wizard)
            await pilot.pause()
            # Walk Tab forward; within ~20 hops (cards + buttons) we should
            # land on the Install & Go button at least once.
            saw_install_focused = False
            for _ in range(20):
                await pilot.press("tab")
                await pilot.pause()
                focused = app.focused
                if isinstance(focused, Button) and focused.id == "setup-action":
                    saw_install_focused = True
                    break
            assert saw_install_focused, (
                "Tab cycling from the chat grid should reach #setup-action; "
                f"last focused: {type(app.focused).__name__}"
            )
            assert isinstance(app.focused, GridSelect) is False


class TestCanonicalModelsDir:
    def test_returns_platform_path(self) -> None:
        from lilbee.platform import canonical_models_dir

        result = canonical_models_dir()
        assert result.name == "models"
        assert "lilbee" in str(result)


class TestRemoteToRow:
    def test_creates(self) -> None:
        from lilbee.model_manager import RemoteModel

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
        assert not any(d == "Catalog" for d in visible)
        assert not any(d == "Status" for d in visible)
        assert not any(d == "Settings" for d in visible)
        assert not any(d == "Theme" for d in visible)

    def test_chat_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        visible = self._visible_bindings(ChatScreen.BINDINGS)
        assert any("command" in d.lower() for d in visible)
        assert len(visible) <= 3

    def test_catalog_tab_bindings_removed(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        keys = {b.key for b in CatalogScreen.BINDINGS if isinstance(b, Binding)}
        for k in ("1", "2", "3", "4"):
            assert k not in keys

    def test_catalog_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        visible = self._visible_bindings(CatalogScreen.BINDINGS)
        assert any("Back" in d for d in visible)
        assert any("Search" in d for d in visible)
        assert any("Delete" in d for d in visible)
        assert len(visible) <= 5

    def test_status_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.status import StatusScreen

        visible = self._visible_bindings(StatusScreen.BINDINGS)
        assert any("Back" in d for d in visible)
        assert len(visible) <= 3

    def test_settings_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        visible = self._visible_bindings(SettingsScreen.BINDINGS)
        assert any("Back" in d for d in visible)
        assert any("Search" in d for d in visible)
        assert len(visible) <= 4


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
        for k in ("1", "2", "3", "4", "f2", "f3", "f4", "ctrl+n", "ctrl+s", "ctrl+e"):
            assert k not in keys


class TestNoRichConsoleInTui:
    """B2: Verify _run_add_background does not import Rich Console."""

    def test_chat_add_uses_copy_files_not_copy_paths(self) -> None:
        import inspect

        from lilbee.cli.tui.screens.chat import ChatScreen

        source = inspect.getsource(ChatScreen._run_add_background)
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
    async def test_model_changed_signal_exists(
        self,
        _fam: mock.MagicMock,
        _cat: mock.MagicMock,
    ) -> None:
        _cat.return_value = CatalogResult(total=0, limit=25, offset=0, models=[])
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert hasattr(app, "model_changed_signal")

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
