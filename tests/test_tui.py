"""Tests for the Textual TUI."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.catalog import CatalogModel, CatalogResult
from lilbee.cli.tui.screens.catalog import _TAB_TO_TASK, ModelRow, RemoteRow
from lilbee.cli.tui.widgets.help_modal import HelpModal
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.chat_model = "test-model"
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    """Patch out embedding model checks so ChatScreen mounts cleanly."""
    with (
        mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=False,
        ),
    ):
        yield


def _make_model(
    name: str = "TestModel",
    task: str = "chat",
    featured: bool = False,
    size_gb: float = 2.0,
    description: str = "A test model",
) -> CatalogModel:
    return CatalogModel(
        name=name,
        hf_repo=f"test/{name.lower().replace(' ', '-')}",
        gguf_filename="*.gguf",
        size_gb=size_gb,
        min_ram_gb=4,
        description=description,
        featured=featured,
        downloads=100,
        task=task,
    )


_EMPTY_CATALOG = CatalogResult(total=0, limit=50, offset=0, models=[])


class TestRunTui:
    @mock.patch("lilbee.cli.tui.app.LilbeeApp.run")
    def test_run_tui_launches_app(self, mock_run: mock.MagicMock) -> None:
        from lilbee.cli.tui import run_tui

        run_tui()
        mock_run.assert_called_once()


class TestUserMessage:
    def test_creates_with_text(self) -> None:
        msg = UserMessage("hello world")
        assert msg is not None
        assert "user-message" in msg.classes


class TestAssistantMessage:
    def test_compose_yields_widgets(self) -> None:
        msg = AssistantMessage()
        children = list(msg.compose())
        assert len(children) == 3  # reasoning, markdown, citation

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


class TestHelpModal:
    def test_creates(self) -> None:
        modal = HelpModal()
        assert modal is not None


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


class TestTabToTask:
    def test_all_maps_to_none(self) -> None:
        assert _TAB_TO_TASK["All"] is None

    def test_chat_maps_to_chat(self) -> None:
        assert _TAB_TO_TASK["Chat"] == "chat"

    def test_embedding_maps_to_embedding(self) -> None:
        assert _TAB_TO_TASK["Embedding"] == "embedding"

    def test_vision_maps_to_vision(self) -> None:
        assert _TAB_TO_TASK["Vision"] == "vision"


class TestModelRow:
    def test_stores_model(self) -> None:
        m = _make_model("Qwen3 8B", featured=True)
        row = ModelRow(m)
        assert row.model is m

    def test_compose_yields_static(self) -> None:
        m = _make_model("TestModel", task="chat", size_gb=5.0)
        row = ModelRow(m)
        children = list(row.compose())
        assert len(children) == 1


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
            await pilot.press("ctrl+q")

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_help_modal(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_push_help()
            await pilot.pause()
            assert len(app.screen_stack) > 1

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_catalog_push(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_push_catalog()
            await pilot.pause()
            assert len(app.screen_stack) > 1

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
            await pilot.press("enter")
            await pilot.pause()

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
            await pilot.press("enter")
            await pilot.pause()

    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    async def test_empty_input_ignored(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = _EMPTY_CATALOG
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.value = ""
            await pilot.press("enter")
            await pilot.pause()


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
            catalog.action_pop_screen()
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
            table = app.screen.query_one("#settings-table")
            assert table is not None


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

        with mock.patch("lilbee.services.get_services", return_value=mock_svc):
            app = LilbeeApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                app.push_screen(StatusScreen())
                await pilot.pause()
                info = app.screen.query_one("#status-info")
                assert info is not None


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


class TestCanonicalModelsDir:
    def test_returns_platform_path(self) -> None:
        from lilbee.platform import canonical_models_dir

        result = canonical_models_dir()
        assert result.name == "models"
        assert "lilbee" in str(result)


class TestRemoteRow:
    def test_creates(self) -> None:
        from lilbee.model_manager import RemoteModel

        rm = RemoteModel(name="mistral:latest", task="chat", family="llama", parameter_size="7.2B")
        row = RemoteRow(rm)
        assert row.remote_model.name == "mistral:latest"


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
            task_bar = app._task_bar  # type: ignore[attr-defined]
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
            screen._streaming = True  # type: ignore[attr-defined]
            await app.action_quit()
            await pilot.pause()
            assert not screen._streaming  # type: ignore[attr-defined]
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
        assert "? help" in visible
        assert "^c cancel/quit" in visible
        # F-key and ctrl-alternative bindings should be hidden
        assert not any("Models" in d for d in visible)
        assert not any("Status" in d for d in visible)
        assert not any("Settings" in d for d in visible)
        assert not any("Theme" in d for d in visible)

    def test_chat_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        visible = self._visible_bindings(ChatScreen.BINDINGS)
        assert "/ commands" in visible
        assert len(visible) <= 3

    def test_catalog_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        visible = self._visible_bindings(CatalogScreen.BINDINGS)
        assert "q back" in visible
        assert "/ search" in visible
        assert "d delete" in visible
        assert len(visible) <= 5

    def test_status_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.status import StatusScreen

        visible = self._visible_bindings(StatusScreen.BINDINGS)
        assert "q back" in visible
        assert len(visible) <= 3

    def test_settings_bindings_minimal(self) -> None:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        visible = self._visible_bindings(SettingsScreen.BINDINGS)
        assert "q back" in visible
        assert len(visible) <= 3


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
