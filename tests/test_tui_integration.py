"""Integration tests for the Textual TUI.

These tests exercise end-to-end flows through the TUI, verifying that
multiple components work together correctly. Unit tests for individual
widgets live in test_tui_widgets.py; screen-level tests in test_tui_screens.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Footer, Input

from lilbee.catalog import CatalogModel, CatalogResult
from lilbee.config import cfg

_EMPTY_CATALOG = CatalogResult(total=0, limit=25, offset=0, models=[])


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.chat_model = "test-model:latest"
    cfg.embedding_model = "test-embed:latest"
    cfg.vision_model = ""
    cfg.chunk_size = 512
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _make_model(
    name: str = "test-7B",
    task: str = "chat",
    featured: bool = False,
    size_gb: float = 4.0,
) -> CatalogModel:
    return CatalogModel(
        name=name,
        hf_repo=f"org/{name}-GGUF",
        gguf_filename="test.gguf",
        size_gb=size_gb,
        min_ram_gb=8.0,
        description="A test model",
        featured=featured,
        downloads=100,
        task=task,
    )


class _ChatApp(App[None]):
    """Minimal app that pushes ChatScreen for integration tests."""

    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


class _FullApp(App[None]):
    """Minimal app with Footer only, for pushing arbitrary screens."""

    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()


class TestCommandRegistry:
    """Verify the command registry is the single source of truth."""

    def test_all_commands_in_dispatch(self) -> None:
        from lilbee.cli.tui.command_registry import COMMANDS, build_dispatch_dict

        dispatch = build_dispatch_dict()
        for cmd in COMMANDS:
            assert cmd.name in dispatch
            for alias in cmd.aliases:
                assert alias in dispatch

    def test_help_text_includes_all_commands(self) -> None:
        from lilbee.cli.tui.command_registry import COMMANDS, help_text

        text = help_text()
        for cmd in COMMANDS:
            assert cmd.name in text

    def test_command_names_are_primary_only(self) -> None:
        from lilbee.cli.tui.command_registry import COMMANDS, command_names

        names = command_names()
        for cmd in COMMANDS:
            assert cmd.name in names
            for alias in cmd.aliases:
                assert alias not in names


class TestMessagesConstants:
    """Verify message constants are used consistently."""

    def test_all_message_constants_are_strings(self) -> None:
        from lilbee.cli.tui import messages

        for name in dir(messages):
            if name.isupper() and not name.startswith("_"):
                assert isinstance(getattr(messages, name), str)

    def test_cmd_unknown_format(self) -> None:
        from lilbee.cli.tui.messages import CMD_UNKNOWN

        result = CMD_UNKNOWN.format(cmd="/foobar")
        assert "/foobar" in result

    def test_sync_file_progress_format(self) -> None:
        from lilbee.cli.tui.messages import SYNC_FILE_PROGRESS

        result = SYNC_FILE_PROGRESS.format(current=1, total=5, file="test.pdf")
        assert "1" in result
        assert "5" in result
        assert "test.pdf" in result


class TestChatScreenIntegration:
    """End-to-end chat screen tests."""

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_startup_shows_input_focused(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)
            assert inp is not None
            assert inp.has_focus

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_send_message_creates_widgets(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app.screen, "_stream_response"):
                inp = app.screen.query_one("#chat-input", Input)
                inp.value = "What is RAG?"
                await pilot.press("enter")
                assert len(app.screen._history) == 1
                assert app.screen._streaming is True

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_empty_input_ignored(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)
            inp.value = ""
            await pilot.press("enter")
            assert len(app.screen._history) == 0

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_input_not_chat_input_ignored(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            event = MagicMock()
            event.input = MagicMock()
            event.input.id = "other-input"
            event.value = "test"
            app.screen.on_input_submitted(event)
            assert len(app.screen._history) == 0

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_cancel_stream_while_idle(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_cancel_stream()
            assert app.screen._streaming is False


class TestSlashCommandIntegration:
    """Slash commands dispatched through the registry end-to-end."""

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_help_shows_modal(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/help")
            await pilot.pause()
            from lilbee.cli.tui.widgets.help_modal import HelpModal

            assert isinstance(app.screen, HelpModal)

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_h_alias(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/h")
            await pilot.pause()
            from lilbee.cli.tui.widgets.help_modal import HelpModal

            assert isinstance(app.screen, HelpModal)

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_quit_exits(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app, "exit") as mock_exit:
                app.screen._handle_slash("/quit")
                mock_exit.assert_called_once()

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_q_alias(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app, "exit") as mock_exit:
                app.screen._handle_slash("/q")
                mock_exit.assert_called_once()

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_exit_alias(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app, "exit") as mock_exit:
                app.screen._handle_slash("/exit")
                mock_exit.assert_called_once()

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_model_switches(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/model qwen3:8b")
            assert cfg.chat_model == "qwen3:8b"

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_model_no_arg_opens_catalog(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with (
                patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
                patch("lilbee.model_manager.classify_ollama_models", return_value=[]),
            ):
                app.screen._handle_slash("/model")
                await pilot.pause()
                from lilbee.cli.tui.screens.catalog import CatalogScreen

                assert isinstance(app.screen, CatalogScreen)

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_set_updates_config(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/set top_k 10")
            assert cfg.top_k == 10

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_set_invalid_value(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            old = cfg.top_k
            app.screen._handle_slash("/set top_k not-a-number")
            assert cfg.top_k == old

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_set_nullable_type(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/set temperature none")
            assert cfg.temperature is None

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_status_shows_info(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.store.get_sources", return_value=[]):
                app.screen._handle_slash("/status")
                await pilot.pause()
                from lilbee.cli.tui.screens.status import StatusScreen

                assert isinstance(app.screen, StatusScreen)

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_unknown_shows_error(self, mock_check: MagicMock) -> None:

        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/foobar")
            # Verify no crash; the notification uses CMD_UNKNOWN format

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_add_nonexistent_path(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/add /nonexistent/path/abc.txt")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_add_with_file(self, mock_check: MagicMock, tmp_path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with (
                patch("lilbee.cli.helpers.copy_paths", return_value=["test.txt"]),
                patch.object(app.screen, "_run_sync"),
            ):
                app.screen._handle_slash(f"/add {test_file}")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_add_copy_exception(self, mock_check: MagicMock, tmp_path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.cli.helpers.copy_paths", side_effect=OSError("disk full")):
                app.screen._handle_slash(f"/add {test_file}")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_delete_removes_document(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with (
                patch("lilbee.cli.tui.screens.chat.get_sources") as mock_get,
                patch("lilbee.cli.tui.screens.chat.delete_by_source") as mock_del_chunks,
                patch("lilbee.cli.tui.screens.chat.delete_source") as mock_del_src,
            ):
                mock_get.return_value = [{"filename": "notes.md", "source": "notes.md"}]
                app.screen._handle_slash("/delete notes.md")
                mock_del_chunks.assert_called_once_with("notes.md")
                mock_del_src.assert_called_once_with("notes.md")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_delete_no_documents(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.store.get_sources", return_value=[]):
                app.screen._handle_slash("/delete x")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_delete_unknown_name(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch(
                "lilbee.store.get_sources",
                return_value=[{"filename": "notes.md", "source": "notes.md"}],
            ):
                app.screen._handle_slash("/delete nonexistent.md")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_delete_no_arg_shows_list(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch(
                "lilbee.store.get_sources",
                return_value=[{"filename": "a.md", "source": "a.md"}],
            ):
                app.screen._handle_slash("/delete")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_cancel(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/cancel")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_version(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.cli.helpers.get_version", return_value="1.0.0"):
                app.screen._handle_slash("/version")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_reset_confirm(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.cli.helpers.perform_reset"):
                app.screen._handle_slash("/reset confirm")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_reset_no_confirm(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/reset")


class TestAutocompleteIntegration:
    """Tab completion from the chat input through the overlay."""

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_slash_prefix_shows_completions(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)
            inp.value = "/he"
            with patch(
                "lilbee.cli.tui.screens.chat.get_completions",
                return_value=["/help"],
            ):
                app.screen.action_complete()
                assert inp.value == "/help"

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    @patch("lilbee.models.list_installed_models", return_value=["qwen3:8b", "mistral:7b"])
    async def test_model_name_completion(
        self, mock_models: MagicMock, mock_check: MagicMock
    ) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)
            inp.value = "/model qw"
            with patch(
                "lilbee.cli.tui.screens.chat.get_completions",
                return_value=["qwen3:8b"],
            ):
                app.screen.action_complete()
                assert "qwen3:8b" in inp.value


class TestGlobalKeybindings:
    """App-level keybindings from LilbeeApp."""

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_f1_opens_help(self, mock_check: MagicMock) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_push_help()
            await pilot.pause()
            from lilbee.cli.tui.widgets.help_modal import HelpModal

            assert isinstance(app.screen, HelpModal)

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_f2_opens_catalog(self, mock_check: MagicMock) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with (
                patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
                patch("lilbee.model_manager.classify_ollama_models", return_value=[]),
            ):
                app.action_push_catalog()
                await pilot.pause()
                from lilbee.cli.tui.screens.catalog import CatalogScreen

                assert isinstance(app.screen, CatalogScreen)

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_f3_opens_status(self, mock_check: MagicMock) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.store.get_sources", return_value=[]):
                app.action_push_status()
                await pilot.pause()
                from lilbee.cli.tui.screens.status import StatusScreen

                assert isinstance(app.screen, StatusScreen)

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_f4_opens_settings(self, mock_check: MagicMock) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_push_settings()
            await pilot.pause()
            from lilbee.cli.tui.screens.settings import SettingsScreen

            assert isinstance(app.screen, SettingsScreen)

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_ctrl_t_cycles_theme(self, mock_check: MagicMock) -> None:
        from lilbee.cli.tui.app import DARK_THEMES, LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_cycle_theme()
            assert app.theme == DARK_THEMES[1]

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_theme_cycles_wraps_around(self, mock_check: MagicMock) -> None:
        from lilbee.cli.tui.app import DARK_THEMES, LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for _ in range(len(DARK_THEMES)):
                app.action_cycle_theme()
            app.action_cycle_theme()
            assert app.theme == DARK_THEMES[1]


class TestChatKeybindings:
    """Chat screen keybindings."""

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_pageup_scrolls_chat(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_scroll_up()

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_pagedown_scrolls_chat(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_scroll_down()

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_escape_cancels_stream(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._streaming = True
            app.screen.action_cancel_stream()
            assert app.screen._streaming is False

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_j_scrolls_down_vim(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.containers import VerticalScroll

            log = app.screen.query_one("#chat-log", VerticalScroll)
            log.focus()
            app.screen.key_j()

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_k_scrolls_up_vim(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.containers import VerticalScroll

            log = app.screen.query_one("#chat-log", VerticalScroll)
            log.focus()
            app.screen.key_k()


class TestCatalogKeybindings:
    """Catalog screen keybindings."""

    async def test_catalog_escape_closes(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            with (
                patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
                patch("lilbee.model_manager.classify_ollama_models", return_value=[]),
            ):
                screen = CatalogScreen()
                app.push_screen(screen)
                await pilot.pause()
                screen.action_pop_screen()
                await pilot.pause()
                assert not isinstance(app.screen, CatalogScreen)

    async def test_catalog_slash_focuses_search(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            with (
                patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
                patch("lilbee.model_manager.classify_ollama_models", return_value=[]),
            ):
                screen = CatalogScreen()
                app.push_screen(screen)
                await pilot.pause()
                screen.action_focus_search()
                assert app.screen.query_one("#catalog-search", Input).has_focus

    async def test_catalog_sort_cycles(self) -> None:
        from textual.widgets import ListView

        from lilbee.cli.tui.screens.catalog import CatalogScreen

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            with (
                patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
                patch("lilbee.model_manager.classify_ollama_models", return_value=[]),
            ):
                screen = CatalogScreen()
                app.push_screen(screen)
                await pilot.pause()
                lv = screen.query_one("#catlist-all", ListView)
                lv.focus()
                await pilot.pause()
                old_sort = screen._current_sort
                screen.action_cycle_sort()
                assert screen._current_sort != old_sort


class TestStatusKeybindings:
    async def test_status_escape_closes(self) -> None:
        from lilbee.cli.tui.screens.status import StatusScreen

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            with patch("lilbee.store.get_sources", return_value=[]):
                screen = StatusScreen()
                app.push_screen(screen)
                await pilot.pause()
                screen.action_pop_screen()
                await pilot.pause()
                assert not isinstance(app.screen, StatusScreen)

    async def test_status_j_k_navigates_table(self) -> None:
        from lilbee.cli.tui.screens.status import StatusScreen

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            with patch(
                "lilbee.store.get_sources",
                return_value=[
                    {"source": "a.md", "chunk_count": 1, "content_type": "text/markdown"},
                    {"source": "b.md", "chunk_count": 2, "content_type": "text/markdown"},
                ],
            ):
                screen = StatusScreen()
                app.push_screen(screen)
                await pilot.pause()
                screen.key_j()
                screen.key_k()


class TestSettingsKeybindings:
    async def test_settings_escape_closes(self) -> None:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = SettingsScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen.action_pop_screen()
            await pilot.pause()
            assert not isinstance(app.screen, SettingsScreen)

    async def test_settings_j_k_navigates_table(self) -> None:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = SettingsScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen.key_j()
            screen.key_k()


class TestHelpModal:
    async def test_help_escape_closes(self) -> None:
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(HelpModal())
            await pilot.pause()
            app.screen.action_close()
            await pilot.pause()
            assert not isinstance(app.screen, HelpModal)

    async def test_help_lists_all_commands(self) -> None:
        from lilbee.cli.tui.command_registry import COMMANDS
        from lilbee.cli.tui.widgets.help_modal import _HELP_TEXT

        for cmd in COMMANDS:
            assert cmd.name in _HELP_TEXT


class TestDownloadModalIntegration:
    @patch("lilbee.catalog.download_model")
    async def test_download_modal_shows_progress(self, mock_dl: MagicMock) -> None:
        from lilbee.cli.tui.widgets.download_modal import DownloadModal

        def capture_progress(model, *, on_progress=None):
            if on_progress:
                on_progress(50, 100)

        mock_dl.side_effect = capture_progress
        m = _make_model("Test")
        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(DownloadModal(m))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()

    async def test_download_modal_escape_cancels(self) -> None:
        import threading

        from lilbee.cli.tui.widgets.download_modal import DownloadModal

        m = _make_model("Test")
        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            with patch("lilbee.catalog.download_model") as mock_dl:
                evt = threading.Event()
                mock_dl.side_effect = lambda *a, **kw: evt.wait(5)
                app.push_screen(DownloadModal(m))
                await pilot.pause()
                app.screen.action_cancel()
                evt.set()
                await pilot.pause()


class TestSetupModalIntegration:
    async def test_setup_modal_lists_embedding_models(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupModal(ollama_embeddings=["nomic:latest"]))
            await pilot.pause()
            assert len(app.screen_stack) == 2

    async def test_setup_modal_cancel_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _FullApp()
        results: list[object] = []
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupModal(), callback=lambda r: results.append(r))
            await pilot.pause()
            app.screen.action_cancel()
            await pilot.pause()
        assert None in results


class TestAssistantMessageIntegration:
    """Tests for AssistantMessage widget branches."""

    async def test_reasoning_widget_none_is_noop(self) -> None:
        from lilbee.cli.tui.widgets.message import AssistantMessage

        am = AssistantMessage()
        am._reasoning_widget = None
        am.append_reasoning("test")
        assert am._reasoning_parts == ["test"]

    async def test_content_widget_none_is_noop(self) -> None:
        from lilbee.cli.tui.widgets.message import AssistantMessage

        am = AssistantMessage()
        am._md_widget = None
        am.append_content("test")
        assert am._content_parts == ["test"]

    async def test_finish_no_reasoning_hides_widget(self) -> None:
        from lilbee.cli.tui.widgets.message import AssistantMessage

        class _MsgApp(App):
            def compose(self) -> ComposeResult:
                self._am = AssistantMessage()
                yield self._am

        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._am.finish(sources=None)
            assert app._am._reasoning_widget.display is False

    async def test_finish_with_sources_shows_citation(self) -> None:
        from lilbee.cli.tui.widgets.message import AssistantMessage

        class _MsgApp(App):
            def compose(self) -> ComposeResult:
                self._am = AssistantMessage()
                yield self._am

        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._am.append_reasoning("think")
            app._am.finish(sources=["doc.pdf:1"])
            assert app._am._finished is True

    async def test_finish_no_sources_hides_citation(self) -> None:
        from lilbee.cli.tui.widgets.message import AssistantMessage

        class _MsgApp(App):
            def compose(self) -> ComposeResult:
                self._am = AssistantMessage()
                yield self._am

        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._am.finish(sources=[])
            assert app._am._citation_widget.display is False


class TestSuggesterIntegration:
    async def test_unknown_command_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        result = await s.get_suggestion("/unknowncmd xyz")
        assert result is None


class TestOverlayEdgeCases:
    async def test_cycle_wraps_around(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        class _App(App):
            def compose(self) -> ComposeResult:
                yield CompletionOverlay()

        app = _App()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            overlay.show_completions(["/a", "/b", "/c"])
            overlay.cycle_next()
            overlay.cycle_next()
            result = overlay.cycle_next()
            assert result == "/a"

    async def test_get_current_out_of_bounds(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        class _App(App):
            def compose(self) -> ComposeResult:
                yield CompletionOverlay()

        app = _App()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            overlay._options = ["/a"]
            overlay._index = 5
            assert overlay.get_current() is None


class TestHistoryManagement:
    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_trim_history(self, mock_check: MagicMock) -> None:
        from lilbee.cli.tui.screens.chat import _MAX_HISTORY_MESSAGES

        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._history = [
                {"role": "user", "content": f"m-{i}"} for i in range(_MAX_HISTORY_MESSAGES + 20)
            ]
            app.screen._trim_history()
            assert len(app.screen._history) == _MAX_HISTORY_MESSAGES

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_trim_history_under_limit_noop(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._history = [{"role": "user", "content": "hi"}]
            app.screen._trim_history()
            assert len(app.screen._history) == 1


class TestAutoSync:
    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    @patch("lilbee.cli.tui.screens.chat.ChatScreen._run_sync")
    async def test_auto_sync_true_triggers_sync(
        self, mock_sync: MagicMock, mock_check: MagicMock
    ) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp(auto_sync=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            mock_sync.assert_called()


class TestVisionCommands:
    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_vision_set(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.settings.set_value"):
                app.screen._cmd_vision("llava:latest")
                assert cfg.vision_model == "llava:latest"

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_vision_off(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            cfg.vision_model = "some-model"
            with patch("lilbee.settings.set_value"):
                app.screen._cmd_vision("off")
                assert cfg.vision_model == ""

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_vision_no_arg_shows_status(self, mock_check: MagicMock) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._cmd_vision("")


class TestThemeCommand:
    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_theme_with_arg(self, mock_check: MagicMock) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/theme dracula")

    @patch("lilbee.cli.tui.screens.chat.ChatScreen._check_embedding_model_async")
    async def test_theme_no_arg_lists_themes(self, mock_check: MagicMock) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/theme")
