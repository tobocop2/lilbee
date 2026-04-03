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
    with (
        patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
    ):
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

    async def test_startup_shows_input_focused(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)
            assert inp is not None
            assert inp.has_focus

    async def test_send_message_creates_widgets(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app.screen, "_stream_response"):
                inp = app.screen.query_one("#chat-input", Input)
                inp.value = "What is RAG?"
                await pilot.press("enter")
                assert len(app.screen._history) == 1
                assert app.screen._streaming is True

    async def test_empty_input_ignored(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)
            inp.value = ""
            await pilot.press("enter")
            assert len(app.screen._history) == 0

    async def test_input_not_chat_input_ignored(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            event = MagicMock()
            event.input = MagicMock()
            event.input.id = "other-input"
            event.value = "test"
            app.screen.on_input_submitted(event)
            assert len(app.screen._history) == 0

    async def test_cancel_stream_while_idle(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_cancel_stream()
            assert app.screen._streaming is False


class TestSlashCommandIntegration:
    """Slash commands dispatched through the registry end-to-end."""

    async def test_slash_help_shows_modal(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/help")
            await pilot.pause()
            from lilbee.cli.tui.widgets.help_modal import HelpModal

            assert isinstance(app.screen, HelpModal)

    async def test_slash_h_alias(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/h")
            await pilot.pause()
            from lilbee.cli.tui.widgets.help_modal import HelpModal

            assert isinstance(app.screen, HelpModal)

    async def test_slash_quit_exits(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app, "exit") as mock_exit:
                app.screen._handle_slash("/quit")
                mock_exit.assert_called_once()

    async def test_slash_q_alias(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app, "exit") as mock_exit:
                app.screen._handle_slash("/q")
                mock_exit.assert_called_once()

    async def test_slash_exit_alias(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch.object(app, "exit") as mock_exit:
                app.screen._handle_slash("/exit")
                mock_exit.assert_called_once()

    async def test_slash_model_switches(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/model qwen3:8b")
            assert cfg.chat_model == "qwen3:8b"

    async def test_slash_model_no_arg_opens_catalog(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with (
                patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
                patch("lilbee.model_manager.classify_remote_models", return_value=[]),
            ):
                app.screen._handle_slash("/model")
                await pilot.pause()
                from lilbee.cli.tui.screens.catalog import CatalogScreen

                assert isinstance(app.screen, CatalogScreen)

    async def test_slash_set_updates_config(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/set top_k 10")
            assert cfg.top_k == 10

    async def test_slash_set_invalid_value(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            old = cfg.top_k
            app.screen._handle_slash("/set top_k not-a-number")
            assert cfg.top_k == old

    async def test_slash_set_nullable_type(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/set temperature none")
            assert cfg.temperature is None

    async def test_slash_status_shows_info(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.services.get_services") as mock_svc:
                mock_svc.return_value.store.get_sources.return_value = []
                app.screen._handle_slash("/status")
                await pilot.pause()
                from lilbee.cli.tui.screens.status import StatusScreen

                assert isinstance(app.screen, StatusScreen)

    async def test_slash_unknown_shows_error(self) -> None:

        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/foobar")
            # Verify no crash; the notification uses CMD_UNKNOWN format

    async def test_slash_add_nonexistent_path(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/add /nonexistent/path/abc.txt")

    async def test_slash_add_with_file(self, tmp_path) -> None:
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

    async def test_slash_add_copy_exception(self, tmp_path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.cli.helpers.copy_paths", side_effect=OSError("disk full")):
                app.screen._handle_slash(f"/add {test_file}")

    async def test_slash_delete_removes_document(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with (
                patch("lilbee.services.get_services") as mock_svc,
            ):
                mock_store = mock_svc.return_value.store
                mock_store.get_sources.return_value = [
                    {"filename": "notes.md", "source": "notes.md"}
                ]
                app.screen._handle_slash("/delete notes.md")
                mock_store.delete_by_source.assert_called_once_with("notes.md")
                mock_store.delete_source.assert_called_once_with("notes.md")

    async def test_slash_delete_no_documents(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.services.get_services") as mock_svc:
                mock_svc.return_value.store.get_sources.return_value = []
                app.screen._handle_slash("/delete x")

    async def test_slash_delete_unknown_name(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.services.get_services") as mock_svc:
                mock_svc.return_value.store.get_sources.return_value = [
                    {"filename": "notes.md", "source": "notes.md"}
                ]
                app.screen._handle_slash("/delete nonexistent.md")

    async def test_slash_delete_no_arg_shows_list(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.services.get_services") as mock_svc:
                mock_svc.return_value.store.get_sources.return_value = [
                    {"filename": "a.md", "source": "a.md"}
                ]
                app.screen._handle_slash("/delete")

    async def test_slash_cancel(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/cancel")

    async def test_slash_version(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.cli.helpers.get_version", return_value="1.0.0"):
                app.screen._handle_slash("/version")

    async def test_slash_reset_confirm(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.cli.helpers.perform_reset"):
                app.screen._handle_slash("/reset confirm")

    async def test_slash_reset_no_confirm(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/reset")


class TestAutocompleteIntegration:
    """Tab completion from the chat input through the overlay."""

    async def test_slash_prefix_shows_completions(self) -> None:
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

    @patch("lilbee.models.list_installed_models", return_value=["qwen3:8b", "mistral:7b"])
    async def test_model_name_completion(self, mock_models: MagicMock) -> None:
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

    async def test_f1_opens_help(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_push_help()
            await pilot.pause()
            from lilbee.cli.tui.widgets.help_modal import HelpModal

            assert isinstance(app.screen, HelpModal)

    async def test_f2_opens_catalog(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with (
                patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
                patch("lilbee.model_manager.classify_remote_models", return_value=[]),
            ):
                app.action_push_catalog()
                await pilot.pause()
                from lilbee.cli.tui.screens.catalog import CatalogScreen

                assert isinstance(app.screen, CatalogScreen)

    async def test_f3_opens_status(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.services.get_services") as mock_svc:
                mock_svc.return_value.store.get_sources.return_value = []
                app.action_push_status()
                await pilot.pause()
                from lilbee.cli.tui.screens.status import StatusScreen

                assert isinstance(app.screen, StatusScreen)

    async def test_f4_opens_settings(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_push_settings()
            await pilot.pause()
            from lilbee.cli.tui.screens.settings import SettingsScreen

            assert isinstance(app.screen, SettingsScreen)

    async def test_ctrl_t_cycles_theme(self) -> None:
        from lilbee.cli.tui.app import DARK_THEMES, LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_cycle_theme()
            assert app.theme == DARK_THEMES[1]

    async def test_theme_cycles_wraps_around(self) -> None:
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

    async def test_pageup_scrolls_chat(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_scroll_up()

    async def test_pagedown_scrolls_chat(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_scroll_down()

    async def test_escape_cancels_stream(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._streaming = True
            app.screen.action_cancel_stream()
            assert app.screen._streaming is False

    async def test_j_scrolls_down_vim(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.containers import VerticalScroll

            log = app.screen.query_one("#chat-log", VerticalScroll)
            log.focus()
            app.screen.key_j()

    async def test_k_scrolls_up_vim(self) -> None:
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
                patch("lilbee.model_manager.classify_remote_models", return_value=[]),
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
                patch("lilbee.model_manager.classify_remote_models", return_value=[]),
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
                patch("lilbee.model_manager.classify_remote_models", return_value=[]),
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
            with patch("lilbee.services.get_services") as mock_svc:
                mock_svc.return_value.store.get_sources.return_value = []
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
            with patch("lilbee.services.get_services") as mock_svc:
                mock_svc.return_value.store.get_sources.return_value = [
                    {"source": "a.md", "chunk_count": 1, "content_type": "text/markdown"},
                    {"source": "b.md", "chunk_count": 2, "content_type": "text/markdown"},
                ]
                screen = StatusScreen()
                app.push_screen(screen)
                await pilot.pause()
                screen.action_cursor_down()
                screen.action_cursor_up()


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
            screen.action_cursor_down()
            screen.action_cursor_up()


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


class TestSetupWizardIntegration:
    async def test_setup_wizard_mounts(self) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = _FullApp()
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            assert len(app.screen_stack) == 2

    async def test_setup_wizard_cancel_returns_skipped(self) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = _FullApp()
        results: list[object] = []
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard(), callback=lambda r: results.append(r))
            await pilot.pause()
            app.screen.action_cancel()
            await pilot.pause()
        assert "skipped" in results


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
    async def test_trim_history(self) -> None:
        from lilbee.cli.tui.screens.chat import _MAX_HISTORY_MESSAGES

        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._history = [
                {"role": "user", "content": f"m-{i}"} for i in range(_MAX_HISTORY_MESSAGES + 20)
            ]
            app.screen._trim_history()
            assert len(app.screen._history) == _MAX_HISTORY_MESSAGES

    async def test_trim_history_under_limit_noop(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._history = [{"role": "user", "content": "hi"}]
            app.screen._trim_history()
            assert len(app.screen._history) == 1


class TestAutoSync:
    @patch("lilbee.cli.tui.screens.chat.ChatScreen._run_sync")
    @patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=True)
    async def test_auto_sync_true_triggers_sync(
        self, mock_embed: MagicMock, mock_sync: MagicMock
    ) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp(auto_sync=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            mock_sync.assert_called()


class TestVisionCommands:
    async def test_vision_set(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with patch("lilbee.settings.set_value"):
                app.screen._cmd_vision("llava:latest")
                assert cfg.vision_model == "llava:latest"

    async def test_vision_off(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            cfg.vision_model = "some-model"
            with patch("lilbee.settings.set_value"):
                app.screen._cmd_vision("off")
                assert cfg.vision_model == ""

    async def test_vision_no_arg_shows_status(self) -> None:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._cmd_vision("")


class TestThemeCommand:
    async def test_theme_with_arg(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/theme dracula")

    async def test_theme_no_arg_lists_themes(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/theme")


# ---------------------------------------------------------------------------
# Integration tests for real download progress (no mocks)
# ---------------------------------------------------------------------------


import pytest

pytestmark = pytest.mark.slow


class TestRealDownloadProgress:
    """Integration tests verifying real download progress works from HuggingFace.
    
    These tests download actual files from HuggingFace (no mocks) to verify
    the full progress callback chain works correctly. They use Qwen3 0.6B
    (~0.5GB) - the smallest featured GGUF model.
    
    Run with: uv run pytest tests/integration/test_tui_integration.py -v -m slow
    """

    @pytest.fixture(autouse=True)
    def _isolated_cfg(self, tmp_path):
        """Set up isolated config for each test."""
        from lilbee.config import cfg
        from lilbee.model_manager import reset_model_manager
        
        snapshot = cfg.model_copy()
        cfg.data_dir = tmp_path / "data"
        cfg.data_root = tmp_path
        cfg.documents_dir = tmp_path / "documents"
        cfg.models_dir = tmp_path / "models"
        cfg.lancedb_dir = tmp_path / "data" / "lancedb"
        cfg.chat_model = "test-model:latest"
        cfg.embedding_model = "test-embed:latest"
        cfg.vision_model = ""
        cfg.chunk_size = 512
        
        for d in [cfg.models_dir, cfg.data_dir, cfg.documents_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        yield
        
        reset_model_manager()
        for field in type(snapshot).model_fields:
            setattr(cfg, field, getattr(snapshot, field))

    async def test_real_download_progress_callback_invoked(self):
        """Download small file from HuggingFace and verify progress callbacks work.
        
        This test verifies the fix for progress_updater routing:
        - Progress callbacks should be invoked with real byte values
        - Not ignored (which was the original bug)
        
        Uses celinah/dummy-xet-testing (~1MB file) for fast testing.
        """
        from lilbee.catalog import CatalogModel, download_model
        from lilbee.config import cfg
        
        # Use small test file from xet-enabled repo
        entry = CatalogModel(
            name="dummy-xet-test",
            hf_repo="celinah/dummy-xet-testing",
            gguf_filename="dummy.safetensors",
            size_gb=0.001,  # ~1MB
            min_ram_gb=1,
            description="Test file for progress verification",
            featured=False,
            downloads=0,
            task="chat",
        )
        
        progress_calls = []
        
        def on_progress(downloaded: int, total: int):
            progress_calls.append((downloaded, total))
        
        # REAL download - no mocks!
        result = download_model(entry, on_progress=on_progress)
        
        # KEY ASSERTIONS - prove the fix works
        assert len(progress_calls) > 0, (
            "Progress callback never invoked! "
            "This means progress_updater is not being passed through correctly. "
            "The fix in huggingface_hub/_download_to_tmp_and_move is not working."
        )
        
        # Verify we received real byte progress (not zeros)
        total_bytes = sum(downloaded for downloaded, total in progress_calls)
        assert total_bytes > 0, (
            "Progress callback received zero bytes! "
            "The callback was invoked but with zero values."
        )
        
        # Verify we downloaded the expected amount (~1MB)
        expected_size = 1 * 1024 * 1024  # ~1MB
        assert total_bytes >= expected_size * 0.8, (
            f"Total bytes {total_bytes} is much less than expected ~{expected_size}"
        )
        
        print(f"\n✓ Download progress verified")
        print(f"  Progress calls: {len(progress_calls)}")
        print(f"  Total bytes: {total_bytes / (1024*1024):.1f} MB")
        print(f"  First callback: {progress_calls[0]}")
        print(f"  Last callback: {progress_calls[-1]}")

    async def test_tui_download_no_fd_error_in_worker_thread(self):
        """Verify download with disable_progress_bars doesn't cause fd error.
        
        The 'bad value(s) in fds_to_keep' error occurred when tqdm
        was used in Textual worker threads. This test verifies the
        fix (disable_progress_bars + tqdm_class=None) works.
        """
        import threading
        from lilbee.catalog import CatalogModel
        from lilbee.config import cfg
        
        # Use small test file (~1MB)
        entry = CatalogModel(
            name="dummy-xet-test",
            hf_repo="celinah/dummy-xet-testing",
            gguf_filename="dummy.safetensors",
            size_gb=0.001,
            min_ram_gb=1,
            description="Test file",
            featured=False,
            downloads=0,
            task="chat",
        )
        
        # Track any errors from the worker thread
        worker_errors = []
        
        def download_in_thread():
            """Run download in a thread like Textual's @work(thread=True)."""
            try:
                # This is what setup.py does - disable progress bars before download
                from huggingface_hub.utils import disable_progress_bars
                disable_progress_bars()
                
                from lilbee.catalog import download_model
                result = download_model(entry, on_progress=lambda d, t: None)
            except Exception as e:
                error_msg = str(e)
                worker_errors.append(error_msg)
                # Check for the specific fd error
                if "fds_to_keep" in error_msg or "bad value" in error_msg.lower():
                    worker_errors.append(f"FD_ERROR: {error_msg}")
        
        # Run in a thread (simulating Textual's @work(thread=True))
        thread = threading.Thread(target=download_in_thread)
        thread.start()
        thread.join(timeout=120)  # 2 minute timeout
        
        # Verify thread completed without fd errors
        assert not thread.is_alive(), "Download timed out"
        
        # Check for fd error specifically
        fd_errors = [e for e in worker_errors if "FD_ERROR" in e or "fds_to_keep" in e]
        assert len(fd_errors) == 0, (
            f"FD error occurred in worker thread: {fd_errors}. "
            "The fix (disable_progress_bars + tqdm_class=None) is not working."
        )
        
        print(f"\n✓ Worker thread download completed without fd errors")

    async def test_setup_wizard_progress_bar_updates_during_download(self):
        """Verify TUI setup wizard progress bar updates during real download.
        
        This is the full integration test - runs the actual TUI with
        real download to verify the complete user flow works.
        """
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.setup import SetupWizard
        from lilbee.catalog import CatalogModel
        from lilbee.config import cfg
        
        # Use small test file (~1MB)
        entry = CatalogModel(
            name="dummy-xet-test",
            hf_repo="celinah/dummy-xet-testing",
            gguf_filename="dummy.safetensors",
            size_gb=0.001,
            min_ram_gb=1,
            description="Test file",
            featured=False,
            downloads=0,
            task="chat",
        )
        
        app = LilbeeApp()
        download_completed = False
        progress_updates = []
        
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            
            setup = app.screen
            
            # Override the progress callback to track updates
            original_download = setup._download_model
            
            def track_progress(model):
                """Wrapper that tracks progress."""
                nonlocal download_completed, progress_updates
                
                # Set up our own progress tracking
                def on_progress(downloaded, total):
                    progress_updates.append((downloaded, total))
                
                # Patch download_model to use our callback
                from unittest.mock import patch
                import lilbee.catalog as catalog_module
                
                original_download(model)
            
            # Run the download (this uses @work(thread=True) internally)
            setup._download_model(entry)
            
            # Wait for download to make progress (give it 30 seconds)
            await pilot.pause(30)
            
            # Verify progress updates occurred
            assert len(progress_updates) > 0, (
                "No progress updates received during TUI download! "
                "The progress callback chain is broken."
            )
            
            # Verify we're making real progress (bytes increasing)
            if len(progress_updates) >= 2:
                first_bytes = progress_updates[0][0]
                last_bytes = progress_updates[-1][0]
                assert last_bytes > first_bytes, (
                    f"Progress not advancing: first={first_bytes}, last={last_bytes}"
                )
            
            print(f"\n✓ TUI progress updates: {len(progress_updates)}")
            print(f"  First: {progress_updates[0] if progress_updates else 'N/A'}")
            print(f"  Last: {progress_updates[-1] if progress_updates else 'N/A'}")
