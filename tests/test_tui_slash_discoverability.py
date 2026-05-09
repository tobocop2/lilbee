"""Tests for the slash-command discoverability surfaces:

* :class:`SlashCommandCatalog` modal (open, filter, select, dismiss)
* :class:`ArgHintLine` widget (visibility states + content)
* :class:`HelpHint` footer chip (renders, click opens modal)
* Chat-screen wiring (``/help`` opens the modal, picked command lands in input)
* Cold-start placeholder copy
"""

from __future__ import annotations

from unittest import mock

import pytest
from textual.app import ComposeResult
from textual.widgets import Input, OptionList

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.command_registry import COMMANDS
from lilbee.cli.tui.widgets.arg_hint import ArgHintLine, _hint_for
from lilbee.cli.tui.widgets.help_hint import HELP_HINT_COMMANDS, HELP_HINT_KEYS, HelpHint
from lilbee.cli.tui.widgets.slash_command_catalog import (
    CATALOG_GROUPS,
    CATALOG_NO_MATCH,
    SlashCommandCatalog,
)
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import LilbeeAppHost


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


@pytest.fixture(autouse=True)
def _suppress_catalog_auto_hf_fetch():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    with mock.patch.object(CatalogScreen, "_fetch_all_hf_models"):
        yield


@pytest.fixture()
def _mock_resolve():
    with mock.patch(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        return_value=cfg.models_dir / "fake.gguf",
    ):
        yield


@pytest.fixture()
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


class _CatalogOnlyApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(SlashCommandCatalog(), self._on_pick)
        self.last_pick: str | None = "<unset>"  # type: ignore[attr-defined]

    def _on_pick(self, name: str | None) -> None:
        self.last_pick = name  # type: ignore[attr-defined]


class _ChatHostApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


class TestArgHintHelper:
    """Pure-function checks for the hint-rendering helper."""

    def test_empty_input_hides(self) -> None:
        assert _hint_for("") is None

    def test_plain_prose_hides(self) -> None:
        assert _hint_for("hello world") is None

    def test_unknown_command_hides(self) -> None:
        assert _hint_for("/notarealcmd") is None
        assert _hint_for("/notarealcmd args") is None

    def test_known_command_no_space_hides(self) -> None:
        # Still typing the command name itself; don't crowd the user.
        assert _hint_for("/mod") is None
        assert _hint_for("/model") is None

    def test_known_command_with_space_renders(self) -> None:
        result = _hint_for("/model ")
        assert result is not None
        assert "/model" in str(result)
        assert "[name]" in str(result)
        assert "Switch chat model" in str(result)

    def test_known_command_with_partial_arg_renders(self) -> None:
        result = _hint_for("/model gpt")
        assert result is not None
        assert "/model" in str(result)

    def test_aliased_command_resolves(self) -> None:
        # /q is an alias of /quit.
        result = _hint_for("/q ")
        assert result is not None
        assert "/quit" in str(result)
        assert "Exit lilbee" in str(result)


class TestArgHintWidgetAsync:
    async def test_starts_hidden(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test():
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            hint = screen.query_one("#arg-hint", ArgHintLine)
            assert hint.display is False

    async def test_typing_slash_with_space_shows_hint(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            chat_input = screen.query_one("#chat-input")
            chat_input.value = "/model "
            await pilot.pause()
            hint = screen.query_one("#arg-hint", ArgHintLine)
            assert hint.display is True

    async def test_clearing_input_hides_hint(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            chat_input = screen.query_one("#chat-input")
            chat_input.value = "/model "
            await pilot.pause()
            chat_input.value = ""
            await pilot.pause()
            hint = screen.query_one("#arg-hint", ArgHintLine)
            assert hint.display is False


class TestSlashCommandCatalogAsync:
    async def test_lists_every_registry_command(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            ids = [
                ol.get_option_at_index(i).id
                for i in range(ol.option_count)
                if ol.get_option_at_index(i).id is not None
            ]
            registry_names = {cmd.name for cmd in COMMANDS}
            # Every registry command appears at least once.
            assert registry_names.issubset(set(ids))

    async def test_shows_all_group_headers(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            seen_titles: list[str] = []
            for i in range(ol.option_count):
                opt = ol.get_option_at_index(i)
                if opt.id is None and opt.disabled:
                    seen_titles.append(str(opt.prompt))
            for group in CATALOG_GROUPS:
                assert any(group.title in title for title in seen_titles), (
                    f"missing header {group.title!r} in {seen_titles}"
                )

    async def test_filter_narrows_to_matching_command(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            filter_input = app.screen.query_one("#catalog-filter", Input)
            filter_input.value = "wiki"
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            runnable_ids = [
                ol.get_option_at_index(i).id
                for i in range(ol.option_count)
                if ol.get_option_at_index(i).id is not None
            ]
            assert "/wiki" in runnable_ids
            # /model, /clear etc. should be filtered out.
            assert "/model" not in runnable_ids
            assert "/clear" not in runnable_ids

    async def test_filter_with_no_match_shows_placeholder(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            filter_input = app.screen.query_one("#catalog-filter", Input)
            filter_input.value = "xxxnotacommandxxx"
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            assert ol.option_count == 1
            only = ol.get_option_at_index(0)
            assert only.id is None
            assert CATALOG_NO_MATCH in str(only.prompt)

    async def test_escape_dismisses_with_none(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.last_pick is None  # type: ignore[attr-defined]


class TestChatScreenIntegrationAsync:
    async def test_help_command_opens_catalog(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            chat_screen = app.screen
            assert isinstance(chat_screen, ChatScreen)
            chat_screen._cmd_help("")
            await pilot.pause()
            assert isinstance(app.screen, SlashCommandCatalog)

    async def test_insert_slash_command_fills_input(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen.insert_slash_command("/wiki")
            await pilot.pause()
            assert screen.query_one("#chat-input").value == "/wiki "

    async def test_help_hint_chip_renders(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test():
            chip = app.screen.query_one("#help-hint", HelpHint)
            rendered = str(chip.render())
            assert HELP_HINT_COMMANDS in rendered
            assert HELP_HINT_KEYS in rendered

    async def test_arg_hint_widget_mounted_in_chat(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test():
            hint = app.screen.query_one("#arg-hint", ArgHintLine)
            assert hint is not None


class TestPlaceholderCopy:
    def test_placeholder_mentions_both_slash_and_question(self) -> None:
        text = msg.CHAT_INPUT_PLACEHOLDER_DEFAULT
        assert "/" in text
        assert "?" in text
