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
from lilbee.cli.tui.widgets.help_hint import HelpHint
from lilbee.cli.tui.widgets.slash_command_catalog import (
    CATALOG_GROUPS,
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
    last_pick: str | None = "<unset>"

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(SlashCommandCatalog(), self._on_pick)

    def _on_pick(self, name: str | None) -> None:
        self.last_pick = name


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
            assert msg.SLASH_CATALOG_NO_MATCH in str(only.prompt)

    async def test_escape_dismisses_with_none(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.last_pick is None

    async def test_filter_match_by_alias(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            filter_input = app.screen.query_one("#catalog-filter", Input)
            # "exit" is an alias of /quit but does NOT appear in any command
            # name; filtering by it must surface /quit via the alias-match
            # branch (not the name-match branch).
            filter_input.value = "exit"
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            ids = [
                ol.get_option_at_index(i).id
                for i in range(ol.option_count)
                if ol.get_option_at_index(i).id is not None
            ]
            assert "/quit" in ids

    async def test_enter_in_filter_picks_first_match(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            filter_input = app.screen.query_one("#catalog-filter", Input)
            filter_input.value = "wiki"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.last_pick == "/wiki"

    async def test_enter_with_no_match_stays_open(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            filter_input = app.screen.query_one("#catalog-filter", Input)
            filter_input.value = "xxxnotacommandxxx"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            # No runnable match means dismissal does not fire.
            assert app.last_pick == "<unset>"

    async def test_action_select_picks_highlighted(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            # Highlight the first runnable option (0 is a header).
            target = None
            for i in range(ol.option_count):
                if ol.get_option_at_index(i).id is not None:
                    target = i
                    break
            assert target is not None
            ol.highlighted = target
            picked = ol.get_option_at_index(target).id
            app.screen.action_select()
            await pilot.pause()
            assert app.last_pick == picked

    async def test_option_list_message_dismisses_with_command(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            # Find a runnable option and synthesize the OptionSelected message
            # the way OptionList itself would on click/enter.
            target_opt = None
            target_index = None
            for i in range(ol.option_count):
                opt = ol.get_option_at_index(i)
                if opt.id is not None:
                    target_opt = opt
                    target_index = i
                    break
            assert target_opt is not None and target_index is not None
            event = OptionList.OptionSelected(option_list=ol, option=target_opt, index=target_index)
            app.screen.on_option_list_option_selected(event)
            await pilot.pause()
            assert app.last_pick == target_opt.id

    async def test_option_list_message_with_disabled_header_no_op(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            header_opt = None
            header_index = None
            for i in range(ol.option_count):
                opt = ol.get_option_at_index(i)
                if opt.id is None:
                    header_opt = opt
                    header_index = i
                    break
            assert header_opt is not None and header_index is not None
            event = OptionList.OptionSelected(option_list=ol, option=header_opt, index=header_index)
            app.screen.on_option_list_option_selected(event)
            await pilot.pause()
            # Header rows have no command id, so dismissal must not fire.
            assert app.last_pick == "<unset>"

    async def test_input_changed_for_unrelated_input_ignored(self) -> None:
        # Synthesize an Input.Changed for a different widget to drive the
        # early-return guard in on_input_changed.
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            other = Input(id="not-the-filter")
            await app.screen.mount(other)
            await pilot.pause()
            # Triggering on_input_changed for a foreign id must not crash
            # or rebuild the list (we just confirm the call returns cleanly).
            app.screen.on_input_changed(Input.Changed(input=other, value="hello"))
            await pilot.pause()
            await other.remove()

    async def test_input_submitted_for_unrelated_input_ignored(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            other = Input(id="not-the-filter")
            await app.screen.mount(other)
            await pilot.pause()
            app.screen.on_input_submitted(Input.Submitted(input=other, value="hello"))
            await pilot.pause()
            await other.remove()
            # Should not have dismissed.
            assert app.last_pick == "<unset>"

    async def test_action_select_swallows_indexerror(self) -> None:
        # Defensive guard: if the highlighted index races with a list
        # rebuild and goes stale, action_select must not crash.
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            ol.highlighted = 0  # set to anything; we'll force the lookup to fail
            with mock.patch.object(ol, "get_option_at_index", side_effect=IndexError):
                app.screen.action_select()
            await pilot.pause()
            assert app.last_pick == "<unset>"

    async def test_action_select_with_no_highlight_falls_through(self) -> None:
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            ol.highlighted = None
            app.screen.action_select()
            await pilot.pause()
            # First runnable option fires.
            assert app.last_pick is not None
            assert str(app.last_pick).startswith("/")

    async def test_unknown_group_member_is_skipped(self, monkeypatch) -> None:
        # If CATALOG_GROUPS lists a name that vanished from the registry,
        # the unknown row is silently skipped (defensive, no crash).
        from lilbee.cli.tui.widgets import slash_command_catalog as scc

        bogus = scc.CatalogGroup("BOGUS", ("/this-does-not-exist",))
        monkeypatch.setattr(scc, "CATALOG_GROUPS", (*scc.CATALOG_GROUPS, bogus))
        app = _CatalogOnlyApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ol = app.screen.query_one("#catalog-list", OptionList)
            # No option carries the bogus id.
            ids = [
                ol.get_option_at_index(i).id
                for i in range(ol.option_count)
                if ol.get_option_at_index(i).id is not None
            ]
            assert "/this-does-not-exist" not in ids


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

    async def test_on_catalog_pick_with_name_inserts(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen._on_catalog_pick("/clear")
            await pilot.pause()
            assert screen.query_one("#chat-input").value == "/clear "

    async def test_on_catalog_pick_with_none_no_op(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            chat_input = screen.query_one("#chat-input")
            chat_input.value = "preserved"
            await pilot.pause()
            screen._on_catalog_pick(None)
            await pilot.pause()
            assert chat_input.value == "preserved"

    async def test_help_hint_chip_renders(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test():
            chip = app.screen.query_one("#help-hint", HelpHint)
            rendered = str(chip.render())
            assert msg.HELP_HINT_COMMANDS in rendered
            assert msg.HELP_HINT_KEYS in rendered

    async def test_arg_hint_widget_mounted_in_chat(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test():
            hint = app.screen.query_one("#arg-hint", ArgHintLine)
            assert hint is not None

    async def test_help_hint_click_opens_catalog(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#help-hint")
            await pilot.pause()
            assert isinstance(app.screen, SlashCommandCatalog)

    async def test_help_hint_click_no_op_off_chat_screen(self) -> None:
        class _NonChatHost(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield HelpHint(id="help-hint")

        bare = _NonChatHost()
        async with bare.run_test() as pilot:
            await pilot.click("#help-hint")
            await pilot.pause()
            assert not isinstance(bare.screen, SlashCommandCatalog)


class TestPlaceholderCopy:
    def test_placeholder_advertises_three_discovery_keys(self) -> None:
        text = msg.CHAT_INPUT_PLACEHOLDER_DEFAULT
        assert "/" in text
        assert "F1" in text
        assert "F2" in text


class TestAutoShowOverlayAsync:
    """The act of typing ``/`` must reveal the command set immediately."""

    async def test_typing_slash_auto_shows_overlay(self, _mock_resolve, _mock_services) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            inp = app.screen.query_one("#chat-input")
            overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
            assert not overlay.is_visible
            inp.value = "/"
            await pilot.pause()
            assert overlay.is_visible

    async def test_overlay_filters_live_as_user_types(self, _mock_resolve, _mock_services) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay, get_completions

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            inp = app.screen.query_one("#chat-input")
            overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
            inp.value = "/"
            await pilot.pause()
            full = list(overlay._options)
            inp.value = "/m"
            await pilot.pause()
            assert overlay.is_visible
            filtered = list(overlay._options)
            assert filtered != full
            for opt in filtered:
                assert opt.startswith("/m")
            assert filtered == get_completions("/m")

    async def test_typing_prose_keeps_overlay_hidden(self, _mock_resolve, _mock_services) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            inp = app.screen.query_one("#chat-input")
            overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
            inp.value = "hello world"
            await pilot.pause()
            assert not overlay.is_visible

    async def test_clearing_slash_hides_overlay(self, _mock_resolve, _mock_services) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            inp = app.screen.query_one("#chat-input")
            overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
            inp.value = "/"
            await pilot.pause()
            assert overlay.is_visible
            inp.value = ""
            await pilot.pause()
            assert not overlay.is_visible


class TestDiscoveryBindingsAsync:
    """The keys that open the discovery surfaces must be visible and live."""

    def test_f1_visible_in_app_bindings(self) -> None:
        from textual.binding import Binding

        from lilbee.cli.tui.app import LilbeeApp

        f1 = next(b for b in LilbeeApp.BINDINGS if isinstance(b, Binding) and b.key == "f1")
        assert f1.show is True
        assert f1.priority is True
        assert f1.action == "push_help"

    async def test_f2_opens_catalog(self, _mock_resolve, _mock_services) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            await pilot.press("f2")
            await pilot.pause()
            assert isinstance(app.screen, SlashCommandCatalog)

    async def test_f2_visible_in_chat_footer(self, _mock_resolve, _mock_services) -> None:
        from textual.binding import Binding

        from lilbee.cli.tui.screens.chat import ChatScreen

        f2 = next(b for b in ChatScreen.BINDINGS if isinstance(b, Binding) and b.key == "f2")
        assert f2.show is True
        assert f2.priority is True
        assert f2.action == "show_command_catalog"

    def test_ctrl_n_is_priority_for_dropdown_navigation(self) -> None:
        """Ctrl+N stays a priority screen binding for vim-style cycle-next."""
        from textual.binding import Binding

        from lilbee.cli.tui.screens.chat import ChatScreen

        ctrl_n = next(
            b for b in ChatScreen.BINDINGS if isinstance(b, Binding) and b.key == "ctrl+n"
        )
        assert ctrl_n.priority is True
        assert ctrl_n.action == "complete_next"

    def test_ctrl_p_handled_via_app_palette_override(self) -> None:
        """Ctrl+P stays the app-level palette binding; the chat screen does NOT
        register its own ``ctrl+p`` binding (so palette opens by default).
        Cycle-prev only fires when the dropdown is visible, via
        :meth:`LilbeeApp.action_command_palette` override."""
        from textual.binding import Binding

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen_ctrl_p = [
            b for b in ChatScreen.BINDINGS if isinstance(b, Binding) and b.key == "ctrl+p"
        ]
        assert screen_ctrl_p == []


class TestDropdownNavigationAsync:
    """Ctrl+N / Ctrl+P / Down / Up cycle the visible dropdown without
    disturbing the input value or collapsing the option list."""

    async def test_ctrl_n_moves_highlight_without_touching_input(
        self, _mock_resolve, _mock_services
    ) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            inp = screen.query_one("#chat-input")
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            inp.value = "/"
            await pilot.pause()
            assert overlay.is_visible
            first = overlay.get_current()
            await pilot.press("ctrl+n")
            await pilot.pause()
            second = overlay.get_current()
            # Input is untouched; highlight moved to the next option;
            # dropdown stays open with the same option list.
            assert inp.value == "/"
            assert overlay.is_visible
            assert second != first

    async def test_ctrl_p_moves_highlight_back(self, _mock_resolve, _mock_services) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            inp = screen.query_one("#chat-input")
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            inp.value = "/"
            await pilot.pause()
            first = overlay.get_current()
            await pilot.press("ctrl+n")
            await pilot.pause()
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert overlay.get_current() == first
            assert inp.value == "/"
            assert overlay.is_visible

    async def test_down_navigates_dropdown_when_visible(
        self, _mock_resolve, _mock_services
    ) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            inp = screen.query_one("#chat-input")
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            inp.value = "/"
            await pilot.pause()
            first = overlay.get_current()
            await pilot.press("down")
            await pilot.pause()
            assert overlay.get_current() != first
            assert inp.value == "/"
            assert overlay.is_visible

    async def test_ctrl_p_opens_palette_when_overlay_hidden(
        self, _mock_resolve, _mock_services
    ) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            assert not overlay.is_visible
            with mock.patch.object(app, "action_command_palette") as palette:
                await pilot.press("ctrl+p")
                await pilot.pause()
            palette.assert_called_once()

    async def test_up_navigates_dropdown_when_visible(self, _mock_resolve, _mock_services) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            inp = screen.query_one("#chat-input")
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            inp.value = "/"
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            mid = overlay.get_current()
            await pilot.press("up")
            await pilot.pause()
            assert overlay.get_current() != mid
            assert inp.value == "/"
            assert overlay.is_visible


class TestEnterAcceptsHighlightAsync:
    """Pressing Enter on a visible dropdown must accept the highlighted command."""

    async def test_enter_accepts_highlight_when_input_differs(
        self, _mock_resolve, _mock_services
    ) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            inp = screen.query_one("#chat-input")
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            inp.value = "/"
            await pilot.pause()
            assert overlay.is_visible
            highlighted = overlay.get_current()
            assert highlighted is not None and highlighted.startswith("/")
            await pilot.press("enter")
            await pilot.pause()
            # Input now holds "<command> " (trailing space invites args), and
            # the message was NOT submitted because Enter was consumed by the
            # accept-on-overlay path.
            assert inp.value == f"{highlighted} "

    async def test_enter_submits_when_selection_matches_input(
        self, _mock_resolve, _mock_services
    ) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            inp = screen.query_one("#chat-input")
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            with mock.patch.object(screen, "_handle_slash") as dispatch:
                inp.value = "/clear"
                await pilot.pause()
                # /clear has no completions when text == one of the names
                # exactly (autocomplete excludes the exact match), so the
                # overlay may or may not be visible. Force-set to a state
                # where the highlighted equals the input to drive the
                # "selection matches input -> submit" branch.
                overlay.show_completions(["/clear"])
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
            dispatch.assert_called_once_with("/clear")
            assert not overlay.is_visible

    async def test_enter_with_empty_overlay_submits_normally(
        self, _mock_resolve, _mock_services
    ) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            inp = screen.query_one("#chat-input")
            with mock.patch.object(screen, "_send_message") as send:
                inp.value = "hello"
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
            send.assert_called_once_with("hello")


class TestOverlayBackoutAsync:
    """Esc on a visible dropdown dismisses it without leaving INSERT mode."""

    async def test_esc_hides_overlay_keeps_insert_mode(self, _mock_resolve, _mock_services) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            inp = screen.query_one("#chat-input")
            inp.value = "/"
            await pilot.pause()
            assert overlay.is_visible
            assert screen._insert_mode is True
            await pilot.press("escape")
            await pilot.pause()
            assert not overlay.is_visible
            assert screen._insert_mode is True

    async def test_second_esc_drops_to_normal_mode(self, _mock_resolve, _mock_services) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            inp = screen.query_one("#chat-input")
            inp.value = "/"
            await pilot.pause()
            assert overlay.is_visible
            await pilot.press("escape")
            await pilot.pause()
            assert not overlay.is_visible
            assert screen._insert_mode is True
            await pilot.press("escape")
            await pilot.pause()
            assert screen._insert_mode is False

    async def test_esc_without_overlay_drops_to_normal_mode(
        self, _mock_resolve, _mock_services
    ) -> None:
        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            assert screen._insert_mode is True
            await pilot.press("escape")
            await pilot.pause()
            assert screen._insert_mode is False


class TestArgCompletionsNotAutoShownAsync:
    """Once the user has entered arg-completion mode (typed a space), the
    overlay must NOT auto-show. Paths and other long lists are intrusive
    and should stay Tab-triggered."""

    async def test_typing_arg_partial_does_not_auto_show(
        self, _mock_resolve, _mock_services
    ) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _ChatHostApp()
        async with app.run_test() as pilot:
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            inp = screen.query_one("#chat-input")
            # Land in arg-completion mode without going through any Tab.
            inp.value = "/add "
            await pilot.pause()
            assert not overlay.is_visible
