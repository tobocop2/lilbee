"""Navigation flow tests: verify keyboard-driven TUI interactions.

Every test uses pilot.press() to simulate actual keystrokes, never
action_* methods directly. This catches key resolution, focus routing,
and event bubbling bugs that unit tests miss.
"""

from __future__ import annotations

from unittest import mock

import pytest
from rich.console import Console
from textual.widgets import Footer, Input

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.fleet import FleetScreen
from lilbee.cli.tui.screens.sessions import SessionsScreen
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.cli.tui.screens.task_center import TaskCenter
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import await_chat, pump_until, shown_footer_keys


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
    # Simulate "already-initialized" state so needs_setup()
    # doesn't push the SetupWizard during tests that exercise chat.
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


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    with (
        mock.patch(
            "lilbee.cli.tui.screens.chat.needs_setup",
            return_value=False,
        ),
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=True,
        ),
    ):
        yield


async def _wait_for_screen(app, pilot, screen_type):
    """Wait (bounded) for a view transition to land on *screen_type*.

    The view-switch guard deliberately drops a switch while the previous
    transition is still in flight, so pressing again after one fixed pause
    races a loaded worker; each press must wait for its transition.
    """
    for _ in range(40):
        if isinstance(app.screen, screen_type):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"Expected {screen_type.__name__}, got {type(app.screen).__name__}")


async def test_bracket_keys_cycle_all_screens():
    """Press ] through all 6 views from normal mode (Escape first on Chat)."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        assert isinstance(app.screen, ChatScreen)

        # Chat starts in insert mode: Escape to normal mode first
        await pilot.press("escape")
        await pilot.pause()

        expected = [
            CatalogScreen,
            StatusScreen,
            SettingsScreen,
            TaskCenter,
            FleetScreen,
            SessionsScreen,
            ChatScreen,
        ]
        for screen_type in expected:
            await pilot.press("right_square_bracket")
            await _wait_for_screen(app, pilot, screen_type)


async def test_view_switch_guard_held_until_transition_completes():
    """The switch guard stays up until Textual's deferred transition completes,
    so a rapid second switch is dropped instead of re-entering switch_screen
    mid-transition and crashing on an empty result-callback stack.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()

        app.switch_view("Catalog")
        # Guard is set synchronously; a re-entrant switch is dropped.
        assert app._switching is True
        app.switch_view("Settings")
        assert app.active_view == "Catalog"

        await pilot.pause()
        # Guard releases only after the transition finishes; now the next
        # switch is accepted.
        assert app._switching is False
        assert isinstance(app.screen, CatalogScreen)
        app.switch_view("Settings")
        await pump_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        assert isinstance(app.screen, SettingsScreen)


async def test_bracket_keys_typed_literally_when_chat_input_focused():
    """Pressing [ or ] with the chat input focused must insert text, not switch screens."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        assert isinstance(app.screen, ChatScreen)

        chat_input = app.screen.query_one("#chat-input", ChatInput)
        assert chat_input.has_focus, "Chat input should auto-focus on mount"
        assert chat_input.value == ""

        await pilot.press("left_square_bracket")
        await pilot.pause()
        await pilot.press("right_square_bracket")
        await pilot.pause()

        assert isinstance(app.screen, ChatScreen), (
            f"Brackets must not navigate while chat input has focus, "
            f"got {type(app.screen).__name__}"
        )
        assert chat_input.value == "[]", (
            f"Brackets must type literally with input focused, got value={chat_input.value!r}"
        )


async def test_bracket_keys_cycle_backward():
    """Press [ to go backward through views."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        # Escape to normal mode so ] works
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("left_square_bracket")
        await _wait_for_screen(app, pilot, SessionsScreen)

        await pilot.press("left_square_bracket")
        await _wait_for_screen(app, pilot, FleetScreen)

        await pilot.press("left_square_bracket")
        await _wait_for_screen(app, pilot, TaskCenter)


async def test_bracket_keys_work_from_settings():
    """Navigate to Settings, press ], verify screen changes to Tasks."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        app.switch_view("Settings")
        await pump_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        assert isinstance(app.screen, SettingsScreen)

        await pilot.press("right_square_bracket")
        await pump_until(pilot, lambda: isinstance(app.screen, TaskCenter))
        assert isinstance(app.screen, TaskCenter)


async def test_bracket_keys_typed_literally_when_catalog_search_focused():
    """Brackets in catalog search input must insert text, not switch screens."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        app.switch_view("Catalog")
        await pilot.pause()

        search = app.screen.query_one("#catalog-search", Input)
        # Catalog kicks off async HF fetches on mount that can steal focus
        # on slow Windows runners; pump frames until the search input
        # actually retains focus before exercising the bracket keys.
        for _ in range(50):
            search.focus()
            await pilot.pause()
            if search.has_focus:
                break
        assert search.has_focus

        await pilot.press("left_square_bracket")
        await pilot.press("right_square_bracket")
        await pilot.pause()

        assert isinstance(app.screen, CatalogScreen), (
            "Brackets must not navigate while catalog search has focus"
        )
        assert search.value == "[]", f"Brackets must type literally; got {search.value!r}"


async def test_settings_escape_returns_to_chat():
    """Escape on Settings switches back to Chat (no filter input to blur)."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        app.switch_view("Settings")
        await pump_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        assert isinstance(app.screen, SettingsScreen)

        await pilot.press("escape")
        await pilot.pause()
        # action_go_back routes back to Chat under LilbeeApp.
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert isinstance(app.screen, ChatScreen)


async def test_rapid_fleet_back_does_not_corrupt_screen_stack():
    """Rapid back-to-back Fleet transitions must not crash or corrupt the stack.

    A raw pop_screen on go_back let a second transition race in and pop Textual's
    result-callback stack while empty (IndexError, bb-ce4). The Fleet view inherits
    the same guarded switch_view back-navigation.
    """
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        for _ in range(5):
            app.switch_view("Fleet")
            await pilot.pause()
            fleet = app.screen
            if not isinstance(fleet, FleetScreen):
                continue
            # Two back actions fired before the first transition settles: the
            # guard must drop the second instead of underflowing the stack.
            fleet.action_go_back()
            fleet.action_go_back()
            await pilot.pause()
        # No IndexError raised; back on Chat. Asserting on Textual's private
        # _result_callbacks is intentional: this regression is about that internal
        # stack underflowing, and one entry confirms it never did.
        assert isinstance(app.screen, ChatScreen)
        assert len(app.screen._result_callbacks) == 1


async def test_slash_catalog_routes_through_switch_view_under_lilbee_app():
    """/models from Chat under LilbeeApp must use switch_view, not push_screen."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat._handle_slash("/models")
        await pump_until(pilot, lambda: isinstance(app.screen, CatalogScreen))
        assert isinstance(app.screen, CatalogScreen)


async def test_slash_settings_routes_through_switch_view_under_lilbee_app():
    """/settings under LilbeeApp must use switch_view, not push_screen."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat._handle_slash("/settings")
        await pump_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        assert isinstance(app.screen, SettingsScreen)


async def test_slash_status_routes_through_switch_view_under_lilbee_app():
    """/status under LilbeeApp must use switch_view, not push_screen."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat._handle_slash("/status")
        await pump_until(pilot, lambda: isinstance(app.screen, StatusScreen))
        assert isinstance(app.screen, StatusScreen)


async def test_grid_arrows_stay_on_catalog():
    """Right arrow in catalog grid mode should move grid cursor, not switch screens."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        app.switch_view("Catalog")
        await pilot.pause()

        await pilot.press("right")
        await pump_until(pilot, lambda: isinstance(app.screen, CatalogScreen))
        assert isinstance(app.screen, CatalogScreen)


def _discover_row(name: str, *, installed: bool = False):
    from lilbee.catalog.types import ModelTask
    from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow

    return LocalCatalogRow(
        name=name,
        task=ModelTask.CHAT,
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=installed,
        sort_downloads=0,
        sort_size=0.0,
        ref=name,
        backend="native",
    )


async def test_discover_arrow_flow_keeps_visible_cursor_on_rail_grids():
    """Arrowing through Discover must always leave focus on a visible rail grid
    with a highlight -- never on a hidden sibling pane's grid, and the cursor
    must come back when arrowing up again (bb-hca4h's Discover symptom)."""
    from lilbee.cli.tui.widgets.discover_rails import DiscoverRails
    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        app.switch_view("Catalog")
        await _wait_for_screen(app, pilot, CatalogScreen)
        screen = app.screen
        assert isinstance(screen, CatalogScreen)

        def _seed_rails() -> None:
            rails = screen.query_one("#discover-rails", DiscoverRails)
            rails.set_rails(
                for_you=[_discover_row("Alpha"), _discover_row("Beta")],
                collection=[_discover_row("Gamma", installed=True)],
                fresh=[_discover_row("Delta"), _discover_row("Epsilon")],
            )

        # Deterministic rail content: worker landings re-seed the same rows.
        screen._populate_discover_rails = _seed_rails  # type: ignore[method-assign]
        await pilot.press("1")  # Discover tab
        await pilot.pause()
        _seed_rails()
        await pilot.pause()

        rails = screen.query_one("#discover-rails", DiscoverRails)
        rail_grids = list(rails.query(ModelGrid))
        visited: list[ModelGrid] = []
        for _ in range(8):
            await pilot.press("down")
            await pilot.pause()
            focused = app.focused
            assert isinstance(focused, ModelGrid), f"focus left the grids: {focused!r}"
            assert focused in rail_grids, "arrows drove a grid outside the Discover pane"
            assert focused.highlighted is not None, "cursor vanished mid-navigation"
            visited.append(focused)
        assert rails.query_one("#discover-grid-fresh", ModelGrid) in visited

        for _ in range(8):
            await pilot.press("up")
            await pilot.pause()
            focused = app.focused
            assert isinstance(focused, ModelGrid)
            assert focused in rail_grids
            assert focused.highlighted is not None
        assert app.focused is rails.query_one("#discover-grid-for-you", ModelGrid)


async def test_grid_cursor_survives_dataset_refresh():
    """A background page landing (set_rows) must not strand the cursor (bb-hca4h)."""
    from textual.app import ComposeResult
    from textual.containers import VerticalScroll

    from lilbee.cli.tui.widgets.model_grid import ModelGrid
    from tests._lilbee_app_test_host import LilbeeAppHost

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            with VerticalScroll():
                yield ModelGrid([_discover_row(f"m{i}") for i in range(6)], id="mg")

    async with _App().run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        grid = pilot.app.query_one("#mg", ModelGrid)
        grid.focus()
        await pilot.pause()
        await pilot.press("right", "right")
        await pilot.pause()
        assert grid.highlighted == 2
        # Simulate an HF page landing: same rows plus a new page appended.
        grid.set_rows([_discover_row(f"m{i}") for i in range(10)])
        await pilot.pause()
        assert grid.highlighted == 2, "refresh stranded the cursor"
        await pilot.press("right")
        await pilot.pause()
        assert grid.highlighted == 3, "cursor teleported after refresh"


async def test_footer_present_on_screens():
    """Every screen should have a Footer widget."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()

        views = ["Chat", "Catalog", "Status", "Settings", "Tasks"]
        for view in views:
            app.switch_view(view)
            # Some screens defer mounting via call_after_refresh, so a
            # single pilot.pause() can race the compose tick on slower
            # Windows runners. Poll until the Footer lands.
            footers = ()
            for _ in range(20):
                await pilot.pause()
                footers = app.screen.query(Footer)
                if len(footers) > 0:
                    break
            assert len(footers) > 0, f"{view} screen has no Footer"


async def test_help_panel_toggle():
    """? opens HelpPanel, ? again closes it."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        # Escape to normal mode so ? isn't typed into Input
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("question_mark")
        await pilot.pause()
        # HelpPanel may be on the screen or app level
        has_panel = bool(app.screen.query("HelpPanel") or app.query("HelpPanel"))
        assert has_panel, "HelpPanel should be visible"

        await pilot.press("question_mark")
        await pilot.pause()
        has_panel = bool(app.screen.query("HelpPanel") or app.query("HelpPanel"))
        assert not has_panel, "HelpPanel should be hidden"


async def test_catalog_nav_noop_when_search_focused():
    """Navigation actions are no-ops when catalog search input is focused."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        app.switch_view("Catalog")
        await pilot.pause()

        screen = app.screen
        # Slow Windows runners can need several frames after action_focus_search
        # before the input takes focus (the input is hidden by default and the
        # action removes -hidden + focuses on the next refresh tick).
        for _ in range(50):
            screen.action_focus_search()
            await pilot.pause()
            if isinstance(screen.focused, Input):
                break
        assert isinstance(screen.focused, Input)

        actions = ("cursor_down", "cursor_up", "page_down", "page_up", "jump_top", "jump_bottom")
        for action in actions:
            getattr(screen, f"action_{action}")()
        await pilot.pause()
        assert isinstance(screen.focused, Input)


async def test_chat_tab_outside_input_advances_focus():
    """Tab from outside the chat input advances the focus chain."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        # Escape to normal mode: input loses focus
        await pilot.press("escape")
        await pilot.pause()
        before = app.focused
        screen = app.screen
        screen.action_complete()
        await pilot.pause()
        assert app.focused is not before, "Tab in normal mode did not advance focus"


async def test_chat_escape_from_model_picker_button():
    """Escape from the focused chat-model picker button returns to chat input."""
    from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        screen = app.screen

        await pilot.press("escape")
        await pilot.pause()
        screen.query_one("#model-pick-chat", ModelPickerButton).focus()
        await pilot.pause()

        assert isinstance(screen.focused, ModelPickerButton)

        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is not None
        assert screen.focused.id == "chat-input"


async def test_app_footer_hides_tasks_hint_when_text_input_focused():
    """`t Tasks` is shown only when no text input has focus -- otherwise the
    focused Input/TextArea swallows `t` as a literal character."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        # Chat boots in INSERT mode with the prompt (a TextArea) focused.
        assert app.check_action("open_tasks", ()) is False
        await pilot.press("escape")
        await pilot.pause()
        # NORMAL mode: focus is off the prompt, so the hint is honest again.
        assert app.check_action("open_tasks", ()) is True


async def test_backward_nav_from_catalog_after_visiting_tasks():
    """Regression: [ from Catalog after visiting Task Center got stuck.

    The bug was that switch_screen is async but active_view updated
    synchronously. Rapid navigation queued conflicting screen switches
    that corrupted the stack. Fixed by adding a _switching guard.
    """
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        assert isinstance(app.screen, ChatScreen)
        await pilot.press("escape")
        await pilot.pause()

        # Forward to Catalog
        await pilot.press("right_square_bracket")
        await pump_until(pilot, lambda: isinstance(app.screen, CatalogScreen))
        assert isinstance(app.screen, CatalogScreen)

        # Forward past Catalog to Tasks (Catalog > Status > Settings > Tasks)
        for _ in range(3):
            await pilot.press("right_square_bracket")
            await pump_until(pilot, lambda: isinstance(app.screen, TaskCenter))
        assert isinstance(app.screen, TaskCenter)

        # Backward back to Catalog (Tasks > Settings > Status > Catalog)
        for _ in range(3):
            await pilot.press("left_square_bracket")
            await pump_until(pilot, lambda: isinstance(app.screen, CatalogScreen))
        assert isinstance(app.screen, CatalogScreen)

        # The critical step: backward from Catalog to Chat
        await pilot.press("left_square_bracket")
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        assert isinstance(app.screen, ChatScreen), (
            f"Expected ChatScreen after [ from Catalog, got {type(app.screen).__name__}"
        )


async def test_switching_guard_blocks_concurrent_switch():
    """The _switching guard drops a second switch_view call while one is pending."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        # Manually set the guard
        app._switching = True
        original_view = app.active_view

        # This should be a no-op
        app.switch_view("Status")

        # active_view unchanged because guard blocked the call
        assert app.active_view == original_view

        # Clean up guard so teardown works
        app._switching = False


async def test_lilbee_app_wires_worker_pool_notifications_on_mount() -> None:
    """``on_mount`` calls ``Services.add_pool_listener`` so server spawn
    lifecycle surfaces as Textual notifications. Verified by replacing the
    Services singleton with a provider whose ``add_spawn_listener`` records the
    callbacks, then firing them from a worker thread (call_from_thread requires a
    different thread) so their notify() bodies execute against the live app."""
    import threading
    from unittest.mock import MagicMock

    from lilbee.app import services as services_mod
    from lilbee.providers.base import LLMProvider
    from lilbee.providers.roles import WorkerRole
    from tests.conftest import make_mock_services

    captured: dict[str, object] = {}

    def _record(*, on_spawning=None, on_spawned=None) -> None:
        captured["on_spawning"] = on_spawning
        captured["on_spawned"] = on_spawned

    provider = MagicMock(spec=LLMProvider)
    provider.add_spawn_listener.side_effect = _record
    services_mod.set_services(make_mock_services(provider=provider))
    try:
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await await_chat(app, pilot)
            await pilot.pause()
            on_spawning = captured.get("on_spawning")
            on_spawned = captured.get("on_spawned")
            assert callable(on_spawning)
            assert callable(on_spawned)
            # call_from_thread refuses to run on the app's own thread; fire
            # the listeners from a worker thread to mimic the real pool's
            # spawn callback site (the pool runtime thread).
            threading.Thread(target=on_spawning, args=(WorkerRole.CHAT,)).start()
            threading.Thread(target=on_spawned, args=(WorkerRole.CHAT,)).start()
            await pilot.pause()
    finally:
        services_mod.set_services(None)


async def test_m_opens_catalog_and_q_returns_to_previous_view():
    """m jumps to the catalog from any view; q returns to the view the user
    came from rather than always landing on Chat."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        app.switch_view("Settings")
        await _wait_for_screen(app, pilot, SettingsScreen)

        await pilot.press("m")
        await _wait_for_screen(app, pilot, CatalogScreen)

        await pilot.press("q")
        await _wait_for_screen(app, pilot, SettingsScreen)


async def test_capital_s_and_m_type_into_chat_input():
    """The S sync binding and the m catalog binding must not steal characters
    from a focused text input."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        chat_input = app.screen.query_one("#chat-input", ChatInput)
        assert chat_input.has_focus, "Chat input should auto-focus on mount"

        with mock.patch.object(LilbeeApp, "action_run_sync") as run_sync:
            await pilot.press("S")
            await pilot.pause()
            assert not run_sync.called, "S must type, not start a document sync"
        await pilot.press("m")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen), "m must not navigate mid-typing"
        assert chat_input.value == "Sm"


async def test_settings_editor_accepts_angle_brackets():
    """< and > must type into a focused settings editor, not switch panes."""
    from textual.widgets import TabbedContent

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        app.switch_view("Settings")
        await _wait_for_screen(app, pilot, SettingsScreen)

        editor = None
        for _ in range(20):
            editors = app.screen.query(Input)
            if editors:
                editor = editors.first()
                break
            await pilot.pause()
        if editor is None:
            pytest.skip("no Input editors mounted on the settings screen")
        editor.focus()
        await pump_until(pilot, lambda: editor.has_focus)
        tabs = app.screen.query_one("#settings-tabs", TabbedContent)
        active_before = tabs.active
        before = editor.value

        await pilot.press("less_than_sign", "greater_than_sign")
        await pilot.pause()
        assert editor.value == f"{before}<>", "angle brackets must type literally"
        assert tabs.active == active_before, "panes must not cycle mid-typing"


async def test_catalog_tab_strip_shows_digit_shortcuts():
    """Tab labels carry the 1-6 numerals so the digit keys are discoverable."""
    from textual.widgets import TabbedContent

    from lilbee.cli.tui.screens.catalog_utils import TAB_CHAT, TAB_DISCOVER, TAB_LIBRARY

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Catalog")
        assert await pump_until(pilot, lambda: isinstance(app.screen, CatalogScreen))
        tabs = app.screen.query_one("#catalog-tabs", TabbedContent)
        assert str(tabs.get_tab(TAB_DISCOVER).label).startswith("1 ")
        assert str(tabs.get_tab(TAB_CHAT).label).startswith("2 ")
        assert str(tabs.get_tab(TAB_LIBRARY).label).startswith("6 ")


def test_catalog_and_settings_share_the_tab_cycle_keys():
    """Both screens bind < / > to tab cycling and both advertise it the same way.

    The catalog used to hide the pair because its row overflowed; grouping the
    app-level bindings freed the columns, so the two screens agree again. A
    user who learns > on one screen must not find it missing on the other.
    """
    from textual.binding import Binding

    def binding(screen_cls, key: str) -> Binding | None:
        return next(
            (b for b in screen_cls.BINDINGS if isinstance(b, Binding) and b.key == key), None
        )

    for key in ("greater_than_sign", "less_than_sign"):
        catalog = binding(CatalogScreen, key)
        settings = binding(SettingsScreen, key)
        assert catalog is not None and settings is not None
        assert catalog.show is True
        assert settings.show is True
        assert catalog.group is not None and settings.group is not None


def test_catalog_help_documents_every_visible_and_hidden_tab_key():
    """Keys must be findable in F1 help, whether or not they reach the footer.

    Asserted in both directions: help that mentions a key no longer bound is
    as wrong as a binding no help mentions. `o` is here because it took over
    the source filter from `c`, which is now the app-wide jump to Chat.
    """
    from textual.binding import Binding

    help_text = CatalogScreen.HELP
    bound = {b.key for b in CatalogScreen.BINDINGS if isinstance(b, Binding)}

    assert "< / >" in help_text
    assert {"less_than_sign", "greater_than_sign"} <= bound
    assert "1-6" in help_text
    assert {str(n) for n in range(1, 7)} <= bound
    assert "- o: cycle source chip" in help_text
    assert "o" in bound
    assert "c" not in bound


def test_settings_help_documents_the_shared_browse_keys():
    """Settings adopted the shared j/k/g/G fragment, so its help must say so.

    Both directions, like the catalog check: the keys must be bound and the
    help must name them.
    """
    from textual.binding import Binding

    help_text = SettingsScreen.HELP
    bound = {b.key for b in SettingsScreen.BINDINGS if isinstance(b, Binding)}

    assert {"j", "k", "g", "G"} <= bound
    for token in ("j / k", "g / G"):
        assert token in help_text

    # Both resets now hold a footer cell under one label, so help has to name
    # both: the group label alone does not say which key does which.
    assert {"ctrl+r", "ctrl+shift+r"} <= bound
    assert "Ctrl+R resets the focused field" in help_text
    assert "Ctrl+Shift+R resets every setting" in help_text


# A footer wider than this scrolls its tail out of view on a normal terminal,
# which is how `^t Theme` once rendered as `^t The`. Measured on the rendered
# row, not on a count of bindings: what overflows a row is columns, and an
# entry count vetoes useful keys while missing a single verbose label.
_FOOTER_COLUMN_BUDGET = 120


def _footer_columns(app) -> tuple[int, str]:
    """Rendered width of the active screen's footer, plus its cells for failures."""
    footer = app.screen.query_one(Footer)
    cells = [getattr(c.render(), "plain", type(c).__name__) for c in footer.children]
    width = sum(c.outer_size.width + c.styles.margin.width for c in footer.children if c.display)
    return width, " | ".join(cells)


async def test_every_footer_fits_the_column_budget():
    """No screen's key row may outgrow a 120-column terminal.

    This is the invariant the per-screen "at most N visible bindings" asserts
    were standing in for. Every view is measured after its footer has laid
    out, so a verbose new label fails here rather than silently pushing the
    rightmost global binding off the row.
    """
    app = LilbeeApp()
    async with app.run_test(size=(_FOOTER_COLUMN_BUDGET, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()

        for view in ("Chat", "Catalog", "Status", "Settings", "Tasks", "Fleet", "Sessions"):
            app.switch_view(view)
            assert await pump_until(pilot, lambda v=view: app.active_view == v), (
                f"never landed on {view}"
            )
            # The footer composes its keys on a later refresh than the switch.
            # Asserted, not merely awaited: an un-laid-out row measures zero,
            # and zero would sail under the budget while proving nothing.
            assert await pump_until(pilot, lambda: _footer_columns(app)[0] > 0), (
                f"{view} footer never laid out"
            )
            width, cells = _footer_columns(app)
            assert width <= _FOOTER_COLUMN_BUDGET, f"{view} footer is {width} cols: {cells}"


async def test_grouped_app_row_keeps_every_view_key_reachable():
    """Grouping the app row must not drop a key, only its per-key label.

    The row collapsed from ten labelled cells to three group cells; the point
    of the exercise was columns, so every key that was bound before must still
    be bound and shown.
    """
    from textual.binding import Binding

    shown = {b.key: b for b in LilbeeApp.BINDINGS if isinstance(b, Binding) and b.show}
    assert set(shown) == {
        "t",
        "m",
        "c",
        "ctrl+g",
        "ctrl+o",
        "left_square_bracket",
        "right_square_bracket",
        "f1",
        "ctrl+t",
        "ctrl+c",
    }
    groups = {key: b.group.description for key, b in shown.items() if b.group is not None}
    assert groups["t"] == groups["m"] == groups["c"] == "Views"
    assert groups["left_square_bracket"] == "Navigate"
    assert groups["f1"] == groups["ctrl+t"] == "App"


async def test_c_returns_to_chat_from_another_view():
    """`c` is the counterpart to t / m for the view users leave most often."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("m")
        await _wait_for_screen(app, pilot, CatalogScreen)
        await pilot.press("c")
        await _wait_for_screen(app, pilot, ChatScreen)


async def test_c_is_hidden_from_the_footer_while_an_input_has_focus():
    """`c` types a literal letter in a text field, so the footer must not claim it.

    Same contract as t / m: the guard exists to keep the footer honest, not to
    change what the key does.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("m")
        await _wait_for_screen(app, pilot, CatalogScreen)

        assert app.check_action("open_chat", ()) is True
        app.screen.query_one("#catalog-search", Input).focus()
        await pilot.pause()
        assert app.check_action("open_chat", ()) is False


async def test_c_does_not_advertise_itself_on_chat():
    """Chat is where `c` has nowhere to go, so it drops off that footer.

    False, not None: Textual excludes a False binding from the row and renders
    a None one greyed but present.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()
        assert app.check_action("open_chat", ()) is False


async def test_catalog_source_filter_moved_off_c():
    """`o` cycles the source chip; `c` leaves for Chat instead of flipping it.

    A hidden binding must never shadow an app-wide key: pressing `c` used to
    change the catalog's filter with nothing on screen to explain it.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("m")
        await _wait_for_screen(app, pilot, CatalogScreen)

        screen = app.screen
        await pilot.press("2")
        await pilot.pause()
        before = dict(screen._source_modes)
        await pilot.press("o")
        await pilot.pause()
        assert dict(screen._source_modes) != before


async def _focus_strip(app, pilot):
    """Put the cursor on the model strip the way a user does, via F6."""
    await pilot.press("f6")
    await pump_until(pilot, lambda: app.screen.focused is not None)
    return app.screen.focused


async def test_f6_reaches_the_model_strip_while_typing():
    """The strip must be reachable mid-sentence, without leaving insert mode.

    A printable key cannot do this: a focused TextArea consumes it first. The
    pickers were mouse-only from the prompt before F6 existed.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        chat_input = app.screen.query_one("#chat-input", ChatInput)
        assert chat_input.has_focus

        focused = await _focus_strip(app, pilot)
        assert focused is not None and focused.id == "model-pick-chat"


async def test_arrows_walk_the_whole_model_strip():
    """Left / Right step every role picker and both mode pills, and clamp at the ends.

    One pair of arrows for the whole strip: the mode toggle used to bind
    left/right to picking a mode, which made the same keys mean two things
    depending on where the cursor sat.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await _focus_strip(app, pilot)

        expected = [
            "model-pick-embed",
            "model-pick-vision",
            "model-pick-rerank",
            "chat-mode-search",
            "chat-mode-chat",
        ]
        for want in expected:
            await pilot.press("right")
            await pilot.pause()
            assert app.screen.focused is not None
            assert app.screen.focused.id == want

        # Clamps rather than wrapping round to the first picker.
        await pilot.press("right")
        await pilot.pause()
        assert app.screen.focused.id == "chat-mode-chat"

        for want in reversed(expected[:-1]):
            await pilot.press("left")
            await pilot.pause()
            assert app.screen.focused.id == want
        await pilot.press("left")
        await pilot.pause()
        assert app.screen.focused.id == "model-pick-chat"
        await pilot.press("left")
        await pilot.pause()
        assert app.screen.focused.id == "model-pick-chat"


async def test_home_and_end_jump_to_the_ends_of_the_strip():
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await _focus_strip(app, pilot)

        await pilot.press("end")
        await pilot.pause()
        assert app.screen.focused.id == "chat-mode-chat"
        await pilot.press("home")
        await pilot.pause()
        assert app.screen.focused.id == "model-pick-chat"


async def test_escape_returns_to_the_prompt_from_any_strip_member():
    """Escape means "back to typing" from every member, not just the pickers.

    From a mode pill it used to drop focus on the chat log instead, so the
    user had to guess a second key to get back to the prompt.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))

        for steps in (0, 3, 5):
            await _focus_strip(app, pilot)
            for _ in range(steps):
                await pilot.press("right")
                await pilot.pause()
            await pilot.press("escape")
            await pump_until(pilot, lambda: app.screen.focused is not None)
            focused = app.screen.focused
            assert isinstance(focused, ChatInput), f"after {steps} steps, got {focused}"


async def test_chat_help_documents_the_strip_keys():
    """Both directions: the keys are bound and help names them."""
    from textual.binding import Binding

    from lilbee.cli.tui.widgets.model_bar import ModelBar

    help_text = ChatScreen.HELP
    screen_keys = {b.key for b in ChatScreen.BINDINGS if isinstance(b, Binding)}
    bar_keys = {b.key for b in ModelBar.BINDINGS if isinstance(b, Binding)}

    assert "f6" in screen_keys
    assert "**F6**" in help_text
    assert {"left", "right", "home", "end", "h", "l"} <= bar_keys
    # The keys that step in from outside the bar live on the screen, not the bar.
    assert {"h", "l", "left", "right"} <= screen_keys
    for token in ("**Left**", "**Right**", "**Home**", "**End**", "**h**", "**l**"):
        assert token in help_text


async def test_the_strip_skips_a_role_the_narrow_layout_hides():
    """Left / Right must not step onto a member the narrow bar has hidden.

    The narrow layout hides switched-off roles on the RoleRow, so the picker
    button's own ``display`` still reads True; the strip has to take its
    membership from the focus chain to see that.
    """
    from lilbee.cli.tui.widgets.model_bar import ModelBar

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        bar = app.screen.query_one("#model-bar", ModelBar)
        # Assert the setup, don't just wait for it: pump_until returns a bool,
        # so an unmet precondition would leave the check below passing on a
        # strip that never had a vision picker to skip in the first place.
        assert await pump_until(pilot, lambda: len(bar.strip) == 6), (
            f"strip never reached six members: {[w.id for w in bar.strip]}"
        )

        # Vision is off in this config, so the narrow class drops its row.
        bar.add_class("-narrow")
        assert await pump_until(
            pilot, lambda: "model-pick-vision" not in {w.id for w in bar.strip}
        ), "the narrow layout never dropped the vision row"

        await _focus_strip(app, pilot)
        seen = [app.screen.focused.id]
        for _ in range(len(bar.strip)):
            await pilot.press("right")
            await pilot.pause()
            seen.append(app.screen.focused.id)
        assert "model-pick-vision" not in seen


async def test_footer_hidden_keys_still_reach_the_f1_key_panel():
    """show=False moves a key off the footer row; it must not hide it outright.

    Trimming the footer is only defensible because Textual's key panel filters
    on `binding.system`, not on `show`. Asserted against what the panel actually
    renders rather than by re-applying that filter here: `active_bindings` keeps
    a `show=False` binding either way, so a re-implemented filter would still
    pass on the day an upgrade starts honouring `show` and hides f4 for real.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("f1")
        assert await pump_until(pilot, lambda: bool(app.screen.query("BindingsTable"))), (
            "F1 never opened the key panel"
        )

        console = Console(width=200, no_color=True)
        with console.capture() as capture:
            console.print(app.screen.query_one("BindingsTable").render())
        rendered = capture.get()
        for key, description in (("f4", "Path/Name"), ("tab", "Complete")):
            assert key in rendered and description in rendered, (
                f"{key} is hidden from the footer AND from the key panel"
            )


async def test_strip_arrows_leave_the_prompt_alone():
    """The strip's arrows must not reach the prompt's own cursor movement.

    What protects the prompt is the focused input consuming these keys first,
    not where the bindings live: the screen itself now binds Left / Right to
    step into the strip from NORMAL mode, and typing Left in a half-written
    prompt still has to move the cursor rather than jump focus into the bar.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        chat_input = app.screen.query_one("#chat-input", ChatInput)
        assert chat_input.has_focus

        await pilot.press("a", "b", "c")
        await pilot.pause()
        assert chat_input.value == "abc"
        for key in ("left", "right", "home", "end"):
            await pilot.press(key)
            await pilot.pause()
            assert app.screen.focused is chat_input, f"{key} moved focus off the prompt"
        assert chat_input.value == "abc"


async def test_the_views_group_only_shows_keys_that_go_somewhere():
    """A grouped key must drop out of the row when its action is unavailable.

    Grouping hides the per-key labels, so the keys themselves are all the row
    has left to be honest with: `c` must be absent on Chat (nowhere to jump)
    and present everywhere else.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()

        assert "c" not in shown_footer_keys(app), "Chat advertises a jump to itself"
        assert {"t", "m"} <= shown_footer_keys(app)

        await pilot.press("m")
        await _wait_for_screen(app, pilot, CatalogScreen)
        assert "c" in shown_footer_keys(app), "Catalog hides the jump back to Chat"


async def test_c_still_cancels_on_tasks_instead_of_leaving():
    """Tasks deliberately shadows the app-wide `c`, so pin that it does.

    The pre-existing e2e coverage for this key asserts `app.screen.is_current`,
    which stays true even if `c` had navigated to Chat, so nothing caught the
    shadow either way. Cancel is why you are on this screen; the shadow is
    allowed only because the footer names it.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("t")
        await _wait_for_screen(app, pilot, TaskCenter)
        with mock.patch.object(app.screen, "action_cancel_task") as cancel:
            await pilot.press("c")
            await pilot.pause()
            assert cancel.called, "c stopped cancelling on Tasks"
        assert isinstance(app.screen, TaskCenter), "c left the Tasks screen"


async def test_enter_on_a_mode_pill_switches_chat_mode():
    """Enter on a focused pill is now the only keyboard route to the mode.

    Removing the toggle's own Left / Right, so one pair of arrows walks the
    whole strip, made this the single key path, and the help text promises it.
    The pre-existing coverage calls `action_select()` directly, which bypasses
    key resolution and focus routing, so nothing pressed a key to prove it.
    """
    app = LilbeeApp()
    with mock.patch(
        "lilbee.cli.tui.widgets.model_bar.is_model_available",
        return_value=True,
    ):
        async with app.run_test(size=(120, 40)) as pilot:
            await await_chat(app, pilot)
            await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
            cfg.chat_mode = "chat"

            await _focus_strip(app, pilot)
            await pump_until(pilot, lambda: app.screen.focused is not None)
            while app.screen.focused.id != "chat-mode-search":
                await pilot.press("right")
                await pilot.pause()

            # pump_until returns a bool and deliberately does not assert, so
            # the assertion has to be the caller's.
            await pilot.press("enter")
            assert await pump_until(pilot, lambda: cfg.chat_mode == "search"), (
                f"Enter on the Search pill left the mode at {cfg.chat_mode!r}"
            )

            await pilot.press("right")
            await pilot.pause()
            assert app.screen.focused.id == "chat-mode-chat"
            await pilot.press("enter")
            assert await pump_until(pilot, lambda: cfg.chat_mode == "chat"), (
                f"Enter on the Chat pill left the mode at {cfg.chat_mode!r}"
            )


async def test_l_walks_into_the_model_strip_from_normal_mode():
    """NORMAL mode reaches the strip sideways, without knowing about F6.

    Escape parks the cursor on the transcript. h / l are free there, and read as
    "move along the row" to anyone already scrolling it with j / k.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("l")
        assert await pump_until(
            pilot, lambda: getattr(app.screen.focused, "id", None) == "model-pick-chat"
        ), f"l from NORMAL mode left the cursor on {app.screen.focused!r}"


async def test_h_enters_the_strip_at_its_far_end():
    """Walking left into the strip lands on the last role, not the first.

    Entering from the side the key points at keeps one more press from being
    needed to get where the direction already said.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("h")
        assert await pump_until(
            pilot, lambda: getattr(app.screen.focused, "id", None) == "chat-mode-chat"
        ), f"h from NORMAL mode left the cursor on {app.screen.focused!r}"


async def test_h_and_l_step_between_roles_once_inside_the_strip():
    """Inside the strip the bar's own h / l win, so the keys keep one meaning.

    Entered from NORMAL mode rather than via F6: F6 leaves the screen in INSERT,
    where a printable key is routed back into the prompt as typing before any
    binding sees it, so h / l are NORMAL-mode keys end to end.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("l")
        assert await pump_until(
            pilot, lambda: getattr(app.screen.focused, "id", None) == "model-pick-chat"
        ), "never entered the strip"

        await pilot.press("l")
        await pilot.pause()
        assert app.screen.focused.id == "model-pick-embed"
        await pilot.press("h")
        await pilot.pause()
        assert app.screen.focused.id == "model-pick-chat"
        # Clamps at the near end instead of wrapping to the mode pills.
        await pilot.press("h")
        await pilot.pause()
        assert app.screen.focused.id == "model-pick-chat"


async def test_typing_h_and_l_in_insert_mode_reaches_the_prompt():
    """The strip keys must not eat two letters out of every prompt.

    A canary on Textual rather than a guard on these bindings: the focused
    prompt keeps printable keys, and it goes on keeping them when the strip
    keys are made ``priority=True`` and when the mode guard below is removed,
    both measured. It stays because binding plain letters is the one real risk
    here, and a Textual upgrade is what would break it.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))

        await pilot.press("h", "e", "l", "l", "o")
        await pilot.pause()
        assert screen._chat_input.value == "hello"
        assert screen._chat_input.has_focus


async def test_the_strip_keys_leave_an_open_drawer_alone():
    """Focus inside a drawer keeps h / l, and the guard must gate the KEY, not
    just the footer.

    ``check_action`` returning False has to stop the binding from firing at all;
    a guard that only greys a footer cell would let h yank the cursor out of the
    drawer. The drawer's pills do not consume letters, so this reaches dispatch.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("ctrl+g")
        for _ in range(12):
            await pilot.press("tab")
            await pilot.pause()
            if app.screen._focus_in_drawer():
                break
        assert app.screen._focus_in_drawer(), (
            f"never tabbed into the fleet drawer, landed on {app.screen.focused!r}"
        )

        before = app.screen.focused
        await pilot.press("h")
        await pilot.pause()
        assert app.screen.focused is before


async def test_enter_on_a_mode_pill_acts_from_normal_mode_too():
    """Walking to a pill must be able to act, not only to arrive.

    NORMAL mode's Enter handler drops back to INSERT for any focus target that
    does not handle Enter itself, and it named two widget types that did not
    include the mode pills. Nothing could reach a pill from NORMAL mode before
    h / l, so pressing Enter there switched to INSERT and left the mode alone.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        cfg.chat_mode = "search"
        await pilot.press("escape")
        await pilot.pause()

        # h enters at the far end of the strip, which is the Chat pill.
        await pilot.press("h")
        assert await pump_until(
            pilot, lambda: getattr(app.screen.focused, "id", None) == "chat-mode-chat"
        ), f"h never reached the Chat pill, landed on {app.screen.focused!r}"

        await pilot.press("enter")
        assert await pump_until(pilot, lambda: cfg.chat_mode == "chat"), (
            f"Enter on the Chat pill left the mode at {cfg.chat_mode!r}"
        )
        assert not app.screen._insert_mode, "Enter fell through to INSERT instead of the pill"


async def test_the_arrows_also_step_into_the_strip_from_normal_mode():
    """Left / Right are the discoverable route in; h / l are the vim one.

    The transcript is a VerticalScroll and binds Left / Right to horizontal
    scrolling, but Widget.action_scroll_left raises SkipAction when there is
    nothing to scroll sideways, so key lookup resumes and reaches the screen.
    Measured, not assumed: the first version of this work bound only h / l on
    the belief that the arrows could never get here.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pump_until(pilot, lambda: isinstance(app.screen, ChatScreen))
        await pilot.press("escape")
        await pilot.pause()
        assert getattr(app.screen.focused, "id", None) == "chat-log"

        await pilot.press("right")
        assert await pump_until(
            pilot, lambda: getattr(app.screen.focused, "id", None) == "model-pick-chat"
        ), f"Right from NORMAL mode left the cursor on {app.screen.focused!r}"

        # And the far end from the other side, after handing the cursor back.
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("left")
        assert await pump_until(
            pilot, lambda: getattr(app.screen.focused, "id", None) == "chat-mode-chat"
        ), f"Left from NORMAL mode left the cursor on {app.screen.focused!r}"
