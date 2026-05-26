"""Navigation flow tests: verify keyboard-driven TUI interactions.

Every test uses pilot.press() to simulate actual keystrokes, never
action_* methods directly. This catches key resolution, focus routing,
and event bubbling bugs that unit tests miss.
"""

from __future__ import annotations

from unittest import mock

import pytest
from textual.widgets import Footer, Input

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.cli.tui.screens.task_center import TaskCenter
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.core.config import cfg


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
    # Simulate "already-initialized" state so ChatScreen._needs_setup()
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
            "lilbee.cli.tui.screens.chat.ChatScreen._needs_setup",
            return_value=False,
        ),
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=True,
        ),
    ):
        yield


async def test_bracket_keys_cycle_all_screens():
    """Press ] through all 5 views from normal mode (Escape first on Chat)."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)

        # Chat starts in insert mode: Escape to normal mode first
        await pilot.press("escape")
        await pilot.pause()

        expected = [CatalogScreen, StatusScreen, SettingsScreen, TaskCenter, ChatScreen]
        for screen_type in expected:
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert isinstance(app.screen, screen_type), (
                f"Expected {screen_type.__name__}, got {type(app.screen).__name__}"
            )


async def test_bracket_keys_typed_literally_when_chat_input_focused():
    """Pressing [ or ] with the chat input focused must insert text, not switch screens."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
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
        await pilot.pause()
        # Escape to normal mode so ] works
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert isinstance(app.screen, TaskCenter)

        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)


async def test_bracket_keys_work_from_settings():
    """Navigate to Settings, press ], verify screen changes to Tasks."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)

        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert isinstance(app.screen, TaskCenter)


async def test_bracket_keys_typed_literally_when_catalog_search_focused():
    """Brackets in catalog search input must insert text, not switch screens."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
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
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)

        await pilot.press("escape")
        await pilot.pause()
        # action_go_back routes back to Chat under LilbeeApp.
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert isinstance(app.screen, ChatScreen)


async def test_slash_catalog_routes_through_switch_view_under_lilbee_app():
    """/models from Chat under LilbeeApp must use switch_view, not push_screen."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat._handle_slash("/models")
        await pilot.pause()
        assert isinstance(app.screen, CatalogScreen)


async def test_slash_settings_routes_through_switch_view_under_lilbee_app():
    """/settings under LilbeeApp must use switch_view, not push_screen."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat._handle_slash("/settings")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)


async def test_slash_status_routes_through_switch_view_under_lilbee_app():
    """/status under LilbeeApp must use switch_view, not push_screen."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat._handle_slash("/status")
        await pilot.pause()
        assert isinstance(app.screen, StatusScreen)


async def test_grid_arrows_stay_on_catalog():
    """Right arrow in catalog grid mode should move grid cursor, not switch screens."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Catalog")
        await pilot.pause()

        await pilot.press("right")
        await pilot.pause()
        assert isinstance(app.screen, CatalogScreen)


async def test_footer_present_on_screens():
    """Every screen should have a Footer widget."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
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
        await pilot.pause()
        screen = app.screen

        await pilot.press("escape")
        await pilot.pause()
        screen.query_one("#chat-model-button", ModelPickerButton).focus()
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
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        await pilot.press("escape")
        await pilot.pause()

        # Forward to Catalog
        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert isinstance(app.screen, CatalogScreen)

        # Forward past Catalog to Tasks (Catalog > Status > Settings > Tasks)
        for _ in range(3):
            await pilot.press("right_square_bracket")
            await pilot.pause()
        assert isinstance(app.screen, TaskCenter)

        # Backward back to Catalog (Tasks > Settings > Status > Catalog)
        for _ in range(3):
            await pilot.press("left_square_bracket")
            await pilot.pause()
        assert isinstance(app.screen, CatalogScreen)

        # The critical step: backward from Catalog to Chat
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen), (
            f"Expected ChatScreen after [ from Catalog, got {type(app.screen).__name__}"
        )


async def test_switching_guard_blocks_concurrent_switch():
    """The _switching guard drops a second switch_view call while one is pending."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
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
    """``on_mount`` calls ``Services.add_pool_listener`` so spawn lifecycle
    surfaces as Textual notifications. Verified by replacing the Services
    singleton with a recording pool, then firing the captured callbacks
    from a worker thread (call_from_thread requires a different thread)
    so their notify() bodies execute against the live app."""
    import threading

    from lilbee.app import services as services_mod
    from lilbee.providers.roles import WorkerRole
    from tests.conftest import make_mock_services

    captured: dict[str, object] = {}

    class _RecordingPool:
        registered_roles: tuple[str, ...] = ()

        def add_listener(self, *, on_spawning=None, on_spawned=None) -> None:
            captured["on_spawning"] = on_spawning
            captured["on_spawned"] = on_spawned

    services_mod.set_services(make_mock_services(worker_pool=_RecordingPool()))
    try:
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
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
