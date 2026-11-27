"""Settings screen: mount, pane cycling, saving values, checkboxes and selects."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from lilbee.core.config import cfg
from tests._async_wait import subtree_mounted, wait_for_widget, wait_until
from tests._lilbee_app_test_host import pump_until, saved_setting
from tests._settings_test_app import LazySettingsTestApp, SettingsTestApp, _wait_for_pane_rows
from tests._tui_screen_fixtures import (  # noqa: F401
    _isolated_cfg,
    _patch_chat_setup,
    _settings_gates_open,
    mock_svc,
)


@pytest.mark.parametrize(
    ("app_class", "eager"), [(SettingsTestApp, True), (LazySettingsTestApp, False)]
)
async def test_a_settings_fixture_mounts_one_screen_once(app_class, eager):
    """Each fixture pushes one Settings screen, and that screen subscribes to changes once."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = app_class()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screens = [s for s in app.screen_stack if isinstance(s, SettingsScreen)]
        assert screens == [app.screen]
        built = [app.screen._pane_is_populated(pane_id) for pane_id in app.screen._pane_groups]
        assert built == [eager] * len(built)
        assert len(app.settings_changed_signal._subscriptions[app.screen]) == 1


async def test_settings_screen_mounts_grouped_sections():
    """Settings screen renders grouped sections with setting rows."""
    app = LazySettingsTestApp("top_k")
    async with app.run_test(size=(120, 40)) as _pilot:
        groups = app.screen.query("TabPane")
        assert len(groups) > 0
        rows = app.screen.query(".setting-row")
        assert len(rows) > 0


async def test_settings_api_keys_group_shows_plaintext_warning():
    """API-Keys group renders a warning naming the config.toml path."""
    from textual.widgets import Static

    app = LazySettingsTestApp("openai_api_key")
    async with app.run_test(size=(120, 60)) as _pilot:
        warnings = list(app.screen.query(".api-keys-warning").results(Static))
        assert len(warnings) == 1
        rendered = str(warnings[0].render())
        assert "plain text" in rendered
        assert "config.toml" in rendered
        assert "sensitive" in rendered.lower()


async def test_settings_hf_token_help_warns_plaintext():
    """hf_token row in the System tab warns the user about plain-text storage."""
    from textual.widgets import Static, TabbedContent

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 60)) as pilot:
        tabbed = app.screen.query_one("#settings-tabs", TabbedContent)
        tabbed.active = "settings-tab-system"
        await pilot.pause(0.2)
        row = app.screen.query_one("#row-hf_token")
        help_widget = row.query_one(".setting-help", Static)
        rendered = str(help_widget.render())
        assert "plain text" in rendered.lower()
        assert "config.toml" in rendered.lower()


async def test_settings_tab_activation_populates_pane():
    """Activating a settings tab mounts its rows.

    With lazy tab bodies, only the active pane composes its editors
    on first paint. Switching to another tab triggers
    ``TabbedContent.TabActivated`` and the pane's body fills in.
    """
    from textual.widgets import TabbedContent

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        tabbed = app.screen.query_one("#settings-tabs", TabbedContent)
        await wait_for_widget(pilot, app, "#profile-tab")
        # Find a non-active tab; activating it should populate its body.
        target_pane_id = None
        for pane in tabbed.query("TabPane"):
            if pane.id and pane.id != tabbed.active:
                target_pane_id = pane.id
                break
        assert target_pane_id is not None
        tabbed.active = target_pane_id
        await _wait_for_pane_rows(app, pilot, target_pane_id)


async def test_settings_bool_renders_checkbox():
    """Boolean settings render as Checkbox widgets."""
    app = LazySettingsTestApp("show_reasoning")
    async with app.run_test(size=(120, 40)) as _pilot:
        checkboxes = app.screen.query("Checkbox.setting-editor")
        assert len(checkboxes) >= 1


async def test_settings_readonly_no_editor():
    """Read-only settings do not have editor widgets."""
    app = LazySettingsTestApp("chat_model")
    async with app.run_test(size=(120, 40)) as _pilot:
        chat_row = app.screen.query_one("#row-chat_model")
        editors = chat_row.query(".setting-editor")
        assert len(editors) == 0


async def test_settings_model_picker_button_renders_for_model_fields():
    """Each model field surfaces a picker button instead of a plain editor."""
    app = LazySettingsTestApp("chat_model")
    async with app.run_test(size=(120, 40)) as _pilot:
        for key in ("chat_model", "embedding_model", "vision_model", "reranker_model"):
            row = app.screen.query_one(f"#row-{key}")
            buttons = list(row.query(".setting-model-picker-button"))
            assert len(buttons) == 1, f"{key} missing its picker button"


async def test_settings_model_picker_button_pushes_modal_after_worker():
    """Clicking the picker runs discovery in a worker, then pushes the modal."""
    from unittest.mock import patch

    from textual.errors import NoWidget
    from textual.widgets import TabbedContent

    from lilbee.catalog.types import ModelTask
    from lilbee.cli.tui.screens.model_picker import ModelPickerModal
    from lilbee.cli.tui.widgets.model_bar import ModelOption

    fake_buckets = {
        ModelTask.CHAT: [ModelOption(label="fake-chat", ref="fake/chat.gguf")],
        ModelTask.EMBEDDING: [],
        ModelTask.VISION: [],
        ModelTask.RERANK: [],
    }
    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen.query_one("#settings-tabs", TabbedContent).active = "settings-tab-models"
        await _wait_for_pane_rows(app, pilot, "settings-tab-models")
        button = app.screen.query_one("#model-pick-chat_model")

        # A mounted button holds no place on the screen until the layout runs.
        def _clickable() -> bool:
            x, y = button.region.center
            try:
                return app.screen.get_widget_at(int(x), int(y))[0] is button
            except NoWidget:
                return False

        assert await wait_until(pilot, _clickable, timeout=5.0)
        with patch(
            "lilbee.cli.tui.widgets.model_bar.classify_installed_models_full",
            return_value=fake_buckets,
        ):
            await pilot.click(button)
            await wait_until(
                pilot,
                lambda: isinstance(app.screen, ModelPickerModal) and subtree_mounted(app.screen),
                timeout=5.0,
            )
        assert isinstance(app.screen, ModelPickerModal), (
            f"expected ModelPickerModal on top, got {type(app.screen).__name__}"
        )


async def test_settings_model_picker_reports_an_unreadable_registry():
    """An unreadable registry opens no picker and says why, instead of listing nothing."""
    from textual.widgets import TabbedContent

    from lilbee.cli.tui.screens.model_picker import ModelPickerModal

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen.query_one("#settings-tabs", TabbedContent).active = "settings-tab-models"
        await _wait_for_pane_rows(app, pilot, "settings-tab-models")
        with (
            patch(
                "lilbee.cli.tui.widgets.model_bar.classify_installed_models_full",
                side_effect=OSError("manifests unreadable"),
            ),
            patch.object(app.screen, "notify") as notify,
        ):
            app.screen._discover_then_open_picker("chat_model", "chat")
            assert await wait_until(pilot, lambda: notify.called, timeout=5.0)
        assert "manifests unreadable" in notify.call_args.args[0]
        assert notify.call_args.kwargs["severity"] == "error"
        assert not isinstance(app.screen, ModelPickerModal)


def test_settings_screen_has_expected_handlers_and_actions() -> None:
    """Structural regression test: SettingsScreen event handlers + action
    bindings must be class methods, not nested functions captured by a
    misplaced module-level helper.

    The polish-pass commit 6ecd206 silently moved a helper into the
    class body with matching indentation, which closed the class early
    and turned every event handler into unreachable dead code. Parsing
    succeeded because the indentation stayed consistent, and all
    behavioural tests stayed red until caught by CI failures on
    release/next-style platforms. See 5ffc5d6 for the fix.

    This test fires before any pilot harness, so if someone re-breaks
    the class boundary it surfaces immediately instead of at runtime.
    """
    from lilbee.cli.tui.screens.settings import SettingsScreen

    expected = (
        "_on_input_save",
        "_on_checkbox_save",
        "_on_select_save",
        "_persist_value",
        "_parse_value",
        "_refresh_help",
        "action_go_back",
        "action_cursor_down",
        "action_cursor_up",
        "action_jump_top",
        "action_jump_bottom",
    )
    missing = [name for name in expected if not callable(getattr(SettingsScreen, name, None))]
    assert not missing, f"SettingsScreen missing expected methods: {missing}"


async def test_settings_persist_on_change():
    """Changing a setting persists the value to cfg."""
    app = LazySettingsTestApp("top_k")
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        editor = app.screen.query_one("#ed-top_k", Input)
        editor.focus()
        assert await wait_until(pilot, lambda: app.focused is editor), "the editor never got focus"
        editor.value = "20"
        await pilot.press("enter")
        await pump_until(pilot, lambda: cfg.top_k == 20)
        assert cfg.top_k == 20


async def test_settings_cycle_pane_wraps_through_strip():
    """> / < jump straight to the next/previous group tab; wraps off either end."""
    from textual.widgets import TabbedContent

    from lilbee.cli.tui.screens.settings import SettingsScreen
    from lilbee.cli.tui.screens.settings_presets import PRESETS_PANE_ID

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        pane_ids = screen._pane_ids
        assert pane_ids[-1] == PRESETS_PANE_ID
        assert len(pane_ids) >= 2
        # Start on the first pane.
        first = pane_ids[0]
        tabs.active = first
        await pilot.pause()
        # > moves to the next pane.
        screen.action_cycle_pane(1)
        await pilot.pause()
        assert tabs.active == pane_ids[1]
        # < wraps back when called from the first pane.
        tabs.active = first
        await pilot.pause()
        screen.action_cycle_pane(-1)
        await pilot.pause()
        assert tabs.active == pane_ids[-1]


async def test_settings_cycle_pane_handles_missing_tabs():
    """If TabbedContent isn't queryable (mid-mount), action_cycle_pane is a no-op
    rather than raising."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    screen = SettingsScreen.__new__(SettingsScreen)
    screen._pane_groups = {}
    with patch.object(SettingsScreen, "query_one", side_effect=Exception("not yet mounted")):
        SettingsScreen.action_cycle_pane(screen, 1)


async def test_settings_cycle_pane_before_compose_stays_on_the_profile_pane():
    """Before compose adds the group panes, cycling has only the Profile pane to land on."""
    from textual.widgets import TabbedContent

    from lilbee.cli.tui.screens.settings import PROFILE_PANE_ID, SettingsScreen

    screen = SettingsScreen()
    fake_tabs = MagicMock(spec=TabbedContent)
    fake_tabs.active = "settings-tab-models"
    with patch.object(SettingsScreen, "query_one", return_value=fake_tabs):
        SettingsScreen.action_cycle_pane(screen, 1)
    assert fake_tabs.active == PROFILE_PANE_ID


async def test_settings_cycle_pane_unknown_active_starts_from_zero():
    """ValueError fallback: if tabs.active isn't in ``_pane_ids`` (stale id
    from a hot-reload / refactor), cycle starts from index 0."""
    from textual.widgets import TabbedContent

    from lilbee.cli.tui.screens.settings import SettingsScreen

    screen = SettingsScreen.__new__(SettingsScreen)
    screen._pane_ids = ["settings-tab-models", "settings-tab-ingest"]
    fake_tabs = MagicMock(spec=TabbedContent)
    fake_tabs.active = "not-in-pane-groups"
    with patch.object(SettingsScreen, "query_one", return_value=fake_tabs):
        SettingsScreen.action_cycle_pane(screen, 1)
    # Fell through to index 0 + delta 1 -> second pane.
    assert fake_tabs.active == "settings-tab-ingest"


async def test_settings_active_pane_body_handles_missing_tabs():
    """``_active_pane_body`` returns None on every error path: TabbedContent
    not queryable, no active pane, and body widget not yet mounted."""
    from textual.widgets import TabbedContent

    from lilbee.cli.tui.screens.settings import SettingsScreen

    screen = SettingsScreen.__new__(SettingsScreen)
    with patch.object(SettingsScreen, "query_one", side_effect=Exception("not mounted")):
        assert screen._active_pane_body() is None
    fake_empty_tabs = MagicMock(spec=TabbedContent)
    fake_empty_tabs.active = ""
    with patch.object(SettingsScreen, "query_one", return_value=fake_empty_tabs):
        assert screen._active_pane_body() is None
    fake_tabs = MagicMock(spec=TabbedContent)
    fake_tabs.active = "settings-tab-models"
    calls = {"n": 0}

    def _fake_query_one(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return fake_tabs
        raise Exception("body not yet mounted")

    with patch.object(SettingsScreen, "query_one", side_effect=_fake_query_one):
        assert screen._active_pane_body() is None


async def test_settings_persist_value_does_not_toast_on_success():
    """Saving a setting (Input.Blurred / Input.Submitted / Checkbox.Changed
    etc.) must NOT pop a success toast. The editor already shows the new
    value and the write is persisted; the toast is noise that fires en
    masse when a user pages through tabs and blurs the prior pane's
    inputs. Errors still toast."""
    from textual.widgets import Input

    app = LazySettingsTestApp("top_k")
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.screen.query_one("#ed-top_k", Input)
        editor.focus()
        assert await wait_until(pilot, lambda: app.focused is editor), "the editor never got focus"
        editor.value = "21"
        with patch.object(app.screen, "notify") as mock_notify:
            await pilot.press("enter")
            await pilot.pause()
        # Persist must have happened (cfg updated)...
        assert cfg.top_k == 21
        # ...but no success toast.
        assert mock_notify.call_count == 0, (
            f"unexpected toast on settings save: {mock_notify.call_args_list}"
        )


async def test_settings_persist_value_still_toasts_on_error():
    """Error path keeps its toast: the user has to know why a value didn't take."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        from lilbee.app.settings_map import SETTINGS_MAP

        defn = SETTINGS_MAP["top_k"]
        with patch.object(screen, "notify") as mock_notify:
            screen._persist_value("top_k", defn, "not_an_int")
        assert mock_notify.call_count == 1
        assert mock_notify.call_args.kwargs.get("severity") == "error"


async def test_settings_first_pane_populate_is_deferred():
    """The first-pane content build must be queued via call_after_refresh,
    not invoked from on_mount. Running mount_all for ~25 editor widgets
    inline with on_mount adds the layout pass to the screen-switch latency
    budget; deferring lets the screen paint empty first.

    Uses a stub instead of the eager test fixture because the fixture's
    ``populate_all_panes`` would itself call ``_populate_pane`` and mask
    the production code path under inspection.
    """
    from unittest.mock import MagicMock, patch

    from lilbee.cli.tui.screens.settings import SettingsScreen

    screen = SettingsScreen.__new__(SettingsScreen)
    screen._eagerly_populate = "settings-tab-models"
    with (
        patch.object(SettingsScreen, "_populate_pane") as mock_populate,
        patch.object(SettingsScreen, "call_after_refresh", MagicMock()) as mock_defer,
        patch.object(SettingsScreen, "app", new_callable=PropertyMock),
    ):
        SettingsScreen.on_mount(screen)
        assert not mock_populate.called, (
            "on_mount must defer first-pane populate via call_after_refresh"
        )
        mock_defer.assert_called_once()
        args = mock_defer.call_args.args
        assert args[1] == "settings-tab-models"


async def test_settings_pane_body_owns_horizontal_padding():
    """Padding on the inner pane VerticalScroll (not the outer Container)
    keeps column 0 stable on rapid wheel-up; putting padding on the
    Container leaked stale glyphs into the leftmost column when the
    inner scroll repainted (filed bug bb-...-tear)."""
    from textual.containers import VerticalScroll
    from textual.widgets import TabbedContent

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Outer Container must NOT carry left/right padding any more.
        outer = app.screen.query_one("#settings-scroll")
        assert outer.styles.padding.left == 0
        assert outer.styles.padding.right == 0
        # Each pane body's VerticalScroll owns the 1-char gutter instead.
        tabs = app.screen.query_one("#settings-tabs", TabbedContent)
        for pane in tabs.query("TabPane"):
            body = pane.query(VerticalScroll).first()
            assert body.styles.padding.left == 1, (
                f"{pane.id} body lost left padding; column-0 tear will return"
            )
            assert body.styles.padding.right == 1


async def test_settings_outer_container_does_not_double_scroll():
    """bb-e1e2: Settings tearing on Wiki tab scroll-up was caused by two
    VerticalScrolls on the same column (outer #settings-scroll wrapping
    the inner _LazyGroupBody per pane). Drop the outer to a Container so
    only the active pane scrolls; this test pins that invariant."""
    from textual.containers import Container, VerticalScroll

    from lilbee.cli.tui.screens.settings import _LazyGroupBody

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        outer = app.screen.query_one("#settings-scroll")
        assert isinstance(outer, Container)
        assert not isinstance(outer, VerticalScroll), (
            "#settings-scroll regressed to a VerticalScroll; nested-scroll "
            "tearing on Settings → Wiki will return."
        )
        # Every pane body is itself a VerticalScroll, and from any pane body
        # to the screen root there must be exactly one VerticalScroll on
        # the chain (the body itself).
        for body in app.screen.query(_LazyGroupBody):
            scroll_ancestors = sum(
                1 for n in body.ancestors_with_self if isinstance(n, VerticalScroll)
            )
            assert scroll_ancestors == 1, (
                f"{body.id}: {scroll_ancestors} VerticalScroll ancestors, expected 1"
            )


async def test_settings_wiki_pane_scroll_clamps_past_top():
    """Wheeling the Wiki settings pane up past its top edge must clamp at
    scroll_y=0 cleanly. With the previous nested-scroll layout, this is
    where Textual reflow tore the screen and lost the tab strip."""
    from textual.widgets import TabbedContent

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        tabs = app.screen.query_one("#settings-tabs", TabbedContent)
        tabs.active = "settings-tab-wiki"
        await pilot.pause()
        body = app.screen.query_one("#settings-tab-wiki-body")
        body.scroll_to(y=0, animate=False)
        await pilot.pause()
        body.scroll_relative(y=-50, animate=False)
        await pilot.pause()
        assert body.scroll_y == 0.0
        # Tabs strip must still be queryable -- the visual symptom of the
        # original bug was that the strip vanished entirely on scroll-up.
        assert app.screen.query_one("#settings-tabs", TabbedContent) is not None


async def test_settings_exposes_wiki_fields():
    """Settings screen renders an editor for every writable wiki config field."""
    app = LazySettingsTestApp("wiki")
    async with app.run_test(size=(120, 40)) as _pilot:
        writable_wiki_keys = [
            "wiki",
            "wiki_prune_raw",
            "wiki_embedding_faithfulness_threshold",
            "wiki_stale_citation_threshold",
            "wiki_drift_threshold",
            "wiki_clusterer",
            "wiki_clusterer_k",
        ]
        for key in writable_wiki_keys:
            assert app.screen.query_one(f"#ed-{key}") is not None
        # wiki_dir is read-only (writing it silently strands existing wiki
        # content), so the row exists but no editor is rendered.
        assert app.screen.query_one("#row-wiki_dir") is not None


async def test_settings_wiki_clusterer_k_persists():
    """Editing wiki_clusterer_k writes through to cfg."""
    app = LazySettingsTestApp("wiki_clusterer_k")
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        editor = app.screen.query_one("#ed-wiki_clusterer_k", Input)
        editor.focus()
        assert await wait_until(pilot, lambda: app.focused is editor), "the editor never got focus"
        editor.value = "8"
        await pilot.press("enter")
        await pump_until(pilot, lambda: cfg.wiki_clusterer_k == 8)
        assert cfg.wiki_clusterer_k == 8


async def test_settings_checkbox_persist():
    """Toggling a checkbox persists the boolean value."""
    app = LazySettingsTestApp("show_reasoning")
    async with app.run_test(size=(120, 40)):
        from textual.widgets import Checkbox

        cb = app.screen.query_one("#ed-show_reasoning", Checkbox)
        original = cfg.show_reasoning
        async with saved_setting(app, "show_reasoning"):
            cb.toggle()
        assert cfg.show_reasoning != original


async def test_settings_tab_reaches_checkbox_and_space_toggles():
    """Tab walks focus to the checkbox in the active group; Space toggles it.

    Settings groups its editors into a TabbedContent, so reaching a
    given editor requires (a) activating its tab via the TabbedContent
    API and (b) tab-walking inside the active pane until focus lands
    on the editor."""
    from textual.widgets import Checkbox, TabbedContent

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        tabs = app.screen.query_one(TabbedContent)
        tabs.active = "settings-tab-display"  # group hosting show_reasoning
        await pilot.pause()
        cb = app.screen.query_one("#ed-show_reasoning", Checkbox)
        original = cfg.show_reasoning
        for _ in range(20):
            await pilot.press("tab")
            await pilot.pause()
            if app.focused is cb:
                break
        assert app.focused is cb, "Tab failed to reach show_reasoning checkbox"
        async with saved_setting(app, "show_reasoning"):
            await pilot.press("space")
        assert cfg.show_reasoning != original


async def test_settings_vim_keys():
    """Vim navigation keys work on the scroll container."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("j")
        await pilot.press("k")
        await pilot.press("g")
        await pilot.press("G")
        assert isinstance(app.screen, SettingsScreen)
        assert app.screen.query("TabPane")


async def test_settings_pop_screen():
    """Pressing q switches the host to the Chat view via SettingsScreen.action_go_back."""
    from unittest.mock import patch

    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        assert isinstance(app.screen, SettingsScreen)
        with patch.object(app, "switch_view") as switch:
            await pilot.press("q")
            switch.assert_called_once_with("Chat")


async def test_settings_effective_value_shows_set_value_or_none():
    """A set value shows as its text; an unset one shows as "None", with no marker."""
    from lilbee.cli.tui.screens.settings_widgets import effective_value

    old_temp = cfg.temperature
    try:
        cfg.temperature = 0.9
        assert effective_value("temperature") == "0.9"
        cfg.temperature = None
        assert effective_value("temperature") == "None"
    finally:
        cfg.temperature = old_temp


async def test_settings_generation_tab_blur_without_edit_leaves_config_untouched(tmp_path):
    """Moving focus through the Generation editors without typing must not write config.toml.

    Several Generation fields are nullable and show blank when the user
    hasn't set a value. Losing focus on an untouched field must not save
    that blank text as if the user had typed it.
    """
    from textual.widgets import TabbedContent

    from lilbee.core import settings as settings_store
    from tests._async_wait import wait_until

    cfg.data_root = tmp_path
    # A baseline write unrelated to the fields under test, so "byte-identical"
    # proves no write happened at all, not just that these keys stayed absent.
    settings_store.update_values(tmp_path, {"top_k": 99})
    baseline = (tmp_path / "config.toml").read_bytes()

    cfg.temperature = None
    cfg.top_p = None
    cfg.top_k_sampling = None
    cfg.repeat_penalty = None
    cfg.num_ctx = None
    cfg.max_tokens = None
    cfg.seed = None

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 60)) as pilot:
        tabs = app.screen.query_one("#settings-tabs", TabbedContent)
        tabs.active = "settings-tab-generation"
        await pilot.pause()
        body = app.screen.query_one("#settings-tab-generation-body")
        # Only an Input or a ListTextArea saves on blur; a Checkbox or Select
        # saves on change and a reset button on press. Each settled focus
        # change rebuilds the footer and restyles the screen, so focus moves
        # through the blur-saving editors in one batch and settles once.
        editors = list(body.query("Input.setting-editor, ListTextArea"))
        assert "ed-temperature" in {editor.id for editor in editors}
        for editor in editors:
            editor.focus()
        reset = app.screen.query_one("#reset-temperature")
        reset.focus()
        assert await wait_until(pilot, lambda: app.screen.focused is reset)
        await pilot.pause()

    assert (tmp_path / "config.toml").read_bytes() == baseline
    assert cfg.temperature is None
    assert cfg.top_p is None
    assert cfg.top_k_sampling is None
    assert cfg.repeat_penalty is None
    assert cfg.num_ctx is None
    assert cfg.max_tokens is None
    assert cfg.seed is None


async def test_settings_edit_over_unset_value_saves(tmp_path):
    """Typing a value into a field that shows blank saves it."""
    from textual.widgets import Input

    cfg.data_root = tmp_path
    cfg.temperature = None

    app = LazySettingsTestApp("temperature")
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.screen.query_one("#ed-temperature", Input)
        assert editor.value == ""
        editor.focus()
        editor.value = "0.85"
        await pilot.press("enter")
        await pump_until(pilot, lambda: cfg.temperature == 0.85)
    assert cfg.temperature == 0.85


async def test_settings_is_writable():
    """is_writable correctly identifies writable vs read-only fields."""
    from lilbee.cli.tui.screens.settings_widgets import is_writable

    assert is_writable("top_k")
    assert is_writable("temperature")
    assert not is_writable("chat_model")
    assert not is_writable("embedding_model")
    assert not is_writable("nonexistent_key_xyz")


async def test_settings_persist_invalid_int():
    """Invalid value for int field shows error and does not change cfg."""
    app = LazySettingsTestApp("top_k")
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        original = cfg.top_k
        editor = app.screen.query_one("#ed-top_k", Input)
        editor.focus()
        assert await wait_until(pilot, lambda: app.focused is editor), "the editor never got focus"
        editor.value = "abc"
        await pilot.press("enter")
        await pilot.pause()
        assert cfg.top_k == original


async def test_settings_repeated_identical_invalid_blur_notifies_twice():
    """Two identical invalid blurs on the same field must both notify.

    The mount-display baseline updates only when ``_persist_value``
    succeeds, so an unparseable value never gets treated as the field's
    shown text; without that, the baseline picked up the invalid text on
    the first blur and a second identical invalid blur compared equal to it
    and stayed silent.
    """
    from textual.widgets import Input

    app = LazySettingsTestApp("top_k")
    async with app.run_test(size=(120, 40)) as pilot:
        original = cfg.top_k
        editor = app.screen.query_one("#ed-top_k", Input)
        with patch.object(app.screen, "notify") as mock_notify:
            editor.focus()
            editor.value = "abc"
            await pilot.press("enter")
            await pump_until(pilot, lambda: mock_notify.call_count >= 1)
            assert mock_notify.call_count == 1

            editor.focus()
            editor.value = "abc"
            await pilot.press("enter")
            await pump_until(pilot, lambda: mock_notify.call_count >= 2)
            assert mock_notify.call_count == 2

        assert cfg.top_k == original


async def test_settings_select_save():
    """_on_select_save routes through _persist_value correctly."""
    from lilbee.app.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        defn = SettingDef(type=str, nullable=False, group="Generation")
        event = MagicMock()
        event.select.name = "test_select"
        event.value = "chosen"
        with (
            patch.dict(
                "lilbee.cli.tui.screens.settings.SETTINGS_MAP",
                {"test_select": defn},
            ),
            patch.object(screen, "_persist_value") as mock_persist,
        ):
            screen._on_select_save(event)
            mock_persist.assert_called_once_with("test_select", defn, "chosen")


async def test_settings_select_save_compares_against_mount_baseline_not_cfg():
    """A Select save compares the new value to the mount baseline, not to cfg.

    The two always agree in practice; this manufactures the divergence to
    prove which one the comparison actually uses.
    """
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp("wiki_clusterer")
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        key = "wiki_clusterer"
        assert screen._mount_display.get(key) is not None
        screen._mount_display[key] = "concepts"
        cfg.wiki_clusterer = "embedding"
        event = MagicMock()
        event.select.name = key
        event.value = "concepts"
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_select_save(event)
            mock_pv.assert_not_called()
