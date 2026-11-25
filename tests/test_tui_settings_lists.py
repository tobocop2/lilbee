"""Settings screen list editors: crawl patterns, browser flags and OCR lists."""

from __future__ import annotations

from unittest.mock import patch

from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.app.settings_map import SettingGroup
from lilbee.core.config import cfg
from tests._async_wait import wait_until
from tests._crawlberg_stub import StubCrawlberg, refusing_patterns
from tests._lilbee_app_test_host import LilbeeAppHost, pump_until, saved_setting
from tests._settings_test_app import LazySettingsTestApp, populate_pane_of
from tests._tui_screen_fixtures import (  # noqa: F401
    _isolated_cfg,
    _patch_chat_setup,
    _settings_gates_open,
    mock_svc,
)


async def test_settings_crawl_exclude_patterns_renders_collapsible():
    """crawl_exclude_patterns renders as a Collapsible with line count in title."""
    from textual.widgets import Collapsible

    from lilbee.core.config import cfg

    app = LazySettingsTestApp("crawl_exclude_patterns")
    async with app.run_test(size=(120, 40)) as _pilot:
        collapsible = app.screen.query_one("#collapsible-crawl_exclude_patterns", Collapsible)
        assert collapsible.collapsed is True
        assert "crawl_exclude_patterns" in collapsible.title
        current = cfg.crawl_exclude_patterns or []
        assert f"({len(current)} lines)" in collapsible.title


async def test_settings_list_editor_can_be_expanded():
    """Setting `collapsed = False` on the public Collapsible API reveals the editor."""
    from textual.widgets import Collapsible

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = LazySettingsTestApp("crawl_exclude_patterns")
    async with app.run_test(size=(120, 40)) as pilot:
        collapsible = app.screen.query_one("#collapsible-crawl_exclude_patterns", Collapsible)
        assert collapsible.collapsed is True
        collapsible.collapsed = False
        await pilot.pause()
        # The TextArea is inside the body once the Collapsible is open.
        ta = collapsible.query_one(ListTextArea)
        assert ta is not None


async def test_settings_list_editor_saves_on_blur():
    """Typing into the list TextArea and blurring persists the parsed list."""

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = LazySettingsTestApp("crawl_exclude_patterns")
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        ta.focus()
        assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
        ta.load_text("foo\nbar")
        async with saved_setting(app, "crawl_exclude_patterns"):
            ta.blur()
        assert cfg.crawl_exclude_patterns == ["foo", "bar"]


async def test_settings_list_editor_blur_without_edit_leaves_config_untouched(tmp_path):
    """Blurring a list-typed editor with no edit must not write config.toml.

    Mirrors the scalar mount-baseline fix: a list editor's blur handler used
    to persist unconditionally, so tabbing past crawl_exclude_patterns
    without editing it rewrote config.toml on every visit.
    """
    from textual.widgets import Input

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea
    from lilbee.core import settings as settings_store

    cfg.data_root = tmp_path
    # A baseline write unrelated to the field under test, so "byte-identical"
    # proves no write happened at all, not just that this key stayed absent.
    settings_store.update_values(tmp_path, {"top_k": 99})
    baseline = (tmp_path / "config.toml").read_bytes()

    app = LazySettingsTestApp("crawl_exclude_patterns", "top_k")
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        ta.focus()
        await pilot.pause()
        app.screen.query_one("#ed-top_k", Input).focus()
        await pilot.pause()

    assert (tmp_path / "config.toml").read_bytes() == baseline


async def test_settings_list_editor_strips_blanks():
    """Blank lines and surrounding whitespace are stripped during parsing."""

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = LazySettingsTestApp("crawl_exclude_patterns")
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        ta.focus()
        assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
        ta.load_text("a\n\nb\n")
        async with saved_setting(app, "crawl_exclude_patterns"):
            ta.blur()
        assert cfg.crawl_exclude_patterns == ["a", "b"]


async def test_settings_list_editor_refused_pattern_blocks_save():
    """A pattern crawlberg refuses shows its line and reason and does not mutate cfg."""
    from textual.widgets import Static

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    refused = r"\Z"
    stub = StubCrawlberg(refuse=refusing_patterns(refused))
    app = LazySettingsTestApp()
    with stub.installed():
        async with app.run_test(size=(120, 40)) as pilot:
            cfg.crawl_exclude_patterns = ["keep"]
            populate_pane_of(app.screen, "crawl_exclude_patterns")
            assert await wait_until(
                pilot, lambda: bool(app.screen.query("#ed-crawl_exclude_patterns"))
            )
            ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
            ta.focus()
            assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
            ta.load_text(f"/ok\n{refused}")
            ta.blur()
            err = app.screen.query_one("#err-crawl_exclude_patterns", Static)
            # Focus change → blur handler → check → error text are all async. Wait
            # for the text the test asserts on, and fail on the wait if it never lands.
            assert await pump_until(pilot, lambda: "line 2" in str(err.render())), (
                "the refused-pattern error never rendered"
            )
            assert err.has_class("-visible")
            shown = str(err.render())
            assert "line 2" in shown
            assert "unrecognized escape sequence" in shown
            assert cfg.crawl_exclude_patterns == ["keep"]


async def test_settings_refresh_after_focus_leaves_keeps_the_unsaved_edit():
    """A refresh that lands before the blur save keeps the typed text, so the save checks it."""
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    key = "crawl_exclude_patterns"
    refused = r"\Z"
    stub = StubCrawlberg(refuse=refusing_patterns(refused))
    app = LazySettingsTestApp(key)
    with stub.installed():
        async with app.run_test(size=(120, 40)) as pilot:
            cfg.crawl_exclude_patterns = ["keep"]
            screen = app.screen
            ta = screen.query_one(f"#ed-{key}", ListTextArea)
            ta.focus()
            assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
            ta.load_text(f"/ok\n{refused}")
            ta.blur()
            # Focus has left; the blur save still waits in the queue when the refresh runs.
            screen._on_setting_changed((key, cfg.crawl_exclude_patterns))
            screen._refresh_changed_rows()
            assert ta.text == f"/ok\n{refused}"
            err = screen.query_one(f"#err-{key}", Static)
            assert await pump_until(pilot, lambda: "line 2" in str(err.render())), (
                "the refused-pattern error never rendered"
            )
            assert cfg.crawl_exclude_patterns == ["keep"]


async def test_settings_refresh_shows_cfg_in_an_editor_without_an_edit():
    """A refresh still puts cfg's value in an editor that holds no unsaved edit."""
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    key = "crawl_exclude_patterns"
    app = LazySettingsTestApp(key)
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        ta = screen.query_one(f"#ed-{key}", ListTextArea)
        cfg.crawl_exclude_patterns = ["/new"]
        screen._on_setting_changed((key, cfg.crawl_exclude_patterns))
        screen._refresh_changed_rows()
        assert ta.text == "/new"


async def test_settings_refresh_after_a_refused_save_shows_the_stored_value():
    """Once a refused save has run, an outside change shows the stored value in the field."""
    from textual.widgets import Input

    app = LazySettingsTestApp("top_k")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        field = screen.query_one("#ed-top_k", Input)
        field.focus()
        assert await wait_until(pilot, lambda: app.focused is field), "the field never got focus"
        field.value = "abc"
        stored = cfg.top_k
        field.blur()
        assert "top_k" in screen._saves_pending
        assert await wait_until(pilot, lambda: "top_k" not in screen._saves_pending)
        assert cfg.top_k == stored
        cfg.top_k = 17
        app.publish_settings(["top_k"])
        assert await pump_until(pilot, lambda: field.value == "17"), field.value


async def test_settings_profile_refresh_replaces_a_refused_pattern():
    """The refresh a profile apply sends shows the profile's value over a refused pattern."""
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    key = "crawl_exclude_patterns"
    refused = r"\Z"
    app = LazySettingsTestApp(key)
    with StubCrawlberg(refuse=refusing_patterns(refused)).installed():
        async with app.run_test(size=(120, 40)) as pilot:
            cfg.crawl_exclude_patterns = ["keep"]
            screen = app.screen
            ta = screen.query_one(f"#ed-{key}", ListTextArea)
            ta.focus()
            assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
            ta.load_text(refused)
            ta.blur()
            err = screen.query_one(f"#err-{key}", Static)
            assert await pump_until(pilot, lambda: "line 1" in str(err.render()))
            cfg.crawl_exclude_patterns = ["/profile"]
            app.publish_settings([key])
            assert await pump_until(pilot, lambda: ta.text == "/profile"), ta.text


async def test_settings_list_editor_saves_a_pattern_only_crawlberg_compiles():
    """Python's re refuses this pattern; crawlberg compiles it, so the screen saves it."""
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    pattern = r"/wiki/\p{L}+:"
    stub = StubCrawlberg(refuse=refusing_patterns(r"\Z"))
    app = LazySettingsTestApp("crawl_exclude_patterns")
    with stub.installed():
        async with app.run_test(size=(120, 40)) as pilot:
            cfg.crawl_exclude_patterns = ["keep"]
            ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
            ta.focus()
            assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
            ta.load_text(pattern)
            async with saved_setting(app, "crawl_exclude_patterns"):
                ta.blur()
            assert cfg.crawl_exclude_patterns == [pattern]


async def test_settings_browser_flags_editor_saves_on_blur():
    """The launch-flag list editor stores each line as one flag."""
    from textual.widgets import Collapsible

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = LazySettingsTestApp("crawl_browser_extra_args")
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-crawl_browser_extra_args", ListTextArea)
        ta.focus()
        assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
        ta.load_text("--lang=fr\n--mute-audio\n--disable-gpu")
        async with saved_setting(app, "crawl_browser_extra_args"):
            ta.blur()
        flags = ["--lang=fr", "--mute-audio", "--disable-gpu"]
        assert cfg.crawl_browser_extra_args == flags
        title = app.screen.query_one("#collapsible-crawl_browser_extra_args", Collapsible).title
        assert "(3 lines)" in title


async def test_settings_browser_flags_editor_stores_a_flag_crawlberg_refuses():
    """The editor stores every line and counts it; crawlberg checks the flags at crawl time."""
    from textual.widgets import Collapsible

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = LazySettingsTestApp("crawl_browser_extra_args")
    async with app.run_test(size=(120, 40)) as pilot:
        cfg.crawl_browser_extra_args = ["--lang=fr"]
        ta = app.screen.query_one("#ed-crawl_browser_extra_args", ListTextArea)
        ta.focus()
        assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
        ta.load_text("--lang=fr\n--headless=new\nmute-audio")
        async with saved_setting(app, "crawl_browser_extra_args"):
            ta.blur()
        flags = ["--lang=fr", "--headless=new", "mute-audio"]
        assert cfg.crawl_browser_extra_args == flags
        title = app.screen.query_one("#collapsible-crawl_browser_extra_args", Collapsible).title
        assert "(3 lines)" in title


async def test_settings_crawl_user_agent_editor_saves_the_override():
    """The crawl user agent editor stores the text it holds."""
    from textual.widgets import Input

    app = LazySettingsTestApp("crawl_user_agent")
    async with app.run_test(size=(120, 40)) as pilot:
        cfg.crawl_user_agent = ""
        editor = app.screen.query_one("#ed-crawl_user_agent", Input)
        editor.focus()
        assert await wait_until(pilot, lambda: app.focused is editor), "the editor never got focus"
        editor.value = "Probe/1.0"
        async with saved_setting(app, "crawl_user_agent"):
            await pilot.press("enter")
        assert cfg.crawl_user_agent == "Probe/1.0"


async def test_settings_extra_browser_wait_editor_saves_the_wait():
    """The extra browser wait editor stores the milliseconds it holds."""
    from textual.widgets import Input

    app = LazySettingsTestApp("crawl_browser_extra_wait_ms")
    async with app.run_test(size=(120, 40)) as pilot:
        cfg.crawl_browser_extra_wait_ms = 1000
        editor = app.screen.query_one("#ed-crawl_browser_extra_wait_ms", Input)
        editor.focus()
        assert await wait_until(pilot, lambda: app.focused is editor), "the editor never got focus"
        editor.value = "2500"
        async with saved_setting(app, "crawl_browser_extra_wait_ms"):
            await pilot.press("enter")
        assert cfg.crawl_browser_extra_wait_ms == 2500


async def test_settings_robots_txt_checkbox_turns_the_setting_off():
    """The robots.txt checkbox starts ticked and unticking it stores off."""
    from textual.widgets import Checkbox

    cfg.crawl_respect_robots_txt = True
    app = LazySettingsTestApp("crawl_respect_robots_txt")
    async with app.run_test(size=(120, 40)):
        checkbox = app.screen.query_one("#ed-crawl_respect_robots_txt", Checkbox)
        assert checkbox.value is True
        async with saved_setting(app, "crawl_respect_robots_txt"):
            checkbox.toggle()
        assert cfg.crawl_respect_robots_txt is False


async def test_settings_list_editor_restore_defaults():
    """Pressing Restore resets both cfg and the TextArea to the defaults."""
    from textual.widgets import Button

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea
    from lilbee.core.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS

    app = LazySettingsTestApp("crawl_exclude_patterns")
    async with app.run_test(size=(120, 40)):
        cfg.crawl_exclude_patterns = ["old"]
        btn = app.screen.query_one("#list-restore-crawl_exclude_patterns", Button)
        async with saved_setting(app, "crawl_exclude_patterns"):
            btn.press()
        assert cfg.crawl_exclude_patterns == list(DEFAULT_CRAWL_EXCLUDE_PATTERNS)
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        assert ta.text == "\n".join(DEFAULT_CRAWL_EXCLUDE_PATTERNS)


async def test_settings_list_editor_restore_then_blur_does_not_resave():
    """Blurring the list editor right after Restore, unedited, must not save again.

    First edit the field so its mount-display baseline diverges from the
    default; if Restore left that stale baseline in place instead of
    updating it, the restored (default) text would still look changed on
    the next blur and get persisted again.
    """
    from textual.widgets import Button, Input

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = LazySettingsTestApp("crawl_exclude_patterns", "top_k")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        ta = screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        ta.focus()
        await pilot.pause()
        ta.load_text("edited")
        ta.blur()
        await pump_until(pilot, lambda: cfg.crawl_exclude_patterns == ["edited"])

        btn = screen.query_one("#list-restore-crawl_exclude_patterns", Button)
        btn.press()
        await pump_until(pilot, lambda: cfg.crawl_exclude_patterns != ["edited"])
        assert screen._mount_display["crawl_exclude_patterns"] == ta.text

        with patch.object(screen, "_persist_value") as mock_pv:
            ta.focus()
            await pilot.pause()
            screen.query_one("#ed-top_k", Input).focus()
            await pilot.pause()
            mock_pv.assert_not_called()


async def test_settings_parse_value_list_branch():
    """_parse_value splits, strips, and drops blanks for list-typed settings."""
    from lilbee.app.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    defn = SettingDef(list, nullable=False, group=SettingGroup.INGEST)
    screen = SettingsScreen()
    result = screen._parse_value(defn, "a\n\nb")
    assert result == ["a", "b"]


async def test_settings_list_editor_persists_through_toml_round_trip(tmp_path):
    """Editing a list-typed setting survives a full settings.save + pydantic reload.

    Guards against the bug where `_persist_value` used `str(parsed)` for lists,
    which wrote Python repr such as "['foo', 'bar']" into the TOML store.
    After reload, the pydantic `splitlines()` validator then produced a
    one-element list with corrupt contents.
    """

    from lilbee.app.settings_map import SETTINGS_MAP
    from lilbee.cli.tui.screens.settings import SettingsScreen
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea
    from lilbee.core import settings

    cfg.data_root = tmp_path
    app = LazySettingsTestApp("crawl_exclude_patterns")
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        ta.focus()
        assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
        ta.load_text("pat-a\npat-b")
        # The blur handler posts Blurred, the screen handles it, and only then
        # is the value written and the change published.
        async with saved_setting(app, "crawl_exclude_patterns"):
            ta.blur()

    # Raw TOML value is a newline-joined string (not Python repr of the list).
    reloaded = settings.load(tmp_path)
    raw = reloaded["crawl_exclude_patterns"]
    assert raw == "pat-a\npat-b"
    assert "[" not in raw  # would indicate list repr leaked into TOML

    # Pydantic's `splitlines()` validator then reconstructs the list cleanly.
    screen = SettingsScreen()
    parsed = screen._parse_value(SETTINGS_MAP["crawl_exclude_patterns"], raw)
    assert parsed == ["pat-a", "pat-b"]


async def test_settings_every_list_setting_gets_the_list_editor():
    """A list-typed setting never lands in a plain Input.

    The Input path is seeded from ``effective_value``, which summarises a list
    as "N lines"; saving the form then wrote that summary back as the value.
    """
    from textual.widgets import Collapsible

    from lilbee.app.settings_map import SETTINGS_MAP
    from lilbee.cli.tui.screens.settings_widgets import group_settings
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    list_keys = [
        key
        for group in group_settings().values()
        for key, defn in group
        if SETTINGS_MAP[key].type is list
    ]
    assert "ocr_language" in list_keys
    app = LazySettingsTestApp(*list_keys)
    async with app.run_test(size=(120, 60)):
        for key in list_keys:
            app.screen.query_one(f"#collapsible-{key}", Collapsible)
            assert isinstance(app.screen.query_one(f"#ed-{key}"), ListTextArea), key


async def test_settings_ocr_language_round_trips_through_the_list_editor(tmp_path):
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea
    from lilbee.core import settings

    cfg.data_root = tmp_path
    app = LazySettingsTestApp("ocr_language")
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-ocr_language", ListTextArea)
        ta.focus()
        assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
        ta.load_text("eng\ndeu")
        async with saved_setting(app, "ocr_language"):
            ta.blur()
    assert cfg.ocr_language == ["eng", "deu"]
    assert settings.load(tmp_path)["ocr_language"] == "eng\ndeu"


async def test_settings_force_ocr_pages_round_trips_through_the_list_editor(tmp_path, monkeypatch):
    """The list editor renders and saves integer items, not only strings."""
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea
    from lilbee.core import settings

    cfg.data_root = tmp_path
    monkeypatch.setattr(cfg, "force_ocr_pages", [1, 3])
    app = LazySettingsTestApp("force_ocr_pages")
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-force_ocr_pages", ListTextArea)
        assert ta.text == "1\n3"
        ta.focus()
        assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
        ta.load_text("5\n2")
        async with saved_setting(app, "force_ocr_pages"):
            ta.blur()
    assert cfg.force_ocr_pages == [2, 5]
    assert settings.load(tmp_path)["force_ocr_pages"] == "2\n5"


def test_list_editor_shows_integer_items_one_per_line():
    from lilbee.cli.tui.screens.settings_widgets import set_widget_value
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    editor = ListTextArea("")
    set_widget_value(editor, [2, 4])
    assert editor.text == "2\n4"


async def test_list_text_area_posts_blurred():
    """ListTextArea posts its Blurred message when focus moves away."""
    from textual.widgets import Input

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    captured: list[ListTextArea.Blurred] = []

    class _TestApp(LilbeeAppHost):
        CSS = ""

        def compose(self) -> ComposeResult:
            yield ListTextArea(id="ta")
            yield Input(id="other")

        def on_list_text_area_blurred(self, event: ListTextArea.Blurred) -> None:
            captured.append(event)

    app = _TestApp()
    async with app.run_test(size=(80, 20)) as pilot:
        ta = app.query_one("#ta", ListTextArea)
        ta.focus()
        assert await wait_until(pilot, lambda: app.focused is ta), "the editor never got focus"
        await pilot.pause()
        assert not captured, "ListTextArea posted Blurred before focus left it"
        app.query_one("#other", Input).focus()
        assert await wait_until(pilot, lambda: bool(captured)), (
            "Expected a Blurred message from ListTextArea"
        )
        assert len(captured) == 1
        assert captured[0].control is ta
