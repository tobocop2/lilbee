"""The /preset slash command and the Presets tab of the Settings screen, over the real service."""

from __future__ import annotations

import json
from unittest import mock

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Checkbox, Input, Label, Select, Static, TabbedContent

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.app.presets import add_preset, list_presets
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.command_registry import build_dispatch_dict, runs_while_streaming
from lilbee.cli.tui.preset_commands import (
    DialogRequest,
    PresetCommandResult,
    dialog_request,
    is_preset_line,
    key_typed,
    remove_target,
    run_preset_command,
)
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.settings_presets import (
    PRESETS_PANE_ID,
    PresetDialog,
    PresetsPane,
)
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost

KEY = "SENTINEL-tui-key-91c4"
LOCAL = "http://127.0.0.1:8000/v1"
OTHER = "http://127.0.0.1:9000/v1"
SIZES = [(80, 30), (140, 40)]


@pytest.fixture(autouse=True)
def _chat_setup():
    """Release the startup gate at once, with no model scan or embedder check."""
    from lilbee.app.services import set_services
    from lilbee.cli.tui.widgets.model_bar import ModelBar
    from tests.conftest import make_mock_services

    set_services(make_mock_services())
    with (
        mock.patch("lilbee.cli.tui.app.chat_ready", return_value=True),
        mock.patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
        mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
        mock.patch.object(ModelBar, "_scan_models"),
    ):
        yield
    set_services(None)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.server_presets = None
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _stored() -> list[dict]:
    raw = settings.load(cfg.data_root).get("server_presets")
    return json.loads(raw) if raw else []


def _seed(name: str = "omlx", url: str = LOCAL, key: str | None = KEY) -> None:
    add_preset(name, url, roles=["chat", "embed"], api_key=key)


class _ChatApp(LilbeeAppHost):
    """An app that opens the chat screen directly, with no startup gate."""

    def __init__(self) -> None:
        super().__init__()
        from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


async def await_chat(app, pilot):
    """The mounted chat screen."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    assert await wait_until(
        pilot, lambda: isinstance(app.screen, ChatScreen) and app.screen.is_mounted
    )
    return app.screen


async def await_dialog(pilot, app) -> bool:
    """Wait until the preset dialog is the top screen and every widget in it is mounted."""
    return await wait_until(
        pilot,
        lambda: (
            isinstance(app.screen, PresetDialog)
            and all(w.is_mounted for w in app.screen.walk_children(with_self=True))
        ),
    )


def _press(app, selector: str) -> None:
    """Press the button *selector* on the top screen."""
    app.screen.query_one(selector, Button).press()


def _screen_text(app) -> str:
    return "\n".join(str(w.content) for w in app.screen.query(Static))


class TestRunPresetCommand:
    @pytest.mark.parametrize(
        "args", ["", "frob", "list extra", "add", "add x", "'open", "remove", "edit"]
    )
    def test_a_malformed_command_gets_the_usage_text(self, args):
        result = run_preset_command(args)
        assert (result.ok, result.text) == (False, msg.PRESET_USAGE)

    @pytest.mark.parametrize(
        "args",
        [
            f"add x1 {LOCAL} --roles chat --bogus 1",
            f"add x1 {LOCAL} --roles",
            f"add x1 {LOCAL} --clear-key",
            f"add x1 {LOCAL} --url {OTHER}",
            "edit omlx extra",
            "remove a b",
        ],
    )
    def test_a_bad_flag_shape_gets_the_usage_text(self, args):
        _seed()
        assert run_preset_command(args) == run_preset_command("")
        assert len(_stored()) == 1

    def test_list_with_nothing_stored(self):
        assert run_preset_command("list") == run_preset_command("list")
        assert run_preset_command("list").text == msg.PRESET_LIST_EMPTY

    def test_list_shows_the_key_state_and_never_the_key(self):
        _seed()
        result = run_preset_command("list")
        assert result.ok
        assert "omlx" in result.text and "key: set" in result.text
        assert KEY not in result.text

    def test_list_shows_the_wire_only_with_the_rerank_role(self):
        add_preset("rr", LOCAL, roles=["rerank"], rerank_wire="tei")
        assert "wire: tei" in run_preset_command("list").text

    def test_add_stores_the_preset(self):
        result = run_preset_command(f"add omlx {LOCAL} --roles chat,embed --wire v1")
        assert (result.ok, result.text) == (True, "Added server preset 'omlx'.")
        assert [(e["name"], e["url"], e["roles"]) for e in _stored()] == [
            ("omlx", LOCAL, ["chat", "embed"])
        ]

    def test_add_refuses_a_key_flag_and_stores_nothing(self):
        result = run_preset_command(f"add x1 {LOCAL} --roles chat --key {KEY}")
        assert (result.ok, result.text) == (False, msg.PRESET_KEY_IN_DIALOG)
        assert _stored() == []

    def test_add_with_no_roles_shows_the_service_text(self):
        result = run_preset_command(f"add x1 {LOCAL}")
        assert result == run_preset_command(f"add x1 {LOCAL} --roles ''")
        assert result.text == "A preset needs at least one role: embed, rerank, chat, vision."

    def test_add_with_a_name_that_holds_a_space_shows_the_name_rule(self):
        result = run_preset_command(f"add a b {LOCAL} --roles chat")
        assert not result.ok and result.text.startswith("A preset name uses")

    def test_edit_changes_the_url_and_keeps_the_key(self):
        _seed()
        result = run_preset_command(f"edit omlx --url {LOCAL}/x --roles chat --wire tei")
        assert (result.ok, result.text) == (True, "Updated server preset 'omlx'.")
        assert _stored() == [
            {
                "name": "omlx",
                "url": f"{LOCAL}/x",
                "roles": ["chat"],
                "rerank_wire": "tei",
                "api_key": KEY,
            }
        ]

    def test_edit_changes_only_the_flags_it_is_given(self):
        add_preset("rr", LOCAL, roles=["rerank"], rerank_wire="tei", api_key=KEY)
        assert run_preset_command("edit rr --roles rerank,chat").ok
        assert _stored() == [
            {
                "name": "rr",
                "url": LOCAL,
                "roles": ["rerank", "chat"],
                "rerank_wire": "tei",
                "api_key": KEY,
            }
        ]

    def test_edit_clear_key(self):
        _seed()
        assert run_preset_command("edit omlx --clear-key").ok
        assert not list_presets()[0].has_key

    def test_edit_with_no_field_shows_the_service_text(self):
        _seed()
        assert run_preset_command("edit omlx").text.startswith("Nothing to change.")

    def test_edit_a_missing_preset(self):
        assert (
            run_preset_command("edit nope --clear-key").text == "No server preset is named 'nope'."
        )

    def test_remove(self):
        _seed()
        result = run_preset_command("remove omlx")
        assert (result.ok, result.text) == (True, "Removed server preset 'omlx'.")
        assert _stored() == []

    def test_a_locked_config_file_shows_the_service_text(self):
        with mock.patch("lilbee.cli.tui.preset_commands.list_presets", side_effect=OSError("busy")):
            result = run_preset_command("list")
        assert not result.ok and result.text

    def test_dialog_request_and_remove_target(self):
        assert dialog_request("add") == DialogRequest(None)
        assert dialog_request("edit omlx") == DialogRequest("omlx")
        assert dialog_request(f"add x {LOCAL}") is None
        assert dialog_request("edit omlx --clear-key") is None
        assert dialog_request("'open") is None
        assert remove_target("remove omlx") == "omlx"
        assert remove_target("remove") is None
        assert remove_target("list") is None


class TestKeyDetection:
    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            (f"add x1 {LOCAL} --key abc", True),
            (f"add x1 {LOCAL} --key=abc", True),
            ("add 'x1 --key abc", True),
            (f"add x1 {LOCAL} --roles chat", False),
            ("add x1 --keyboard 1", False),
            ("", False),
        ],
    )
    def test_key_typed_finds_the_flag_in_either_form(self, args, expected):
        assert key_typed(args) is expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("/preset list", True),
            ("/preset\tadd x1 u --key abc", True),
            ("/preset   add x1 u --api-key abc", True),
            ("  /preset add x1 u --key=abc", True),
            ("/PRESET", True),
            ("/presets list", False),
            ("/set --key abc", False),
            ("hello /preset", False),
            ("", False),
        ],
    )
    def test_only_a_preset_line_counts_whatever_the_spacing(self, text, expected):
        assert is_preset_line(text) is expected

    @pytest.mark.parametrize("args", [f"add x1 {LOCAL} --key=abc", "edit omlx --key abc"])
    def test_run_refuses_either_form_with_the_fixed_text(self, args):
        _seed()
        assert run_preset_command(args) == PresetCommandResult(False, msg.PRESET_KEY_IN_DIALOG)
        assert len(_stored()) == 1


class TestRegistry:
    def test_preset_is_registered_and_not_allowed_mid_answer(self):
        assert build_dispatch_dict()["/preset"] == "_cmd_preset"
        assert runs_while_streaming("/preset") is False


class TestChatHandler:
    async def test_list_notifies_the_listing(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            with mock.patch.object(chat, "notify") as notify:
                chat._cmd_preset("list")
                assert await wait_until(pilot, lambda: notify.called)
        text, kwargs = notify.call_args.args[0], notify.call_args.kwargs
        assert "omlx" in text and KEY not in text
        assert kwargs == {"severity": "information"}

    async def test_a_refusal_notifies_as_an_error(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            with mock.patch.object(chat, "notify") as notify:
                chat._cmd_preset("frob")
                assert await wait_until(pilot, lambda: notify.called)
        assert notify.call_args.args[0] == msg.PRESET_USAGE
        assert notify.call_args.kwargs == {"severity": "error"}

    async def test_remove_asks_first_and_cancel_keeps_the_preset(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            chat._cmd_preset("remove omlx")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
            assert await wait_until(pilot, lambda: bool(app.screen.query("#confirm-message")))
            assert "omlx" in str(app.screen.query_one("#confirm-message", Label).content)
            await pilot.press("n")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, ConfirmDialog))
        assert len(_stored()) == 1

    async def test_remove_after_yes_removes_the_preset(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            chat._cmd_preset("remove omlx")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
            await pilot.press("y")
            assert await wait_until(pilot, lambda: _stored() == [])

    async def test_add_with_no_arguments_opens_the_dialog(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            chat._cmd_preset("add")
            assert await await_dialog(pilot, app)
            assert app.screen.query("#preset-name-input")

    async def test_edit_with_a_name_opens_the_dialog_for_that_preset(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            chat._cmd_preset("edit omlx")
            assert await await_dialog(pilot, app)
            assert app.screen.query_one("#preset-url-input", Input).value == LOCAL

    async def test_edit_of_a_missing_preset_notifies_the_service_text(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            with mock.patch.object(chat, "notify") as notify:
                chat._cmd_preset("edit nope")
                assert await wait_until(pilot, lambda: notify.called)
            assert not isinstance(app.screen, PresetDialog)
        assert notify.call_args.args[0] == "No server preset is named 'nope'."

    async def test_edit_dialog_result_is_notified(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            with mock.patch.object(chat, "notify") as notify:
                chat._cmd_preset("edit omlx")
                assert await await_dialog(pilot, app)
                app.screen.query_one("#preset-key-input", Input).value = "NEWKEY"
                _press(app, "#preset-save")
                assert await wait_until(pilot, lambda: notify.called)
        assert notify.call_args.args[0] == "Updated server preset 'omlx'."

    async def test_a_lookup_failure_notifies_the_service_text(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            with (
                mock.patch("lilbee.app.presets.get_preset", side_effect=OSError("busy")),
                mock.patch.object(chat, "notify") as notify,
            ):
                chat._cmd_preset("edit omlx")
                assert await wait_until(pilot, lambda: notify.called)
        assert notify.call_args.kwargs == {"severity": "error"}


async def _open_pane(pilot, app) -> PresetsPane:
    await await_chat(app, pilot)
    app.push_screen(SettingsScreen())
    assert await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
    screen = app.screen
    assert await wait_until(pilot, lambda: bool(screen.query("#settings-tabs")))
    screen.query_one("#settings-tabs", TabbedContent).active = PRESETS_PANE_ID
    return screen.query_one(PresetsPane)


class TestPane:
    @pytest.mark.parametrize("size", SIZES)
    async def test_the_pane_lists_presets_with_the_key_state_only(self, size):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=size) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-line")))
            line = str(pane.query_one(".preset-line", Static).content)
            assert "omlx" in line and "key: set" in line
            assert KEY not in _screen_text(app) and "None" not in line
            assert {str(b.label) for b in pane.query(Button)} >= {"Add preset", "Edit", "Remove"}

    async def test_the_pane_says_when_nothing_is_stored(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            pane = await _open_pane(pilot, app)
            status = pane.query_one("#presets-status", Static)
            assert await wait_until(
                pilot, lambda: str(status.content) == msg.SETTINGS_PRESETS_EMPTY
            )

    async def test_the_pane_shows_the_service_text_when_the_read_fails(self):
        app = _ChatApp()
        with mock.patch(
            "lilbee.cli.tui.screens.settings_presets.list_presets",
            side_effect=ValueError("bad store"),
        ):
            async with app.run_test(size=(120, 40)) as pilot:
                pane = await _open_pane(pilot, app)
                status = pane.query_one("#presets-status", Static)
                assert await wait_until(pilot, lambda: str(status.content) == "bad store")

    async def test_add_through_the_dialog(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            _press(app, "#presets-add")
            assert await await_dialog(pilot, app)
            dialog = app.screen
            key_input = dialog.query_one("#preset-key-input", Input)
            assert key_input.password is True
            dialog.query_one("#preset-name-input", Input).value = "omlx"
            dialog.query_one("#preset-url-input", Input).value = LOCAL
            dialog.query_one("#preset-role-chat", Checkbox).value = True
            key_input.value = KEY
            _press(app, "#preset-save")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, PresetDialog))
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-line")))
            assert KEY not in _screen_text(app)
        assert [(e["name"], e["api_key"]) for e in _stored()] == [("omlx", KEY)]

    async def test_the_dialog_shows_the_service_text_and_stays_open(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            await _open_pane(pilot, app)
            _press(app, "#presets-add")
            assert await await_dialog(pilot, app)
            app.screen.query_one("#preset-name-input", Input).value = "omlx"
            app.screen.query_one("#preset-url-input", Input).value = LOCAL
            _press(app, "#preset-save")
            error = app.screen.query_one("#preset-error", Static)
            assert await wait_until(pilot, lambda: bool(str(error.content)))
            assert (
                str(error.content)
                == "A preset needs at least one role: embed, rerank, chat, vision."
            )
            assert not app.screen.query_one("#preset-save", Button).disabled
        assert _stored() == []

    async def test_the_wire_select_shows_with_the_rerank_role(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            await _open_pane(pilot, app)
            _press(app, "#presets-add")
            assert await await_dialog(pilot, app)
            select = app.screen.query_one("#preset-wire-select", Select)
            assert select.display is False
            app.screen.query_one("#preset-role-rerank", Checkbox).value = True
            assert await wait_until(pilot, lambda: select.display is True)

    async def test_cancel_closes_the_dialog_and_changes_nothing(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            await _open_pane(pilot, app)
            _press(app, "#presets-add")
            assert await await_dialog(pilot, app)
            _press(app, "#preset-cancel")
            assert await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        assert _stored() == []

    async def test_edit_sends_only_what_changed_and_keeps_the_key(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-edit")))
            _press(app, ".preset-edit")
            assert await await_dialog(pilot, app)
            assert KEY not in _screen_text(app)
            assert app.screen.query_one("#preset-key-input", Input).value == ""
            app.screen.query_one("#preset-role-vision", Checkbox).value = True
            _press(app, "#preset-save")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, PresetDialog))
        stored = _stored()
        assert (stored[0]["roles"], stored[0]["api_key"]) == (["embed", "chat", "vision"], KEY)

    async def test_edit_with_a_new_url_sends_the_url_and_keeps_the_key(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-edit")))
            _press(app, ".preset-edit")
            assert await await_dialog(pilot, app)
            app.screen.query_one("#preset-url-input", Input).value = f"{LOCAL}/x"
            _press(app, "#preset-save")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, PresetDialog))
        stored = _stored()
        assert (stored[0]["url"], stored[0]["roles"], stored[0]["api_key"]) == (
            f"{LOCAL}/x",
            ["chat", "embed"],
            KEY,
        )

    async def test_edit_with_no_change_shows_the_nothing_to_change_text(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-edit")))
            _press(app, ".preset-edit")
            assert await await_dialog(pilot, app)
            _press(app, "#preset-save")
            error = app.screen.query_one("#preset-error", Static)
            assert await wait_until(
                pilot, lambda: str(error.content).startswith("Nothing to change.")
            )

    async def test_edit_with_a_new_wire_sends_the_wire(self):
        add_preset("rr", LOCAL, roles=["rerank"])
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-edit")))
            _press(app, ".preset-edit")
            assert await await_dialog(pilot, app)
            app.screen.query_one("#preset-wire-select", Select).value = "tei"
            _press(app, "#preset-save")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, PresetDialog))
        assert _stored()[0]["rerank_wire"] == "tei"

    async def test_clear_key_button_clears_the_key_alone(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-edit")))
            _press(app, ".preset-edit")
            assert await await_dialog(pilot, app)
            assert "Key: set" in _screen_text(app)
            _press(app, "#preset-clear-key")
            assert await wait_until(pilot, lambda: not isinstance(app.screen, PresetDialog))
        assert "api_key" not in _stored()[0]

    async def test_edit_dialog_has_no_clear_key_button_without_a_key(self):
        _seed(key=None)
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-edit")))
            _press(app, ".preset-edit")
            assert await await_dialog(pilot, app)
            assert not app.screen.query("#preset-clear-key")
            assert "Key: none" in _screen_text(app)

    async def test_remove_asks_then_removes(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-remove")))
            _press(app, ".preset-remove")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
            await pilot.press("n")
            assert await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
            assert len(_stored()) == 1
            _press(app, ".preset-remove")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
            await pilot.press("y")
            assert await wait_until(pilot, lambda: _stored() == [])
            assert await wait_until(pilot, lambda: not pane.query(".preset-line"))

    async def test_a_failed_remove_notifies_the_service_text(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-remove")))
            with (
                mock.patch(
                    "lilbee.cli.tui.screens.settings_presets.remove_preset",
                    side_effect=ValueError("no"),
                ),
                mock.patch.object(pane, "notify") as notify,
            ):
                _press(app, ".preset-remove")
                assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
                await pilot.press("y")
                assert await wait_until(pilot, lambda: notify.called)
        assert notify.call_args.args[0] == "no"
        assert len(_stored()) == 1

    async def test_a_failed_add_write_shows_the_service_text(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            await _open_pane(pilot, app)
            _press(app, "#presets-add")
            assert await await_dialog(pilot, app)
            with mock.patch(
                "lilbee.cli.tui.screens.settings_presets.add_preset", side_effect=OSError("busy")
            ):
                _press(app, "#preset-save")
                error = app.screen.query_one("#preset-error", Static)
                assert await wait_until(pilot, lambda: bool(str(error.content)))

    async def test_row_buttons_name_their_preset(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 50)) as pilot:
            pane = await _open_pane(pilot, app)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-edit")))
            names = {b.name for b in pane.query(".preset-edit, .preset-remove")}
            assert names == {"omlx"}


async def _open_settings(pilot, app) -> TabbedContent:
    """The Settings tab strip with the first group active, opened with no click."""
    await await_chat(app, pilot)
    app.push_screen(SettingsScreen())
    assert await wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
    assert await wait_until(pilot, lambda: bool(app.screen.query("#settings-tabs")))
    return app.screen.query_one("#settings-tabs", TabbedContent)


class TestPaneKeyboardAccess:
    async def test_greater_than_walks_every_group_then_reaches_presets_then_wraps(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            tabs = await _open_settings(pilot, app)
            groups = list(app.screen._pane_groups)
            assert len(groups) > 1 and tabs.active == groups[0]
            seen = [tabs.active]
            for _ in groups:
                await pilot.press(">")
                assert await wait_until(pilot, lambda: tabs.active != seen[-1])
                seen.append(tabs.active)
            assert seen == [*groups, PRESETS_PANE_ID]
            await pilot.press(">")
            assert await wait_until(pilot, lambda: tabs.active == groups[0])

    async def test_less_than_from_the_first_group_reaches_presets(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            tabs = await _open_settings(pilot, app)
            await pilot.press("<")
            assert await wait_until(pilot, lambda: tabs.active == PRESETS_PANE_ID)

    async def test_tab_past_the_last_group_field_reaches_presets_and_back(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            tabs = await _open_settings(pilot, app)
            screen = app.screen
            last_group = list(screen._pane_groups)[-1]
            tabs.active = last_group
            assert await wait_until(pilot, lambda: bool(screen.query(f"#{last_group}-body > *")))
            screen._focus_pane_edge(last_group, -1)
            await pilot.press("tab")
            assert await wait_until(pilot, lambda: tabs.active == PRESETS_PANE_ID)
            assert await wait_until(
                pilot, lambda: getattr(app.focused, "id", None) == "presets-add"
            )
            await pilot.press("shift+tab")
            assert await wait_until(pilot, lambda: tabs.active == last_group)

    async def test_tab_inside_presets_reaches_a_row_button_by_keys_alone(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            tabs = await _open_settings(pilot, app)
            await pilot.press("<")
            assert await wait_until(pilot, lambda: tabs.active == PRESETS_PANE_ID)
            pane = app.screen.query_one(PresetsPane)
            assert await wait_until(pilot, lambda: bool(pane.query(".preset-edit")))
            pane.query_one("#presets-add", Button).focus()
            await pilot.press("tab")
            assert await wait_until(
                pilot, lambda: "preset-edit" in getattr(app.focused, "classes", ())
            )
            await pilot.press("enter")
            assert await await_dialog(pilot, app)


SECRET = "SENTINEL-slash-key-5d7e"


def _text_under(root) -> str:
    """Every file below *root* as one text, so a leak into any store shows up."""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in root.rglob("*")
        if path.is_file()
    )


class TestSlashKeyStaysOutOfEveryStore:
    @pytest.mark.parametrize(
        "typed",
        [
            f"/preset add x1 {LOCAL} --roles chat --key {SECRET}",
            f"/preset add x1 {LOCAL} --roles chat --key={SECRET}",
            f"/preset edit omlx --key {SECRET}",
            f"/preset\tadd x1 {LOCAL} --roles chat --key {SECRET}",
            f"/preset    add x1 {LOCAL} --roles chat --key {SECRET}",
            f"  /preset add x1 {LOCAL} --roles chat --key {SECRET}",
        ],
    )
    async def test_a_typed_key_reaches_no_history_log_or_screen(self, typed, caplog):
        _seed()
        caplog.set_level("DEBUG")
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            chat._enter_insert_mode()
            chat._input_history.append("earlier line")
            chat._chat_input.value = typed
            with (
                mock.patch.object(chat, "notify") as notify,
                mock.patch.object(chat, "_run_preset_command") as worker,
            ):
                chat._submit_draft(chat._chat_input, typed)
                assert await wait_until(pilot, lambda: notify.called)
            assert not worker.called
            assert notify.call_args.args[0] == msg.PRESET_KEY_IN_DIALOG
            assert SECRET not in repr(notify.mock_calls)
            chat.action_history_prev()
            assert chat._chat_input.value == "earlier line"
            assert SECRET not in repr(chat._input_history)
            assert SECRET not in repr(chat._history)
            on_disk = _text_under(cfg.data_root)
            assert KEY in on_disk
            assert SECRET not in on_disk
            assert SECRET not in caplog.text
            assert SECRET not in _screen_text(app)
        assert [e["api_key"] for e in _stored()] == [KEY]

    async def test_an_unknown_key_flag_stays_out_of_the_input_history(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            chat._enter_insert_mode()
            chat._input_history.append("earlier line")
            typed = f"/preset add x1 {LOCAL} --roles chat --api-key {SECRET}"
            with mock.patch.object(chat, "_run_preset_command"):
                chat._submit_draft(chat._chat_input, typed)
            assert chat._input_history == ["earlier line"]
            assert SECRET not in repr(chat._history)

    async def test_a_plain_preset_line_stays_out_of_the_input_history_too(self):
        _seed()
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            chat._enter_insert_mode()
            chat._input_history.append("earlier line")
            with mock.patch.object(chat, "_run_preset_command") as worker:
                chat._submit_draft(chat._chat_input, "/preset list")
            assert worker.called
            assert chat._input_history == ["earlier line"]


class TestUpArrowReturnsOnlyWhatIsSafeToRecall:
    """Press Up after a submit and read what the input holds."""

    PASSWORD = "SENTINEL-crawl-password-5e1d"
    TOKEN = "hf_SENTINEL-login-token-77aa"

    @staticmethod
    async def _recalled(typed: str) -> str:
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            chat._enter_insert_mode()
            chat._input_history.append("earlier line")
            with (
                mock.patch.object(chat, "notify"),
                mock.patch.object(chat, "_run_preset_command"),
                mock.patch.object(chat, "_run_hf_login"),
                mock.patch.object(chat, "_send_message"),
            ):
                chat._submit_draft(chat._chat_input, typed)
            chat._chat_input.focus()
            await pilot.press("up")
            assert await wait_until(pilot, lambda: bool(chat._chat_input.value))
            return chat._chat_input.value

    @pytest.mark.parametrize(
        "typed",
        [
            f"/login {TOKEN}",
            f"/LOGIN   {TOKEN}",
            f"/set crawl_password {PASSWORD}",
            f"/set  crawl_password\t{PASSWORD}",
            f"/preset add x1 {LOCAL} --roles chat --key {PASSWORD}",
        ],
    )
    async def test_a_line_with_a_secret_is_not_recalled(self, typed):
        assert await self._recalled(typed) == "earlier line"

    @pytest.mark.parametrize(
        "typed",
        ["/login", "/set chunk_size 512", "/version", "hello there"],
    )
    async def test_a_line_without_a_secret_is_recalled(self, typed):
        assert await self._recalled(typed) == typed

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("/login tok", True),
            ("/login", False),
            ("/set crawl_password x", True),
            ("/set crawl_password", True),
            ("/set chunk_size 512", False),
            ("/set nonsense x", False),
            ("/set", False),
            ("/preset list", True),
            ("/version", False),
            ("", False),
        ],
    )
    def test_the_rule_names_each_secret_line(self, text, expected):
        from lilbee.cli.tui.secret_lines import carries_secret

        assert carries_secret(text) is expected


class TestEngineHandler:
    async def test_the_slash_command_notifies_the_roles_through_the_service(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            with mock.patch.object(chat, "notify") as notify:
                chat._cmd_engine("roles")
                assert await wait_until(pilot, lambda: notify.called)
        assert "chat: " in notify.call_args.args[0]
        assert notify.call_args.kwargs == {"severity": "information"}

    async def test_a_refusal_notifies_as_an_error(self):
        app = _ChatApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            with mock.patch.object(chat, "notify") as notify:
                chat._cmd_engine("prefer nonesuch")
                assert await wait_until(pilot, lambda: notify.called)
        assert "No engine is named 'nonesuch'" in notify.call_args.args[0]
        assert notify.call_args.kwargs == {"severity": "error"}

    def test_engine_is_registered_and_not_allowed_mid_answer(self):
        assert build_dispatch_dict()["/engine"] == "_cmd_engine"
        assert runs_while_streaming("/engine") is False
