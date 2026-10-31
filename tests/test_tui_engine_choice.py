"""The TUI shows the engine of each role, lets the user change it, and shows the notices."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from rich.console import Console
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import KeyPanel, OptionList, Static
from textual.widgets._key_panel import BindingsTable

from lilbee.app import engine_service
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_commands import announce, run_engine_command
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.engine_picker import EnginePickerModal
from lilbee.cli.tui.screens.model_info import ModelInfoModal
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.model_bar import ModelBar, ModelPickerButton
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.data.store.canary import CanaryState
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.retrieval import canary_probe
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost, await_chat
from tests.conformance.fake_adapter import FakeAdapter

GGUF_REF = "org/repo/file.gguf"
EMBED_REF = "org/repo/embed.gguf"


class _Gguf(FakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})


class _Openvino(FakeAdapter):
    name = "openvino"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})


class _Mlx(FakeAdapter):
    name = "mlx"
    formats = frozenset({"mlx"})
    architectures = frozenset({"mlx_arch"})


class _BarApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield ModelBar()


@pytest.fixture(autouse=True)
def _engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    cfg.data_root = tmp_path
    cfg.chat_model = GGUF_REF
    cfg.embedding_model = EMBED_REF
    cfg.vision_model = ""
    cfg.reranker_model = ""
    cfg.preferred_engine = EngineSetting.AUTO
    cfg.engine_choices = None
    for name, adapter in (
        (EngineName.LLAMA, _Gguf()),
        (EngineName.OPENVINO, _Openvino()),
        (EngineName.MLX, _Mlx()),
    ):
        monkeypatch.setitem(choice.ADAPTERS, name, adapter)  # type: ignore[arg-type]
    yield
    for field in type(snapshot).model_fields:
        setattr(cfg, field, getattr(snapshot, field))


def _text(root: Widget, selector: str) -> str:
    return str(root.query_one(selector, Static).render())


def _toasts(app: LilbeeAppHost) -> list[str]:
    return [n.message for n in app._notifications]


async def _open_picker(app: LilbeeAppHost, pilot) -> None:
    app.query_one("#model-pick-chat", ModelPickerButton).focus()
    await pilot.press("e")
    assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_the_chips_show_the_engine_of_each_role_that_has_a_model(
    size: tuple[int, int],
) -> None:
    app = _BarApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        text = _text(app, "#model-bar-engine-chips")
        assert text == "chat: llama.cpp  embed: llama.cpp"
        assert "None" not in text
        assert app.query_one("#model-bar-engine-chips").region.width <= size[0]


async def test_the_chips_follow_the_stored_choice_without_a_restart() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        cfg.engine_choices = json.dumps({GGUF_REF: "openvino"})
        app.query_one(ModelBar).refresh_engines()
        assert _text(app, "#model-bar-engine-chips") == "chat: openvino  embed: llama.cpp"


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_the_key_e_opens_the_picker_and_saves_the_pick(size: tuple[int, int]) -> None:
    app = _BarApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _open_picker(app, pilot)
        options = app.screen.query_one(OptionList)
        labels = [str(options.get_option_at_index(i).prompt) for i in range(options.option_count)]
        assert labels == ["auto (let lilbee choose)", "llama.cpp (now)", "openvino"]
        await pilot.press("down", "down", "enter")
        assert await wait_until(pilot, lambda: not isinstance(app.screen, EnginePickerModal))
        stored = lambda: settings.load(cfg.data_root).get("engine_choices")  # noqa: E731
        assert await wait_until(pilot, lambda: stored() is not None)
        assert json.loads(str(stored())) == {GGUF_REF: "openvino"}
        assert await wait_until(
            pilot, lambda: _text(app, "#model-bar-engine-chips").startswith("chat: openvino")
        )
        assert any(t.endswith("uses engine openvino.") for t in _toasts(app))


async def test_a_refused_save_shows_the_refusal_and_keeps_the_live_choice() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_picker(app, pilot)
        cfg.chat_model = "org/repo#mlx"  # the model changes while the picker is open
        await pilot.press("down", "down", "enter")  # openvino cannot load mlx
        assert await wait_until(
            pilot, lambda: any("cannot load the mlx format" in t for t in _toasts(app))
        )
        assert cfg.engine_choices is None
        assert settings.load(cfg.data_root).get("engine_choices") is None


async def test_a_failed_write_shows_the_not_saved_text_and_keeps_the_live_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(settings, "mutate_value", _boom)
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_picker(app, pilot)
        await pilot.press("down", "down", "enter")
        assert await wait_until(
            pilot, lambda: any("The engine choice was not saved." in t for t in _toasts(app))
        )
        assert cfg.engine_choices is None
        assert "read-only" not in " ".join(_toasts(app))
        assert _text(app, "#model-bar-engine-chips").startswith("chat: llama.cpp")


async def test_escape_leaves_the_choice_alone() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_picker(app, pilot)
        await pilot.press("escape")
        assert await wait_until(pilot, lambda: not isinstance(app.screen, EnginePickerModal))
        assert cfg.engine_choices is None


async def test_a_role_with_no_model_says_so_instead_of_opening_the_picker() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#model-pick-vision", ModelPickerButton).focus()
        await pilot.press("e")
        assert await wait_until(
            pilot, lambda: any("No vision model is set." in t for t in _toasts(app))
        )
        assert not isinstance(app.screen, EnginePickerModal)


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_both_notices_show_on_the_model_bar(size: tuple[int, int]) -> None:
    cfg.engine_choices = "{not json"
    app = _BarApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        notice = app.query_one("#model-bar-engine-notice", Static)
        assert notice.has_class("-visible")
        assert "The saved engine choices cannot be read." in _text(app, "#model-bar-engine-notice")
    cfg.engine_choices = json.dumps({GGUF_REF: "mlx"})
    app = _BarApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        shown = _text(app, "#model-bar-engine-notice")
        assert app.query_one("#model-bar-engine-notice", Static).has_class("-visible")
        assert "cannot load the gguf format" in shown


async def test_no_notice_is_shown_when_the_choice_holds() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert not app.query_one("#model-bar-engine-notice", Static).has_class("-visible")


async def test_the_catalog_dialog_names_the_engine_of_the_model() -> None:
    from lilbee.cli.tui.screens.model_picker import _browse_catalog_row

    row = _browse_catalog_row()
    row.ref = GGUF_REF  # type: ignore[misc]
    cfg.engine_choices = json.dumps({GGUF_REF: "openvino"})
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(ModelInfoModal(row))
        await pilot.pause()
        assert "**Engine:** openvino" in app.screen._build_markdown()  # type: ignore[attr-defined]


class _FleetApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.fleet_body import FleetBody

        yield FleetBody()


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_the_fleet_body_shows_the_chips_and_the_fallback_notice(
    size: tuple[int, int],
) -> None:
    cfg.engine_choices = json.dumps({GGUF_REF: "mlx"})
    app = _FleetApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        shown = _text(app, "#placement-engines")
        assert shown.splitlines()[0] == "chat: no engine  embed: llama.cpp"
        assert "Engine mlx for chat cannot load the gguf format" in shown


async def test_the_catalog_dialog_says_when_no_engine_loads_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.cli.tui.screens.model_picker import _browse_catalog_row

    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX)
    row = _browse_catalog_row()
    row.ref = "org/repo#mlx"  # type: ignore[misc]
    app = _BarApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await app.push_screen(ModelInfoModal(row))
        await pilot.pause()
        assert "**Engine:** No installed engine loads the mlx format" in str(
            app.screen._build_markdown()  # type: ignore[attr-defined]
        )


class _ChatApp(LilbeeAppHost):
    """An app that opens the chat screen directly, with no startup gate."""

    def __init__(self) -> None:
        super().__init__()
        from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_the_slash_command_repaints_the_chips_and_clears_the_notice(
    size: tuple[int, int],
) -> None:
    cfg.engine_choices = json.dumps({GGUF_REF: "mlx"})
    app = _ChatApp()
    async with app.run_test(size=size) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        assert _text(chat, "#model-bar-engine-chips") == "chat: no engine  embed: llama.cpp"
        assert "cannot load the gguf format" in _text(chat, "#model-bar-engine-notice")
        chat._cmd_engine(f"use chat {GGUF_REF} openvino")
        assert await wait_until(
            pilot,
            lambda: _text(chat, "#model-bar-engine-chips") == "chat: openvino  embed: llama.cpp",
        )
        assert _text(chat, "#model-bar-engine-notice") == ""
        assert any("uses engine openvino" in t for t in _toasts(app))


async def test_the_slash_prefer_repaints_the_chips() -> None:
    app = _ChatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        chat._cmd_engine("prefer openvino")
        assert await wait_until(
            pilot,
            lambda: _text(chat, "#model-bar-engine-chips") == "chat: openvino  embed: openvino",
        )


async def test_a_refused_slash_command_leaves_the_chips_alone() -> None:
    app = _ChatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        chat._cmd_engine(f"use chat {GGUF_REF} mlx")
        assert await wait_until(pilot, lambda: any("cannot load" in t for t in _toasts(app)))
        assert _text(chat, "#model-bar-engine-chips") == "chat: llama.cpp  embed: llama.cpp"


async def test_the_fleet_body_repaints_after_an_engine_command() -> None:
    app = _FleetApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert _text(app, "#placement-engines") == "chat: llama.cpp  embed: llama.cpp"
        announce(app, run_engine_command(f"use chat {GGUF_REF} openvino"))
        assert await wait_until(
            pilot, lambda: _text(app, "#placement-engines") == "chat: openvino  embed: llama.cpp"
        )


def test_the_picker_offers_only_the_engines_that_load_the_model() -> None:
    assert engine_service.engine_names(GGUF_REF) == ["auto", "llama.cpp", "openvino"]
    assert engine_service.engine_names("org/repo#mlx") == ["auto", "mlx"]


async def test_the_tooltip_and_the_help_panel_name_the_key_e() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        button = app.query_one("#model-pick-chat", ModelPickerButton)
        assert str(button.tooltip).endswith(msg.ENGINE_KEY_HINT)
        button.focus()
        app.action_show_help_panel()
        await pilot.pause()
        panel = app.screen.query_one(KeyPanel)
        console = Console(width=120, file=io.StringIO(), record=True)
        console.print(panel.query_one(BindingsTable).render())
        assert "Pick engine" in console.export_text()


class _Store:
    def source_summary(self, _newest: int) -> SimpleNamespace:
        return SimpleNamespace(document_count=7)

    def canary_state(self) -> CanaryState | None:
        return CanaryState([[1.0]], "an-older-trigger", True)


@pytest.fixture
def embed_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A running library of 7 documents whose baseline differs from the candidate embedder."""
    model_file = tmp_path / "embed.gguf"
    model_file.write_bytes(b"x")
    monkeypatch.setattr(canary_probe, "resolve_model_path", lambda *_: model_file)
    monkeypatch.setattr(canary_probe, "engine_build_id", lambda: "build")
    services = SimpleNamespace(store=_Store(), registry=SimpleNamespace(), provider=mock.Mock())
    monkeypatch.setattr(engine_service, "peek_services", lambda: services)


async def _pick_openvino_for_embed(app: _ChatApp, pilot, chat: ChatScreen) -> None:
    chat.query_one("#model-pick-embed", ModelPickerButton).action_pick_engine()
    assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
    await pilot.press("down", "down", "enter")
    assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))


def _stored_choices() -> object:
    return settings.load(cfg.data_root).get("engine_choices")


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_the_picker_asks_with_the_library_and_count_then_writes_on_yes(
    embed_library: None, size: tuple[int, int]
) -> None:
    app = _ChatApp()
    async with app.run_test(size=size) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        await _pick_openvino_for_embed(app, pilot, chat)
        text = _text(app.screen, "#confirm-message")
        assert f"Library {cfg.data_root.name} holds 7 documents." in text
        assert "None" not in text
        assert _stored_choices() is None
        await pilot.press("y")
        assert await wait_until(pilot, lambda: _stored_choices() is not None)
        assert json.loads(str(_stored_choices())) == {EMBED_REF: "openvino"}
        assert app.focused is chat._chat_input


async def test_cancelling_the_confirm_leaves_the_file_unchanged_and_returns_focus(
    embed_library: None,
) -> None:
    app = _ChatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        await _pick_openvino_for_embed(app, pilot, chat)
        await pilot.press("n")
        assert await wait_until(pilot, lambda: not isinstance(app.screen, ConfirmDialog))
        assert await wait_until(pilot, lambda: "The engine is unchanged." in _toasts(app))
        assert _stored_choices() is None
        assert cfg.engine_choices is None
        assert app.focused is chat._chat_input
