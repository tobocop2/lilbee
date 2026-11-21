"""The TUI shows the engine of each role, lets the user change it, and shows the notices."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from rich.console import Console
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import KeyPanel, OptionList, Static
from textual.widgets._key_panel import BindingsTable

from lilbee.app import engine_service
from lilbee.catalog.models import DownloadProgress
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_commands import (
    announce,
    engine_line,
    install_name,
    run_engine_command,
)
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.engine_picker import EnginePickerModal
from lilbee.cli.tui.screens.model_info import ModelInfoModal
from lilbee.cli.tui.task_queue import TaskStatus
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.model_bar import (
    ChatModePill,
    EngineLine,
    ModelBar,
    ModelPickerButton,
)
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.data.store.canary import CanaryState
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.roles import WorkerRole
from lilbee.retrieval import canary_probe
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost, await_chat
from tests.conformance.fake_adapter import BundledFakeAdapter

GGUF_REF = "org/repo/file.gguf"
EMBED_REF = "org/repo/embed.gguf"


class _Gguf(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})
    served_roles = frozenset(WorkerRole)


class _Openvino(BundledFakeAdapter):
    name = "openvino"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})


class _Mlx(BundledFakeAdapter):
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
    fakes = (
        (EngineName.LLAMA, _Gguf()),
        (EngineName.OPENVINO, _Openvino()),
        (EngineName.MLX, _Mlx()),
    )
    # The picker also offers an engine with no build, so no real adapter may stay beside the fakes.
    for name in set(choice.ADAPTERS) - {name for name, _ in fakes}:
        monkeypatch.delitem(choice.ADAPTERS, name)  # type: ignore[attr-defined]
    for name, adapter in fakes:
        monkeypatch.setitem(choice.ADAPTERS, name, adapter)  # type: ignore[arg-type]
    monkeypatch.setattr(engine_service, "pack_backend", lambda _host: "metal")
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
        assert text == "engine: chat ■ llama.cpp  embed ■ llama.cpp"
        assert "None" not in text
        assert app.query_one("#model-bar-engine-chips").region.width <= size[0]


async def test_the_chips_follow_the_stored_choice_without_a_restart() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        cfg.engine_choices = json.dumps({GGUF_REF: "openvino"})
        app.query_one(ModelBar).refresh_engines()
        assert _text(app, "#model-bar-engine-chips") == "engine: chat ■ openvino  embed ■ llama.cpp"


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_the_key_e_opens_the_picker_and_saves_the_pick(size: tuple[int, int]) -> None:
    app = _BarApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _open_picker(app, pilot)
        options = app.screen.query_one(OptionList)
        labels = [str(options.get_option_at_index(i).prompt) for i in range(options.option_count)]
        assert labels == [
            "auto (now llama.cpp)",
            "■ llama.cpp (now)  bundled · metal",
            "■ openvino  bundled",
        ]
        await pilot.press("down", "down", "enter")
        assert await wait_until(pilot, lambda: not isinstance(app.screen, EnginePickerModal))
        stored = lambda: settings.load(cfg.data_root).get("engine_choices")  # noqa: E731
        assert await wait_until(pilot, lambda: stored() is not None)
        assert json.loads(str(stored())) == {GGUF_REF: "openvino"}
        assert await wait_until(
            pilot,
            lambda: _text(app, "#model-bar-engine-chips").startswith("engine: chat ■ openvino"),
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
        assert _text(app, "#model-bar-engine-chips").startswith("engine: chat ■ llama.cpp")


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
        assert _text(app.screen, "#info-engine").splitlines()[1].startswith("■ openvino")


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
        assert "No installed engine loads the mlx format" in _text(app.screen, "#info-engine")


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
async def test_key_e_on_a_focused_pill_opens_the_picker_in_insert_mode(
    size: tuple[int, int],
) -> None:
    app = _ChatApp()
    async with app.run_test(size=size) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        assert chat._insert_mode is True
        chat.query_one("#model-pick-chat", ModelPickerButton).focus()
        await pilot.press("e")
        assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
        assert chat._chat_input.value == ""


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_a_key_the_focused_pill_does_not_bind_still_types_into_the_prompt(
    size: tuple[int, int],
) -> None:
    app = _ChatApp()
    async with app.run_test(size=size) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        chat.query_one("#model-pick-chat", ModelPickerButton).focus()
        await pilot.press("x")
        assert await wait_until(pilot, lambda: chat._chat_input.value == "x")
        assert chat._chat_input.has_focus
        assert not isinstance(app.screen, EnginePickerModal)


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_key_e_off_the_strip_still_types_into_the_prompt(size: tuple[int, int]) -> None:
    app = _ChatApp()
    async with app.run_test(size=size) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        chat._chat_log.focus()
        await pilot.press("e")
        assert await wait_until(pilot, lambda: chat._chat_input.value == "e")
        assert not isinstance(app.screen, EnginePickerModal)


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_a_key_a_fleet_toggle_binds_still_types_into_the_prompt(
    monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    """Only the model strip keeps its keys: a Fleet toggle's space still goes to the prompt."""
    from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
    from lilbee.cli.tui.widgets import fleet_body, gpu_fleet_panel
    from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

    view = PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "NVIDIA A40", 44 * 1024**3, 44 * 1024**3),),
        roles=(RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (0,), None, 1, "llama.cpp"),),
        unplaceable=(),
        manual=False,
        spec_json=None,
    )
    monkeypatch.setattr(fleet_body, "get_placement", lambda: view)
    monkeypatch.setattr(gpu_fleet_panel, "probe_gpu_stats", lambda devices: {})
    app = _ChatApp()
    async with app.run_test(size=size) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        await pilot.press("ctrl+g")
        assert await wait_until(
            pilot, lambda: bool(chat.query(FleetDrawer) and chat.query(".dev-toggle"))
        )
        toggle = chat.query(".dev-toggle").first()
        toggle.focus()
        assert await wait_until(pilot, lambda: chat.focused is toggle)
        assert "space" in chat.active_bindings
        await pilot.press("space")
        assert await wait_until(pilot, lambda: chat._chat_input.value == " ")
        assert chat._chat_input.has_focus


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_space_on_a_focused_mode_pill_picks_the_mode_and_does_not_type(
    monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    monkeypatch.setattr(cfg, "chat_mode", "search")
    app = _ChatApp()
    async with app.run_test(size=size) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        assert chat._insert_mode is True
        pill = chat.query_one("#chat-mode-chat", ChatModePill)
        pill.focus()
        assert await wait_until(pilot, lambda: chat.focused is pill)
        assert "space" in chat.active_bindings
        await pilot.press("space")
        assert await wait_until(pilot, lambda: cfg.chat_mode == "chat")
        assert chat._chat_input.value == ""
        assert chat.focused is pill


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_the_slash_command_repaints_the_chips_and_clears_the_notice(
    size: tuple[int, int],
) -> None:
    cfg.engine_choices = json.dumps({GGUF_REF: "mlx"})
    app = _ChatApp()
    async with app.run_test(size=size) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        assert _text(chat, "#model-bar-engine-chips") == (
            "engine: chat ▲ needs an engine  embed ■ llama.cpp"
        )
        assert "cannot load the gguf format" in _text(chat, "#model-bar-engine-notice")
        chat._cmd_engine(f"use chat {GGUF_REF} openvino")
        assert await wait_until(
            pilot,
            lambda: (
                _text(chat, "#model-bar-engine-chips")
                == "engine: chat ■ openvino  embed ■ llama.cpp"
            ),
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
            lambda: (
                _text(chat, "#model-bar-engine-chips")
                == "engine: chat ■ openvino  embed ■ openvino"
            ),
        )


async def test_a_refused_slash_command_leaves_the_chips_alone() -> None:
    app = _ChatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        chat._cmd_engine(f"use chat {GGUF_REF} mlx")
        assert await wait_until(pilot, lambda: any("cannot load" in t for t in _toasts(app)))
        assert (
            _text(chat, "#model-bar-engine-chips") == "engine: chat ■ llama.cpp  embed ■ llama.cpp"
        )


async def test_the_slash_install_shows_its_bytes_in_the_task_bar_and_each_retry_as_a_toast() -> (
    None
):
    seen: list[tuple[str, str, float, str]] = []
    app = _ChatApp()

    def install(
        _name: str, _wheel: object = None, *, on_update: Any, on_retry: Any, cancel: Any
    ) -> engine_service.InstalledBuild:
        assert cancel.is_set() is False
        on_retry("Transfer failed: timed out. Attempt 1 of 3 failed; trying again.")
        on_update(DownloadProgress(percent=40.0, detail="4/10 MB", is_cache_hit=False))
        seen.extend(
            (t.name, t.task_type, t.progress, t.detail) for t in app.task_bar.queue.active_tasks
        )
        return engine_service.InstalledBuild("openvino", "1.0", "cpu", "")

    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        with mock.patch.object(engine_service, "install", side_effect=install):
            chat._cmd_engine("install openvino")
            assert await wait_until(pilot, lambda: any("Installed" in t for t in _toasts(app)))
        assert seen == [("Install engine openvino", "engine", 40.0, "openvino: 4/10 MB")]
        assert any("Attempt 1 of 3 failed; trying again." in t for t in _toasts(app))


async def test_a_refused_slash_install_marks_its_task_failed() -> None:
    app = _ChatApp()

    def install(_name: str, _wheel: object = None, **_kw: Any) -> engine_service.InstalledBuild:
        raise engine_service.EngineRefusedError("Engine openvino cannot run on this host.")

    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        with mock.patch.object(engine_service, "install", side_effect=install):
            chat._cmd_engine("install openvino")
            assert await wait_until(
                pilot,
                lambda: [t.status for t in app.task_bar.queue.history] == [TaskStatus.FAILED],
            )
        assert any("cannot run on this host" in t for t in _toasts(app))


def test_only_an_install_with_an_engine_name_is_an_install_command() -> None:
    assert install_name("install openvino") == "openvino"
    assert install_name("install openvino --yes") == "openvino"
    assert [install_name(a) for a in ("install", "roles", "prefer openvino", "install 'x")] == [
        None,
        None,
        None,
        None,
    ]


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
    names = [option.name for option in engine_service.engine_options(GGUF_REF)]
    assert names == ["auto", "llama.cpp", "openvino"]
    names = [option.name for option in engine_service.engine_options("org/repo#mlx")]
    assert names == ["auto", "mlx"]


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


async def test_the_confirm_opens_on_cancel_so_one_enter_writes_nothing_and_yes_then_enter_writes(
    embed_library: None,
) -> None:
    app = _ChatApp()
    async with app.run_test(size=(80, 24)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        await _pick_openvino_for_embed(app, pilot, chat)
        assert app.focused is not None
        assert app.focused.id == "confirm-no"
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: not isinstance(app.screen, ConfirmDialog))
        assert await wait_until(pilot, lambda: "The engine is unchanged." in _toasts(app))
        assert _stored_choices() is None
        assert cfg.engine_choices is None

        await _pick_openvino_for_embed(app, pilot, chat)
        await pilot.press("left")
        assert await wait_until(
            pilot, lambda: app.focused is not None and app.focused.id == "confirm-yes"
        )
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: _stored_choices() is not None)
        assert json.loads(str(_stored_choices())) == {EMBED_REF: "openvino"}


_DOWNLOAD_BYTES = 180_000_000


def _offer_openvino_for_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """The picker offers openvino as an engine with no build, 180 MB to download."""
    monkeypatch.setattr(
        engine_service,
        "engine_options",
        lambda _model: [
            engine_service.EngineOption("auto", auto_now="llama.cpp"),
            engine_service.EngineOption("llama.cpp", bundled=True, backend="metal"),
            engine_service.EngineOption(
                "openvino", installed=False, backend="cpu", download_bytes=_DOWNLOAD_BYTES
            ),
        ],
    )


def _engine_tasks(app: LilbeeAppHost) -> list[Any]:
    queue = app.task_bar.queue
    return [t for t in [*queue.active_tasks, *queue.history] if t.task_type == "engine"]


async def _pick_the_download(app: LilbeeAppHost, pilot) -> None:
    await _open_picker(app, pilot)
    await pilot.press("down", "down", "enter")
    assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))


async def test_the_picker_rows_show_state_and_backend_and_the_size_of_a_missing_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offer_openvino_for_download(monkeypatch)
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_picker(app, pilot)
        options = app.screen.query_one(OptionList)
        labels = [str(options.get_option_at_index(i).prompt) for i in range(options.option_count)]
        assert labels == [
            "auto (now llama.cpp)",
            "■ llama.cpp (now)  bundled · metal",
            "■ openvino  download 180 MB · cpu",
        ]
        assert "The engine is the program that runs the model file." in _text(
            app.screen, "#engine-picker-note"
        )


async def test_a_missing_engine_asks_to_download_then_installs_as_a_task_then_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    _offer_openvino_for_download(monkeypatch)
    release = threading.Event()
    installed: list[str] = []

    def _install(name: str, **kwargs: Any) -> engine_service.InstalledBuild:
        kwargs["on_update"](DownloadProgress(50.0, "90 MB of 180 MB", False))
        assert release.wait(timeout=10)
        installed.append(name)
        return engine_service.InstalledBuild(name, "1", "cpu")

    monkeypatch.setattr(engine_service, "install", _install)
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _pick_the_download(app, pilot)
        assert _text(app.screen, "#confirm-message") == (
            "Download openvino (180 MB) and use it for this model?"
        )
        await pilot.press("y")
        assert await wait_until(pilot, lambda: bool(_engine_tasks(app)))
        task = _engine_tasks(app)[0]
        assert task.name == "Install engine openvino"
        assert await wait_until(pilot, lambda: task.progress > 0)
        assert cfg.engine_choices is None
        assert _text(app, "#model-bar-engine-chips").startswith("engine: chat ■ llama.cpp")
        release.set()
        assert await wait_until(pilot, lambda: cfg.engine_choices is not None)
        assert installed == ["openvino"]
        assert json.loads(str(cfg.engine_choices)) == {GGUF_REF: "openvino"}
        assert task.status is TaskStatus.DONE


async def test_declining_the_download_installs_nothing_and_keeps_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offer_openvino_for_download(monkeypatch)
    monkeypatch.setattr(engine_service, "install", mock.Mock())
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _pick_the_download(app, pilot)
        await pilot.press("n")
        assert await wait_until(pilot, lambda: not isinstance(app.screen, ConfirmDialog))
        await pilot.pause()
        assert _engine_tasks(app) == []
        engine_service.install.assert_not_called()  # type: ignore[attr-defined]
        assert cfg.engine_choices is None


async def test_a_failed_install_fails_the_task_and_keeps_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offer_openvino_for_download(monkeypatch)

    def _refused(*_a: object, **_k: object) -> None:
        raise engine_service.EngineInstallRefusedError("The download of openvino failed.")

    monkeypatch.setattr(engine_service, "install", _refused)
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _pick_the_download(app, pilot)
        await pilot.press("y")
        assert await wait_until(
            pilot, lambda: any(t.status is TaskStatus.FAILED for t in _engine_tasks(app))
        )
        assert _engine_tasks(app)[0].detail == "The download of openvino failed."
        assert cfg.engine_choices is None


_DOCS_URL = "https://example.org/install-lemonade"


def _offer_engines_with_no_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """openvino has no published pack; lemonade is installed by hand."""
    monkeypatch.setattr(
        engine_service,
        "engine_options",
        lambda _model: [
            engine_service.EngineOption("auto", auto_now="llama.cpp"),
            engine_service.EngineOption("llama.cpp", bundled=True, backend="metal"),
            engine_service.EngineOption(
                "openvino", installed=False, route=engine_service.InstallRoute.NO_PACK
            ),
            engine_service.EngineOption(
                "lemonade",
                installed=False,
                route=engine_service.InstallRoute.BY_HAND,
                docs_url=_DOCS_URL,
            ),
        ],
    )


async def test_an_engine_with_no_download_is_listed_with_its_reason_and_the_docs_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offer_engines_with_no_download(monkeypatch)
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_picker(app, pilot)
        options = app.screen.query_one(OptionList)
        labels = [str(options.get_option_at_index(i).prompt) for i in range(options.option_count)]
        assert labels[2:] == [
            "■ openvino  no download yet",
            f"■ lemonade  install it yourself · {_DOCS_URL}",
        ]


async def test_choosing_an_engine_with_no_download_asks_nothing_and_starts_no_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offer_engines_with_no_download(monkeypatch)
    monkeypatch.setattr(engine_service, "install", mock.Mock())
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_picker(app, pilot)
        options = app.screen.query_one(OptionList)
        assert [options.get_option_at_index(i).disabled for i in range(4)] == [
            False,
            False,
            True,
            True,
        ]
        await pilot.press("down", "down", "down", "enter")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmDialog)
        assert _engine_tasks(app) == []
        engine_service.install.assert_not_called()  # type: ignore[attr-defined]


def _views(*pairs: tuple[WorkerRole, str, str]) -> list[engine_service.RoleView]:
    return [
        engine_service.RoleView(role=r, model=m, format="gguf", engine=e, notice="")
        for r, m, e in pairs
    ]


def test_the_engine_line_colors_each_mark_and_warns_for_a_role_with_no_engine() -> None:
    line = engine_line(_views((WorkerRole.CHAT, "m", ""), (WorkerRole.EMBED, "e", "llama.cpp")))
    styles = {str(span.style) for span in line.spans}
    assert "$warning" in styles
    assert "$engine-llama-cpp" in styles
    assert line.plain == "engine: chat ▲ needs an engine  embed ■ llama.cpp"


def test_the_engine_line_is_empty_when_no_role_has_a_model() -> None:
    assert engine_line(_views((WorkerRole.CHAT, " ", "llama.cpp"))).plain == ""


async def test_the_e_key_shows_in_the_footer_for_a_role_button() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#model-pick-chat", ModelPickerButton).focus()
        await pilot.pause()
        binding = app.screen.active_bindings["e"].binding
        assert binding.show
        assert binding.description == "Pick engine"


async def test_a_click_on_a_role_in_the_engine_line_opens_the_picker_of_that_role() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        line = app.query_one(EngineLine)
        assert str(line.render()).index("embed") == 26
        await pilot.click(EngineLine, offset=(28, 0))
        assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
        assert _text(app.screen, "#engine-picker-title") == "Engine for the embed model"


@pytest.mark.parametrize("role", ["chat", "embed"])
async def test_a_click_on_a_role_opens_one_picker_for_that_role_only(role: str) -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        offset = str(app.query_one(EngineLine).render()).index(role) + 1
        await pilot.click(EngineLine, offset=(offset, 0))
        assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
        await pilot.pause()
        assert _text(app.screen, "#engine-picker-title") == f"Engine for the {role} model"
        assert sum(isinstance(screen, EnginePickerModal) for screen in app.screen_stack) == 1


# --- model info: the Engine section and the e key

_DOWNLOAD = engine_service.EngineOption("openvino", installed=False, download_bytes=180_000_000)


async def _open_info(app: LilbeeAppHost, pilot) -> ModelInfoModal:
    from lilbee.cli.tui.screens.model_picker import _browse_catalog_row

    row = _browse_catalog_row()
    row.ref = GGUF_REF  # type: ignore[misc]
    row.installed = True  # type: ignore[misc]
    modal = ModelInfoModal(row)
    await app.push_screen(modal)
    await pilot.pause()
    return modal


def _stored() -> dict[str, str]:
    return json.loads(settings.load(cfg.data_root).get("engine_choices") or "{}")


async def test_model_info_puts_the_engine_section_after_status_and_keeps_every_field() -> None:
    from textual.widgets import Markdown

    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = await _open_info(app, pilot)
        body = modal.query_one("#info-body").children
        assert [type(w).__name__ for w in body] == ["Markdown", "Static", "Markdown"]
        head, tail = (modal.query_one(i, Markdown).source for i in ("#info-md", "#info-md-tail"))
        assert head.rstrip().endswith("**Status:** installed")
        assert "huggingface.co/" in tail
        assert _text(modal, "#info-engine") == (
            "Engine\n■ llama.cpp  bundled · metal\nAlso loads it: ■ openvino bundled"
        )
        assert msg.MODEL_INFO_HINT.startswith("e to change the engine")
        assert "e to change" not in _text(modal, "#info-engine")


async def test_key_e_in_model_info_saves_the_pick_for_that_model() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = await _open_info(app, pilot)
        await pilot.press("e")
        assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
        await pilot.press("down", "down", "enter")
        assert await wait_until(pilot, lambda: _stored() == {GGUF_REF: "openvino"})
        assert await wait_until(
            pilot, lambda: _text(modal, "#info-engine").splitlines()[1].startswith("■ openvino")
        )
        assert any(t == f"Model {GGUF_REF} uses engine openvino." for t in _toasts(app))


async def test_an_engine_that_is_not_installed_is_saved_only_after_the_confirm_and_install() -> (
    None
):
    installed: list[str] = []

    def install(name: str, _wheel: object = None, **_kw: Any) -> engine_service.InstalledBuild:
        assert _stored() == {}
        installed.append(name)
        return engine_service.InstalledBuild(name, "1.0", "cpu", "")

    options = [
        engine_service.EngineOption("auto", auto_now="llama.cpp"),
        engine_service.EngineOption("llama.cpp", bundled=True, backend="metal"),
        _DOWNLOAD,
    ]
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = await _open_info(app, pilot)
        with (
            mock.patch.object(engine_service, "engine_options", return_value=options),
            mock.patch.object(engine_service, "install", side_effect=install),
        ):
            await pilot.press("e")
            assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
            await pilot.press("down", "down", "enter")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
            assert installed == []
            assert _stored() == {}
            await pilot.press("y")
            assert await wait_until(pilot, lambda: _stored() == {GGUF_REF: "openvino"})
        assert installed == ["openvino"]
        assert modal.is_attached


async def test_declining_the_download_saves_nothing() -> None:
    options = [
        engine_service.EngineOption("auto", auto_now="llama.cpp"),
        engine_service.EngineOption("llama.cpp", bundled=True, backend="metal"),
        _DOWNLOAD,
    ]
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_info(app, pilot)
        with (
            mock.patch.object(engine_service, "engine_options", return_value=options),
            mock.patch.object(engine_service, "install") as install,
        ):
            await pilot.press("e")
            assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
            await pilot.press("down", "down", "enter")
            assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
            await pilot.press("escape")
            await pilot.pause()
        install.assert_not_called()
        assert _stored() == {}


def test_choosing_an_engine_that_cannot_load_the_model_returns_the_refusal_text() -> None:
    from lilbee.cli.tui.engine_commands import choose_engine_for_model

    result = choose_engine_for_model(GGUF_REF, "mlx")
    assert not result.ok
    assert result.text.startswith("Engine mlx cannot load the gguf format")
    assert _stored() == {}


async def test_model_info_says_so_when_no_other_engine_can_load_the_model() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = await _open_info(app, pilot)
        only_auto = [engine_service.EngineOption("auto", auto_now="llama.cpp")]
        with mock.patch.object(engine_service, "engine_options", return_value=only_auto):
            await pilot.press("e")
            await pilot.pause()
        assert app.screen is modal
        assert msg.MODEL_INFO_NO_ENGINE_CHOICE in _toasts(app)


async def test_escape_in_the_model_info_picker_saves_nothing() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = await _open_info(app, pilot)
        await pilot.press("e")
        assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
        await pilot.press("escape")
        assert await wait_until(pilot, lambda: app.screen is modal)
        assert _stored() == {}


async def test_the_picker_marks_no_engine_as_current_when_none_can_load_the_model() -> None:
    options = [engine_service.EngineOption("auto"), _DOWNLOAD]
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        with (
            mock.patch(
                "lilbee.cli.tui.screens.model_info.engine_for_ref",
                side_effect=choice.NoEngineError("No installed engine loads the gguf format."),
            ),
            mock.patch.object(engine_service, "engine_options", return_value=options),
        ):
            modal = await _open_info(app, pilot)
            assert _text(modal, "#info-engine").endswith(
                "No installed engine loads the gguf format."
            )
            await pilot.press("e")
            assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
        picked = app.screen.query_one(OptionList)
        labels = [str(picked.get_option_at_index(i).prompt) for i in range(picked.option_count)]
        assert not any("(now)" in label for label in labels)


async def test_a_repaint_after_model_info_closed_does_nothing() -> None:
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        modal = await _open_info(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        modal._repaint_engine()


def _confirming_use(calls: list[bool]):
    def use(model: str, name: str, confirmed: bool = False) -> str:
        calls.append(confirmed)
        if not confirmed:
            raise engine_service.EmbedConfirmRequiredError("docs", 3)
        return name

    return use


async def _pick_openvino_in_model_info(app: LilbeeAppHost, pilot) -> None:
    from lilbee.cli.tui.widgets.confirm_dialog import CancelFirstConfirmDialog

    await _open_info(app, pilot)
    await pilot.press("e")
    assert await wait_until(pilot, lambda: isinstance(app.screen, EnginePickerModal))
    await pilot.press("down", "down", "enter")
    assert await wait_until(pilot, lambda: isinstance(app.screen, CancelFirstConfirmDialog))


async def test_a_change_that_can_orphan_the_index_vectors_is_saved_only_after_a_yes() -> None:
    calls: list[bool] = []
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        with mock.patch.object(engine_service, "use", side_effect=_confirming_use(calls)):
            await _pick_openvino_in_model_info(app, pilot)
            assert calls == [False]
            await pilot.press("y")
            assert await wait_until(pilot, lambda: calls == [False, True])
        assert await wait_until(
            pilot, lambda: any(t.endswith("uses engine openvino.") for t in _toasts(app))
        )


async def test_a_no_to_the_index_vector_question_keeps_the_engine() -> None:
    calls: list[bool] = []
    app = _BarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        with mock.patch.object(engine_service, "use", side_effect=_confirming_use(calls)):
            await _pick_openvino_in_model_info(app, pilot)
            await pilot.press("n")
            assert await wait_until(pilot, lambda: msg.ENGINE_EMBED_CANCELLED in _toasts(app))
        assert calls == [False]
