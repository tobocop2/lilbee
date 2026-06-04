"""Tests for the chat screen's ``/remember`` and ``/memories`` handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.app import ComposeResult

from lilbee.app.services import set_services
from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    cfg.memory_enabled = True
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def mock_svc():
    store = MagicMock()
    store.add_memory.return_value = "id123"
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    embedder.embedding_available.return_value = True
    services = make_mock_services(store=store, embedder=embedder)
    set_services(services)
    yield services
    set_services(None)


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    from lilbee.cli.tui.widgets.model_bar import ModelBar

    with (
        patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
        patch.object(ModelBar, "_scan_models"),
    ):
        yield


class ChatTestApp(LilbeeAppHost):
    CSS = ""

    def __init__(self) -> None:
        super().__init__()
        from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


async def test_cmd_remember_stores_via_worker(mock_svc):
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        set_services(mock_svc)
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_remember("uses rust")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
        mock_svc.store.add_memory.assert_called_once()
        assert mock_notify.call_args[0][0] == msg.CMD_REMEMBER_SUCCESS.format(kind="fact")


async def test_cmd_remember_disabled_notifies(mock_svc):
    cfg.memory_enabled = False
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        set_services(mock_svc)
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_remember("uses rust")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
        mock_svc.store.add_memory.assert_not_called()
        assert mock_notify.call_args.kwargs["severity"] == "warning"


async def test_cmd_memories_pushes_screen():
    from lilbee.cli.tui.screens.memories import MemoriesScreen

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen._cmd_memories("")
        await pilot.pause()
        assert isinstance(app.screen, MemoriesScreen)
