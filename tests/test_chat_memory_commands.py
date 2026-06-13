"""Tests for the chat screen's ``/remember`` and ``/memories`` handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.app import ComposeResult

from lilbee.app.services import set_services
from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from lilbee.providers.base import ChatResult, FinishReason
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


async def test_maybe_extract_skips_when_disabled(mock_svc):
    cfg.memory_auto_extract = False
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch.object(app.screen, "_extract_memories_worker") as worker:
            app.screen._maybe_extract_memories("q", "a")
            await pilot.pause()
            worker.assert_not_called()


async def test_maybe_extract_skips_empty_answer(mock_svc):
    cfg.memory_enabled = True
    cfg.memory_auto_extract = True
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch.object(app.screen, "_extract_memories_worker") as worker:
            app.screen._maybe_extract_memories("q", "")
            await pilot.pause()
            worker.assert_not_called()


def _patch_active_tasks(monkeypatch, queue, tasks):
    """Replace the read-only ``active_tasks`` property for one test."""
    monkeypatch.setattr(type(queue), "active_tasks", property(lambda _self: tasks))


def _fake_active_task(task_type):
    """A real Task the TaskBar can render. A bare stub drifts from the render
    interface (the bar reads name/progress/indeterminate when it repaints
    mid-test), which made these tests flaky."""
    from lilbee.cli.tui.task_queue import Task, TaskStatus

    return Task(
        task_id=f"t-{task_type}",
        name=task_type,
        task_type=task_type,
        fn=lambda: None,
        status=TaskStatus.ACTIVE,
    )


async def test_maybe_extract_skips_while_indexing(mock_svc, monkeypatch):
    from lilbee.cli.tui.task_queue import Task, TaskType

    cfg.memory_enabled = True
    cfg.memory_auto_extract = True
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        # A real Task carries every field the task-bar timer reads when a tick
        # lands mid-test; a bare SimpleNamespace stub raced the timer (bb-5ze).
        task = Task(task_id="t1", name="Indexing", task_type=TaskType.SYNC.value, fn=lambda: None)
        _patch_active_tasks(monkeypatch, app.task_bar.queue, [task])
        with patch.object(app.screen, "_extract_memories_worker") as worker:
            app.screen._maybe_extract_memories("q", "a")
            await pilot.pause()
            worker.assert_not_called()


async def test_maybe_extract_runs_when_idle(mock_svc):
    cfg.memory_enabled = True
    cfg.memory_auto_extract = True
    mock_svc.provider.chat.return_value = ChatResult(
        text='[{"text": "the user prefers rust", "kind": "fact"}]',
        tool_calls=(),
        finish_reason=FinishReason.STOP,
    )
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        set_services(mock_svc)
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._maybe_extract_memories("I love rust", "Rust is great.")
            await app.screen.workers.wait_for_complete()
            await pilot.pause()
        mock_svc.store.add_memory.assert_called_once()
        assert mock_notify.call_args[0][0] == msg.MEMORY_AUTO_EXTRACTED.format(count=1)


async def test_indexing_active_reflects_queue(mock_svc, monkeypatch):
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        _patch_active_tasks(monkeypatch, app.task_bar.queue, [])
        assert app.screen._indexing_active() is False
        _patch_active_tasks(monkeypatch, app.task_bar.queue, [_fake_active_task("add")])
        assert app.screen._indexing_active() is True
