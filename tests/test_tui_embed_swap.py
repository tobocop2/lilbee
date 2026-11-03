"""One embedder swap path: the model bar and every catalog site ask, write, reload and rebuild."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from textual.pilot import Pilot

from conftest import TEST_EMBED_REF
from lilbee.app.services import peek_services, set_services
from lilbee.app.settings import SettingsUpdateResult, apply_settings_update, setting_sources
from lilbee.catalog.types import KeyStatus, ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.catalog_utils import FrontierCatalogRow
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.task_queue import Task, TaskStatus, TaskType
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.model_bar import ModelBar, ModelPickerButton
from lilbee.cli.tui.widgets.model_pick import apply_model_pick
from lilbee.config_meta import REINDEX_FIELDS
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from lilbee.core.config.enums import SettingSource
from lilbee.core.config.resolve import builtin_value
from lilbee.modelhub.model_manager.types import RemoteModel
from lilbee.providers.roles import WorkerRole
from tests._lilbee_app_test_host import LilbeeAppHost, pump_until
from tests._profile_fixtures import (
    gate_releases_at_once,  # noqa: F401 -- autouse fixture, applied by import
    isolated_cfg,  # noqa: F401 -- autouse fixture, applied by import
)
from tests.test_tui_model_bar import _ChatTestApp
from tests.test_tui_widgets import _ModelBarApp

_OLD = TEST_EMBED_REF
_NEW = "ollama/new-embed:latest"
_TAGLESS = "ollama/new-embed"
_DOWNLOADING = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
_DOWNLOADING_LABEL = "Qwen2.5 0.5B"


async def _pick_on_bar(app: LilbeeAppHost, pilot: Pilot, ref: str) -> None:
    app.query_one("#model-pick-embed", ModelPickerButton)._on_picker_dismissed(ref)


async def _catalog(app: LilbeeAppHost, pilot: Pilot) -> CatalogScreen:
    screen = CatalogScreen()
    await app.push_screen(screen)
    await pilot.pause()
    return screen


async def _pick_remote(app: LilbeeAppHost, pilot: Pilot, ref: str) -> None:
    remote = RemoteModel(
        name=ref, task=ModelTask.EMBEDDING, family="", parameter_size="", provider="Ollama"
    )
    row = SimpleNamespace(
        kind=None, variant=None, family=None, catalog_model=None, remote_model=remote, ref=ref
    )
    (await _catalog(app, pilot))._select_row(row)  # type: ignore[arg-type]


async def _pick_frontier(app: LilbeeAppHost, pilot: Pilot, ref: str) -> None:
    row = FrontierCatalogRow(
        name=ref,
        ref=ref,
        task=ModelTask.EMBEDDING,
        provider="Gemini",
        provider_id="gemini",
        key_status=KeyStatus.READY,
    )
    (await _catalog(app, pilot))._select_row(row)


async def _pick_first_download(app: LilbeeAppHost, pilot: Pilot, ref: str) -> None:
    model = SimpleNamespace(task=ModelTask.EMBEDDING, ref=ref)
    (await _catalog(app, pilot))._adopt_first_download(model)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _Site:
    """A place that picks an embedder, with the embedder set before the pick."""

    pick: Callable[[LilbeeAppHost, Pilot, str], Awaitable[None]]
    before: str
    reloads_role: bool
    catalog_toast: str | None = None


_SITES = {
    "model_bar": _Site(_pick_on_bar, _OLD, reloads_role=True),
    "catalog_remote": _Site(
        _pick_remote, _OLD, reloads_role=True, catalog_toast=msg.CATALOG_USING_REMOTE.split("{")[0]
    ),
    "catalog_frontier": _Site(
        _pick_frontier,
        _OLD,
        reloads_role=True,
        catalog_toast=msg.CATALOG_USING_FRONTIER.split("{")[0],
    ),
    "catalog_first_download": _Site(_pick_first_download, "", reloads_role=False),
}
_every_site = pytest.mark.parametrize("site", _SITES.values(), ids=_SITES.keys())


@dataclass
class _Harness:
    """A running app whose index was built by *built_by*, with its dialogs and toasts counted."""

    app: LilbeeAppHost
    pilot: Pilot
    services: mock.MagicMock
    start_rebuild: mock.MagicMock
    notify: mock.MagicMock
    dialogs: list[ConfirmDialog]

    def toasts(self) -> list[str]:
        return [str(call.args[0]) for call in self.notify.call_args_list]

    def toasted(self, text: str) -> bool:
        return any(text in toast for toast in self.toasts())

    async def settle(self) -> None:
        await self.app.workers.wait_for_complete()
        await self.pilot.pause()

    async def dialog(self) -> None:
        assert await pump_until(self.pilot, lambda: isinstance(self.app.screen, ConfirmDialog))

    async def answer(self, key: str) -> None:
        await self.dialog()
        await self.pilot.press(key)
        assert await pump_until(self.pilot, lambda: not isinstance(self.app.screen, ConfirmDialog))


def _index_built_by(services: mock.MagicMock, ref: str) -> None:
    """Make the store's mismatch verdict compare the asked ref, or cfg, against *ref*."""

    def _mismatch(model: str | None = None, dim: int | None = None) -> str | None:
        return None if (model or cfg.embedding_model) == ref else "built under another embedder"

    services.store.index_mismatch.side_effect = _mismatch


@pytest.fixture
def services() -> Iterator[mock.MagicMock]:
    previous = peek_services()
    fake = mock.MagicMock()
    fake.store.has_chunks.return_value = True
    _index_built_by(fake, _OLD)
    set_services(fake)
    with mock.patch("lilbee.cli.tui.widgets.model_pick.get_services", return_value=fake):
        yield fake
    set_services(previous)


class _Running:
    """Enter a test app with its dialogs, toasts and rebuild starts counted."""

    def __init__(self, app: LilbeeAppHost, services: mock.MagicMock) -> None:
        self._app = app
        self._services = services

    async def __aenter__(self) -> _Harness:
        self._context = self._app.run_test(size=(120, 40))
        pilot = await self._context.__aenter__()
        app = self._app
        await pilot.pause()
        dialogs: list[ConfirmDialog] = []
        real_push = app.push_screen

        def _push(screen: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(screen, ConfirmDialog):
                dialogs.append(screen)
            return real_push(screen, *args, **kwargs)

        self._patches = [
            mock.patch.object(app, "push_screen", _push),
            mock.patch.object(app, "start_rebuild"),
            mock.patch.object(app, "notify", wraps=app.notify),
        ]
        _, start, notify = [patch.start() for patch in self._patches]
        return _Harness(app, pilot, self._services, start, notify, dialogs)

    async def __aexit__(self, *exc: object) -> None:
        for patch in self._patches:
            patch.stop()
        await self._context.__aexit__(*exc)


@_every_site
async def test_a_differing_pick_asks_once_and_no_changes_nothing(
    site: _Site, services: mock.MagicMock
) -> None:
    cfg.embedding_model = site.before
    async with _Running(_ModelBarApp(), services) as run:
        await site.pick(run.app, run.pilot, _NEW)
        await run.dialog()
        assert cfg.embedding_model == site.before
        await run.answer("n")
        await run.settle()
        assert cfg.embedding_model == site.before
        services.reload_role.assert_not_called()
        run.start_rebuild.assert_not_called()
        assert len(run.dialogs) == 1
        assert not run.toasted(msg.MODEL_SWAP_DONE.split("{")[0])
        if site.catalog_toast is not None:
            assert not run.toasted(site.catalog_toast)


@_every_site
async def test_yes_swaps_and_starts_one_rebuild_with_no_second_question(
    site: _Site, services: mock.MagicMock
) -> None:
    cfg.embedding_model = site.before
    async with _Running(_ModelBarApp(), services) as run:
        await site.pick(run.app, run.pilot, _NEW)
        await run.answer("y")
        assert await pump_until(run.pilot, lambda: run.start_rebuild.call_count == 1)
        await run.settle()
        assert cfg.embedding_model == _NEW
        run.start_rebuild.assert_called_once_with()
        assert len(run.dialogs) == 1
        reloaded = mock.call(WorkerRole.EMBED, wait=True) in services.reload_role.mock_calls
        assert reloaded is site.reloads_role


@_every_site
async def test_a_pick_that_matches_the_index_asks_nothing(
    site: _Site, services: mock.MagicMock
) -> None:
    cfg.embedding_model = site.before
    _index_built_by(services, _NEW)
    async with _Running(_ModelBarApp(), services) as run:
        await site.pick(run.app, run.pilot, _NEW)
        assert await pump_until(run.pilot, lambda: cfg.embedding_model == _NEW)
        await run.settle()
        assert run.dialogs == []
        run.start_rebuild.assert_not_called()


async def test_a_pick_that_differs_from_the_config_only_by_form_asks_nothing(
    services: mock.MagicMock,
) -> None:
    cfg.embedding_model = _NEW
    _index_built_by(services, _NEW)
    async with _Running(_ModelBarApp(), services) as run:
        await _pick_on_bar(run.app, run.pilot, _TAGLESS)
        await run.settle()
        assert cfg.embedding_model == _NEW
        assert run.dialogs == []
        run.start_rebuild.assert_not_called()


async def test_yes_on_a_ref_the_config_canonicalizes_starts_the_rebuild(
    services: mock.MagicMock,
) -> None:
    async with _Running(_ModelBarApp(), services) as run:
        await _pick_on_bar(run.app, run.pilot, _TAGLESS)
        await run.answer("y")
        assert await pump_until(run.pilot, lambda: run.start_rebuild.call_count == 1)
        await run.settle()
        assert cfg.embedding_model == _NEW
        run.start_rebuild.assert_called_once_with()


async def test_a_swap_nothing_asked_about_offers_the_rebuild_after(
    services: mock.MagicMock,
) -> None:
    """No chunks means no question, so a stale verdict after the write becomes an offer."""
    services.store.has_chunks.return_value = False
    async with _Running(_ModelBarApp(), services) as run:
        await _pick_on_bar(run.app, run.pilot, _NEW)
        await run.answer("n")
        await run.settle()
        assert cfg.embedding_model == _NEW
        assert run.dialogs[0]._message == msg.SETTINGS_REINDEX_MESSAGE
        assert len(run.dialogs) == 1
        run.start_rebuild.assert_not_called()


async def test_a_failed_reload_restores_the_old_model_and_starts_no_rebuild(
    services: mock.MagicMock,
) -> None:

    def _fail_the_waited_reload(role: WorkerRole, *, wait: bool = False) -> None:
        if wait:
            raise RuntimeError("fleet down")

    services.reload_role.side_effect = _fail_the_waited_reload
    done = mock.Mock()
    async with _Running(_ModelBarApp(), services) as run:
        apply_model_pick(run.app.screen, key="embedding_model", ref=_NEW, on_done=done)
        await run.answer("y")
        await run.settle()
        assert cfg.embedding_model == _OLD
        run.start_rebuild.assert_not_called()
        done.assert_not_called()
        assert len(run.dialogs) == 1
        assert run.toasted(msg.MODEL_SWAP_FAILED.format(error="fleet down"))
        assert not run.toasted(msg.MODEL_SWAP_DONE.split("{")[0])


_REFUSING_SITES = {name: site for name, site in _SITES.items() if name != "catalog_first_download"}


@pytest.mark.parametrize("site", _REFUSING_SITES.values(), ids=_REFUSING_SITES.keys())
async def test_a_ref_still_downloading_shows_no_success_toast(
    site: _Site, services: mock.MagicMock
) -> None:
    _index_built_by(services, _DOWNLOADING)
    async with _Running(_ModelBarApp(), services) as run:
        run.app.task_bar.queue.enqueue(lambda: None, _DOWNLOADING_LABEL, TaskType.DOWNLOAD.value)
        await site.pick(run.app, run.pilot, _DOWNLOADING)
        await run.settle()
        assert cfg.embedding_model == _OLD
        assert run.toasted(msg.MODEL_BEING_DOWNLOADED.format(name=_DOWNLOADING_LABEL))
        assert not run.toasted(msg.MODEL_SWAP_DONE.split("{")[0])
        if site.catalog_toast is not None:
            assert not run.toasted(site.catalog_toast)
        services.reload_role.assert_not_called()


async def test_yes_then_a_refused_write_starts_no_rebuild(services: mock.MagicMock) -> None:
    async with _Running(_ModelBarApp(), services) as run:
        await _pick_on_bar(run.app, run.pilot, _DOWNLOADING)
        await run.dialog()
        run.app.task_bar.queue.enqueue(lambda: None, _DOWNLOADING_LABEL, TaskType.DOWNLOAD.value)
        await run.answer("y")
        await run.settle()
        assert cfg.embedding_model == _OLD
        run.start_rebuild.assert_not_called()
        assert not run.toasted(msg.MODEL_SWAP_DONE.split("{")[0])


async def test_a_disk_error_on_the_write_toasts_and_changes_nothing(
    services: mock.MagicMock,
) -> None:
    _index_built_by(services, _NEW)
    done = mock.Mock()
    async with _Running(_ModelBarApp(), services) as run:
        with mock.patch(
            "lilbee.app.settings.persistent_settings.update_values",
            side_effect=OSError("disk full"),
        ):
            apply_model_pick(run.app.screen, key="embedding_model", ref=_NEW, on_done=done)
            await run.settle()
        assert cfg.embedding_model == _OLD
        done.assert_not_called()
        services.reload_role.assert_not_called()
        assert run.toasted(msg.MODEL_ASSIGN_REJECTED.format(error="disk full"))


def test_a_chat_model_swap_never_needs_a_rebuild() -> None:
    """The chat picker keeps no offer, so the chat role must stay outside the reindex set."""
    assert "chat_model" not in REINDEX_FIELDS


async def test_set_active_model_returns_the_verdict_and_offers_nothing() -> None:
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    result = SettingsUpdateResult(updated=["embedding_model"], reindex_required=True, warnings=[])
    with (
        mock.patch("lilbee.cli.tui.app.apply_settings_update", return_value=result),
        mock.patch.object(app, "offer_rebuild") as offer,
    ):
        assert app.set_active_model("embedding_model", _NEW) is result
    offer.assert_not_called()


def _start_sync(screen: ChatScreen) -> None:
    screen.run_sync()


def _start_add(screen: ChatScreen) -> None:
    screen._submit_add([Path("x.md")], force=False)


def _start_import(screen: ChatScreen) -> None:
    screen._cmd_import("x.tar")


_WORK = {"sync": _start_sync, "add": _start_add, "import": _start_import}


class _BlockedWork:
    """Stand in for the three kinds of sync work; the first waits until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.rebuilds = 0
        self.syncs = 0

    def sync(self, reporter: Any, *, force_rebuild: bool = False, **_: Any) -> None:
        if force_rebuild:
            self.rebuilds += 1
            return
        self.syncs += 1
        self.release.wait(timeout=30)

    def other(self, *args: Any, **kwargs: Any) -> None:
        self.release.wait(timeout=30)

    def patches(self) -> list[Any]:
        return [
            mock.patch.object(ChatScreen, "_do_sync", side_effect=self.sync),
            mock.patch.object(ChatScreen, "_do_add", side_effect=self.other),
            mock.patch.object(ChatScreen, "_do_import", side_effect=self.other),
        ]


async def _all_work_ended(app: LilbeeAppHost, pilot: Pilot) -> bool:
    return await pump_until(
        pilot,
        lambda: (
            not app.screen._sync_work_live()
            and not any(thread.is_alive() for thread in app.task_bar._workers.values())
        ),
    )


@pytest.mark.parametrize("start", _WORK.values(), ids=_WORK.keys())
async def test_a_rebuild_asked_during_running_work_runs_once_when_the_work_ends(
    start: Callable[[ChatScreen], None],
) -> None:
    work = _BlockedWork()
    patches = work.patches()
    for patch in patches:
        patch.start()
    try:
        app = _ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            start(screen)
            assert screen._sync_work_live()
            with mock.patch.object(app, "notify", wraps=app.notify) as notify:
                screen.run_sync(force_rebuild=True)
                screen.run_sync(force_rebuild=True)
            assert work.rebuilds == 0
            assert [call.args[0] for call in notify.call_args_list] == [msg.REBUILD_OWED] * 2
            work.release.set()
            assert await pump_until(pilot, lambda: work.rebuilds == 1)
            assert await _all_work_ended(app, pilot)
            await pilot.pause()
            assert work.rebuilds == 1
    finally:
        work.release.set()
        for patch in patches:
            patch.stop()


async def test_yes_during_a_running_sync_starts_the_rebuild_when_the_sync_ends(
    services: mock.MagicMock,
) -> None:
    work = _BlockedWork()
    patches = work.patches()
    for patch in patches:
        patch.start()
    try:
        app = _ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen.run_sync()
            with mock.patch.object(app, "chat_screen", return_value=screen):
                apply_model_pick(screen, key="embedding_model", ref=_NEW, on_done=lambda: None)
                assert await pump_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
                await pilot.press("y")
                assert await pump_until(pilot, lambda: screen._rebuild_owed)
            assert cfg.embedding_model == _NEW
            assert work.rebuilds == 0
            work.release.set()
            assert await pump_until(pilot, lambda: work.rebuilds == 1)
            assert await _all_work_ended(app, pilot)
            await pilot.pause()
            assert work.rebuilds == 1
    finally:
        work.release.set()
        for patch in patches:
            patch.stop()


async def test_a_plain_sync_during_running_work_is_refused_and_owes_nothing() -> None:
    work = _BlockedWork()
    patches = work.patches()
    for patch in patches:
        patch.start()
    try:
        app = _ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen.run_sync()
            with mock.patch.object(app, "notify", wraps=app.notify) as notify:
                screen.run_sync()
            assert notify.call_args.args[0] == msg.SYNC_ALREADY_ACTIVE
            work.release.set()
            assert await _all_work_ended(app, pilot)
            assert work.rebuilds == 0
    finally:
        work.release.set()
        for patch in patches:
            patch.stop()


async def test_a_rebuild_asked_while_a_cancelled_sync_unwinds_waits_for_its_worker() -> None:
    """A cancelled row has left the queue, but its worker still writes until it ends."""
    work = _BlockedWork()
    patches = work.patches()
    for patch in patches:
        patch.start()
    try:
        app = _ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen.run_sync()
            running = app.task_bar.queue.active_tasks[0]
            app.task_bar.cancel_task(running.task_id)
            assert running.status is TaskStatus.CANCELLED
            screen.run_sync(force_rebuild=True)
            await pilot.pause()
            assert work.rebuilds == 0
            assert screen._rebuild_owed
            work.release.set()
            assert await pump_until(pilot, lambda: work.rebuilds == 1)
            assert await _all_work_ended(app, pilot)
            await pilot.pause()
            assert work.rebuilds == 1
    finally:
        work.release.set()
        for patch in patches:
            patch.stop()


@dataclass
class _RealChat:
    """The real app on its installed chat screen, with its sync work blocked until released."""

    app: LilbeeApp
    pilot: Pilot
    screen: ChatScreen
    work: _BlockedWork

    async def cancel_on_tasks(self) -> None:
        """Switch to Tasks with t and cancel the running row with c, as a user does."""
        from lilbee.cli.tui.screens.task_center import TaskCenter

        assert await pump_until(self.pilot, lambda: self.app.screen is self.screen)
        await self.pilot.press("t")
        assert await pump_until(self.pilot, lambda: isinstance(self.app.screen, TaskCenter))
        await self.pilot.press("c")
        queue = self.app.task_bar.queue
        assert await pump_until(self.pilot, lambda: not queue.active_tasks)
        assert await pump_until(
            self.pilot, lambda: any(t.status is TaskStatus.CANCELLED for t in queue.history)
        )


@asynccontextmanager
async def _real_chat(size: tuple[int, int] = (120, 40)) -> AsyncIterator[_RealChat]:
    """Run the real app through its startup gate to the chat screen it installs by name.

    The blocked work ignores a cancel until it is released, as native work does, so a
    cancelled row leaves the queue while its worker still runs.
    """
    from tests._lilbee_app_test_host import await_chat

    # The container the startup gate releases on, so the real app reaches chat.
    services = peek_services()
    assert services is not None
    services.store.has_chunks.return_value = True
    _index_built_by(services, _OLD)
    work = _BlockedWork()
    patches = [
        *work.patches(),
        mock.patch("lilbee.cli.tui.widgets.model_pick.get_services", return_value=services),
        mock.patch.object(ChatScreen, "_embedding_ready", return_value=True),
        mock.patch.object(ModelBar, "_scan_models"),
    ]
    for patch in patches:
        patch.start()
    try:
        app = LilbeeApp()
        async with app.run_test(size=size) as pilot:
            await await_chat(app, pilot)
            screen = app.chat_screen()
            assert screen is not None
            assert app.screen is screen
            await pilot.press("escape")
            try:
                yield _RealChat(app, pilot, screen, work)
            finally:
                # Before run_test exits, so a failing body ends the app, not a hang.
                work.release.set()
    finally:
        work.release.set()
        for patch in patches:
            patch.stop()


async def test_a_swap_owed_rebuild_runs_when_the_sync_is_cancelled_from_the_tasks_screen() -> None:
    async with _real_chat() as run:
        run.screen.run_sync()
        apply_model_pick(run.screen, key="embedding_model", ref=_NEW, on_done=lambda: None)
        assert await pump_until(run.pilot, lambda: isinstance(run.app.screen, ConfirmDialog))
        await run.pilot.press("y")
        assert await pump_until(run.pilot, lambda: run.screen._rebuild_owed)
        await run.cancel_on_tasks()
        assert run.work.rebuilds == 0
        assert run.screen._sync_work_live()
        run.work.release.set()
        assert await pump_until(run.pilot, lambda: run.work.rebuilds == 1)
        assert await pump_until(run.pilot, lambda: not run.screen._sync_work_live())
        await run.pilot.pause()
        assert run.work.rebuilds == 1
        assert not run.screen._rebuild_owed
        assert cfg.embedding_model == _NEW


async def test_a_sync_after_an_add_cancelled_on_tasks_is_refused_until_its_worker_ends() -> None:
    async with _real_chat() as run:
        run.screen._submit_add([Path("x.md")], force=False)
        await run.cancel_on_tasks()
        await run.pilot.press("C")
        assert [t.task_type for t in run.app.task_bar.queue.history] == [TaskType.ADD.value]
        with mock.patch.object(run.app, "notify", wraps=run.app.notify) as notify:
            await run.pilot.press("S")
            assert await pump_until(run.pilot, lambda: bool(notify.call_args_list))
        assert [call.args[0] for call in notify.call_args_list] == [
            msg.SYNC_CANCELLING.format(kind="add")
        ]
        assert run.work.syncs == 0
        run.work.release.set()
        assert await pump_until(run.pilot, lambda: not run.screen._sync_work_live())
        await run.pilot.press("S")
        assert await pump_until(run.pilot, lambda: run.work.syncs == 1)


def _task_frame(run: _RealChat) -> str:
    """The text of the Tasks screen's task rows and the task bar, as the user reads them."""
    from textual.widgets import Label

    heads = [str(label.render()) for label in run.app.screen.query("#row-head").results(Label)]
    bar = str(run.app.screen.query_one("#task-status-label", Label).render())
    return "\n".join([*heads, bar])


async def test_a_cancelled_sync_shows_cancelling_until_its_worker_exits_then_cancelled() -> None:
    async with _real_chat(size=(80, 24)) as run:
        run.screen.run_sync()
        await run.cancel_on_tasks()
        assert run.app.task_bar.cancelling_tasks()
        assert await pump_until(run.pilot, lambda: msg.TASK_STATUS_CANCELLING in _task_frame(run))
        frame = _task_frame(run)
        assert f"{msg.TASKBAR_CANCELLING} {msg.TASK_NAME_SYNC}" in frame
        assert "cancelled" not in frame
        with mock.patch.object(run.app, "notify", wraps=run.app.notify) as notify:
            run.screen.run_sync()
        assert [call.args[0] for call in notify.call_args_list] == [
            msg.SYNC_CANCELLING.format(kind="sync")
        ]
        run.work.release.set()
        assert await pump_until(run.pilot, lambda: not run.app.task_bar.cancelling_tasks())
        assert await pump_until(
            run.pilot, lambda: msg.TASK_STATUS_CANCELLING not in _task_frame(run)
        )
        assert "cancelled" in _task_frame(run)


async def test_clearing_the_history_keeps_a_cancelling_row_and_counts_it() -> None:
    from textual.widgets import Label

    def counts() -> str:
        return str(run.app.screen.query_one("#task-center-counts", Label).render())

    async with _real_chat(size=(80, 24)) as run:
        run.screen.run_sync()
        await run.cancel_on_tasks()
        await run.pilot.press("C")
        assert await pump_until(run.pilot, lambda: "1 cancelling" in counts())
        assert "0 running  ·  0 queued  ·  0 done  ·  1 cancelling" in counts()
        assert msg.TASK_STATUS_CANCELLING in _task_frame(run)
        run.work.release.set()
        assert await pump_until(run.pilot, lambda: not run.app.task_bar.cancelling_tasks())
        assert await pump_until(run.pilot, lambda: "cancelling" not in counts())
        assert await pump_until(run.pilot, lambda: "cancelled" in _task_frame(run))


class _ChatRun:
    """Run a chat screen with its sync work blocked until released."""

    async def __aenter__(self) -> _ChatRun:
        self.work = _BlockedWork()
        self._patches = self.work.patches()
        for patch in self._patches:
            patch.start()
        self.app = _ChatTestApp()
        self._context = self.app.run_test(size=(120, 40))
        self.pilot = await self._context.__aenter__()
        await self.pilot.pause()
        screen = self.app.screen
        assert isinstance(screen, ChatScreen)
        self.screen = screen
        self._chat_screen = mock.patch.object(self.app, "chat_screen", return_value=screen)
        self._chat_screen.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.work.release.set()
        self._chat_screen.stop()
        await self._context.__aexit__(*exc)
        for patch in self._patches:
            patch.stop()

    def queued_syncs(self) -> list[Task]:
        queued = self.app.task_bar.queue.queued_tasks
        return [task for task in queued if task.task_type == TaskType.SYNC.value]

    async def swap_with_yes(self, ref: str) -> None:
        apply_model_pick(self.screen, key="embedding_model", ref=ref, on_done=lambda: None)
        assert await pump_until(self.pilot, lambda: isinstance(self.app.screen, ConfirmDialog))
        await self.pilot.press("y")
        assert await pump_until(self.pilot, lambda: cfg.embedding_model == ref)

    async def end_all_work(self) -> None:
        self.work.release.set()
        assert await _all_work_ended(self.app, self.pilot)
        await self.pilot.pause()


async def test_an_owed_rebuild_never_queues_behind_the_ending_worker() -> None:
    """At the end of a sync's worker, no rebuild row exists for the user to cancel and strand."""
    async with _ChatRun() as run:
        reached = threading.Event()
        leave = threading.Event()

        def _hold(*_: object) -> None:
            reached.set()
            leave.wait(timeout=30)

        with mock.patch.object(run.app.task_bar, "start_detect_pending", side_effect=_hold):
            run.screen.run_sync()
            run.screen.run_sync(force_rebuild=True)
            run.work.release.set()
            assert await pump_until(run.pilot, reached.is_set)
            await run.pilot.pause()
            assert run.queued_syncs() == []
            leave.set()
            assert await pump_until(run.pilot, lambda: run.work.rebuilds == 1)
            assert await _all_work_ended(run.app, run.pilot)
            run.screen.run_sync()
            assert await pump_until(run.pilot, lambda: run.work.syncs == 2)


async def test_a_cancelled_queued_rebuild_never_runs_and_a_later_sync_starts() -> None:
    async with _ChatRun() as run:
        ran = threading.Event()
        run.screen.run_sync()
        queued = run.app.task_bar.start_task(
            msg.TASK_NAME_REBUILD, TaskType.SYNC, lambda _reporter: ran.set()
        )
        assert [task.task_id for task in run.queued_syncs()] == [queued]
        run.app.task_bar.cancel_task(queued)
        await run.end_all_work()
        assert not ran.is_set()
        run.screen.run_sync()
        assert await pump_until(run.pilot, lambda: run.work.syncs == 2)
        await run.end_all_work()
        assert run.work.rebuilds == 0


async def test_a_cancelled_queued_sync_with_a_rebuild_owed_runs_the_rebuild_once() -> None:
    async with _ChatRun() as run:
        queued = run.app.task_bar.add_task(msg.TASK_NAME_SYNC, TaskType.SYNC)
        run.screen.run_sync(force_rebuild=True)
        assert run.screen._rebuild_owed
        assert run.work.rebuilds == 0
        run.app.task_bar.cancel_task(queued)
        assert await pump_until(run.pilot, lambda: run.work.rebuilds == 1)
        await run.end_all_work()
        assert run.work.rebuilds == 1
        assert not run.screen._rebuild_owed


async def test_cancelling_the_running_and_the_queued_sync_runs_the_owed_rebuild_once() -> None:
    async with _ChatRun() as run:
        run.screen.run_sync()
        running = run.app.task_bar.queue.active_tasks[0].task_id
        queued = run.app.task_bar.start_task(
            msg.TASK_NAME_SYNC, TaskType.SYNC, lambda _reporter: None
        )
        run.screen.run_sync(force_rebuild=True)
        run.app.task_bar.cancel_task(queued)
        run.app.task_bar.cancel_task(running)
        await run.pilot.pause()
        assert run.work.rebuilds == 0
        run.work.release.set()
        assert await pump_until(run.pilot, lambda: run.work.rebuilds == 1)
        await run.end_all_work()
        assert run.work.rebuilds == 1


async def test_a_listener_subscribed_twice_runs_once_and_never_after_unsubscribe() -> None:
    async with _ChatRun() as run:
        controller = run.app.task_bar
        ends = mock.Mock()
        controller.subscribe_work_end(ends)
        controller.subscribe_work_end(ends)
        controller.cancel_task(controller.add_task("Sync", TaskType.SYNC))
        assert ends.call_count == 1
        controller.unsubscribe_work_end(ends)
        controller.unsubscribe_work_end(ends)
        controller.cancel_task(controller.add_task("Sync", TaskType.SYNC))
        assert ends.call_count == 1


def _fail(_reporter: object) -> None:
    raise ValueError("boom")


async def _end_with_a_worker(controller: Any, target: Callable[[Any], None]) -> str:
    return controller.start_task("Wiki", TaskType.WIKI, target)


async def _cancel_a_running_worker(controller: Any, _target: object) -> str:
    entered = threading.Event()
    release = threading.Event()

    def _run(reporter: Any) -> None:
        entered.set()
        release.wait(timeout=30)
        reporter.check_cancelled()

    task_id = controller.start_task("Wiki", TaskType.WIKI, _run)
    assert entered.wait(timeout=5)
    controller.cancel_task(task_id)
    release.set()
    return task_id


async def _cancel_a_queued_worker(controller: Any, _target: object) -> str:
    controller.start_task("Wiki", TaskType.WIKI, lambda _reporter: None)
    queued = controller.start_task("Wiki 2", TaskType.WIKI, lambda _reporter: None)
    controller.cancel_task(queued)
    return queued


async def _end_a_row_with_no_worker(controller: Any, end: Callable[[Any, str], None]) -> str:
    task_id = controller.add_task("Sync", TaskType.SYNC)
    controller.queue.advance(TaskType.SYNC.value)
    end(controller, task_id)
    return task_id


_ENDS: dict[str, tuple[Callable[..., Awaitable[str]], Any, int]] = {
    "worker_done": (_end_with_a_worker, lambda _reporter: None, 1),
    "worker_failed": (_end_with_a_worker, _fail, 1),
    "running_cancel": (_cancel_a_running_worker, None, 1),
    "queued_cancel": (_cancel_a_queued_worker, None, 2),
    "no_worker_complete": (_end_a_row_with_no_worker, lambda c, t: c.complete_task(t), 1),
    "no_worker_fail": (_end_a_row_with_no_worker, lambda c, t: c.fail_task(t, "x"), 1),
    "no_worker_cancel": (_end_a_row_with_no_worker, lambda c, t: c.cancel_task(t), 1),
}


@pytest.mark.parametrize(("end", "arg", "ends"), _ENDS.values(), ids=_ENDS.keys())
async def test_the_work_end_hook_runs_once_per_end(
    end: Callable[..., Awaitable[str]], arg: Any, ends: int
) -> None:
    """A queued cancel ends two tasks: the cancelled row and the worker it waited behind."""
    async with _ChatRun() as run:
        controller = run.app.task_bar
        listener = mock.Mock()
        controller.subscribe_work_end(listener)
        await end(controller, arg)
        assert await pump_until(run.pilot, lambda: not controller._workers)
        assert await pump_until(run.pilot, lambda: listener.call_count >= ends)
        for _ in range(5):
            await run.pilot.pause()
        assert listener.call_count == ends


async def test_a_raising_work_end_listener_is_logged_and_the_next_one_still_runs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _ChatRun() as run:
        controller = run.app.task_bar
        later = mock.Mock()
        controller.subscribe_work_end(mock.Mock(side_effect=ValueError("listener bug")))
        controller.subscribe_work_end(later)
        controller.cancel_task(controller.add_task("Sync", TaskType.SYNC))
        assert later.call_count == 1
        controller.start_task("Wiki", TaskType.WIKI, lambda _reporter: None)
        assert await pump_until(run.pilot, lambda: later.call_count == 2)
        assert run.app.is_running
    raised = [r for r in caplog.records if r.getMessage() == "A work-end listener raised"]
    assert len(raised) == 2


async def test_a_cancel_whose_row_was_cleared_still_stops_the_worker() -> None:
    """The Task Center's C between a cancel and the worker's next checkpoint keeps the cancel."""
    async with _ChatRun() as run:
        controller = run.app.task_bar
        entered = threading.Event()
        release = threading.Event()
        steps: list[int] = []

        def _job(reporter: Any) -> None:
            entered.set()
            release.wait(timeout=30)
            for index in range(5):
                reporter.check_cancelled()
                steps.append(index)

        task_id = controller.start_task("Wiki", TaskType.WIKI, _job)
        assert entered.wait(timeout=5)
        controller.cancel_task(task_id)
        controller.queue.clear_history()
        assert controller.queue.get_task(task_id) is None
        release.set()
        assert await pump_until(run.pilot, lambda: not controller._workers)
        assert steps == []


async def test_a_worker_whose_row_was_cleared_counts_as_live_for_its_own_type_only() -> None:
    async with _ChatRun() as run:
        controller = run.app.task_bar
        release = threading.Event()
        task_id = controller.start_task(
            "Pull", TaskType.DOWNLOAD, lambda _reporter: release.wait(timeout=30)
        )
        controller.cancel_task(task_id)
        controller.queue.clear_history()
        assert controller.queue.get_task(task_id) is None
        assert controller.has_live_work({TaskType.DOWNLOAD.value})
        assert not controller.has_live_work({TaskType.SYNC.value})
        run.screen.run_sync()
        assert await pump_until(run.pilot, lambda: run.work.syncs == 1)
        release.set()
        assert await pump_until(
            run.pilot, lambda: not controller.has_live_work({TaskType.DOWNLOAD.value})
        )


async def test_a_swap_during_a_sync_runs_one_rebuild_when_the_work_ends(
    services: mock.MagicMock,
) -> None:
    async with _ChatRun() as run:
        run.screen.run_sync()
        await run.swap_with_yes(_NEW)
        run.work.release.set()
        assert await pump_until(run.pilot, lambda: run.work.rebuilds == 1)
        await run.end_all_work()
        assert run.work.rebuilds == 1


async def test_a_direct_rebuild_during_a_sync_runs_when_the_work_ends(
    services: mock.MagicMock,
) -> None:
    async with _ChatRun() as run:
        run.screen.run_sync()
        run.screen.run_sync(force_rebuild=True)
        run.work.release.set()
        assert await pump_until(run.pilot, lambda: run.work.rebuilds == 1)
        await run.end_all_work()
        assert run.work.rebuilds == 1


_STORE_ERROR = "lance: table corrupt"


def _saved_embedder() -> str | None:
    return persistent_settings.load(cfg.data_root).get("embedding_model")


def _still_up(app: LilbeeAppHost) -> bool:
    return app.is_running and app.return_code is None


@pytest.mark.parametrize("error", [RuntimeError, OSError, ValueError])
async def test_a_store_error_at_the_question_toasts_and_keeps_the_tui(
    error: type[Exception], services: mock.MagicMock
) -> None:
    services.store.index_mismatch.side_effect = error(_STORE_ERROR)
    async with _Running(_ModelBarApp(), services) as run:
        await _pick_on_bar(run.app, run.pilot, _NEW)
        assert await pump_until(
            run.pilot, lambda: run.toasted(msg.MODEL_ASSIGN_REJECTED.format(error=_STORE_ERROR))
        )
        assert _still_up(run.app)
        assert run.dialogs == []
        assert cfg.embedding_model == _OLD


async def test_a_store_error_before_the_write_toasts_and_changes_nothing(
    services: mock.MagicMock,
) -> None:
    services.store.has_chunks.return_value = False
    services.store.initialize_meta_if_legacy.side_effect = RuntimeError(_STORE_ERROR)
    async with _Running(_ModelBarApp(), services) as run:
        await _pick_on_bar(run.app, run.pilot, _NEW)
        assert await pump_until(
            run.pilot, lambda: run.toasted(msg.MODEL_ASSIGN_REJECTED.format(error=_STORE_ERROR))
        )
        assert _still_up(run.app)
        assert cfg.embedding_model == _OLD
        assert _saved_embedder() is None


@pytest.mark.parametrize("saved", [None, "ollama/saved-embed:latest"], ids=["unsaved", "saved"])
async def test_a_store_error_at_the_verdict_puts_the_config_back(
    saved: str | None, services: mock.MagicMock
) -> None:
    if saved is not None:
        persistent_settings.update_values(cfg.data_root, {"embedding_model": saved})
    services.store.has_chunks.return_value = False
    services.store.index_mismatch.side_effect = RuntimeError(_STORE_ERROR)
    async with _Running(_ModelBarApp(), services) as run:
        await _pick_on_bar(run.app, run.pilot, _NEW)
        assert await pump_until(
            run.pilot, lambda: run.toasted(msg.MODEL_ASSIGN_REJECTED.format(error=_STORE_ERROR))
        )
        await run.settle()
        assert _still_up(run.app)
        assert cfg.embedding_model == _OLD
        assert _saved_embedder() == saved
        run.start_rebuild.assert_not_called()


def test_a_failed_config_undo_still_raises_the_first_error(services: mock.MagicMock) -> None:
    services.store.index_mismatch.side_effect = RuntimeError(_STORE_ERROR)
    real_update = persistent_settings.update_values
    calls: list[str] = []

    def _second_write_fails(root: Path, updates: dict[str, Any]) -> None:
        calls.append(updates["embedding_model"])
        if len(calls) > 1:
            raise OSError("disk full")
        real_update(root, updates)

    persistent_settings.update_values(cfg.data_root, {"embedding_model": _OLD})
    with (
        mock.patch(
            "lilbee.app.settings.persistent_settings.update_values",
            side_effect=_second_write_fails,
        ),
        pytest.raises(RuntimeError, match=_STORE_ERROR),
    ):
        apply_settings_update({"embedding_model": _NEW})
    assert calls == [_NEW, _OLD]


def _fail_the_waited_reload(role: WorkerRole, *, wait: bool = False) -> None:
    if wait:
        raise RuntimeError("fleet down")


_RELOADING_SITES = {name: site for name, site in _SITES.items() if site.reloads_role}


@pytest.mark.parametrize("site", _RELOADING_SITES.values(), ids=_RELOADING_SITES.keys())
async def test_a_failed_reload_at_every_site_restores_the_model_and_shows_no_success(
    site: _Site, services: mock.MagicMock
) -> None:
    services.reload_role.side_effect = _fail_the_waited_reload
    async with _Running(_ModelBarApp(), services) as run:
        await site.pick(run.app, run.pilot, _NEW)
        await run.answer("y")
        assert await pump_until(
            run.pilot, lambda: run.toasted(msg.MODEL_SWAP_FAILED.format(error="fleet down"))
        )
        await run.settle()
        assert cfg.embedding_model == _OLD
        assert _saved_embedder() is None
        run.start_rebuild.assert_not_called()
        assert not run.toasted(msg.MODEL_SWAP_DONE.split("{")[0])
        if site.catalog_toast is not None:
            assert not run.toasted(site.catalog_toast)


@pytest.mark.parametrize(
    ("saved", "source"),
    [(None, SettingSource.BUILT_IN), (_OLD, SettingSource.USER)],
    ids=["unsaved", "saved"],
)
async def test_a_failed_reload_restores_the_source_of_the_old_model(
    saved: str | None, source: SettingSource, services: mock.MagicMock
) -> None:
    """The rollback puts back the config entry as it was, so the setting is not "set by you"."""
    builtin = builtin_value("embedding_model")
    cfg.embedding_model = builtin if saved is None else saved
    if saved is not None:
        persistent_settings.update_values(cfg.data_root, {"embedding_model": saved})
    before = cfg.embedding_model
    services.reload_role.side_effect = _fail_the_waited_reload
    async with _Running(_ModelBarApp(), services) as run:
        apply_model_pick(run.app.screen, key="embedding_model", ref=_NEW, on_done=mock.Mock())
        await run.settle()
        if isinstance(run.app.screen, ConfirmDialog):
            await run.answer("y")
        assert await pump_until(
            run.pilot, lambda: run.toasted(msg.MODEL_SWAP_FAILED.format(error="fleet down"))
        )
        await run.settle()
        assert cfg.embedding_model == before
        assert _saved_embedder() == saved
        expected = {source, SettingSource.AUTO} if saved is None else {source}
        assert setting_sources()["embedding_model"] in expected


@pytest.mark.parametrize("error", [ValueError, OSError, RuntimeError])
async def test_a_failed_rollback_names_the_model_now_in_use(
    error: type[Exception], services: mock.MagicMock
) -> None:
    services.reload_role.side_effect = _fail_the_waited_reload
    async with _Running(_ModelBarApp(), services) as run:
        with mock.patch(
            "lilbee.cli.tui.widgets.model_pick.restore_settings", side_effect=error("disk full")
        ):
            apply_model_pick(run.app.screen, key="embedding_model", ref=_NEW, on_done=mock.Mock())
            await run.answer("y")
            await run.settle()
            stuck = msg.MODEL_SWAP_STUCK.format(error="fleet down", name=_NEW)
            assert await pump_until(run.pilot, lambda: run.toasted(stuck))
        assert _still_up(run.app)
        assert cfg.embedding_model == _NEW
        assert _saved_embedder() == _NEW
        assert len([toast for toast in run.toasts() if "fleet down" in toast]) == 1
