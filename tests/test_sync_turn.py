"""Syncs on one library take turns: a second sync waits for the running one and names it."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import pytest
from filelock import FileLock

from lilbee.core.config import cfg
from lilbee.data.ingest import SyncResult
from lilbee.data.ingest.pipeline import _marks_sync_running
from lilbee.runtime.lock import (
    LibraryUnlockableError,
    SyncTurnOwner,
    read_sync_turn_owner,
    sync_turn,
)
from lilbee.runtime.progress import (
    EventType,
    FileStartEvent,
    ProgressEvent,
    SyncWaitEvent,
)

# Ends a test that hangs; no wait in these tests is a sleep.
_HANG_LIMIT_S = 10.0
_TURN_LOCK = "sync.turn.lock"
_TURN_OWNER = "sync.turn.owner.json"
_WAIT_FOR_ADD = (
    f"A sync started by lilbee add (process {os.getpid()}) is running on this library. "
    "Waiting for it to finish. Press Ctrl-C to stop waiting."
)


@dataclass
class _Library:
    """A library whose sync plans from the store and stores what it embeds, like the real one."""

    disk: set[str] = field(default_factory=set)
    stored: set[str] = field(default_factory=set)
    embedded: list[str] = field(default_factory=list)

    async def run(
        self,
        *,
        planned: asyncio.Event | None = None,
        gate: asyncio.Event | None = None,
        **_kwargs: Any,
    ) -> SyncResult:
        new = sorted(self.disk - self.stored)
        if planned is not None:
            planned.set()
        if gate is not None:
            await gate.wait()
        self.embedded.extend(new)
        self.stored |= set(new)
        return SyncResult()


@pytest.fixture
def library() -> Iterator[_Library]:
    lib = _Library()
    with (
        mock.patch("lilbee.data.ingest.pipeline.get_services", return_value=mock.MagicMock()),
        mock.patch("lilbee.data.ingest.pipeline.settings.overlay_persisted_roots"),
    ):
        yield lib


def _sync(lib: _Library):
    return _marks_sync_running(lib.run)


def _waits(seen: list[ProgressEvent], waiting: asyncio.Event | threading.Event):
    def _on_progress(event_type: EventType, data: ProgressEvent) -> None:
        if event_type == "sync_wait":
            seen.append(data)
            waiting.set()

    return _on_progress


@pytest.fixture
async def turn_held_by_another_add() -> AsyncIterator[FileLock]:
    """Hold the library's sync turn as another ``lilbee add`` in this process would."""
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    holder = FileLock(cfg.data_root / _TURN_LOCK, thread_local=False)
    holder.acquire(timeout=_HANG_LIMIT_S)
    (cfg.data_root / _TURN_OWNER).write_text(
        json.dumps({"started_by": "lilbee add", "pid": os.getpid()}), encoding="utf-8"
    )
    try:
        yield holder
    finally:
        holder.release()


async def test_two_syncs_on_one_library_run_one_after_the_other_and_sync_no_file_twice(
    library: _Library,
) -> None:
    library.disk |= {f"first-{n}" for n in range(6)}
    first_planned, release_first = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(
        _sync(library)(planned=first_planned, gate=release_first, started_by="lilbee add")
    )
    await asyncio.wait_for(first_planned.wait(), _HANG_LIMIT_S)
    # The second add registers its files while the first add's files are not stored yet.
    library.disk |= {f"second-{n}" for n in range(3)}
    second_planned_or_waiting = asyncio.Event()
    second = asyncio.create_task(
        _sync(library)(
            planned=second_planned_or_waiting,
            on_progress=_waits([], second_planned_or_waiting),
            started_by="lilbee add",
        )
    )
    await asyncio.wait_for(second_planned_or_waiting.wait(), _HANG_LIMIT_S)
    release_first.set()
    await asyncio.wait_for(asyncio.gather(first, second), _HANG_LIMIT_S)
    assert sorted(library.embedded) == sorted(library.disk)
    assert len(library.embedded) == 9


async def test_a_waiting_sync_names_the_running_sync_once(
    library: _Library, turn_held_by_another_add: FileLock
) -> None:
    seen: list[ProgressEvent] = []
    waiting = asyncio.Event()
    task = asyncio.create_task(_sync(library)(on_progress=_waits(seen, waiting)))
    await asyncio.wait_for(waiting.wait(), _HANG_LIMIT_S)
    assert not task.done()
    turn_held_by_another_add.release()
    await asyncio.wait_for(task, _HANG_LIMIT_S)
    assert seen == [SyncWaitEvent(message=_WAIT_FOR_ADD, holder="lilbee add", pid=os.getpid())]


async def test_a_waiting_sync_with_no_readable_owner_says_another_sync_runs(
    library: _Library, turn_held_by_another_add: FileLock
) -> None:
    (cfg.data_root / _TURN_OWNER).write_text("not json", encoding="utf-8")
    seen: list[ProgressEvent] = []
    waiting = asyncio.Event()
    task = asyncio.create_task(_sync(library)(on_progress=_waits(seen, waiting)))
    await asyncio.wait_for(waiting.wait(), _HANG_LIMIT_S)
    turn_held_by_another_add.release()
    await asyncio.wait_for(task, _HANG_LIMIT_S)
    assert seen == [
        SyncWaitEvent(
            message=(
                "Another sync is running on this library. "
                "Waiting for it to finish. Press Ctrl-C to stop waiting."
            )
        )
    ]


async def test_the_sync_that_holds_the_turn_names_itself_and_clears_the_record_after(
    library: _Library,
) -> None:
    seen_owner: list[dict[str, object]] = []

    async def _run(**_kwargs: Any) -> SyncResult:
        seen_owner.append(json.loads((cfg.data_root / _TURN_OWNER).read_text(encoding="utf-8")))
        return SyncResult()

    await asyncio.wait_for(_marks_sync_running(_run)(started_by="lilbee serve"), _HANG_LIMIT_S)
    assert seen_owner == [{"started_by": "lilbee serve", "pid": os.getpid()}]
    assert not (cfg.data_root / _TURN_OWNER).exists()


async def test_a_cancel_during_the_wait_stops_the_sync_and_frees_the_turn(
    library: _Library, turn_held_by_another_add: FileLock
) -> None:
    cancel = threading.Event()
    task = asyncio.create_task(_sync(library)(on_progress=_waits([], cancel), cancel=cancel))
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, _HANG_LIMIT_S)
    assert library.embedded == []
    turn_held_by_another_add.release()
    # The cancelled waiter holds nothing: the next sync takes the turn and runs.
    library.disk.add("doc")
    await asyncio.wait_for(_sync(library)(), _HANG_LIMIT_S)
    assert library.embedded == ["doc"]


async def test_a_fan_out_worker_runs_under_its_parents_turn_without_taking_it(
    library: _Library, turn_held_by_another_add: FileLock
) -> None:
    library.disk.add("slice")
    seen: list[ProgressEvent] = []
    await asyncio.wait_for(
        _sync(library)(shard=mock.sentinel.shard, on_progress=_waits(seen, asyncio.Event())),
        _HANG_LIMIT_S,
    )
    assert library.embedded == ["slice"]
    assert seen == []


def test_lilbee_add_prints_the_wait_on_stderr_naming_the_running_sync(
    library: _Library, capsys: pytest.CaptureFixture[str]
) -> None:
    from lilbee.cli.commands import ingest_sync

    cfg.data_root.mkdir(parents=True, exist_ok=True)
    holder = FileLock(cfg.data_root / _TURN_LOCK, thread_local=False)
    holder.acquire(timeout=_HANG_LIMIT_S)
    (cfg.data_root / _TURN_OWNER).write_text(
        json.dumps({"started_by": "lilbee add", "pid": os.getpid()}), encoding="utf-8"
    )

    def _free_the_turn(event_type: EventType, _data: ProgressEvent) -> None:
        if event_type == "sync_wait":
            holder.release()

    try:
        with mock.patch("lilbee.data.ingest.sync", _sync(library)):
            ingest_sync._run_sync(
                threading.Event(), on_progress=_free_the_turn, started_by="lilbee add"
            )
    finally:
        holder.release()
    err = capsys.readouterr().err
    assert " ".join(err.split()) == _WAIT_FOR_ADD


async def test_the_sync_stream_of_serve_sends_the_wait_then_its_own_result(
    library: _Library, turn_held_by_another_add: FileLock
) -> None:
    from lilbee.server.handlers.ingest import sync_stream

    frames: list[str] = []
    with mock.patch("lilbee.data.ingest.sync", _sync(library)):
        async for frame in sync_stream():
            frames.append(frame)
            if frame.startswith("event: sync_wait"):
                turn_held_by_another_add.release()
    wait_frame = (
        f"event: sync_wait\ndata: "
        f"{json.dumps({'message': _WAIT_FOR_ADD, 'holder': 'lilbee add', 'pid': os.getpid()})}"
        "\n\n"
    )
    assert frames[0] == wait_frame
    assert frames[-1].startswith("event: done")
    assert len(frames) == 2


_NOTICE = SyncWaitEvent(message=_WAIT_FOR_ADD, holder="lilbee add", pid=os.getpid())


def test_the_background_sync_printer_prints_the_wait_notice() -> None:
    from lilbee.cli import sync as cli_sync

    con = mock.MagicMock()
    cli_sync._sync_progress_printer(con)(EventType.SYNC_WAIT, _NOTICE)
    assert con.print.call_args.args == (_WAIT_FOR_ADD,)


def test_the_chat_status_bar_shows_the_wait_notice() -> None:
    from lilbee.cli import sync as cli_sync

    status = cli_sync.SyncStatus()
    status.adjust_pending(1)
    cli_sync._chat_sync_callback(status)(EventType.SYNC_WAIT, _NOTICE)
    assert status.text == f"⟳ {_WAIT_FOR_ADD} (+1 queued)"


@pytest.mark.parametrize("build", ["build_add_progress_callback", "build_sync_progress_callback"])
def test_the_tui_task_bar_says_it_waits_for_the_running_sync(build: str) -> None:
    from lilbee.cli.tui.screens import chat_helpers
    from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter

    reporter = mock.MagicMock(spec=ProgressReporter)
    getattr(chat_helpers, build)(reporter)(EventType.SYNC_WAIT, _NOTICE)
    reporter.update.assert_called_once_with(0, "Waiting for the running sync", indeterminate=True)


def test_the_background_sync_printer_rejects_a_wait_event_of_the_wrong_type() -> None:
    from lilbee.cli import sync as cli_sync

    bad = FileStartEvent(file="x", total_files=1, current_file=1)
    with pytest.raises(TypeError, match="Expected SyncWaitEvent"):
        cli_sync._sync_progress_printer(mock.MagicMock())(EventType.SYNC_WAIT, bad)


def test_the_chat_status_bar_rejects_a_wait_event_of_the_wrong_type() -> None:
    from lilbee.cli import sync as cli_sync

    bad = FileStartEvent(file="x", total_files=1, current_file=1)
    with pytest.raises(TypeError, match="Expected SyncWaitEvent"):
        cli_sync._chat_sync_callback(cli_sync.SyncStatus())(EventType.SYNC_WAIT, bad)


def _no_wait_notice(_owner: SyncTurnOwner | None) -> None:
    raise AssertionError("the turn was free, so nobody should wait")


async def test_a_holder_whose_owner_record_cannot_be_written_still_runs_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    # A directory in the record's place makes both the write and the delete fail.
    (cfg.data_root / _TURN_OWNER).mkdir()
    ran = False
    with caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"):
        async with sync_turn(cfg.data_root, None, started_by="lilbee add", on_wait=_no_wait_notice):
            ran = True
    assert ran
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Could not record the sync owner") for m in messages)
    assert any(m.startswith("Could not delete") for m in messages)
    # The turn is free after the failed delete: another holder takes it at once.
    async with sync_turn(cfg.data_root, None, started_by="lilbee add", on_wait=_no_wait_notice):
        pass


async def test_a_turn_lock_the_system_refuses_ends_the_sync_with_a_library_error() -> None:
    class _RefusingLock:
        lock_file = str(cfg.data_root / _TURN_LOCK)
        is_locked = False

        def __init__(self, _path: object) -> None:
            pass

        async def acquire(self, *, timeout: float) -> None:
            raise PermissionError(13, "Permission denied")

    cfg.data_root.mkdir(parents=True, exist_ok=True)
    with (
        mock.patch("lilbee.runtime.lock.AsyncFileLock", _RefusingLock),
        pytest.raises(LibraryUnlockableError, match="Permission denied"),
    ):
        async with sync_turn(cfg.data_root, None, started_by="lilbee add", on_wait=_no_wait_notice):
            pytest.fail("the block must not run when the turn cannot be locked")


async def test_the_foreground_add_sync_names_its_command() -> None:
    from lilbee.cli.helpers import _run_foreground_sync

    with mock.patch("lilbee.data.ingest.sync", new_callable=mock.AsyncMock) as sync:
        await asyncio.to_thread(_run_foreground_sync)
    sync.assert_awaited_once_with(started_by="lilbee add")


# The child holds the turn like a running `lilbee add`, announces it, and never lets go.
_HOLDER_CHILD = """
import asyncio, sys
from pathlib import Path
from lilbee.runtime.lock import sync_turn

async def main():
    async with sync_turn(
        Path(sys.argv[1]), None, started_by="child add", on_wait=lambda owner: None
    ):
        print("held", flush=True)
        await asyncio.sleep(3600)

asyncio.run(main())
"""


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGKILL to send")
async def test_a_waiter_takes_the_turn_when_its_holder_is_killed_with_sigkill() -> None:
    import signal

    cfg.data_root.mkdir(parents=True, exist_ok=True)
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _HOLDER_CHILD,
        str(cfg.data_root),
        stdout=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        assert child.stdout is not None
        line = await asyncio.wait_for(child.stdout.readline(), _HANG_LIMIT_S)
        assert line.strip() == b"held"
        owners: list[SyncTurnOwner | None] = []
        waiting = asyncio.Event()

        def _on_wait(owner: SyncTurnOwner | None) -> None:
            owners.append(owner)
            waiting.set()

        async def _wait_for_the_turn() -> SyncTurnOwner | None:
            async with sync_turn(cfg.data_root, None, started_by="waiter", on_wait=_on_wait):
                return read_sync_turn_owner(cfg.data_root)

        waiter = asyncio.create_task(_wait_for_the_turn())
        await asyncio.wait_for(waiting.wait(), _HANG_LIMIT_S)
        assert not waiter.done()
        assert owners == [SyncTurnOwner(started_by="child add", pid=child.pid)]
        os.kill(child.pid, signal.SIGKILL)
        taken_by = await asyncio.wait_for(waiter, _HANG_LIMIT_S)
        assert taken_by == SyncTurnOwner(started_by="waiter", pid=os.getpid())
        assert await asyncio.wait_for(child.wait(), _HANG_LIMIT_S) == -signal.SIGKILL
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
