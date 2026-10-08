"""Tests for /add cancel cleanup.

When a user cancels an in-flight /add, the source root it registered must be
un-registered so the next sync does not silently re-ingest it. The source bytes
on disk are never touched.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from lilbee.cli.tui.screens.chat_helpers import unregister_added_roots
from lilbee.core.config import cfg
from lilbee.runtime.cancellation import TaskCancelledError
from tests._async_wait import wait_until


@pytest.fixture
def isolated_documents(tmp_path):
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_root = tmp_path
    cfg.linked_roots = {}
    try:
        yield cfg.documents_dir
    finally:
        for field_name in type(snapshot).model_fields:
            setattr(cfg, field_name, getattr(snapshot, field_name))


class TestUnregisterAddedRoots:
    def test_unregisters_root_without_touching_source(self, isolated_documents, tmp_path):
        from lilbee.core import settings

        source = tmp_path / "corpus"
        source.mkdir()
        (source / "a.txt").write_text("keep me")
        settings.set_value(cfg.data_root, "linked_roots", {"corpus": str(source)})

        unregister_added_roots(["corpus"])

        assert "corpus" not in cfg.linked_roots  # registry entry dropped
        assert (source / "a.txt").read_text() == "keep me"  # source bytes untouched

    def test_tolerates_unknown_label(self, isolated_documents):
        # User may have removed the source concurrently; do not raise.
        unregister_added_roots(["never-registered"])
        assert cfg.linked_roots == {}

    def test_leaves_other_roots_alone(self, isolated_documents, tmp_path):
        from lilbee.core import settings

        settings.set_value(
            cfg.data_root,
            "linked_roots",
            {"keep": str(tmp_path / "keep"), "drop": str(tmp_path / "drop")},
        )
        unregister_added_roots(["drop"])
        assert "keep" in cfg.linked_roots
        assert "drop" not in cfg.linked_roots

    def test_empty_list_is_a_noop(self, isolated_documents, tmp_path):
        cfg.linked_roots = {"keep": str(tmp_path / "keep")}
        unregister_added_roots([])
        assert "keep" in cfg.linked_roots

    def test_drops_skip_records_under_the_root(self, isolated_documents, tmp_path):
        from lilbee.app.ingest import register_sources
        from lilbee.data.ingest.skip_marker import (
            load_skip_markers,
            load_skip_reasons,
            write_skip_markers,
            write_skip_reasons,
        )

        scan = tmp_path / "scan.pdf"
        scan.write_bytes(b"")
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_bytes(b"")
        register_sources([scan, corpus])
        other = cfg.documents_dir / "other.md"
        other.write_bytes(b"")
        write_skip_markers(
            cfg.data_root, {"scan.pdf": "h1", "corpus/a.txt": "h2", "other.md": "h3"}
        )
        write_skip_reasons(
            cfg.data_root, {"scan.pdf": "no text", "corpus/a.txt": "no text", "other.md": "x"}
        )

        unregister_added_roots(["scan.pdf", "corpus"])

        assert load_skip_markers(cfg.data_root) == {"other.md": "h3"}
        assert load_skip_reasons(cfg.data_root) == {"other.md": "x"}
        assert cfg.linked_roots == {}


class TestDoAddCancelCleanup:
    """When the sync under /add raises (cancel or crash), the root it registered
    must be un-registered so the next sync does not silently re-ingest it."""

    def test_a_failing_sync_drops_the_root(self, isolated_documents, tmp_path):
        from unittest.mock import MagicMock, patch

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock()
        reporter.is_set.return_value = False  # a failure, not a cancel
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]

        source = tmp_path / "corpus"
        source.mkdir()
        (source / "a.txt").write_text("big file contents")

        def _run(coro):
            coro.close()
            raise OSError("disk gone")

        with (
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            pytest.raises(OSError, match="disk gone"),
        ):
            screen._do_add([source], reporter)

        # The root registered by this /add must be gone after the failure.
        assert "corpus" not in cfg.linked_roots
        assert (source / "a.txt").exists()  # source bytes never touched

    def test_relocated_result_notifies(self, isolated_documents, tmp_path):
        # A successful sync that relocated a source notifies the user with the
        # relocated count (chat.py relocated branch).
        from unittest.mock import MagicMock, patch

        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.data.ingest import SyncResult

        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock()
        reporter.is_set.return_value = False  # a failure, not a cancel
        screen.notify = MagicMock()

        source = tmp_path / "corpus"
        source.mkdir()
        (source / "a.txt").write_text("moved content")
        relocated_result = SyncResult(
            added=[], updated=[], removed=[], unchanged=0, relocated=["corpus/a.txt"]
        )

        def _run(coro):
            coro.close()
            return relocated_result

        with (
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            patch("lilbee.cli.tui.screens.chat.call_from_thread") as cft,
        ):
            screen._do_add([source], reporter)

        sent = [c.args[2] for c in cft.call_args_list if len(c.args) >= 3]
        assert msg.CMD_ADD_RELOCATED.format(count=1) in sent

    def test_sync_result_failed_triggers_cleanup(self, isolated_documents, tmp_path):
        """A SyncResult with failed entries must also un-register the root.

        Without this, a failing sync would leave the root registered, ready for
        the next sync to re-ingest.
        """
        from unittest.mock import MagicMock, patch

        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.data.ingest import SyncResult

        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock()
        reporter.is_set.return_value = False  # a failure, not a cancel
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]

        source = tmp_path / "corpus"
        source.mkdir()
        (source / "a.txt").write_text("hello")

        failing_result = SyncResult(
            added=[], updated=[], removed=[], unchanged=0, failed=["corpus/a.txt"]
        )

        def _run(coro):
            coro.close()
            return failing_result

        with (
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            pytest.raises(RuntimeError, match="Sync failed"),
        ):
            screen._do_add([source], reporter)

        assert "corpus" not in cfg.linked_roots
        assert (source / "a.txt").exists()

    def test_failed_add_leaves_no_skip_record(self, isolated_documents, tmp_path):
        """The skip record the failed sync wrote for the added file goes with its root."""
        from unittest.mock import MagicMock, patch

        from lilbee.app.status import held_out_sources
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.data.ingest import SyncResult
        from lilbee.data.ingest.skip_marker import (
            load_skip_markers,
            load_skip_reasons,
            write_skip_markers,
            write_skip_reasons,
        )

        screen = ChatScreen.__new__(ChatScreen)
        source = tmp_path / "scan.pdf"
        source.write_bytes(b"")

        def _sync_that_skips(coro):
            coro.close()
            write_skip_markers(cfg.data_root, {"scan.pdf": "e3b0"})
            write_skip_reasons(cfg.data_root, {"scan.pdf": "no text extracted (0 chunks)"})
            return SyncResult(skipped=["scan.pdf"])

        with (
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_sync_that_skips),
            patch("lilbee.cli.tui.screens.chat.call_from_thread"),
            pytest.raises(RuntimeError, match=r"scan\.pdf"),
        ):
            screen._do_add([source], MagicMock(**{"is_set.return_value": False}))

        assert cfg.linked_roots == {}
        assert load_skip_markers(cfg.data_root) == {}
        assert load_skip_reasons(cfg.data_root) == {}
        assert held_out_sources() == ([], 0)
        assert source.exists()


@pytest.mark.parametrize(
    ("registered", "result", "expected"),
    [
        pytest.param(["corpus"], {"added": ["corpus/a.py"]}, True, id="file_under_root"),
        pytest.param(["scan.pdf"], {"added": ["scan.pdf"]}, True, id="single_file_root"),
        pytest.param(["corpus"], {"updated": ["corpus/a.py"]}, True, id="updated_counts"),
        pytest.param(["corpus"], {"relocated": ["corpus/a.py"]}, True, id="relocated_counts"),
        pytest.param(
            ["scan.pdf"], {"added": ["other.md"], "unchanged": 5}, False, id="other_sources_only"
        ),
        pytest.param(
            ["corpus"], {"added": ["corpus2/a.py"]}, False, id="prefix_not_a_path_boundary"
        ),
        pytest.param(["corpus"], {}, False, id="nothing_indexed"),
    ],
)
def test_add_indexed_anything(registered, result, expected):
    from lilbee.cli.tui.screens.chat_helpers import add_indexed_anything
    from lilbee.data.types import SyncResult

    assert add_indexed_anything(registered, SyncResult(**result)) is expected


@pytest.mark.parametrize(
    ("not_added", "expected"),
    [
        (["scan.pdf"], "Add cancelled. scan.pdf was not added."),
        (["a.pdf", "b.pdf"], "Add cancelled. a.pdf, b.pdf were not added."),
        ([], "Sync cancelled."),
    ],
)
def test_add_rollback_message_names_what_was_not_added(not_added, expected) -> None:
    from lilbee.app.ingest import AddRollback

    assert AddRollback(not_added).message("Sync cancelled.") == expected


@pytest.mark.parametrize(
    ("exc", "expected", "logged"),
    [
        pytest.param(
            OSError(28, "No space left"),
            "It also hit an error: [Errno 28] No space left.",
            True,
            id="error",
        ),
        pytest.param(RuntimeError(), "It also hit an error: RuntimeError.", True, id="no_text"),
        pytest.param(asyncio.CancelledError(), "Sync cancelled.", False, id="asyncio_cancel"),
        pytest.param(TaskCancelledError("x"), "Sync cancelled.", False, id="task_cancel"),
        pytest.param(KeyboardInterrupt(), "Sync cancelled.", False, id="ctrl_c"),
    ],
)
def test_an_error_after_the_stop_is_named_and_logged_but_a_cancel_is_not(
    caplog, exc, expected, logged
) -> None:
    from lilbee.app.ingest import AddRollback

    rollback = AddRollback()
    with caplog.at_level(logging.WARNING, logger="lilbee.app.ingest"):
        rollback.note_error(exc)
    assert rollback.message("Sync cancelled.").endswith(expected)
    assert [r.exc_info[1] for r in _cancel_error_records(caplog)] == ([exc] if logged else [])


def _screen_text(app) -> str:
    """The text the terminal shows for the current screen."""
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


class _ParkedSync:
    """A sync that parks until its cancel is set, then raises the cancel."""

    def __init__(self) -> None:
        import threading

        self.syncing = threading.Event()
        self.unwound = threading.Event()
        self.registered_during_sync: list[dict[str, str]] = []

    async def run(self, *, cancel, **_kwargs):
        import asyncio

        self.registered_during_sync.append(dict(cfg.linked_roots))
        self.syncing.set()
        try:
            while not cancel.is_set():
                await asyncio.sleep(0.02)
            raise asyncio.CancelledError
        finally:
            self.unwound.set()


def _chat_app():
    """A LilbeeApp host that opens on the chat screen."""
    from textual.app import ComposeResult

    from lilbee.cli.tui.screens.chat import ChatScreen
    from tests._lilbee_app_test_host import LilbeeAppHost

    class _ChatApp(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield from ()

        def on_mount(self) -> None:
            self.push_screen(ChatScreen())

    return _ChatApp()


async def _start_add(app, pilot, scan, parked: _ParkedSync):
    """Type /add for *scan* on the chat screen and return its task once the sync is running."""
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.task_queue import TaskType
    from lilbee.cli.tui.widgets.chat_input import ChatInput

    await wait_until(pilot, lambda: isinstance(app.screen, ChatScreen))
    prompt = app.screen.query_one(ChatInput)
    prompt.focus()
    prompt.value = f"/add {scan}"
    await pilot.press("enter")
    # The worker thread starts the sync after file-locked writes no pause drives.
    assert await wait_until(pilot, parked.syncing.is_set, timeout=_SETTLE_SECONDS)
    (task,) = [t for t in app.task_bar.queue.active_tasks if t.task_type == TaskType.ADD.value]
    return task


@pytest.mark.asyncio
async def test_quitting_the_tui_mid_add_keeps_the_source_for_the_next_sync(
    isolated_documents, tmp_path
) -> None:
    """Quitting stops the add as an exit, not a user cancel, so its source stays registered."""
    import asyncio
    from unittest.mock import MagicMock, patch

    from lilbee.app.services import set_services
    from lilbee.cli.tui.task_queue import CancelOrigin, TaskStatus

    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.4")
    parked = _ParkedSync()
    services = MagicMock()
    services.store.get_sources.return_value = []
    set_services(services)
    app = _chat_app()
    try:
        with patch("lilbee.data.ingest.sync", side_effect=parked.run):
            async with app.run_test() as pilot:
                task = await _start_add(app, pilot, scan, parked)
                await pilot.press("ctrl+c")
                assert await wait_until(pilot, lambda: app.return_code == 0)  # the app quit
                assert task.status is TaskStatus.ACTIVE  # quitting alone stops nothing
            # run_tui stops the tasks this way once the app has exited, off the app's loop.
            await asyncio.to_thread(app.task_bar.stop_all)
    finally:
        set_services(None)
    assert parked.unwound.is_set()
    assert cfg.linked_roots == {"scan.pdf": str(scan.resolve())}  # the next sync resumes it
    assert task.status is TaskStatus.CANCELLED
    assert task.cancel_origin is CancelOrigin.EXIT


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)], ids=["wide", "narrow"])
async def test_a_cancelled_add_row_names_the_file_it_did_not_add(
    isolated_documents, tmp_path, caplog, size
) -> None:
    """Cancelling the /add task un-registers the unfinished file and the row says so."""
    from unittest.mock import MagicMock, patch

    from lilbee.app.services import set_services
    from lilbee.cli.tui.screens.task_center import TaskCenter
    from lilbee.cli.tui.task_queue import CancelOrigin, TaskStatus

    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.4")
    expected = "Add cancelled. scan.pdf was not added."
    parked = _ParkedSync()
    services = MagicMock()
    services.store.get_sources.return_value = []
    set_services(services)
    app = _chat_app()
    try:
        async with app.run_test(size=size) as pilot:
            with patch("lilbee.data.ingest.sync", side_effect=parked.run):
                task = await _start_add(app, pilot, scan, parked)
                app.task_bar.cancel_task(task.task_id)
                assert await wait_until(pilot, lambda: task.detail == expected)
            assert task.status is TaskStatus.CANCELLED
            assert task.cancel_origin is CancelOrigin.USER
            app.push_screen(TaskCenter())
            assert await wait_until(pilot, lambda: expected in _screen_text(app))
    finally:
        set_services(None)
    assert parked.registered_during_sync == [{"scan.pdf": str(scan.resolve())}]
    assert cfg.linked_roots == {}  # the next sync does not resume it
    assert _cancel_error_records(caplog) == []  # a plain cancel is no error


_RESUME = "Sync cancelled. Press S to resume."
_SETTLE_SECONDS = 10.0
_NOT_ADDED = "Add cancelled. corpus was not added."


def _cancel_the_add(app, origin, stopped: list) -> None:
    """Stop the running /add as the Task Center cancel (USER) or as the app exit (EXIT)."""
    from lilbee.cli.tui.task_queue import CancelOrigin, TaskType

    (task,) = [t for t in app.task_bar.queue.active_tasks if t.task_type == TaskType.ADD.value]
    stopped.append(task)
    if origin is CancelOrigin.USER:
        app.task_bar.cancel_task(task.task_id)
    else:
        app.task_bar.queue.cancel(task.task_id, CancelOrigin.EXIT)


def _stage_patch(stage: str, stop):
    """Patch the add so *stop* runs at *stage*, then the real step continues."""
    from unittest.mock import patch

    from lilbee.app import ingest as app_ingest
    from lilbee.data.ingest import pipeline

    if stage == "registration":
        real_register = app_ingest.register_sources

        def _register(paths, **kwargs):
            result = real_register(paths, **kwargs)
            stop()
            return result

        return patch.object(app_ingest, "register_sources", _register)
    if stage == "planning":
        real_discover = pipeline.discover_corpus

        def _discover(*args, **kwargs):
            stop()
            return real_discover(*args, **kwargs)

        return patch.object(pipeline, "discover_corpus", _discover)
    real_passes = pipeline._run_post_ingest_passes

    async def _passes(*args, **kwargs):
        stop()
        return await real_passes(*args, **kwargs)

    return patch.object(pipeline, "_run_post_ingest_passes", _passes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "origin_name", "expected_detail", "kept"),
    [
        pytest.param("registration", "USER", _NOT_ADDED, False, id="registration-user"),
        pytest.param("registration", "EXIT", _RESUME, True, id="registration-exit"),
        pytest.param("planning", "USER", _NOT_ADDED, False, id="planning-user"),
        pytest.param("planning", "EXIT", _RESUME, True, id="planning-exit"),
        pytest.param("post_ingest", "USER", _RESUME, True, id="post_ingest-user"),
        pytest.param("post_ingest", "EXIT", _RESUME, True, id="post_ingest-exit"),
    ],
)
async def test_a_stopped_add_follows_one_rule_whatever_the_stage(
    isolated_documents, tmp_path, stage, origin_name, expected_detail, kept
) -> None:
    """A user cancel drops only a root with no finished file; an exit keeps every root."""
    from lilbee.app.services import set_services
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.task_queue import CancelOrigin, TaskStatus
    from lilbee.cli.tui.widgets.chat_input import ChatInput
    from tests._ingesting_services import ingesting_services

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.txt").write_text("hello world " * 50, encoding="utf-8")
    services, sources = ingesting_services()
    set_services(services)
    app = _chat_app()
    stopped: list = []
    try:
        with _stage_patch(stage, lambda: _cancel_the_add(app, CancelOrigin[origin_name], stopped)):
            async with app.run_test() as pilot:
                assert await wait_until(
                    pilot, lambda: isinstance(app.screen, ChatScreen), timeout=_SETTLE_SECONDS
                )
                prompt = app.screen.query_one(ChatInput)
                prompt.focus()
                prompt.value = f"/add {corpus}"
                await pilot.press("enter")
                assert await wait_until(pilot, lambda: bool(stopped), timeout=_SETTLE_SECONDS)
                (task,) = stopped
                assert await wait_until(
                    pilot, lambda: task.status is TaskStatus.CANCELLED, timeout=_SETTLE_SECONDS
                )
                assert await wait_until(
                    pilot, lambda: task.detail == expected_detail, timeout=_SETTLE_SECONDS
                ), task.detail
    finally:
        set_services(None)
    assert ("corpus/notes.txt" in sources) is (stage == "post_ingest")
    assert ("corpus" in cfg.linked_roots) is kept


class _SyncFailingAfterTheStop(_ParkedSync):
    """A sync that parks until its cancel is set, then fails on a full disk."""

    async def run(self, *, cancel, **_kwargs):
        import asyncio
        import errno

        self.syncing.set()
        try:
            while not cancel.is_set():
                await asyncio.sleep(0.02)
            raise OSError(errno.ENOSPC, "No space left on device")
        finally:
            self.unwound.set()


_DISK_FULL_NOTE = " It also hit an error: [Errno 28] No space left on device."
_CLOSED_NOTE = " It also hit an error: store is closed."


def _cancel_error_records(caplog) -> list[logging.LogRecord]:
    """The WARNING-or-higher records the add's cancel scope logged with an exception."""
    return [
        r
        for r in caplog.records
        if r.name == "lilbee.app.ingest" and r.levelno >= logging.WARNING and r.exc_info
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("origin_name", "expected_detail", "kept"),
    [
        pytest.param("EXIT", _RESUME + _DISK_FULL_NOTE, True, id="exit"),
        pytest.param(
            "USER", "Add cancelled. scan.pdf was not added." + _DISK_FULL_NOTE, False, id="user"
        ),
    ],
)
async def test_a_sync_failing_after_a_stop_is_the_stop(
    isolated_documents, tmp_path, caplog, origin_name, expected_detail, kept
) -> None:
    """An error after the stop follows the stop's rule, and the row and the log name it."""
    from unittest.mock import MagicMock, patch

    from lilbee.app.services import set_services
    from lilbee.cli.tui.task_queue import CancelOrigin, TaskStatus

    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.4")
    failing = _SyncFailingAfterTheStop()
    services = MagicMock()
    services.store.get_sources.return_value = []
    set_services(services)
    app = _chat_app()
    stopped: list = []
    try:
        with (
            caplog.at_level(logging.WARNING, logger="lilbee.app.ingest"),
            patch("lilbee.data.ingest.sync", side_effect=failing.run),
        ):
            async with app.run_test() as pilot:
                task = await _start_add(app, pilot, scan, failing)
                _cancel_the_add(app, CancelOrigin[origin_name], stopped)
                assert await wait_until(
                    pilot, lambda: task.status is TaskStatus.CANCELLED, timeout=_SETTLE_SECONDS
                )
                assert await wait_until(
                    pilot, lambda: task.detail == expected_detail, timeout=_SETTLE_SECONDS
                ), task.detail
    finally:
        set_services(None)
    assert failing.unwound.is_set()
    assert ("scan.pdf" in cfg.linked_roots) is kept
    assert [type(r.exc_info[1]) for r in _cancel_error_records(caplog)] == [OSError]


def test_the_rollback_stamps_come_from_the_rows_not_a_stale_cached_map(
    tmp_path, monkeypatch
) -> None:
    """A reader straddling a write caches a stale map; the rollback still sees the write."""
    import threading

    from lilbee.app.ingest import indexed_stamps
    from lilbee.app.services import set_services
    from lilbee.data.store import Store
    from tests.conftest import make_mock_services

    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    store = Store(cfg)
    store.upsert_source("old.txt", "h0", 1)
    real_get = store.get_sources
    read_done, resume = threading.Event(), threading.Event()

    def _straddling_get():
        rows = real_get()
        read_done.set()
        resume.wait(5)
        return rows

    monkeypatch.setattr(store, "get_sources", _straddling_get)
    reader = threading.Thread(target=store.source_ingested_at_map)
    reader.start()
    assert read_done.wait(5)
    monkeypatch.setattr(store, "get_sources", real_get)
    store.upsert_source("new.txt", "h1", 1)
    resume.set()
    reader.join(5)
    assert "new.txt" not in store.source_ingested_at_map()  # the cache holds the stale map
    set_services(make_mock_services(store=store))
    try:
        assert set(indexed_stamps(["new.txt", "old.txt"])) == {"new.txt", "old.txt"}
    finally:
        set_services(None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("origin_name", "expected_detail", "kept"),
    [
        pytest.param("EXIT", _RESUME + _CLOSED_NOTE, True, id="exit"),
        pytest.param(
            "USER", "Add cancelled. scan.pdf was not added." + _CLOSED_NOTE, False, id="user"
        ),
    ],
)
async def test_a_stop_while_the_add_reads_its_snapshot_follows_the_stop(
    isolated_documents, tmp_path, caplog, origin_name, expected_detail, kept
) -> None:
    """The store read before the sync fails after the stop; the row and the log name it."""
    from unittest.mock import MagicMock

    from lilbee.app.services import set_services
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.task_queue import CancelOrigin, TaskStatus
    from lilbee.cli.tui.widgets.chat_input import ChatInput

    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.4")
    app = _chat_app()
    stopped: list = []
    reads: list[int] = []

    def _stopped_then_closed():
        reads.append(1)
        if len(reads) == 1:
            _cancel_the_add(app, CancelOrigin[origin_name], stopped)
            raise RuntimeError("store is closed")
        return []

    services = MagicMock()
    services.store.get_sources.side_effect = _stopped_then_closed
    set_services(services)
    try:
        with caplog.at_level(logging.WARNING, logger="lilbee.app.ingest"):
            async with app.run_test() as pilot:
                assert await wait_until(
                    pilot, lambda: isinstance(app.screen, ChatScreen), timeout=_SETTLE_SECONDS
                )
                prompt = app.screen.query_one(ChatInput)
                prompt.focus()
                prompt.value = f"/add {scan}"
                await pilot.press("enter")
                assert await wait_until(pilot, lambda: bool(stopped), timeout=_SETTLE_SECONDS)
                (task,) = stopped
                assert await wait_until(
                    pilot, lambda: task.status is TaskStatus.CANCELLED, timeout=_SETTLE_SECONDS
                )
                assert await wait_until(
                    pilot, lambda: task.detail == expected_detail, timeout=_SETTLE_SECONDS
                ), task.detail
    finally:
        set_services(None)
    assert ("scan.pdf" in cfg.linked_roots) is kept
    assert [type(r.exc_info[1]) for r in _cancel_error_records(caplog)] == [RuntimeError]


def test_a_snapshot_read_that_fails_with_no_stop_fails_the_add(isolated_documents, tmp_path):
    """Only the snapshot read fails; the sync after it would succeed."""
    from unittest.mock import MagicMock, patch

    from lilbee.app.services import set_services
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.ingest import SyncResult

    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text("hello", encoding="utf-8")
    services = MagicMock()
    services.store.get_sources.side_effect = [RuntimeError("store is closed"), []]
    set_services(services)
    screen = ChatScreen.__new__(ChatScreen)
    synced: list[bool] = []

    def _succeeding_sync(coro):
        coro.close()
        synced.append(True)
        return SyncResult(added=["corpus/a.txt"])

    try:
        with (
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_succeeding_sync),
            patch("lilbee.cli.tui.screens.chat.call_from_thread"),
            pytest.raises(RuntimeError, match="store is closed"),
        ):
            screen._do_add([source], MagicMock(**{"is_set.return_value": False}))
    finally:
        set_services(None)
    assert synced == []
    assert cfg.linked_roots == {}
