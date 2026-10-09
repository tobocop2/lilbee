"""Coverage for the chat screen's TaskBarController-backed flows.

These exercise the public entry points (``_cmd_add``, ``_start_crawl``,
``_run_sync``) and the worker bodies (``_do_add``, ``_do_crawl``,
``_do_sync``) that the old screen-owned @work paths no longer cover.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.catalog import CatalogModel
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.task_queue import TaskStatus, TaskType
from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter, TaskBarController
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import await_chat, pump_until, ready_services

# The longest a test waits for work a task worker thread does.
_SETTLE_SECONDS = 10.0


@pytest.fixture(autouse=True)
def _gate_releases_at_once():
    """Bind a ready chat role so the startup gate hands over on mount."""
    with ready_services():
        yield


def _fake_model() -> CatalogModel:
    return CatalogModel(
        hf_repo="o/r-GGUF",
        gguf_filename="f.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task="chat",
    )


@pytest.mark.asyncio
async def test_reporter_task_id_property_exposes_id() -> None:
    """ProgressReporter.task_id returns the id it was bound to."""
    app = LilbeeApp()
    async with app.run_test():
        controller = TaskBarController(app)
        tid = controller.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        reporter = ProgressReporter(controller, tid)
        assert reporter.task_id == tid


@pytest.mark.asyncio
async def test_on_success_exception_is_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    """An on_success failure leaves the task DONE and is reported in the log."""
    caplog.set_level("WARNING", logger="lilbee.cli.tui.widgets.task_bar_controller")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await await_chat(app, pilot)
        controller = TaskBarController(app)

        def _oops() -> None:
            raise RuntimeError("boom")

        task_id = controller.start_task("demo", TaskType.SYNC, lambda r: None, on_success=_oops)

        def _finalized() -> bool:
            task = controller.queue.get_task(task_id)
            return (
                task is not None
                and task.status == TaskStatus.DONE
                and any("on_success" in r.message for r in caplog.records)
            )

        assert await pump_until(pilot, _finalized), "the task never finalized"

    task = controller.queue.get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.DONE
    raised = [r for r in caplog.records if "on_success" in r.message]
    assert raised, "the swallowed on_success failure was never logged"
    assert "boom" in raised[0].exc_text


@pytest.mark.asyncio
async def test_queue_unsubscribe_removes_callback() -> None:
    """TaskQueue.unsubscribe removes a previously registered callback."""
    from lilbee.cli.tui.task_queue import TaskQueue

    q = TaskQueue()
    called = []

    def cb() -> None:
        called.append(1)

    q.subscribe(cb)
    q.unsubscribe(cb)
    q.enqueue(lambda: None, "demo", TaskType.SYNC.value)
    assert called == []


@pytest.mark.asyncio
async def test_do_add_reports_progress_and_runs_sync(tmp_path: Path) -> None:
    """_do_add copies files, reports indeterminate progress, and runs sync."""

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None

        reporter = MagicMock(spec=ProgressReporter)

        from lilbee.app.ingest import RegisterResult

        reg_result = RegisterResult(registered=[src.name])

        import threading as _th

        exc: list[Exception] = []

        from lilbee.data.ingest import SyncResult

        def _worker() -> None:
            try:
                with (
                    patch("lilbee.app.ingest.register_sources", return_value=reg_result),
                    patch("lilbee.data.ingest.sync", new=MagicMock(return_value=None)),
                    patch(
                        "lilbee.runtime.asyncio_loop.run", new=MagicMock(return_value=SyncResult())
                    ),
                ):
                    screen._do_add([src], reporter)
            except Exception as e:  # pragma: no cover
                exc.append(e)

        t = _th.Thread(target=_worker, daemon=True)
        t.start()
        for _ in range(40):
            await pilot.pause()
            if reporter.update.call_count >= 2:
                break
        assert not exc, f"_do_add raised: {exc[0]}"
        assert reporter.update.call_count >= 2


@pytest.mark.asyncio
async def test_do_add_force_propagates_to_register_sources(tmp_path: Path) -> None:
    """After overwrite-confirm ``_do_add`` must pass ``force=True`` through."""

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None

        reporter = MagicMock(spec=ProgressReporter)

        from lilbee.app.ingest import RegisterResult

        reg_result = RegisterResult(registered=[src.name])

        import threading as _th

        exc: list[Exception] = []
        mock_register = MagicMock(return_value=reg_result)

        def _worker() -> None:
            try:
                with (
                    patch("lilbee.app.ingest.register_sources", new=mock_register),
                    patch(
                        "lilbee.runtime.asyncio_loop.run",
                        new=MagicMock(
                            return_value=__import__(
                                "lilbee.data.ingest", fromlist=["SyncResult"]
                            ).SyncResult()
                        ),
                    ),
                ):
                    screen._do_add([src], reporter, force=True)
            except Exception as e:  # pragma: no cover
                exc.append(e)

        t = _th.Thread(target=_worker, daemon=True)
        t.start()
        for _ in range(40):
            await pilot.pause()
            if mock_register.called:
                break
        assert not exc, f"_do_add raised: {exc[0]}"
        assert mock_register.called
        _, kwargs = mock_register.call_args
        assert kwargs.get("force") is True


@pytest.mark.asyncio
async def test_do_add_passes_taken_names_through_register_result(tmp_path: Path) -> None:
    """_do_add observes register_sources' name_taken list and keeps running."""

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None

        reporter = MagicMock(spec=ProgressReporter)

        from lilbee.app.ingest import RegisterResult

        reg_result = RegisterResult(registered=[src.name], name_taken=["exists.pdf"])

        import threading as _th

        exc: list[Exception] = []
        mock_register = MagicMock(return_value=reg_result)

        def _worker() -> None:
            try:
                with (
                    patch("lilbee.app.ingest.register_sources", new=mock_register),
                    patch(
                        "lilbee.runtime.asyncio_loop.run",
                        new=MagicMock(
                            return_value=__import__(
                                "lilbee.data.ingest", fromlist=["SyncResult"]
                            ).SyncResult()
                        ),
                    ),
                ):
                    screen._do_add([src], reporter)
            except Exception as e:  # pragma: no cover
                exc.append(e)

        t = _th.Thread(target=_worker, daemon=True)
        t.start()
        # Worker may block on call_from_thread (app loop is pinned in the
        # test harness); we only need to confirm register_sources was reached.
        for _ in range(40):
            await pilot.pause()
            if mock_register.called:
                break
        assert mock_register.called
        assert reporter.update.call_count >= 1


def test_do_crawl_reports_setup_progress() -> None:
    """_do_crawl wires SETUP_START and SETUP_PROGRESS through reporter.update."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.runtime.progress import EventType, SetupProgressEvent

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_crawl(
        url,
        *,
        depth,
        max_pages,
        on_progress,
        cancel=None,
        quiet=False,
        include_subdomains=False,
        render_mode=None,
    ):
        on_progress(EventType.SETUP_START, object())
        on_progress(
            EventType.SETUP_PROGRESS,
            SetupProgressEvent(
                component="chromium", downloaded_bytes=5_000_000, total_bytes=10_000_000
            ),
        )
        on_progress(
            EventType.SETUP_PROGRESS,
            SetupProgressEvent(component="chromium", downloaded_bytes=1_000_000, total_bytes=None),
        )
        return []

    exc: list[Exception] = []

    def _worker() -> None:
        try:
            # The screen is unmounted (__new__ without __init__), so the trailing
            # success notify cannot reach a live app; patch the dispatch so the
            # test isolates _do_crawl's progress wiring.
            with (
                patch("lilbee.cli.tui.screens.chat.call_from_thread"),
                patch("lilbee.crawler.crawl_and_save", side_effect=fake_crawl),
            ):
                screen._do_crawl("https://x", 0, 2, reporter)
        except Exception as e:  # pragma: no cover - re-raised
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not exc, f"worker raised: {exc[0]}"
    assert reporter.update.call_count >= 3


def test_do_crawl_reports_page_progress() -> None:
    """_do_crawl wires CrawlPageEvent through reporter.update."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.runtime.progress import CrawlPageEvent, EventType

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_crawl(
        url,
        *,
        depth,
        max_pages,
        on_progress,
        cancel=None,
        quiet=False,
        include_subdomains=False,
        render_mode=None,
    ):
        on_progress(
            EventType.CRAWL_PAGE,
            CrawlPageEvent(url="https://x/a", current=1, total=2),
        )
        return [Path("/tmp/a")]

    exc: list[Exception] = []

    def _worker() -> None:
        try:
            # The screen is unmounted (__new__ without __init__), so the trailing
            # success notify cannot reach a live app; patch the dispatch so the
            # test isolates _do_crawl's progress wiring.
            with (
                patch("lilbee.cli.tui.screens.chat.call_from_thread"),
                patch("lilbee.crawler.crawl_and_save", side_effect=fake_crawl),
            ):
                screen._do_crawl("https://x", 0, 2, reporter)
        except Exception as e:  # pragma: no cover - re-raised
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not exc, f"worker raised: {exc[0]}"
    assert reporter.update.call_count >= 2


def test_do_crawl_notifies_page_failures() -> None:
    """_do_crawl surfaces per-page failures as a warning notify after the crawl."""
    import threading

    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.runtime.progress import CrawlPageFailedEvent, EventType

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_crawl(
        url,
        *,
        depth,
        max_pages,
        on_progress,
        cancel=None,
        quiet=False,
        include_subdomains=False,
        render_mode=None,
    ):
        on_progress(
            EventType.CRAWL_PAGE_FAILED,
            CrawlPageFailedEvent(url="https://x/a", reason="403 Forbidden"),
        )
        return []

    exc: list[Exception] = []
    cft_calls: list[tuple] = []

    def _worker() -> None:
        try:
            # The screen is unmounted (__new__ without __init__), so notifies
            # cannot reach a live app; capture the dispatch instead.
            with (
                patch(
                    "lilbee.cli.tui.screens.chat.call_from_thread",
                    side_effect=lambda *a, **kw: cft_calls.append((a, kw)),
                ),
                patch("lilbee.crawler.crawl_and_save", side_effect=fake_crawl),
            ):
                screen._do_crawl("https://x", 0, 2, reporter)
        except Exception as e:  # pragma: no cover - re-raised
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not exc, f"worker raised: {exc[0]}"
    expected = msg.CMD_CRAWL_PAGES_FAILED.format(count=1, reason="403 Forbidden")
    assert any(expected in a and kw.get("severity") == "warning" for a, kw in cft_calls)


@contextlib.contextmanager
def _chat_screen_with_task_bar():
    """A bare ChatScreen whose read-only ``_task_bar`` property yields a MagicMock."""
    from unittest.mock import PropertyMock

    from lilbee.cli.tui.screens.chat import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
    bar = MagicMock()
    with patch.object(ChatScreen, "_task_bar", new_callable=PropertyMock, return_value=bar):
        yield screen, bar


def test_start_crawl_browser_persists_mode_and_bootstraps_chromium(monkeypatch) -> None:
    """Browser mode differing from config persists the choice and ensures Chromium first."""
    from lilbee.core.config import cfg
    from lilbee.core.config.enums import CrawlRenderMode

    monkeypatch.setattr(cfg, "crawl_render_mode", CrawlRenderMode.HTTP)
    with (
        _chat_screen_with_task_bar() as (screen, bar),
        patch("lilbee.app.settings.apply_settings_update") as mock_apply,
    ):
        screen._start_crawl("https://x", 0, 5, render_mode=CrawlRenderMode.BROWSER)
    mock_apply.assert_called_once_with({"crawl_render_mode": "browser"})
    bar.ensure_chromium.assert_called_once()
    bar.start_task.assert_not_called()


def test_start_crawl_http_skips_chromium_and_persists(monkeypatch) -> None:
    """HTTP mode differing from config persists and kicks off without Chromium bootstrap."""
    from lilbee.core.config import cfg
    from lilbee.core.config.enums import CrawlRenderMode

    monkeypatch.setattr(cfg, "crawl_render_mode", CrawlRenderMode.BROWSER)
    with (
        _chat_screen_with_task_bar() as (screen, bar),
        patch("lilbee.app.settings.apply_settings_update") as mock_apply,
    ):
        screen._start_crawl("https://x", 0, 5, render_mode=CrawlRenderMode.HTTP)
    mock_apply.assert_called_once_with({"crawl_render_mode": "http"})
    bar.ensure_chromium.assert_not_called()
    bar.start_task.assert_called_once()


def test_start_crawl_none_uses_config_without_persisting(monkeypatch) -> None:
    """render_mode=None inherits cfg and does not re-persist the setting."""
    from lilbee.core.config import cfg
    from lilbee.core.config.enums import CrawlRenderMode

    monkeypatch.setattr(cfg, "crawl_render_mode", CrawlRenderMode.HTTP)
    with (
        _chat_screen_with_task_bar() as (screen, bar),
        patch("lilbee.app.settings.apply_settings_update") as mock_apply,
    ):
        screen._start_crawl("https://x", 0, 5)
    mock_apply.assert_not_called()
    bar.ensure_chromium.assert_not_called()
    bar.start_task.assert_called_once()


def test_persist_crawl_render_mode_swallows_write_errors() -> None:
    """A failed settings write is logged, not raised, so the crawl still proceeds."""
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.core.config.enums import CrawlRenderMode

    screen = ChatScreen.__new__(ChatScreen)
    with patch("lilbee.app.settings.apply_settings_update", side_effect=OSError("disk full")):
        screen._persist_crawl_render_mode(CrawlRenderMode.BROWSER)


def test_do_sync_reports_file_and_embed_progress() -> None:
    """_do_sync routes FileStart / FileDone / Embed events through reporter.update."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.runtime.progress import EmbedEvent, EventType, FileDoneEvent, FileStartEvent

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    from lilbee.data.ingest import SyncResult

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        on_progress(
            EventType.FILE_START,
            FileStartEvent(file="a.pdf", current_file=1, total_files=2),
        )
        on_progress(EventType.FILE_DONE, FileDoneEvent(file="a.pdf", status="ok", chunks=5))
        on_progress(EventType.EMBED, EmbedEvent(file="a.pdf", chunk=1, total_chunks=10))
        return SyncResult()

    exc: list[Exception] = []

    def _worker() -> None:
        try:
            with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
                screen._do_sync(reporter)
        except Exception as e:  # pragma: no cover - re-raised
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not exc, f"worker raised: {exc[0]}"
    assert reporter.update.call_count >= 3


def test_do_sync_done_event_reports_completion() -> None:
    """_do_sync routes EventType.SYNC_DONE through reporter.update at 100% so the
    Task Center row flashes 'just-completed' (regression for bb-7enj)."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.ingest import SyncResult
    from lilbee.runtime.progress import EventType, SyncDoneEvent

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        on_progress(
            EventType.SYNC_DONE,
            SyncDoneEvent(added=3, updated=1, removed=0, failed=0),
        )
        return SyncResult()

    exc: list[Exception] = []

    def _worker() -> None:
        try:
            with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
                screen._do_sync(reporter)
        except Exception as e:  # pragma: no cover - re-raised
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not exc, f"worker raised: {exc[0]}"
    # At least one call should hit pct=100 with indeterminate=False.
    completion_calls = [
        call for call in reporter.update.call_args_list if call.args and call.args[0] == 100
    ]
    assert completion_calls, "no reporter.update(100, ...) call observed"
    last = completion_calls[-1]
    assert last.kwargs.get("indeterminate") is False
    # Detail string shows total count: added + updated + removed (failed dropped).
    from lilbee.cli.tui import messages as msg

    assert str(last.args[1]) == msg.SYNC_STATUS_DONE.format(count=4)


def test_do_sync_reports_what_pruning_dropped() -> None:
    """/prune-ignored must say how many documents it removed, or that none matched."""
    import threading

    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.ingest import SyncResult

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    notes: list[str] = []

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        assert prune_ignored is True
        return SyncResult(removed=["vendor/lib.min.js"])

    def _worker() -> None:
        with (
            patch("lilbee.data.ingest.sync", side_effect=fake_sync),
            patch(
                "lilbee.cli.tui.screens.chat.call_from_thread",
                side_effect=lambda _s, _fn, text, **kw: notes.append(text),
            ),
        ):
            screen._do_sync(reporter, prune_ignored=True)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert notes == [msg.prune_ignored_message(1)]
    assert "1" in notes[0]


def test_prune_ignored_message_distinguishes_nothing_matched() -> None:
    """A prune that matched nothing must not read as a successful removal."""
    from lilbee.cli.tui import messages as msg

    assert msg.prune_ignored_message(0) == msg.CMD_PRUNE_IGNORED_NONE
    assert "2" in msg.prune_ignored_message(2)


def test_do_sync_raises_on_sync_failed() -> None:
    """bb-vb28 parallel: auto-sync worker raises when SyncResult.failed is non-empty."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.ingest import SyncResult

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        return SyncResult(failed=["broken.pdf"])

    captured: list[Exception] = []

    def _worker() -> None:
        try:
            with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
                screen._do_sync(reporter)
        except Exception as e:
            captured.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert captured, "worker should have raised"
    assert isinstance(captured[0], RuntimeError)
    assert "broken.pdf" in str(captured[0])


def test_do_sync_reports_a_cancel_with_the_resume_hint() -> None:
    """asyncio.CancelledError becomes a task cancel that carries the resume hint."""
    import threading

    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.runtime.cancellation import TaskCancelledError

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        import asyncio as _asyncio

        raise _asyncio.CancelledError

    captured: list[Exception] = []

    def _worker() -> None:
        try:
            with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
                screen._do_sync(reporter)
        except Exception as e:
            captured.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert captured, "_do_sync should have raised"
    assert isinstance(captured[0], TaskCancelledError)
    assert str(captured[0]) == msg.SYNC_CANCELLED_RESUME


def test_do_sync_hands_the_task_reporter_to_sync_as_its_cancel() -> None:
    """Cancelling the task reaches the sync itself, not only its progress callback."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.ingest import SyncResult

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    received: list[object] = []

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        received.append(cancel)
        return SyncResult()

    def _worker() -> None:
        with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
            screen._do_sync(reporter)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert received == [reporter]


def test_do_add_hands_the_task_reporter_to_sync_as_its_cancel(tmp_path: Path) -> None:
    """Cancelling an /add task reaches the sync itself, not only its progress callback."""
    import threading

    from lilbee.app.ingest import RegisterResult
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.ingest import SyncResult

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    received: list[object] = []

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        received.append(cancel)
        return SyncResult(added=[src.name])

    def _worker() -> None:
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
        with (
            patch(
                "lilbee.app.ingest.register_sources",
                return_value=RegisterResult(registered=[src.name]),
            ),
            patch("lilbee.data.ingest.sync", side_effect=fake_sync),
        ):
            screen._do_add([src], reporter)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert received == [reporter]


@pytest.mark.asyncio
async def test_cmd_add_missing_path_notifies(tmp_path: Path) -> None:
    """_cmd_add on a non-existent path shows an error."""

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        notified: list[str] = []
        screen.notify = lambda *a, **kw: notified.append(str(a[0]))  # type: ignore[assignment]
        screen._cmd_add(str(tmp_path / "nope.pdf"))
        assert any("not found" in n.lower() for n in notified)


@pytest.mark.asyncio
async def test_cmd_add_submits_task_to_controller(tmp_path: Path) -> None:
    """_cmd_add routes real work through TaskBarController.start_task."""

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._cmd_add(str(src))
        assert mock_start.called
        call_args = mock_start.call_args
        assert call_args.args[1] == TaskType.ADD


@pytest.mark.asyncio
async def test_cmd_add_prompts_before_overwriting_existing_file(tmp_path: Path) -> None:
    """A duplicate in documents_dir opens ConfirmDialog; confirm spawns the task."""
    from lilbee.core.config import cfg as _cfg

    # Seed a copy already in documents_dir so _cmd_add detects a duplicate.
    _cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    (_cfg.documents_dir / "doc.pdf").write_bytes(b"existing")

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"new")

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None

        captured_callbacks: list[object] = []
        real_push = app.push_screen

        def _capture_push(screen_or_name, callback=None, **kwargs):  # type: ignore[no-untyped-def]
            captured_callbacks.append(callback)
            return real_push(screen_or_name, callback, **kwargs)

        app.push_screen = _capture_push  # type: ignore[assignment]

        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._cmd_add(str(src))
            # Dialog pushed, task NOT yet submitted.
            assert captured_callbacks, "confirm dialog should have been pushed"
            assert not mock_start.called, "start_task must wait for confirmation"

            # Simulate user confirming: the captured callback runs with True.
            confirm_callback = captured_callbacks[0]
            assert callable(confirm_callback)
            confirm_callback(True)
            assert mock_start.called, "confirmed dialog should spawn the add task"


@pytest.mark.asyncio
async def test_cmd_add_re_adding_registered_path_skips_dialog(tmp_path: Path) -> None:
    """Re-adding the path already registered under its label is idempotent: the
    add proceeds with no overwrite dialog (the label is held by this very file)."""
    from lilbee.core.config import cfg as _cfg

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    _cfg.linked_roots = {"doc.pdf": str(src.resolve())}

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None

        pushed: list[object] = []
        real_push = app.push_screen

        def _capture_push(screen_or_name, callback=None, **kwargs):  # type: ignore[no-untyped-def]
            pushed.append(screen_or_name)
            return real_push(screen_or_name, callback, **kwargs)

        app.push_screen = _capture_push  # type: ignore[assignment]

        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._cmd_add(str(src))
            assert not pushed, "re-adding the same path must not open a confirm dialog"
            assert mock_start.called, "the idempotent add should submit directly"


@pytest.mark.asyncio
async def test_cmd_add_overwrite_rejected_keeps_existing_copy(tmp_path: Path) -> None:
    """When the user answers No to the overwrite dialog, no task is spawned."""
    from lilbee.core.config import cfg as _cfg

    _cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    (_cfg.documents_dir / "doc.pdf").write_bytes(b"existing")

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"new")

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None

        captured_callbacks: list[object] = []
        real_push = app.push_screen

        def _capture_push(screen_or_name, callback=None, **kwargs):  # type: ignore[no-untyped-def]
            captured_callbacks.append(callback)
            return real_push(screen_or_name, callback, **kwargs)

        app.push_screen = _capture_push  # type: ignore[assignment]

        notified: list[str] = []
        screen.notify = lambda *a, **kw: notified.append(str(a[0]))  # type: ignore[assignment]

        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._cmd_add(str(src))
            assert captured_callbacks
            confirm_callback = captured_callbacks[0]
            assert callable(confirm_callback)
            # User rejects the overwrite.
            confirm_callback(False)
            assert not mock_start.called, "start_task must not fire when user declines"
            assert any("kept existing" in n.lower() for n in notified)


@pytest.mark.asyncio
async def test_cmd_add_rejects_when_sync_active(tmp_path: Path) -> None:
    """_cmd_add refuses when another sync is already running."""

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        screen._sync_active = True
        notified: list[str] = []
        screen.notify = lambda *a, **kw: notified.append(str(a[0]))  # type: ignore[assignment]
        screen._cmd_add(str(src))
        assert any("sync in progress" in n.lower() for n in notified)


@pytest.mark.asyncio
async def test_start_crawl_submits_task_to_controller() -> None:
    """_start_crawl routes through TaskBarController.start_task with CRAWL type.

    After bb-wq8g, _start_crawl first calls ensure_chromium which may
    spawn a SETUP task. This test patches chromium_installed=True so
    ensure_chromium short-circuits and the subsequent start_task call
    lands with the CRAWL type.
    """

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        with (
            patch(
                "lilbee.cli.tui.widgets.task_bar_controller.chromium_installed",
                return_value=True,
            ),
            patch.object(app.task_bar, "start_task", return_value="tid") as mock_start,
        ):
            screen._start_crawl("https://x", 0, 5)
        assert mock_start.called
        assert mock_start.call_args.args[1] == TaskType.CRAWL


@pytest.mark.asyncio
async def test_run_sync_submits_task_to_controller() -> None:
    """_run_sync routes through TaskBarController.start_task with SYNC type."""

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._run_sync()
        assert mock_start.called
        assert mock_start.call_args.args[1] == TaskType.SYNC


@pytest.mark.asyncio
async def test_cmd_import_submits_task_to_controller() -> None:
    """_cmd_import routes through TaskBarController.start_task with IMPORT type."""

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._cmd_import("/tmp/pages.parquet")
        assert mock_start.called
        assert mock_start.call_args.args[1] == TaskType.IMPORT


@pytest.mark.asyncio
async def test_cmd_export_submits_task_to_controller() -> None:
    """_cmd_export routes through TaskBarController.start_task with EXPORT type."""

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._cmd_export("/tmp/pages.parquet")
        assert mock_start.called
        assert mock_start.call_args.args[1] == TaskType.EXPORT


@pytest.mark.asyncio
async def test_cmd_import_rejects_when_sync_active() -> None:
    """_cmd_import refuses while an ingest task holds the store."""

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        screen._sync_active = True
        notified: list[str] = []
        screen.notify = lambda *a, **kw: notified.append(str(a[0]))  # type: ignore[assignment]
        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._cmd_import("/tmp/pages.parquet")
        assert not mock_start.called
        assert notified


@pytest.mark.asyncio
async def test_indexing_active_true_during_import_task() -> None:
    """Memory extraction stays gated while an import re-embeds."""

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        tid = app.task_bar.queue.enqueue(lambda: None, "Import x", TaskType.IMPORT.value)
        app.task_bar.queue.advance(TaskType.IMPORT.value)
        assert screen._indexing_active() is True
        app.task_bar.queue.complete_task(tid)
        assert screen._indexing_active() is False


@pytest.mark.asyncio
async def test_indexing_active_true_during_wiki_task() -> None:
    """Wiki builds and draft accepts re-chunk and embed, the contention the gate
    exists to avoid."""

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        tid = app.task_bar.queue.enqueue(lambda: None, "Wikify", TaskType.WIKI.value)
        app.task_bar.queue.advance(TaskType.WIKI.value)
        assert screen._indexing_active() is True
        app.task_bar.queue.complete_task(tid)
        assert screen._indexing_active() is False


@pytest.mark.asyncio
async def test_run_sync_rejects_when_already_active() -> None:
    """_run_sync refuses when another sync is already running."""

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        screen._sync_active = True
        notified: list[str] = []
        screen.notify = lambda *a, **kw: notified.append(str(a[0]))  # type: ignore[assignment]
        screen._run_sync()
        assert any("sync in progress" in n.lower() for n in notified)


@pytest.mark.asyncio
async def test_catalog_enqueue_download_calls_start_download_and_notifies() -> None:
    """Inside a LilbeeApp, _enqueue_download calls start_download + notifies."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(CatalogScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, CatalogScreen)
        notified: list[str] = []
        screen.notify = lambda *a, **kw: notified.append(str(a[0]))  # type: ignore[assignment]
        with patch.object(app.task_bar, "start_download", return_value="tid") as mock_start:
            screen._enqueue_download(_fake_model())
        mock_start.assert_called_once()
        assert any("fake" in n.lower() or "queued" in n.lower() for n in notified)


def test_do_add_on_progress_updates_reporter_on_file_start(tmp_path: Path) -> None:
    """The nested on_progress inside _do_add wires FILE_START to reporter.update."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.runtime.progress import EventType, FileStartEvent

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    from lilbee.app.ingest import RegisterResult

    reg_result = RegisterResult(registered=[src.name])

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        on_progress(
            EventType.FILE_START,
            FileStartEvent(file="a.pdf", current_file=1, total_files=1),
        )

    exc: list[Exception] = []

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with (
                patch("lilbee.app.ingest.register_sources", return_value=reg_result),
                patch("lilbee.data.ingest.sync", side_effect=fake_sync),
            ):
                screen._do_add([src], reporter)
        except Exception as e:  # pragma: no cover
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    # The "Syncing {file}..." update is reached only via the FILE_START branch.
    assert any("Syncing a.pdf" in str(call) for call in reporter.update.call_args_list)


def test_do_add_on_progress_surfaces_per_page_progress(tmp_path: Path) -> None:
    """BATCH_PROGRESS events from the vision-OCR subprocess become per-page reporter updates."""
    import asyncio
    import threading

    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import SyncResult
    from lilbee.runtime.progress import (
        BatchProgressEvent,
        BatchStatus,
        EventType,
        FileStartEvent,
    )

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    from lilbee.app.ingest import RegisterResult

    reg_result = RegisterResult(registered=[src.name])

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        # Per-page rasterization progress fires while the file is being
        # processed (FILE_START has already named it via the relative source
        # name); the BATCH_PROGRESS event itself is emitted by the OCR
        # subprocess pump with the *absolute* path in data.file (see
        # data/ingest/extract.py:_pump_pdf_progress), so the two strings
        # do not match and identity-based dispatch would skip the per-page
        # branch entirely. The realistic shape catches that regression.
        on_progress(
            EventType.FILE_START,
            FileStartEvent(file="a.pdf", current_file=1, total_files=1),
        )
        on_progress(
            EventType.BATCH_PROGRESS,
            BatchProgressEvent(
                file="/abs/path/to/documents/a.pdf",
                status=BatchStatus.RASTERIZING,
                current=2,
                total=10,
            ),
        )
        return SyncResult()

    def _worker() -> None:
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
        with (
            patch("lilbee.app.ingest.register_sources", return_value=reg_result),
            patch("lilbee.data.ingest.sync", side_effect=fake_sync),
            # Run the coroutine inline so on_progress fires; bypass asyncio_loop
            # which may not be primed inside this worker thread (Windows CI).
            patch("lilbee.runtime.asyncio_loop.run", side_effect=lambda coro: asyncio.run(coro)),
        ):
            screen._do_add([src], reporter)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    expected_detail = msg.ADD_PAGE_PROGRESS.format(
        status=BatchStatus.RASTERIZING.capitalize(), current=2, total=10
    )
    page_updates = [call for call in reporter.update.call_args_list if expected_detail in str(call)]
    assert page_updates


def test_do_add_progress_label_pins_to_oldest_in_flight_file(tmp_path: Path) -> None:
    """With concurrent file ingestion, the progress label pins to the oldest
    file still in flight rather than tracking the just-completed file."""
    import asyncio
    import threading

    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import SyncResult
    from lilbee.runtime.progress import (
        BatchProgressEvent,
        BatchStatus,
        EventType,
        FileDoneEvent,
        FileStartEvent,
    )

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    from lilbee.app.ingest import RegisterResult

    reg_result = RegisterResult(registered=[src.name])

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        # Three files start concurrently. The pipeline emits FILE_START for each.
        on_progress(
            EventType.FILE_START, FileStartEvent(file="a.pdf", current_file=1, total_files=3)
        )
        on_progress(
            EventType.FILE_START, FileStartEvent(file="b.pdf", current_file=2, total_files=3)
        )
        on_progress(
            EventType.FILE_START, FileStartEvent(file="c.pdf", current_file=3, total_files=3)
        )
        # b finishes first (out of order). Pipeline fires FILE_DONE then BATCH_PROGRESS.
        on_progress(EventType.FILE_DONE, FileDoneEvent(file="b.pdf", status="ok", chunks=2))
        on_progress(
            EventType.BATCH_PROGRESS,
            BatchProgressEvent(file="b.pdf", status=BatchStatus.INGESTED, current=1, total=3),
        )
        # a finishes next.
        on_progress(EventType.FILE_DONE, FileDoneEvent(file="a.pdf", status="ok", chunks=4))
        on_progress(
            EventType.BATCH_PROGRESS,
            BatchProgressEvent(file="a.pdf", status=BatchStatus.INGESTED, current=2, total=3),
        )
        # c finishes last (in-flight is now empty).
        on_progress(EventType.FILE_DONE, FileDoneEvent(file="c.pdf", status="ok", chunks=1))
        on_progress(
            EventType.BATCH_PROGRESS,
            BatchProgressEvent(file="c.pdf", status=BatchStatus.INGESTED, current=3, total=3),
        )
        return SyncResult()

    def _worker() -> None:
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
        with (
            patch("lilbee.app.ingest.register_sources", return_value=reg_result),
            patch("lilbee.data.ingest.sync", side_effect=fake_sync),
            patch("lilbee.runtime.asyncio_loop.run", side_effect=lambda coro: asyncio.run(coro)),
        ):
            screen._do_add([src], reporter)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)

    # Reduce the call list to the detail strings reporter.update saw, in order.
    details = [call.args[1] for call in reporter.update.call_args_list]

    syncing_a = msg.ADD_SYNCING_FILE.format(file="a.pdf")
    syncing_b = msg.ADD_SYNCING_FILE.format(file="b.pdf")
    syncing_c = msg.ADD_SYNCING_FILE.format(file="c.pdf")

    # All three FILE_STARTs reported syncing_a. Label never advanced
    # to b or c just because they started, because a is the oldest.
    assert syncing_a in details
    assert syncing_b not in details  # b never became oldest
    assert syncing_c in details  # c becomes oldest after a finishes

    # b's BATCH_PROGRESS came in while a was still oldest, so the detail
    # at that point must still point at a, not b.
    assert details.index(syncing_a) < details.index(syncing_c)

    # The very last batch tick (c done, in-flight empty) shows the done label.
    assert details[-1] == msg.ADD_FILE_DONE.format(file="c.pdf")


def test_do_sync_notifies_on_skipped(tmp_path: Path) -> None:
    """Auto-sync surfaces skipped files via notify so the user knows about them."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import SyncResult

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    screen.notify = lambda body, **kw: None  # type: ignore[assignment]
    notify_calls: list[tuple[str, ...]] = []

    def _worker() -> None:
        with (
            patch(
                "lilbee.runtime.asyncio_loop.run",
                new=MagicMock(return_value=SyncResult(skipped=["scan.pdf"])),
            ),
            patch(
                "lilbee.cli.tui.screens.chat.call_from_thread",
                side_effect=lambda *a, **kw: notify_calls.append(a),
            ),
        ):
            screen._do_sync(reporter)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    # call_from_thread(self, self.notify, message, severity="warning") was invoked.
    assert notify_calls
    assert any("scan.pdf" in str(call) for call in notify_calls)


def test_do_sync_names_ocr_off_for_a_scan_skipped_with_ocr_off(tmp_path: Path) -> None:
    """With OCR off the skip toast says so, even when a vision model is configured."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import OcrReport, SyncResult
    from lilbee.runtime.progress import OcrBackendUsed

    cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    result = SyncResult(
        skipped=["scan.pdf"], skipped_ocr={"scan.pdf": OcrReport(backend=OcrBackendUsed.NONE)}
    )
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    notify_calls: list[tuple[object, ...]] = []

    def _worker() -> None:
        with (
            patch("lilbee.runtime.asyncio_loop.run", new=MagicMock(return_value=result)),
            patch(
                "lilbee.cli.tui.screens.chat.call_from_thread",
                side_effect=lambda *a, **kw: notify_calls.append(a),
            ),
        ):
            screen._do_sync(reporter)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    texts = [str(call) for call in notify_calls]
    assert any("OCR is off" in text and "scan.pdf" in text for text in texts)
    assert not any("vision OCR returned no text" in text for text in texts)


def test_do_sync_names_the_tui_log_for_a_scan_vision_could_not_read(tmp_path: Path) -> None:
    """The vision-failed skip toast names the log the TUI itself writes.

    The TUI runs sync in-process, so the underlying error is in tui.log, not
    in server.log, which a standalone ``lilbee serve`` process would write.
    """
    import threading

    from lilbee.cli.tui.log_routing import tui_log_path
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import OcrReport, SyncResult
    from lilbee.runtime.progress import OcrBackendUsed

    result = SyncResult(
        skipped=["scan.pdf"],
        skipped_ocr={"scan.pdf": OcrReport(backend=OcrBackendUsed.VISION, pages=3)},
    )
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    notify_calls: list[tuple[object, ...]] = []

    def _worker() -> None:
        with (
            patch("lilbee.runtime.asyncio_loop.run", new=MagicMock(return_value=result)),
            patch(
                "lilbee.cli.tui.screens.chat.call_from_thread",
                side_effect=lambda *a, **kw: notify_calls.append(a),
            ),
        ):
            screen._do_sync(reporter)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    # The message is call_from_thread's third positional arg. str() of the
    # whole call tuple reprs it, which doubles backslashes in a Windows path
    # and breaks a raw substring check; read the message argument itself.
    messages = [str(call[2]) for call in notify_calls]
    assert any(str(tui_log_path()) in message for message in messages)
    assert not any("server.log" in message for message in messages)


def test_do_add_names_ocr_off_when_the_only_file_skipped_with_ocr_off(tmp_path: Path) -> None:
    """A failed /add raises the OCR-off message, not the vision one."""
    import threading

    from lilbee.app.ingest import RegisterResult
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import OcrReport, SyncResult
    from lilbee.runtime.progress import OcrBackendUsed

    cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    src = tmp_path / "scan.pdf"
    src.write_bytes(b"x")
    result = SyncResult(
        skipped=["scan.pdf"], skipped_ocr={"scan.pdf": OcrReport(backend=OcrBackendUsed.NONE)}
    )
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    reporter.is_set.return_value = False  # a failed add, not a cancel
    captured: list[Exception] = []

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with (
                patch(
                    "lilbee.app.ingest.register_sources",
                    return_value=RegisterResult(registered=[src.name]),
                ),
                patch("lilbee.runtime.asyncio_loop.run", new=MagicMock(return_value=result)),
                patch("lilbee.cli.tui.screens.chat.unregister_added_roots"),
            ):
                screen._do_add([src], reporter)
        except Exception as e:
            captured.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert captured and "OCR is off" in str(captured[0]) and "scan.pdf" in str(captured[0])


def test_do_add_skipped_alongside_indexed_is_partial_success(tmp_path: Path) -> None:
    """Skipped files beside indexed ones finish the add: warn, keep roots, no raise."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import SyncResult

    src = tmp_path / "corpus"
    src.mkdir()
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    from lilbee.app.ingest import RegisterResult

    reg_result = RegisterResult(registered=[src.name])
    captured: list[Exception] = []
    notify_calls: list[tuple[object, ...]] = []
    unregister = MagicMock()

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with (
                patch("lilbee.app.ingest.register_sources", return_value=reg_result),
                patch(
                    "lilbee.runtime.asyncio_loop.run",
                    new=MagicMock(
                        return_value=SyncResult(
                            added=["corpus/store.py"], skipped=["corpus/__init__.py"]
                        )
                    ),
                ),
                patch("lilbee.cli.tui.screens.chat.unregister_added_roots", new=unregister),
                patch(
                    "lilbee.cli.tui.screens.chat.call_from_thread",
                    side_effect=lambda *a, **kw: notify_calls.append(a),
                ),
            ):
                screen._do_add([src], reporter)
        except Exception as e:
            captured.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not captured, f"partial success must not raise: {captured}"
    unregister.assert_not_called()
    assert any("__init__.py" in str(call) for call in notify_calls)


def test_do_add_names_the_tui_log_for_a_scan_vision_could_not_read(tmp_path: Path) -> None:
    """The /add skip warning names the log the TUI itself writes, not the server's."""
    import threading

    from lilbee.app.ingest import RegisterResult
    from lilbee.cli.tui.log_routing import tui_log_path
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import OcrReport, SyncResult
    from lilbee.runtime.progress import OcrBackendUsed

    src = tmp_path / "corpus"
    src.mkdir()
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    reg_result = RegisterResult(registered=[src.name])
    captured: list[Exception] = []
    notify_calls: list[tuple[object, ...]] = []

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with (
                patch("lilbee.app.ingest.register_sources", return_value=reg_result),
                patch(
                    "lilbee.runtime.asyncio_loop.run",
                    new=MagicMock(
                        return_value=SyncResult(
                            added=["corpus/store.py"],
                            skipped=["corpus/scan.pdf"],
                            skipped_ocr={
                                "corpus/scan.pdf": OcrReport(backend=OcrBackendUsed.VISION, pages=2)
                            },
                        )
                    ),
                ),
                patch("lilbee.cli.tui.screens.chat.unregister_added_roots"),
                patch(
                    "lilbee.cli.tui.screens.chat.call_from_thread",
                    side_effect=lambda *a, **kw: notify_calls.append(a),
                ),
            ):
                screen._do_add([src], reporter)
        except Exception as e:
            captured.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not captured, f"partial success must not raise: {captured}"
    # The message is call_from_thread's third positional arg. str() of the
    # whole call tuple reprs it, which doubles backslashes in a Windows path
    # and breaks a raw substring check; read the message argument itself.
    messages = [str(call[2]) for call in notify_calls]
    assert any(str(tui_log_path()) in message for message in messages)
    assert not any("server.log" in message for message in messages)


def test_do_add_raises_when_nothing_indexed(tmp_path: Path) -> None:
    """Every file skipped and nothing indexed: the add failed and its roots are dropped."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import SyncResult

    src = tmp_path / "scan.pdf"
    src.write_bytes(b"x")
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    reporter.is_set.return_value = False  # a failed add, not a cancel

    from lilbee.app.ingest import RegisterResult

    reg_result = RegisterResult(registered=[src.name])
    captured: list[Exception] = []
    unregister = MagicMock()

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with (
                patch("lilbee.app.ingest.register_sources", return_value=reg_result),
                patch(
                    "lilbee.runtime.asyncio_loop.run",
                    new=MagicMock(return_value=SyncResult(skipped=["scan.pdf"])),
                ),
                patch("lilbee.cli.tui.screens.chat.unregister_added_roots", new=unregister),
            ):
                screen._do_add([src], reporter)
        except Exception as e:
            captured.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert captured and "scan.pdf" in str(captured[0])
    unregister.assert_called_once()


def test_do_add_other_sources_do_not_mask_a_dead_add(tmp_path: Path) -> None:
    """Sync is global: activity on other sources must not turn an all-skipped add into a success."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.types import SyncResult

    src = tmp_path / "scan.pdf"
    src.write_bytes(b"x")
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)
    reporter.is_set.return_value = False  # a failed add, not a cancel

    from lilbee.app.ingest import RegisterResult

    reg_result = RegisterResult(registered=[src.name])
    captured: list[Exception] = []
    unregister = MagicMock()

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with (
                patch("lilbee.app.ingest.register_sources", return_value=reg_result),
                patch(
                    "lilbee.runtime.asyncio_loop.run",
                    new=MagicMock(
                        return_value=SyncResult(
                            added=["other-doc.md"], unchanged=5, skipped=["scan.pdf"]
                        )
                    ),
                ),
                patch("lilbee.cli.tui.screens.chat.unregister_added_roots", new=unregister),
            ):
                screen._do_add([src], reporter)
        except Exception as e:
            captured.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert captured and "scan.pdf" in str(captured[0])
    unregister.assert_called_once()


@pytest.mark.asyncio
async def test_cmd_crawl_with_valid_url_routes_to_start_crawl() -> None:
    """/crawl with a valid URL (explicit https) triggers _start_crawl."""

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = await await_chat(app, pilot)
        assert screen is not None
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch.object(screen, "_start_crawl") as mock_start,
        ):
            screen._cmd_crawl("https://example.com")
        mock_start.assert_called_once()


@pytest.mark.asyncio
async def test_cancelling_a_tui_crawl_stops_its_sync(tmp_path, monkeypatch) -> None:
    """Cancelling a /crawl task stops the crawl's sync and skips the follow-up sync."""
    import asyncio
    import threading
    from unittest.mock import AsyncMock

    from lilbee.app.services import CrawlerSyncState, get_services
    from lilbee.cli.tui import messages as msg
    from lilbee.core.config.enums import CrawlRenderMode
    from lilbee.crawler.models import CrawlResult
    from tests._async_wait import wait_until

    monkeypatch.setattr(cfg, "documents_dir", tmp_path / "documents")
    monkeypatch.setattr(cfg, "data_dir", tmp_path / "data")
    monkeypatch.setattr(cfg, "crawl_render_mode", CrawlRenderMode.HTTP)
    monkeypatch.setattr(cfg, "crawl_sync_interval", 1)
    (tmp_path / "documents").mkdir()
    (tmp_path / "data").mkdir()
    sync_started = threading.Event()
    seen: list[object] = []

    async def _sync_until_cancelled(**kwargs):
        cancel = kwargs.get("cancel")
        seen.append(cancel)
        sync_started.set()
        for _ in range(200):
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError
            await asyncio.sleep(0.01)
        raise AssertionError("the crawl's sync never saw the task's cancel")

    async def _fake_single(url, *, quiet=False, on_progress=None, render_mode=None):
        return CrawlResult(url=url, markdown="# page")

    app = LilbeeApp()
    async with app.run_test() as pilot:
        screen = await await_chat(app, pilot)
        assert screen is not None
        services = get_services()
        services.crawler_semaphore = None
        services.crawler_sync_state = CrawlerSyncState()
        notified: list[str] = []
        screen.notify = lambda *a, **kw: notified.append(str(a[0]))  # type: ignore[assignment]
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch("lilbee.cli.tui.screens.chat.require_valid_crawl_url"),
            patch("lilbee.crawler.runner._ensure_crawler_ready", new_callable=AsyncMock),
            patch("lilbee.crawler.runner.crawl_single", side_effect=_fake_single),
            patch("lilbee.data.ingest.sync", _sync_until_cancelled),
            patch.object(screen, "_run_sync") as follow_up_sync,
        ):
            screen._cmd_crawl("https://example.com/c --depth 0")
            assert await wait_until(pilot, sync_started.is_set, max_pauses=500)
            crawl = next(
                t for t in app.task_bar.queue.active_tasks if t.task_type == TaskType.CRAWL.value
            )
            app.task_bar.cancel_task(crawl.task_id)
            worker = next(t for t in threading.enumerate() if t.name == f"task-{crawl.task_id}")
            await asyncio.to_thread(worker.join, 5)
            assert not worker.is_alive()
            await pilot.pause()
        assert len(seen) == 1
        assert seen[0] is not None and seen[0].is_set()
        follow_up_sync.assert_not_called()
        success = msg.CMD_CRAWL_SUCCESS.format(count=1, url="https://example.com/c")
        assert success not in notified


def test_do_sync_throttles_rapid_embed_events() -> None:
    """Two EMBED events within the throttle window → only the first updates."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.runtime.progress import EmbedEvent, EventType

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress, cancel, force_rebuild=False, prune_ignored=False):
        on_progress(EventType.EMBED, EmbedEvent(file="a.pdf", chunk=1, total_chunks=10))
        on_progress(EventType.EMBED, EmbedEvent(file="a.pdf", chunk=2, total_chunks=10))

    exc: list[Exception] = []

    def _worker() -> None:
        try:
            with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
                screen._do_sync(reporter)
        except Exception as e:  # pragma: no cover
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    # Initial SYNC_STATUS_SYNCING + one EMBED (second EMBED throttled).
    assert reporter.update.call_count == 2


@pytest.mark.asyncio
async def test_run_task_worker_noop_when_target_popped_before_start() -> None:
    """Race guard: _run_task_worker returns silently if the entry is gone."""
    app = LilbeeApp()
    async with app.run_test():
        controller = TaskBarController(app)
        task_id = controller.queue.enqueue(lambda: None, "demo", TaskType.SYNC.value)
        # Simulate the race: entry popped before worker body runs.
        controller._task_targets.pop(task_id, None)
        controller._run_task_worker(task_id)  # must not raise


@pytest.mark.asyncio
async def test_an_add_whose_sync_is_cancelled_ends_as_a_cancelled_task(tmp_path: Path) -> None:
    """An /add whose sync raises CancelledError finalizes its row as cancelled."""
    import asyncio

    from lilbee.app.ingest import RegisterResult
    from lilbee.cli.tui.screens.chat import ChatScreen
    from tests._async_wait import wait_until

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")

    async def fake_sync(**_kwargs):
        raise asyncio.CancelledError

    app = LilbeeApp()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        screen = ChatScreen.__new__(ChatScreen)
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
        with (
            patch(
                "lilbee.app.ingest.register_sources",
                return_value=RegisterResult(registered=[src.name]),
            ),
            patch("lilbee.cli.tui.screens.chat.unregister_added_roots"),
            patch("lilbee.data.ingest.sync", side_effect=fake_sync),
        ):
            task_id = controller.start_task(
                "Add doc.pdf", TaskType.ADD, lambda reporter: screen._do_add([src], reporter)
            )
            await wait_until(
                pilot,
                lambda: controller.queue.get_task(task_id).status is not TaskStatus.ACTIVE,
                timeout=_SETTLE_SECONDS,
            )
        assert controller.queue.get_task(task_id).status is TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_a_cancelled_sync_row_shows_the_resume_hint() -> None:
    """Cancelling a TUI sync leaves the row cancelled with the hint to press S."""
    import asyncio
    import threading

    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.chat import ChatScreen
    from tests._async_wait import wait_until

    async def fake_sync(*, cancel, **_kwargs):
        while not cancel.is_set():
            await asyncio.sleep(0.02)
        raise asyncio.CancelledError

    app = LilbeeApp()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)
        screen = ChatScreen.__new__(ChatScreen)
        started = threading.Event()

        def _target(reporter: ProgressReporter) -> None:
            started.set()
            screen._do_sync(reporter)

        with patch("lilbee.data.ingest.sync", side_effect=fake_sync):
            task_id = controller.start_task("Sync", TaskType.SYNC, _target)
            assert await wait_until(pilot, started.is_set, timeout=_SETTLE_SECONDS)
            controller.cancel_task(task_id)
            task = controller.queue.get_task(task_id)
            # The worker posts the hint from its own thread, which no pause drives.
            assert await wait_until(
                pilot, lambda: task.detail == msg.SYNC_CANCELLED_RESUME, timeout=_SETTLE_SECONDS
            )
        assert task.status is TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_stop_all_cancels_queued_and_running_tasks_within_its_budget() -> None:
    """A wedged worker costs at most the budget; queued rows are cancelled too."""
    import threading
    import time

    from lilbee.cli.tui.task_queue import TaskStatus

    app = LilbeeApp()
    async with app.run_test():
        controller = TaskBarController(app)
        release = threading.Event()
        running = controller.start_task(
            "wedged", TaskType.SYNC, lambda _reporter: release.wait(10.0)
        )
        queued = controller.start_task("next", TaskType.SYNC, lambda _reporter: None)
        started = time.monotonic()
        controller.stop_all(budget_s=0.2)
        elapsed = time.monotonic() - started
        release.set()
        assert elapsed < 2.0
        statuses = {
            task_id: controller.queue.get_task(task_id).status for task_id in (running, queued)
        }
        assert statuses == {running: TaskStatus.CANCELLED, queued: TaskStatus.CANCELLED}


@pytest.mark.asyncio
async def test_stop_all_spends_one_budget_on_the_drain_and_the_joins() -> None:
    """A coroutine that outlasts the budget costs the exit the budget once, not once per wait."""
    import asyncio
    import threading
    import time

    from lilbee.runtime import asyncio_loop

    app = LilbeeApp()
    async with app.run_test():
        controller = TaskBarController(app)
        waiting = threading.Event()
        release = threading.Event()

        async def _slow_to_unwind() -> None:
            try:
                waiting.set()
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(15.0)  # a write that outlasts the exit budget
                raise

        def _target(_reporter: ProgressReporter) -> None:
            asyncio.run_coroutine_threadsafe(_slow_to_unwind(), asyncio_loop.get_loop())
            release.wait(30.0)

        controller.start_task("Sync", TaskType.SYNC, _target)
        assert waiting.wait(5.0)
        started = time.monotonic()
        controller.stop_all(budget_s=2.0)
        elapsed = time.monotonic() - started
        release.set()
        assert 1.5 < elapsed < 3.5  # two waits of the budget take 4 seconds


class _OwnedLock:
    """A lock that knows which thread holds it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.owner: int | None = None

    def __enter__(self) -> None:
        self._lock.acquire()
        self.owner = threading.get_ident()

    def __exit__(self, *_exc: object) -> None:
        self.owner = None
        self._lock.release()


class _WatchedWorkers(dict):
    """A worker map that records, per operation, whether its thread held the lock."""

    def __init__(self, lock: _OwnedLock) -> None:
        super().__init__()
        self._lock = lock
        self.held: dict[str, list[bool]] = {
            "start": [],
            "spawn": [],
            "worker_exit": [],
            "stop_all": [],
        }
        self.started_at_spawn: list[bool] = []

    def _note(self, site: str) -> None:
        self.held[site].append(self._lock.owner == threading.get_ident())

    def watch_starts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Record, for each task worker started, whether the starting thread held the lock."""
        real_start = threading.Thread.start

        def _start(thread: threading.Thread) -> None:
            if thread.name.startswith("task-"):
                self._note("start")
            real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", _start)

    def __setitem__(self, task_id, thread) -> None:
        self._note("spawn")
        self.started_at_spawn.append(thread.ident is not None)
        super().__setitem__(task_id, thread)

    def pop(self, *args):
        self._note("worker_exit")
        return super().pop(*args)

    def values(self):
        self._note("stop_all")
        return super().values()


async def _spawn_end_and_stop(pilot, monkeypatch: pytest.MonkeyPatch) -> _WatchedWorkers:
    """Run one task through spawn, worker exit and the stop path on a watched worker map."""
    from tests._async_wait import wait_until

    controller = TaskBarController(pilot.app)
    controller._workers_lock = _OwnedLock()
    controller._workers = workers = _WatchedWorkers(controller._workers_lock)
    workers.watch_starts(monkeypatch)
    controller.start_task("watched", TaskType.SYNC, lambda _reporter: None)
    assert await wait_until(pilot, lambda: workers.held["worker_exit"] != [], timeout=5.0)
    controller.stop_all(budget_s=0.1)
    return workers


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["spawn", "worker_exit", "stop_all"])
async def test_the_worker_map_is_read_and_written_under_its_lock(
    site: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each site touches the worker map on the thread that holds the lock at that moment."""
    app = LilbeeApp()
    async with app.run_test() as pilot:
        workers = await _spawn_end_and_stop(pilot, monkeypatch)
    assert workers.held[site] == [True]


@pytest.mark.asyncio
async def test_a_task_worker_is_started_in_the_lock_before_it_enters_the_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stop path joins every thread in the map, and a join before the start raises."""
    app = LilbeeApp()
    async with app.run_test() as pilot:
        workers = await _spawn_end_and_stop(pilot, monkeypatch)
    assert workers.held["start"] == [True]
    assert workers.started_at_spawn == [True]


class _StartsATaskOnRelease:
    """A lock whose release, once armed, starts a task before any other thread can run."""

    def __init__(self, start_task) -> None:
        self._lock = threading.Lock()
        self._start_task = start_task
        self.armed = False

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *_exc: object) -> None:
        self._lock.release()
        if self.armed:
            self.armed = False
            self._start_task()


class _ArmsOnCopy(dict):
    """A worker map that arms its lock when the stop path copies it."""

    def __init__(self, lock: _StartsATaskOnRelease) -> None:
        super().__init__()
        self._lock = lock

    def values(self):
        self._lock.armed = True
        return super().values()


@pytest.mark.asyncio
@pytest.mark.parametrize("moment", ["after_the_copy", "after_the_stop"])
async def test_a_task_started_once_the_stop_has_begun_gets_no_worker(moment: str) -> None:
    """The stop joins the workers it copied; a later start is refused and cancelled as an exit."""
    from lilbee.cli.tui.task_queue import CancelOrigin

    app = LilbeeApp()
    async with app.run_test():
        controller = TaskBarController(app)
        release = threading.Event()
        started: list[str] = []

        def _wait(_reporter: ProgressReporter) -> None:
            release.wait(5.0)

        def _start_late() -> None:
            started.append(controller.start_task("late", TaskType.SYNC, _wait))

        if moment == "after_the_copy":
            controller._workers_lock = lock = _StartsATaskOnRelease(_start_late)
            controller._workers = _ArmsOnCopy(lock)
        try:
            controller.stop_all(budget_s=0.2)
            if moment == "after_the_stop":
                _start_late()
            (late,) = started
            task = controller.queue.get_task(late)
            assert dict(controller._workers) == {}
            assert (task.status, task.cancel_origin) == (TaskStatus.CANCELLED, CancelOrigin.EXIT)
            assert late not in controller._task_targets
        finally:
            release.set()


@pytest.mark.asyncio
async def test_a_sync_stopped_at_exit_starts_no_pending_detection() -> None:
    """The re-detect a sync starts on its way out does not run into the exit teardown."""
    import threading

    app = LilbeeApp()
    async with app.run_test():
        controller = TaskBarController(app)
        detected: list[bool] = []

        def _sync_target(reporter: ProgressReporter) -> None:
            try:
                while not reporter.is_set():
                    threading.Event().wait(0.02)
            finally:
                controller.start_detect_pending()

        with patch.object(controller, "_run_detect_pending", lambda: detected.append(True)):
            controller.start_task("Sync", TaskType.SYNC, _sync_target)
            controller.stop_all(budget_s=2.0)
        assert detected == []
        assert controller._detect_thread is None


@pytest.mark.asyncio
async def test_stop_all_waits_for_a_detection_already_running() -> None:
    """A detection in flight when the app exits finishes before the teardown."""
    import threading

    app = LilbeeApp()
    async with app.run_test():
        controller = TaskBarController(app)
        release = threading.Event()
        finished = threading.Event()

        def _detect() -> None:
            release.wait(5.0)
            finished.set()

        with patch.object(controller, "_run_detect_pending", _detect):
            controller.start_detect_pending()
            threading.Timer(0.2, release.set).start()
            controller.stop_all(budget_s=3.0)
        assert finished.is_set()
