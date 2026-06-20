"""Coverage for chat + wiki flows after migration to TaskBarController.start_task.

These exercise the public entry points (``_cmd_add``, ``_start_crawl``,
``_run_sync``) and the worker bodies (``_do_add``, ``_do_crawl``,
``_do_sync``) that the old screen-owned @work paths no longer cover.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.catalog import CatalogModel
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.task_queue import TaskStatus, TaskType
from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter, TaskBarController


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
async def test_on_success_exception_is_swallowed() -> None:
    """An exception raised inside on_success must not propagate."""
    app = LilbeeApp()
    async with app.run_test() as pilot:
        controller = TaskBarController(app)

        def _oops() -> None:
            raise RuntimeError("boom")

        task_id = controller.start_task("demo", TaskType.SYNC, lambda r: None, on_success=_oops)
        for _ in range(20):
            await pilot.pause()
            task = controller.queue.get_task(task_id)
            if task is not None and task.status == TaskStatus.DONE:
                break
        # Test passes as long as we didn't blow up.


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
    from lilbee.cli.tui.screens.chat import ChatScreen

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        assert screen is not None

        reporter = MagicMock(spec=ProgressReporter)

        from lilbee.app.ingest import CopyResult

        copy_result = CopyResult(copied=[str(src)], skipped=[])

        import threading as _th

        exc: list[Exception] = []

        from lilbee.data.ingest import SyncResult

        def _worker() -> None:
            try:
                with (
                    patch("lilbee.app.ingest.copy_files", return_value=copy_result),
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
async def test_do_add_force_propagates_to_copy_files(tmp_path: Path) -> None:
    """After overwrite-confirm ``_do_add`` must pass ``force=True`` through."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        assert screen is not None

        reporter = MagicMock(spec=ProgressReporter)

        from lilbee.app.ingest import CopyResult

        copy_result = CopyResult(copied=[str(src)], skipped=[])

        import threading as _th

        exc: list[Exception] = []
        mock_copy = MagicMock(return_value=copy_result)

        def _worker() -> None:
            try:
                with (
                    patch("lilbee.app.ingest.copy_files", new=mock_copy),
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
            if mock_copy.called:
                break
        assert not exc, f"_do_add raised: {exc[0]}"
        assert mock_copy.called
        _, kwargs = mock_copy.call_args
        assert kwargs.get("force") is True


@pytest.mark.asyncio
async def test_do_add_passes_skipped_files_through_copy_result(tmp_path: Path) -> None:
    """_do_add observes copy_files' skipped list and keeps running."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        assert screen is not None

        reporter = MagicMock(spec=ProgressReporter)

        from lilbee.app.ingest import CopyResult

        copy_result = CopyResult(copied=[str(src)], skipped=["exists.pdf"])

        import threading as _th

        exc: list[Exception] = []
        mock_copy = MagicMock(return_value=copy_result)

        def _worker() -> None:
            try:
                with (
                    patch("lilbee.app.ingest.copy_files", new=mock_copy),
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
        # test harness); we only need to confirm copy_files was reached.
        for _ in range(40):
            await pilot.pause()
            if mock_copy.called:
                break
        assert mock_copy.called
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
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with patch("lilbee.crawler.crawl_and_save", side_effect=fake_crawl):
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
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with patch("lilbee.crawler.crawl_and_save", side_effect=fake_crawl):
                screen._do_crawl("https://x", 0, 2, reporter)
        except Exception as e:  # pragma: no cover - re-raised
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not exc, f"worker raised: {exc[0]}"
    assert reporter.update.call_count >= 2


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

    async def fake_sync(*, quiet, on_progress, force_rebuild=False):
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
    """_do_sync routes EventType.DONE through reporter.update at 100% so the
    Task Center row flashes 'just-completed' (regression for bb-7enj)."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.ingest import SyncResult
    from lilbee.runtime.progress import EventType, SyncDoneEvent

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress, force_rebuild=False):
        on_progress(
            EventType.DONE,
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


def test_do_sync_raises_on_sync_failed() -> None:
    """bb-vb28 parallel: auto-sync worker raises when SyncResult.failed is non-empty."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.ingest import SyncResult

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress, force_rebuild=False):
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


def test_do_sync_translates_cancellation() -> None:
    """asyncio.CancelledError becomes a RuntimeError the controller can surface."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress, force_rebuild=False):
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
    assert isinstance(captured[0], RuntimeError)
    assert "cancelled" in str(captured[0]).lower()


@pytest.mark.asyncio
async def test_cmd_add_missing_path_notifies(tmp_path: Path) -> None:
    """_cmd_add on a non-existent path shows an error."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        assert screen is not None
        notified: list[str] = []
        screen.notify = lambda *a, **kw: notified.append(str(a[0]))  # type: ignore[assignment]
        screen._cmd_add(str(tmp_path / "nope.pdf"))
        assert any("not found" in n.lower() for n in notified)


@pytest.mark.asyncio
async def test_cmd_add_submits_task_to_controller(tmp_path: Path) -> None:
    """_cmd_add routes real work through TaskBarController.start_task."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        assert screen is not None
        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._cmd_add(str(src))
        assert mock_start.called
        call_args = mock_start.call_args
        assert call_args.args[1] == TaskType.ADD


@pytest.mark.asyncio
async def test_cmd_add_prompts_before_overwriting_existing_file(tmp_path: Path) -> None:
    """A duplicate in documents_dir opens ConfirmDialog; confirm spawns the task."""
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.core.config import cfg as _cfg

    # Seed a copy already in documents_dir so _cmd_add detects a duplicate.
    _cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    (_cfg.documents_dir / "doc.pdf").write_bytes(b"existing")

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"new")

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
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
async def test_cmd_add_overwrite_rejected_keeps_existing_copy(tmp_path: Path) -> None:
    """When the user answers No to the overwrite dialog, no task is spawned."""
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.core.config import cfg as _cfg

    _cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    (_cfg.documents_dir / "doc.pdf").write_bytes(b"existing")

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"new")

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
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
    from lilbee.cli.tui.screens.chat import ChatScreen

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
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
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
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
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        assert screen is not None
        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._run_sync()
        assert mock_start.called
        assert mock_start.call_args.args[1] == TaskType.SYNC


@pytest.mark.asyncio
async def test_run_sync_rejects_when_already_active() -> None:
    """_run_sync refuses when another sync is already running."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
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

    from lilbee.app.ingest import CopyResult

    copy_result = CopyResult(copied=[str(src)], skipped=[])

    async def fake_sync(*, quiet, on_progress, force_rebuild=False):
        on_progress(
            EventType.FILE_START,
            FileStartEvent(file="a.pdf", current_file=1, total_files=1),
        )

    exc: list[Exception] = []

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with (
                patch("lilbee.app.ingest.copy_files", return_value=copy_result),
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
    from lilbee.data.ingest.types import SyncResult
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

    from lilbee.app.ingest import CopyResult

    copy_result = CopyResult(copied=[str(src)], skipped=[])

    async def fake_sync(*, quiet, on_progress, force_rebuild=False):
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
            patch("lilbee.app.ingest.copy_files", return_value=copy_result),
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
    from lilbee.data.ingest.types import SyncResult
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

    from lilbee.app.ingest import CopyResult

    copy_result = CopyResult(copied=[str(src)], skipped=[])

    async def fake_sync(*, quiet, on_progress, force_rebuild=False):
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
            patch("lilbee.app.ingest.copy_files", return_value=copy_result),
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
    from lilbee.data.ingest.types import SyncResult

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


def test_do_add_raises_on_skipped(tmp_path: Path) -> None:
    """When sync returns skipped files, _do_add raises so the worker surfaces the error."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.data.ingest.types import SyncResult

    src = tmp_path / "scan.pdf"
    src.write_bytes(b"x")
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    from lilbee.app.ingest import CopyResult

    copy_result = CopyResult(copied=[str(src)], skipped=[])
    captured: list[Exception] = []

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with (
                patch("lilbee.app.ingest.copy_files", return_value=copy_result),
                patch(
                    "lilbee.runtime.asyncio_loop.run",
                    new=MagicMock(return_value=SyncResult(skipped=["scan.pdf"])),
                ),
                patch("lilbee.cli.tui.screens.chat.remove_copied_files"),
            ):
                screen._do_add([src], reporter)
        except Exception as e:
            captured.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert captured and "scan.pdf" in str(captured[0])


@pytest.mark.asyncio
async def test_cmd_crawl_with_valid_url_routes_to_start_crawl() -> None:
    """/crawl with a valid URL (explicit https) triggers _start_crawl."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        assert screen is not None
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch.object(screen, "_start_crawl") as mock_start,
        ):
            screen._cmd_crawl("https://example.com")
        mock_start.assert_called_once()


def test_do_sync_throttles_rapid_embed_events() -> None:
    """Two EMBED events within the throttle window → only the first updates."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.runtime.progress import EmbedEvent, EventType

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress, force_rebuild=False):
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
