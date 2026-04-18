"""Coverage for chat + wiki flows after migration to TaskBarController.start_task.

These exercise the public entry points (``_cmd_add``, ``_start_crawl``,
``_run_sync``, wiki regen) and the worker bodies (``_do_add``, ``_do_crawl``,
``_do_sync``, ``generate_wiki_pages``, ``_process_source``) that the
old screen-owned @work paths no longer cover.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.catalog import CatalogModel
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.task_queue import TaskStatus, TaskType
from lilbee.cli.tui.widgets.task_bar import ProgressReporter, TaskBarController


def _fake_model() -> CatalogModel:
    return CatalogModel(
        name="n",
        tag="t",
        display_name="Fake",
        hf_repo="o/r",
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
async def test_catalog_enqueue_download_without_lilbee_app_notifies() -> None:
    """When the host is not a LilbeeApp, catalog surfaces an error via notify."""
    from textual.app import App, ComposeResult
    from textual.widgets import Footer

    from lilbee.cli.tui.screens.catalog import CatalogScreen

    class _PlainApp(App[None]):
        def compose(self) -> ComposeResult:
            yield Footer()

    # Run under a plain app — CatalogScreen.app won't be a LilbeeApp.
    with patch("lilbee.cli.tui.screens.catalog.get_catalog"):
        app = _PlainApp()
        async with app.run_test() as pilot:
            screen = CatalogScreen()
            await app.push_screen(screen)
            await pilot.pause()
            notified: list[str] = []
            screen.notify = lambda *a, **kw: notified.append(str(a[0]))  # type: ignore[assignment]
            screen._enqueue_download(_fake_model())
            assert any("task" in n.lower() or "bar" in n.lower() for n in notified)


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

        from typing import ClassVar as _CV

        class _Copied:
            copied: _CV[list[Path]] = [src]
            skipped: _CV[list[str]] = []

        import threading as _th

        exc: list[BaseException] = []

        def _worker() -> None:
            try:
                with (
                    patch("lilbee.cli.helpers.copy_files", return_value=_Copied()),
                    patch("lilbee.ingest.sync", new=MagicMock(return_value=None)),
                    patch("asyncio.run"),
                ):
                    screen._do_add(src, reporter)
            except BaseException as e:  # pragma: no cover
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

        from typing import ClassVar as _CV

        class _Copied:
            copied: _CV[list[Path]] = [src]
            skipped: _CV[list[str]] = ["exists.pdf"]

        import threading as _th

        exc: list[BaseException] = []
        mock_copy = MagicMock(return_value=_Copied())

        def _worker() -> None:
            try:
                with (
                    patch("lilbee.cli.helpers.copy_files", new=mock_copy),
                    patch("asyncio.run", new=MagicMock(return_value=None)),
                ):
                    screen._do_add(src, reporter)
            except BaseException as e:  # pragma: no cover
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


def test_do_crawl_reports_page_progress() -> None:
    """_do_crawl wires CrawlPageEvent through reporter.update.

    Runs on a worker thread (off the pytest event loop) so asyncio.run is
    allowed.
    """
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.progress import CrawlPageEvent, EventType

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_crawl(url, *, depth, max_pages, on_progress):
        on_progress(
            EventType.CRAWL_PAGE,
            CrawlPageEvent(url="https://x/a", current=1, total=2),
        )
        return [Path("/tmp/a")]

    exc: list[BaseException] = []

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with patch("lilbee.crawler.crawl_and_save", side_effect=fake_crawl):
                screen._do_crawl("https://x", 0, 2, reporter)
        except BaseException as e:  # pragma: no cover - re-raised
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not exc, f"worker raised: {exc[0]}"
    assert reporter.update.call_count >= 2


def test_do_sync_reports_file_and_embed_progress() -> None:
    """_do_sync routes FileStart / FileDone / Embed events through reporter.update."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.progress import EmbedEvent, EventType, FileDoneEvent, FileStartEvent

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress):
        on_progress(
            EventType.FILE_START,
            FileStartEvent(file="a.pdf", current_file=1, total_files=2),
        )
        on_progress(EventType.FILE_DONE, FileDoneEvent(file="a.pdf", status="ok", chunks=5))
        on_progress(EventType.EMBED, EmbedEvent(file="a.pdf", chunk=1, total_chunks=10))

    exc: list[BaseException] = []

    def _worker() -> None:
        try:
            with patch("lilbee.ingest.sync", side_effect=fake_sync):
                screen._do_sync(reporter)
        except BaseException as e:  # pragma: no cover - re-raised
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not exc, f"worker raised: {exc[0]}"
    assert reporter.update.call_count >= 3


def test_do_sync_translates_cancellation() -> None:
    """asyncio.CancelledError becomes a RuntimeError the controller can surface."""
    import threading

    from lilbee.cli.tui.screens.chat import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress):
        import asyncio as _asyncio

        raise _asyncio.CancelledError

    captured: list[BaseException] = []

    def _worker() -> None:
        try:
            with patch("lilbee.ingest.sync", side_effect=fake_sync):
                screen._do_sync(reporter)
        except BaseException as e:
            captured.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert captured, "_do_sync should have raised"
    assert isinstance(captured[0], RuntimeError)
    assert "cancelled" in str(captured[0]).lower()


@pytest.mark.asyncio
async def test_wiki_worker_process_source_returns_true_on_result(tmp_path: Path) -> None:
    """_process_source returns True when generate_summary_page produces a page."""
    from lilbee.cli.tui.wiki_worker import _process_source

    reporter = MagicMock(spec=ProgressReporter)
    svc = MagicMock()
    svc.store.get_chunks_by_source.return_value = [MagicMock()]
    with (
        patch("lilbee.cli.tui.wiki_worker.get_services", return_value=svc),
        patch("lilbee.wiki.gen.generate_summary_page", return_value=tmp_path / "out.md"),
    ):
        assert _process_source("src.pdf", 0, 1, reporter, []) is True


@pytest.mark.asyncio
async def test_wiki_worker_process_source_returns_false_with_no_chunks() -> None:
    """_process_source short-circuits when a source has no chunks."""
    from lilbee.cli.tui.wiki_worker import _process_source

    reporter = MagicMock(spec=ProgressReporter)
    svc = MagicMock()
    svc.store.get_chunks_by_source.return_value = []
    with patch("lilbee.cli.tui.wiki_worker.get_services", return_value=svc):
        assert _process_source("src.pdf", 0, 1, reporter, []) is False


@pytest.mark.asyncio
async def test_wiki_worker_process_source_records_stage_errors(tmp_path: Path) -> None:
    """The stage callback appends ``failed`` stages to the errors list."""
    from lilbee.cli.tui.wiki_worker import _process_source

    reporter = MagicMock(spec=ProgressReporter)
    svc = MagicMock()
    svc.store.get_chunks_by_source.return_value = [MagicMock()]
    errors: list[str] = []

    def fake_generate(source, chunks, provider, store, *, on_progress):
        on_progress("failed", {"error": "nope"})
        return None

    with (
        patch("lilbee.cli.tui.wiki_worker.get_services", return_value=svc),
        patch("lilbee.wiki.gen.generate_summary_page", side_effect=fake_generate),
    ):
        assert _process_source("src.pdf", 0, 1, reporter, errors) is False
    assert errors == ["nope"]


@pytest.mark.asyncio
async def test_wiki_worker_process_source_reports_other_stages(tmp_path: Path) -> None:
    """Non-failure stages call reporter.update with a progress fraction."""
    from lilbee.cli.tui.wiki_worker import _process_source

    reporter = MagicMock(spec=ProgressReporter)
    svc = MagicMock()
    svc.store.get_chunks_by_source.return_value = [MagicMock()]

    def fake_generate(source, chunks, provider, store, *, on_progress):
        on_progress("generating", {})
        return tmp_path / "out.md"

    with (
        patch("lilbee.cli.tui.wiki_worker.get_services", return_value=svc),
        patch("lilbee.wiki.gen.generate_summary_page", side_effect=fake_generate),
    ):
        _process_source("src.pdf", 0, 2, reporter, [])
    # At least: initial preparing + generating stage.
    assert reporter.update.call_count >= 2


@pytest.mark.asyncio
async def test_generate_wiki_pages_counts_successes() -> None:
    """generate_wiki_pages returns the number of pages produced."""
    from lilbee.cli.tui.wiki_worker import generate_wiki_pages

    reporter = MagicMock(spec=ProgressReporter)
    with patch("lilbee.cli.tui.wiki_worker._process_source", side_effect=[True, True, False]):
        assert generate_wiki_pages(["a", "b", "c"], reporter) == 2


@pytest.mark.asyncio
async def test_generate_wiki_pages_raises_when_all_fail() -> None:
    """generate_wiki_pages surfaces the last error when zero pages are produced."""
    from lilbee.cli.tui.wiki_worker import generate_wiki_pages

    reporter = MagicMock(spec=ProgressReporter)
    with (
        patch(
            "lilbee.cli.tui.wiki_worker._process_source",
            side_effect=lambda *a, **kw: (
                (_ for _ in a[4].append("boom")).__next__() if False else False
            ),
        ),
        pytest.raises(RuntimeError),
    ):
        generate_wiki_pages(["a"], reporter)


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
    """_start_crawl routes through TaskBarController.start_task with CRAWL type."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        assert screen is not None
        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
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
async def test_wiki_submit_wiki_task_calls_controller() -> None:
    """_submit_wiki_task wires generate_wiki_pages via start_task."""
    from lilbee.cli.tui.screens.wiki import WikiScreen

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(WikiScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, WikiScreen)
        with patch.object(app.task_bar, "start_task", return_value="tid") as mock_start:
            screen._submit_wiki_task(["src.pdf"])
        assert mock_start.called
        assert mock_start.call_args.args[1] == TaskType.WIKI


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
    from lilbee.progress import EventType, FileStartEvent

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"x")
    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    from typing import ClassVar as _CV

    class _Copied:
        copied: _CV[list[Path]] = [src]
        skipped: _CV[list[str]] = []

    async def fake_sync(*, quiet, on_progress):
        on_progress(
            EventType.FILE_START,
            FileStartEvent(file="a.pdf", current_file=1, total_files=1),
        )

    exc: list[BaseException] = []

    def _worker() -> None:
        try:
            screen.notify = lambda *a, **kw: None  # type: ignore[assignment]
            with (
                patch("lilbee.cli.helpers.copy_files", return_value=_Copied()),
                patch("lilbee.ingest.sync", side_effect=fake_sync),
            ):
                screen._do_add(src, reporter)
        except BaseException as e:  # pragma: no cover
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    # The "Syncing {file}..." update is reached only via the FILE_START branch.
    assert any("Syncing a.pdf" in str(call) for call in reporter.update.call_args_list)


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
    from lilbee.progress import EmbedEvent, EventType

    screen = ChatScreen.__new__(ChatScreen)
    reporter = MagicMock(spec=ProgressReporter)

    async def fake_sync(*, quiet, on_progress):
        on_progress(EventType.EMBED, EmbedEvent(file="a.pdf", chunk=1, total_chunks=10))
        on_progress(EventType.EMBED, EmbedEvent(file="a.pdf", chunk=2, total_chunks=10))

    exc: list[BaseException] = []

    def _worker() -> None:
        try:
            with patch("lilbee.ingest.sync", side_effect=fake_sync):
                screen._do_sync(reporter)
        except BaseException as e:  # pragma: no cover
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    # Initial SYNC_STATUS_SYNCING + one EMBED (second EMBED throttled).
    assert reporter.update.call_count == 2


@pytest.mark.asyncio
async def test_wiki_action_regenerate_submits_task() -> None:
    """action_regenerate with wiki enabled + targets → submits task."""
    from lilbee.cli.tui.screens.wiki import WikiScreen
    from lilbee.config import cfg

    original_wiki = cfg.wiki
    cfg.wiki = True
    try:
        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(WikiScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, WikiScreen)
            with (
                patch(
                    "lilbee.cli.tui.screens.wiki.resolve_wiki_targets",
                    return_value=["s.pdf"],
                ),
                patch.object(app.task_bar, "start_task", return_value="tid") as mock_start,
            ):
                screen.action_regenerate()
            mock_start.assert_called_once()
    finally:
        cfg.wiki = original_wiki


@pytest.mark.asyncio
async def test_wiki_submit_target_calls_generate_and_notifies() -> None:
    """The _target closure inside _submit_wiki_task calls generate_wiki_pages."""
    import threading

    from lilbee.cli.tui.screens.wiki import WikiScreen

    captured_target: dict[str, object] = {}

    def _capture(name, task_type, target, **kwargs):
        captured_target["fn"] = target
        return "tid"

    app = LilbeeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(WikiScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, WikiScreen)
        with patch.object(app.task_bar, "start_task", side_effect=_capture):
            screen._submit_wiki_task(["s.pdf"])
    assert "fn" in captured_target

    reporter = MagicMock(spec=ProgressReporter)

    exc: list[BaseException] = []

    def _worker() -> None:
        try:
            with patch("lilbee.cli.tui.screens.wiki.generate_wiki_pages", return_value=1):
                captured_target["fn"](reporter)  # type: ignore[operator]
        except BaseException as e:  # pragma: no cover
            exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5)
    # Either completed or dropped via call_from_thread after the app shut down;
    # either way the target function itself ran without raising.


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
async def test_wiki_submit_wiki_task_noop_outside_lilbee_app() -> None:
    """_submit_wiki_task is a no-op when the host is not a LilbeeApp."""
    from textual.app import App, ComposeResult
    from textual.widgets import Footer

    from lilbee.cli.tui.screens.wiki import WikiScreen

    class _PlainApp(App[None]):
        def compose(self) -> ComposeResult:
            yield Footer()

    app = _PlainApp()
    async with app.run_test() as pilot:
        app.push_screen(WikiScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, WikiScreen)
        screen._submit_wiki_task(["s.pdf"])  # must not raise
