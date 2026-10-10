"""The TUI shows a partly read sync or add in amber and lists its files on request."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Label, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.chat_helpers import ingest_report, report_ingest
from lilbee.cli.tui.screens.task_center import TaskCenter
from lilbee.cli.tui.screens.task_detail import TaskDetailModal
from lilbee.cli.tui.task_queue import (
    STATUS_ICONS,
    Task,
    TaskQueue,
    TaskReport,
    TaskStatus,
    TaskType,
)
from lilbee.cli.tui.widgets.task_bar import TaskBar
from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter, TaskBarController
from lilbee.cli.tui.widgets.task_row import TaskRow, _build_head
from lilbee.core.config import cfg
from lilbee.data.store import FailedPage
from lilbee.data.types import PartialFile, SyncResult
from tests._lilbee_app_test_host import LilbeeAppHost

_PAGE_3 = FailedPage(page=3, error="[vision] timed out", recovered=False)
_PAGE_3_LINE = "page 3: [vision] timed out"
_PARTLY_READ = SyncResult(
    added=["scan.pdf"],
    partial=[PartialFile("scan.pdf", [_PAGE_3])],
    failed=["bad.pdf"],
    skipped=["blank.pdf", "odd.bin"],
    reasons={"bad.pdf": "the file is encrypted", "blank.pdf": "no text extracted (0 chunks)"},
)


def _report(**overrides: object) -> TaskReport:
    fields: dict[str, object] = {
        "partial": (PartialFile("scan.pdf", [_PAGE_3]),),
        "failed": {"bad.pdf": "the file is encrypted"},
        "skipped": {"odd.bin": ""},
        "indexed": 1,
    }
    return TaskReport(**{**fields, **overrides})  # type: ignore[arg-type]


class _BarHost(LilbeeAppHost):
    def __init__(self) -> None:
        super().__init__()
        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield TaskBar(id="tbar")


class _RowHost(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="host")


def _finish(queue: TaskQueue, status: TaskStatus, *, report: TaskReport | None = None) -> str:
    """Put one finished sync task with *status* on *queue* and return its id."""
    task_id = queue.enqueue(lambda: None, "Sync documents", TaskType.SYNC.value)
    queue.advance(TaskType.SYNC.value)
    if report is not None:
        queue.set_report(task_id, report)
    if status is TaskStatus.FAILED:
        queue.fail_task(task_id, "Sync failed for bad.pdf")
    else:
        queue.complete_task(task_id, status)
    return task_id


async def _open_task_center(app: LilbeeAppHost, pilot) -> TaskCenter:
    await pilot.pause()
    app.push_screen(TaskCenter())
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, TaskCenter)
    return screen


class TestTaskReport:
    def test_a_report_with_indexed_files_and_problems_is_partial(self) -> None:
        assert _report().is_partial is True

    def test_a_report_that_indexed_nothing_is_not_partial(self) -> None:
        report = _report(indexed=0)
        assert (report.has_problems, report.is_partial) == (True, False)

    def test_a_report_without_problems_is_not_partial(self) -> None:
        report = TaskReport(indexed=3)
        assert (report.has_problems, report.is_partial) == (False, False)

    def test_ingest_report_carries_reasons_and_counts_what_reached_the_index(self) -> None:
        report = ingest_report(_PARTLY_READ)
        assert report.partial == (PartialFile("scan.pdf", [_PAGE_3]),)
        assert report.failed == {"bad.pdf": "the file is encrypted"}
        assert report.skipped == {"blank.pdf": "no text extracted (0 chunks)", "odd.bin": ""}
        assert report.indexed == 1

    def test_report_ingest_attaches_the_report_and_shows_its_counts(self) -> None:
        reporter = MagicMock(spec=ProgressReporter)
        report = report_ingest(reporter, _PARTLY_READ)
        reporter.set_report.assert_called_once_with(report)
        reporter.update.assert_called_once_with(
            100, "1 partly read  ·  1 failed  ·  2 skipped  ·  i for details", indeterminate=False
        )

    def test_report_ingest_leaves_a_clean_sync_s_row_alone(self) -> None:
        reporter = MagicMock(spec=ProgressReporter)
        report_ingest(reporter, SyncResult(added=["a.md"]))
        reporter.set_report.assert_not_called()
        reporter.update.assert_not_called()


class TestReportText:
    def test_the_summary_names_only_the_kinds_that_occurred(self) -> None:
        summary = msg.task_report_summary(_report(failed={}, skipped={}))
        assert summary == "1 partly read  ·  i for details"

    def test_the_detail_lists_each_failed_page_then_failed_and_skipped_reasons(self) -> None:
        assert msg.task_report_text(_report()).splitlines() == [
            "Partly read",
            "  scan.pdf",
            f"    {_PAGE_3_LINE}",
            "",
            "Failed",
            "  bad.pdf: the file is encrypted",
            "",
            "Skipped",
            "  odd.bin: no reason recorded",
        ]

    def test_the_detail_omits_the_partly_read_section_when_no_file_was(self) -> None:
        text = msg.task_report_text(_report(partial=(), skipped={}))
        assert text.splitlines() == ["Failed", "  bad.pdf: the file is encrypted"]


class TestQueuePartialStatus:
    def test_a_task_completed_as_partial_is_terminal_and_in_history(self) -> None:
        queue = TaskQueue()
        task_id = _finish(queue, TaskStatus.PARTIAL, report=_report())
        task = queue.get_task(task_id)
        assert task is not None
        assert (task.status, task.progress) == (TaskStatus.PARTIAL, 100)
        assert queue.history == [task]
        assert queue.cancel(task_id) is False
        assert task.status is TaskStatus.PARTIAL

    def test_set_report_ignores_an_unknown_task(self) -> None:
        queue = TaskQueue()
        queue.set_report("missing", _report())
        assert queue.get_task("missing") is None

    def test_the_partial_icon_is_one_cell_and_its_own(self) -> None:
        icon = STATUS_ICONS[TaskStatus.PARTIAL]
        assert len(icon) == 1
        assert list(STATUS_ICONS.values()).count(icon) == 1


class TestControllerOutcome:
    async def _run(self, target) -> Task:
        app = _BarHost()
        async with app.run_test() as pilot:
            task_id = app.task_bar.start_task("Sync documents", TaskType.SYNC, target)
            for _ in range(200):
                task = app.task_bar.queue.get_task(task_id)
                if task is not None and task.completed_at is not None:
                    break
                await pilot.pause(delay=0.01)
        assert task is not None
        return task

    @pytest.mark.asyncio
    async def test_a_worker_that_reports_a_partial_run_ends_partial(self) -> None:
        task = await self._run(lambda reporter: reporter.set_report(_report()))
        assert task.status is TaskStatus.PARTIAL
        assert task.report == _report()

    @pytest.mark.asyncio
    async def test_a_worker_whose_report_indexed_nothing_ends_done(self) -> None:
        task = await self._run(lambda reporter: reporter.set_report(_report(indexed=0)))
        assert task.status is TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_a_worker_with_no_report_ends_done(self) -> None:
        task = await self._run(lambda reporter: None)
        assert (task.status, task.report) == (TaskStatus.DONE, None)

    @pytest.mark.asyncio
    async def test_a_worker_that_reports_then_raises_ends_failed_with_its_report(self) -> None:
        def _target(reporter: ProgressReporter) -> None:
            reporter.set_report(_report(indexed=0))
            raise RuntimeError("Sync failed for bad.pdf")

        task = await self._run(_target)
        assert (task.status, task.detail) == (TaskStatus.FAILED, "Sync failed for bad.pdf")
        assert task.report == _report(indexed=0)


class TestTaskRowPartial:
    def test_the_head_names_the_partial_status(self) -> None:
        task = Task("t1", "Sync documents", "sync", lambda: None, status=TaskStatus.PARTIAL)
        assert "partial" in _build_head(task, "00:05").plain

    @pytest.mark.asyncio
    async def test_a_partial_row_takes_the_partial_class_and_keeps_its_counts_line(self) -> None:
        app = _RowHost()
        async with app.run_test() as pilot:
            row = TaskRow(task_id="t1")
            await app.query_one("#host").mount(row)
            await pilot.pause()
            task = Task(
                "t1",
                "Sync documents",
                "sync",
                lambda: None,
                status=TaskStatus.PARTIAL,
                progress=100,
                detail=msg.task_report_summary(_report()),
            )
            row.update(task, 0)
            await pilot.pause()
            assert row.has_class("-partial")
            assert not row.has_class("-done")
            meta = str(row.query_one("#row-meta", Label).content)
            assert "1 partly read" in meta
            assert "100.0%" not in meta


class TestTaskRowReportHint:
    async def _meta(self, report: TaskReport | None) -> str:
        app = _RowHost()
        async with app.run_test() as pilot:
            row = TaskRow(task_id="t1")
            await app.query_one("#host").mount(row)
            await pilot.pause()
            task = Task(
                "t1",
                "Sync documents",
                "sync",
                lambda: None,
                status=TaskStatus.DONE,
                progress=100,
                detail="2 skipped  ·  i for details",
                report=report,
            )
            row.update(task, 0)
            await pilot.pause()
            return str(row.query_one("#row-meta", Label).content)

    @pytest.mark.asyncio
    async def test_a_done_row_with_a_report_keeps_the_line_that_offers_the_detail(self) -> None:
        skipped_only = _report(partial=(), failed={}, indexed=0)
        assert "i for details" in await self._meta(skipped_only)

    @pytest.mark.asyncio
    async def test_a_done_row_without_a_report_hides_its_last_progress_line(self) -> None:
        assert await self._meta(None) == ""


class TestTaskBarFlash:
    async def _flash(self, *statuses: TaskStatus) -> tuple[TaskBar, str]:
        app = _BarHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            for status in statuses:
                _finish(app.task_bar.queue, status)
            bar = app.query_one(TaskBar)
            bar._refresh_display()
            label = bar.query_one("#task-status-label", Label)
            return bar, label.content.markup

    @pytest.mark.asyncio
    async def test_a_partial_finish_flashes_amber_with_its_own_copy(self) -> None:
        bar, text = await self._flash(TaskStatus.PARTIAL)
        assert bar._flash_outcome is TaskStatus.PARTIAL
        assert "$warning" in text
        assert "some files not fully read" in text

    @pytest.mark.asyncio
    async def test_a_partial_finish_outranks_a_clean_one(self) -> None:
        bar, _ = await self._flash(TaskStatus.DONE, TaskStatus.PARTIAL)
        assert bar._flash_outcome is TaskStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_a_failure_outranks_a_partial_finish(self) -> None:
        bar, text = await self._flash(TaskStatus.PARTIAL, TaskStatus.FAILED)
        assert bar._flash_outcome is TaskStatus.FAILED
        assert "$error" in text
        assert "1 task failed" in text

    @pytest.mark.asyncio
    async def test_a_clean_finish_still_flashes_green(self) -> None:
        bar, text = await self._flash(TaskStatus.DONE)
        assert bar._flash_outcome is TaskStatus.DONE
        assert "$success" in text


class TestTaskDetail:
    @pytest.mark.asyncio
    async def test_i_on_a_partial_row_opens_the_detail_and_escape_closes_it(self) -> None:
        app = LilbeeAppHost()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_task_center(app, pilot)
            task_id = _finish(app.task_bar.queue, TaskStatus.PARTIAL, report=_report())
            await pilot.pause()
            screen._rows[task_id].focus()
            await pilot.press("i")
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, TaskDetailModal)
            body = str(modal.query_one("#task-detail-text", Static).content)
            assert _PAGE_3_LINE in body
            assert "bad.pdf: the file is encrypted" in body
            assert str(modal.query_one("#task-detail-title", Static).content) == "Sync documents"
            assert "1 partly read" in str(modal.query_one("#task-detail-counts", Static).content)
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen

    @pytest.mark.asyncio
    async def test_i_on_a_failed_row_with_a_report_opens_its_detail(self) -> None:
        app = LilbeeAppHost()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_task_center(app, pilot)
            task_id = _finish(app.task_bar.queue, TaskStatus.FAILED, report=_report(indexed=0))
            await pilot.pause()
            screen._rows[task_id].focus()
            await pilot.press("i")
            await pilot.pause()
            assert isinstance(app.screen, TaskDetailModal)
            await pilot.press("i")
            await pilot.pause()
            assert app.screen is screen

    @pytest.mark.asyncio
    async def test_i_on_a_row_without_a_report_says_there_is_no_detail(self) -> None:
        app = LilbeeAppHost()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_task_center(app, pilot)
            task_id = _finish(app.task_bar.queue, TaskStatus.DONE)
            await pilot.pause()
            screen._rows[task_id].focus()
            with patch.object(screen, "notify") as notify:
                await pilot.press("i")
                await pilot.pause()
            notify.assert_called_once_with(msg.TASK_DETAIL_NONE)
            assert app.screen is screen

    @pytest.mark.asyncio
    async def test_i_with_no_row_in_focus_opens_nothing(self) -> None:
        app = LilbeeAppHost()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_task_center(app, pilot)
            with patch.object(screen, "notify") as notify:
                await pilot.press("i")
                await pilot.pause()
            notify.assert_not_called()
            assert app.screen is screen

    @pytest.mark.asyncio
    async def test_a_click_on_a_partial_row_opens_the_detail(self) -> None:
        app = LilbeeAppHost()
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_task_center(app, pilot)
            task_id = _finish(app.task_bar.queue, TaskStatus.PARTIAL, report=_report())
            await pilot.pause()
            await pilot.click(f"#task-{task_id}")
            await pilot.pause()
            assert isinstance(app.screen, TaskDetailModal)

    @pytest.mark.asyncio
    async def test_a_click_on_a_row_without_a_report_does_nothing(self) -> None:
        app = LilbeeAppHost()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_task_center(app, pilot)
            task_id = _finish(app.task_bar.queue, TaskStatus.DONE)
            await pilot.pause()
            with patch.object(screen, "notify") as notify:
                await pilot.click(f"#task-{task_id}")
                await pilot.pause()
            notify.assert_not_called()
            assert app.screen is screen

    @pytest.mark.asyncio
    async def test_the_counts_strip_counts_a_partial_task_as_done(self) -> None:
        app = LilbeeAppHost()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_task_center(app, pilot)
            _finish(app.task_bar.queue, TaskStatus.PARTIAL, report=_report())
            _finish(app.task_bar.queue, TaskStatus.DONE)
            await pilot.pause()
            counts = str(screen.query_one("#task-center-counts", Label).content)
            assert "2 done" in counts


def _run_in_thread(body) -> list[BaseException]:
    """Run *body* on a worker thread, as the task queue does, and return what it raised."""
    raised: list[BaseException] = []

    def _worker() -> None:
        try:
            body()
        except Exception as exc:
            raised.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=5)
    return raised


@pytest.fixture()
def linked_corpus(tmp_path):
    """A data root with no linked roots, and one source directory to add."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path / "root"
    cfg.documents_dir = tmp_path / "root" / "documents"
    cfg.documents_dir.mkdir(parents=True)
    cfg.linked_roots = {}
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text("readable", encoding="utf-8")
    (source / "bad.pdf").write_bytes(b"%PDF-1.4")
    yield source
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestAddTaskColour:
    def _do_add(
        self, source, result: SyncResult
    ) -> tuple[list[BaseException], MagicMock, list[str]]:
        """Run an add whose sync returns *result*; return what raised, the reporter, the toasts."""
        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock(spec=ProgressReporter)
        reporter.is_set.return_value = False  # a failed add, not a cancel

        def _run(coro):
            coro.close()
            return result

        with (
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            patch("lilbee.cli.tui.screens.chat.call_from_thread") as marshal,
        ):
            raised = _run_in_thread(lambda: screen._do_add([source], reporter))
        toasts = [call.args[2] for call in marshal.call_args_list if len(call.args) > 2]
        return raised, reporter, toasts

    def test_a_failed_file_beside_an_indexed_one_keeps_the_added_root(self, linked_corpus) -> None:
        result = SyncResult(added=["corpus/a.txt"], failed=["corpus/bad.pdf"])
        raised, reporter, _ = self._do_add(linked_corpus, result)
        assert raised == []
        assert "corpus" in cfg.linked_roots
        assert reporter.set_report.call_args.args[0].is_partial is True

    def test_an_add_with_a_partly_read_file_says_so_in_a_toast(self, linked_corpus) -> None:
        result = SyncResult(
            added=["corpus/a.txt"], partial=[PartialFile("corpus/a.txt", [_PAGE_3])]
        )
        raised, _, toasts = self._do_add(linked_corpus, result)
        assert raised == []
        assert msg.SYNC_PARTLY_READ.format(count=1) in toasts

    def test_a_clean_add_shows_no_partly_read_toast(self, linked_corpus) -> None:
        raised, _, toasts = self._do_add(linked_corpus, SyncResult(added=["corpus/a.txt"]))
        assert raised == []
        assert toasts == [msg.CMD_ADD_SUCCESS.format(count=1)]

    def test_an_add_whose_root_indexed_nothing_fails_and_drops_the_root(
        self, linked_corpus
    ) -> None:
        result = SyncResult(added=["other.md"], failed=["corpus/bad.pdf"])
        raised, _, _ = self._do_add(linked_corpus, result)
        assert [str(exc) for exc in raised] == [
            msg.SYNC_FAILED_FILES.format(files="corpus/bad.pdf")
        ]
        assert "corpus" not in cfg.linked_roots

    def test_an_add_of_a_tracked_source_fails_when_files_failed_and_nothing_was_indexed(
        self, linked_corpus
    ) -> None:
        from lilbee.app.ingest import RegisterResult

        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock(spec=ProgressReporter)
        reporter.is_set.return_value = False  # a failed add, not a cancel
        tracked = RegisterResult(registered=[], tracked=["corpus"])

        def _run(coro):
            coro.close()
            return SyncResult(failed=["corpus/bad.pdf"])

        with (
            patch("lilbee.app.ingest.register_sources", return_value=tracked),
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            patch("lilbee.cli.tui.screens.chat.call_from_thread") as marshal,
        ):
            raised = _run_in_thread(lambda: screen._do_add([linked_corpus], reporter))

        assert [str(exc) for exc in raised] == [
            msg.SYNC_FAILED_FILES.format(files="corpus/bad.pdf")
        ]
        sent = [call.args[2] for call in marshal.call_args_list if len(call.args) > 2]
        assert msg.CMD_ADD_SUCCESS.format(count=0) not in sent

    def test_an_add_of_a_tracked_source_with_a_failed_file_beside_indexed_ones_warns(
        self, linked_corpus
    ) -> None:
        from lilbee.app.ingest import RegisterResult

        screen = ChatScreen.__new__(ChatScreen)
        tracked = RegisterResult(registered=[], tracked=["corpus"])

        def _run(coro):
            coro.close()
            return SyncResult(added=["corpus/a.txt"], failed=["corpus/bad.pdf"])

        with (
            patch("lilbee.app.ingest.register_sources", return_value=tracked),
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            patch("lilbee.cli.tui.screens.chat.call_from_thread"),
        ):
            raised = _run_in_thread(
                lambda: screen._do_add([linked_corpus], MagicMock(spec=ProgressReporter))
            )

        assert raised == []
