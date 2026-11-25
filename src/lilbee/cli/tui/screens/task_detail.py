"""Modal listing the files a finished ingest task did not fully index."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.task_queue import TaskReport


class TaskDetailModal(ModalScreen[None]):
    """Read-only modal of a task's partly read, failed and skipped files."""

    CSS_PATH: ClassVar[str] = "task_detail.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss(None)", msg.TASK_DETAIL_CLOSE, show=True),
        Binding("q", "dismiss(None)", msg.TASK_DETAIL_CLOSE, show=False),
        Binding("i", "dismiss(None)", msg.TASK_DETAIL_CLOSE, show=False),
    ]

    def __init__(self, task_name: str, report: TaskReport) -> None:
        super().__init__()
        self._task_name = task_name
        self._report = report

    def compose(self) -> ComposeResult:
        with Vertical(id="task-detail-root"):
            yield Static(self._task_name, id="task-detail-title", markup=False)
            yield Static(
                msg.task_report_summary(self._report), id="task-detail-counts", markup=False
            )
            with VerticalScroll(id="task-detail-body"):
                yield Static(
                    msg.task_report_text(self._report), id="task-detail-text", markup=False
                )
            yield Static(msg.TASK_DETAIL_HINT, id="task-detail-hint")
