"""Single-task row widget for the Task Center.

Three lines per row — see ``docs/task-center-design.md`` (or the
brainstorm HTML mock committed alongside this file) for the aesthetic.

The widget is pure-presentation: ``update(task, tick)`` writes the
three labels from a ``Task`` snapshot. ``TaskCenter._poll`` calls it
at 10 Hz; the same tick drives the left-rail 1 Hz pulse on active rows.
"""

from __future__ import annotations

from time import monotonic

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label, Static

from lilbee.cli.tui.task_queue import Task, TaskStatus
from lilbee.cli.tui.widgets.progress_cell import indeterminate_cell, progress_cell

# 1 Hz rail pulse at a 10 Hz poll cadence = 5 ticks on, 5 off.
_PULSE_HALF_TICKS = 5

_STATUS_CLASS: dict[TaskStatus, str] = {
    TaskStatus.QUEUED: "-queued",
    TaskStatus.ACTIVE: "-active",
    TaskStatus.DONE: "-done",
    TaskStatus.FAILED: "-failed",
    TaskStatus.CANCELLED: "-cancelled",
}

_STATUS_CLASSES: tuple[str, ...] = tuple(_STATUS_CLASS.values())


def _format_elapsed(task: Task) -> str:
    """Return elapsed time as MM:SS, a status tag, or empty.

    Terminal states (DONE / FAILED / CANCELLED) freeze at ``completed_at``
    so the timer doesn't keep climbing for rows that are just waiting out
    their 2-second flash before removal.
    """
    if task.status == TaskStatus.QUEUED:
        return "queued"
    if task.started_at is None:
        return ""
    end = task.completed_at if task.completed_at is not None else monotonic()
    seconds = max(0, int(end - task.started_at))
    mm, ss = divmod(seconds, 60)
    return f"{mm:02d}:{ss:02d}"


class TaskRow(Widget, can_focus=True):
    """One task, rendered as three stacked lines.

    Focusable so ``Tab`` / ``j`` / ``k`` in the Task Center moves between
    rows and ``c`` cancels the focused task.
    """

    DEFAULT_CSS = ""  # all styling lives in task_center.tcss

    def __init__(self, task_id: str, **kwargs: object) -> None:
        super().__init__(id=f"task-{task_id}", **kwargs)  # type: ignore[arg-type]
        self._task_id = task_id

    def compose(self) -> ComposeResult:
        # Widget with yielded children lays them out vertically by default.
        # An explicit Vertical wrapper would inherit ``height: 1fr`` and
        # stretch each row to fill the scroll viewport, painting the
        # border-left rail down the whole empty stretch.
        yield Label("", id="row-head", classes="row-head")
        yield Label("", id="row-meta", classes="row-meta")
        yield Static("", id="row-bar", classes="row-bar")

    def update(self, task: Task, tick: int) -> None:
        """Re-render from a Task snapshot. Safe to call every poll tick.

        Quietly no-ops until the row's child labels have mounted, so the
        first few poll ticks (before compose settles) don't error.
        """
        # State class: exactly one of the 5 modifier classes is active.
        target_class = _STATUS_CLASS.get(task.status, "")
        for cls in _STATUS_CLASSES:
            self.set_class(cls == target_class, cls)
        # 1 Hz rail pulse on the active row only.
        self.set_class(
            task.status == TaskStatus.ACTIVE and (tick // _PULSE_HALF_TICKS) % 2 == 0,
            "-pulse",
        )

        try:
            head = self.query_one("#row-head", Label)
            meta = self.query_one("#row-meta", Label)
            bar = self.query_one("#row-bar", Static)
        except Exception:
            return  # compose hasn't finished; retry on next poll

        elapsed = _format_elapsed(task)
        head_left = f"[b]{task.name}[/b] · [i]{task.task_type}[/i]"
        head_text = f"{head_left}  [dim]{elapsed}[/dim]" if elapsed else head_left
        head.update(head_text)

        detail = task.detail or ""
        pct = "" if task.indeterminate else f"[b]{task.progress:.1f}%[/b]"
        meta_parts = [detail, pct]
        meta.update("  ".join(p for p in meta_parts if p))

        if task.indeterminate:
            bar.update(indeterminate_cell(tick))
        else:
            bar.update(progress_cell(task.progress))

    def flash_completed(self) -> None:
        """Mark the row as 'just completed' for a 2-second visual flash."""
        self.add_class("-just-completed")
