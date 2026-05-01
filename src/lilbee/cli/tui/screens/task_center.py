"""Task Center screen: flight-deck-style background task monitor.

Each task renders as a ``TaskRow`` with a three-line body (title +
type, detail + percent, block-char bar) and a thick left rail in the
state's color. On the active row the rail pulses at ~1 Hz, which is
the only motion in the screen beyond the bar filling.

State refresh is event-driven: the screen subscribes to ``TaskQueue``
and ``_refresh_rows`` runs whenever a task is enqueued, advanced,
updated, completed, or cancelled. A separate slow timer advances the
spinner frame and the rail pulse so the visual heartbeat stays alive
while the queue is idle.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Footer, Label

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.task_queue import Task, TaskStatus
from lilbee.cli.tui.widgets.task_row import TaskRow

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

log = logging.getLogger(__name__)

# Spinner advance cadence. Decoupled from queue-state refresh: queue
# events drive _refresh_rows directly, this timer only advances the
# rotating glyph and the active-row pulse so they keep moving while
# the queue itself is idle.
_TICK_INTERVAL_SECONDS = 0.25


class TaskQueueChanged(Message):
    """Posted by TaskCenter._on_queue_change when the queue notifies.

    Posting a Textual Message is thread-safe, so the queue can call the
    subscriber from any thread; the message is processed on the
    screen's main-thread message pump.
    """


# Quarter-circle rotation cycles every 4 ticks (~0.4 s). Visible motion
# in the counts strip confirms background work is live when rows are
# running (bb-18y3).
_COUNTS_SPINNER_FRAMES = ("◐", "◓", "◑", "◒")


class TaskCenter(Screen[None]):
    """Live view of active + queued + recently completed tasks."""

    CSS_PATH = "task_center.tcss"
    AUTO_FOCUS = "#task-rows"
    HELP = "Background task monitor.\n\nPress r to refresh, c to cancel the focused task."

    app: LilbeeApp

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        Binding("r", "refresh_tasks", "Refresh", show=True),
        Binding("c", "cancel_task", "Cancel", show=True),
        Binding("C", "clear_history", "Clear done", show=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        yield Label(msg.TASK_CENTER_TITLE, id="task-center-title")
        yield Label("", id="task-center-counts")
        yield VerticalScroll(id="task-rows")
        yield Label(
            f"{msg.TASK_CENTER_EMPTY_HEADLINE}\n{msg.TASK_CENTER_EMPTY_DETAIL}",
            id="task-center-empty",
        )
        with BottomBars():
            yield Label(msg.TASK_CENTER_HINT, id="task-center-hint")
            yield TaskBar()
            yield Footer()

    def action_go_back(self) -> None:
        """Return to Chat (or pop if we're on a detached test app)."""
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.app.switch_view("Chat")
        else:
            self.app.pop_screen()

    def on_mount(self) -> None:
        self._tick: int = 0
        self._rows: dict[str, TaskRow] = {}
        self._tick_timer: Timer | None = None
        self._refresh_rows()
        self._focus_initial_row()

    def on_show(self) -> None:
        # Subscribe + tick only while visible; install_screen keeps this
        # instance alive across switch_view, so anchoring either on
        # on_mount would fire into a detached DOM after navigating away.
        self.app.task_bar.queue.subscribe(self._on_queue_change)
        if self._tick_timer is None:
            self._tick_timer = self.set_interval(_TICK_INTERVAL_SECONDS, self._advance_tick)
        self._refresh_rows()

    def on_hide(self) -> None:
        with contextlib.suppress(Exception):
            self.app.task_bar.queue.unsubscribe(self._on_queue_change)
        if self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer = None

    def _on_queue_change(self) -> None:
        """Queue notification: post a thread-safe message to the screen."""
        self.post_message(TaskQueueChanged())

    def on_task_queue_changed(self, _event: TaskQueueChanged) -> None:
        """Reconcile rows when the queue posts a change."""
        self._refresh_rows()

    def _focus_initial_row(self) -> None:
        """Land initial focus on the topmost active/queued row.

        Users open the Task Center to manage live work, not to review
        history. Without this, focus lands on the first row regardless
        of status, so an accidental ``c`` on a terminal row is a
        no-op rather than a status flip.

        Falls back to the first row if there are no active/queued
        tasks; falls back to no-op if the screen has no rows at all.
        """
        queue = self.app.task_bar.queue
        for task in queue.active_tasks + queue.queued_tasks:
            row = self._rows.get(task.task_id)
            if row is not None:
                row.focus()
                return
        # No active/queued work: leave focus on whatever AUTO_FOCUS
        # picked (the scroll container, or the first row if one exists).

    def action_refresh_tasks(self) -> None:
        """Manual refresh (r). The subscription drives most updates; this
        gives the user a way to force a reconcile if anything ever drifts."""
        self._refresh_rows()

    def action_clear_history(self) -> None:
        """Drop all DONE/FAILED/CANCELLED rows (bound to capital ``C``).

        ``clear_history`` itself emits a notification so the subscription
        triggers the row reconcile; no manual refresh needed here."""
        self.app.task_bar.queue.clear_history()

    def action_cancel_task(self) -> None:
        """Cancel the task whose row currently has focus.

        Falls back to the first active task if no row has focus.
        """
        focused = self.focused
        if isinstance(focused, TaskRow) and focused.id:
            self.app.task_bar.queue.cancel(focused.id.removeprefix("task-"))
            return
        active = self.app.task_bar.queue.active_task
        if active is not None:
            self.app.task_bar.queue.cancel(active.task_id)

    def action_cursor_down(self) -> None:
        self.focus_next()

    def action_cursor_up(self) -> None:
        self.focus_previous()

    def _all_tasks(self) -> list[Task]:
        """Tasks in display order: active first, then queued, then history."""
        queue = self.app.task_bar.queue
        return queue.active_tasks + queue.queued_tasks + list(reversed(queue.history))

    def _advance_tick(self) -> None:
        """Bump the spinner frame and re-render counts + active row pulse."""
        self._tick += 1
        tasks = self._all_tasks()
        for task in tasks:
            row = self._rows.get(task.task_id)
            if row is not None:
                row.update(task, self._tick)
        self._update_counts(tasks)

    def _refresh_rows(self) -> None:
        """Reconcile rows against the queue: add new, update existing, remove stale."""
        container = self.query_one("#task-rows", VerticalScroll)
        tasks = self._all_tasks()
        seen: set[str] = set()
        for task in tasks:
            seen.add(task.task_id)
            row = self._rows.get(task.task_id)
            if row is None:
                row = TaskRow(task_id=task.task_id)
                self._rows[task.task_id] = row
                container.mount(row)
            row.update(task, self._tick)
        for tid in list(self._rows):
            if tid not in seen:
                row = self._rows.pop(tid)
                try:
                    row.remove()
                except Exception:
                    log.debug("Row %s already removed", tid, exc_info=True)
        self._update_counts(tasks)
        # Swap which widget occupies the 1fr row slot: scroll when
        # there are tasks, headline when the list is empty. Hiding one
        # of the pair (not both) keeps the empty-state headline centred
        # in the available height instead of crowded under a ghost
        # scroll that still claims the space.
        empty = self.query_one("#task-center-empty", Label)
        rows = self.query_one("#task-rows", VerticalScroll)
        has_tasks = bool(tasks)
        empty.display = not has_tasks
        rows.display = has_tasks

    def _update_counts(self, tasks: list[Task]) -> None:
        """Top-right status strip: N running · M queued · K done.

        Prepends a rotating spinner glyph when any task is active so
        the header visibly moves. The rail pulse alone is too subtle
        to communicate 'work in progress' at a glance (bb-18y3).
        """
        counts_label = self.query_one("#task-center-counts", Label)
        active = queued = done = 0
        for t in tasks:
            if t.status == TaskStatus.ACTIVE:
                active += 1
            elif t.status == TaskStatus.QUEUED:
                queued += 1
            elif t.status == TaskStatus.DONE:
                done += 1
        body = msg.TASK_CENTER_COUNTS.format(active=active, queued=queued, done=done)
        if active > 0:
            spinner = _COUNTS_SPINNER_FRAMES[self._tick % len(_COUNTS_SPINNER_FRAMES)]
            counts_label.update(f"{spinner}  {body}")
        else:
            counts_label.update(body)
