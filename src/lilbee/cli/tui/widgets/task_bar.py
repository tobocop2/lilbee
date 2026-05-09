"""TaskBar widget: slim 1-line status indicator polling the shared TaskQueue."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.timer import Timer
from textual.widgets import Label, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus
from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

log = logging.getLogger(__name__)

_CSS_FILE = Path(__file__).parent / "task_bar.tcss"

_DONE_FLASH_SECONDS = 2.0
_POLL_INTERVAL_SECONDS = 0.1

# Pulsing-dot cadence: on/off flip at half of this tick count.
# 10 Hz poll x 5 = 500 ms per half cycle, which is a 1 Hz dot pulse,
# matching the active-row rail pulse in the Task Center.
_DOT_PULSE_HALF_TICKS = 5
_DOT_GLYPH = "●"


class TaskBar(Static):
    """Slim 1-line status indicator for background tasks.

    Shows a compact summary when tasks are active and hides when idle.
    Detailed progress (spinners, progress bars, task panels) lives in
    the Task Center screen, accessible via ``t``.
    """

    app: LilbeeApp  # type: ignore[assignment]

    # NOTE: no ``dock: bottom`` here. TaskBar is always mounted inside a
    # ``BottomBars`` container that owns the dock; multiple dock-bottom
    # siblings overlap at the same row in Textual (see BottomBars docstring).
    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._tick_count = 0
        # Timestamp (tick count) at which the current flash started.
        # None when no flash is active. The 2 s completion/failure
        # flash holds the coloured dot + summary past queue drain.
        self._flash_until_tick: int | None = None
        self._flash_outcome: TaskStatus | None = None
        # Task ids we've already flashed on. Task Center rows linger in
        # history after DONE/FAILED/CANCELLED so the user can review
        # recent work; without this gate the bar would re-flash the same
        # task every poll because ``history[-1]`` keeps matching.
        self._flashed_ids: set[str] = set()
        # Fingerprint of the most recently painted label state. Each
        # tick fires at 10 Hz; if nothing visible has changed (no new
        # tasks, no progress shift, no pulse-phase flip) the heavy
        # ``Label.update`` -- which re-segments + re-styles the line --
        # is skipped. Visible idle cost drops from "every tick" to "on
        # actual change", recovering ~5-8 ms/sec on idle screens.
        self._last_render_fingerprint: tuple[object, ...] | None = None
        # Poll handle. Set in on_mount and cleared in on_unmount; declared
        # here so on_unmount can read it directly without a getattr fallback.
        self._interval: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Label("", id="task-status-label")

    def on_mount(self) -> None:
        self._refresh_display()
        # Capture the handle so we can cancel the poll on unmount. Without
        # this, a screen push/pop cycle leaves the previous TaskBar's
        # interval firing against a detached widget, racing with the new
        # TaskBar and occasionally setting ``display=False`` on the live
        # instance (bb-3uzp).
        self._interval = self.set_interval(_POLL_INTERVAL_SECONDS, self._tick)

    def on_unmount(self) -> None:
        if self._interval is not None:
            self._interval.stop()
            self._interval = None

    @property
    def _controller(self) -> TaskBarController:
        return self.app.task_bar

    @property
    def queue(self) -> TaskQueue:
        """Expose the shared queue for callers that iterate or advance it."""
        return self._controller.queue

    def add_task(
        self,
        name: str,
        task_type: str,
        fn: Callable[[], None] | None = None,
        *,
        indeterminate: bool = False,
    ) -> str:
        """Enqueue a task via the app's controller. Returns the task_id."""
        return self._controller.add_task(name, task_type, fn, indeterminate=indeterminate)

    def update_task(
        self,
        task_id: str,
        progress: float,
        detail: str = "",
        *,
        indeterminate: bool | None = None,
    ) -> None:
        self._controller.update_task(task_id, progress, detail, indeterminate=indeterminate)

    def complete_task(self, task_id: str) -> None:
        self._controller.complete_task(task_id)

    def fail_task(self, task_id: str, detail: str = "") -> None:
        self._controller.fail_task(task_id, detail)

    def cancel_task(self, task_id: str) -> None:
        self._controller.cancel_task(task_id)

    def _tick(self) -> None:
        """Poll the shared queue at 10 Hz and re-render."""
        self._tick_count += 1
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Rebuild the 1-line status label from the shared queue.

        Visual language:
        - Leading ``●`` pulses ``$primary`` <-> ``$primary-lighten-2`` at 1 Hz
          when anything is active. Dim ``$text-muted`` when only queued tasks
          remain, ``$success`` during a completion flash, ``$error`` during
          a failure flash.
        - The text either reads ``{name} {pct}`` (one active, zero queued),
          ``{N} tasks running`` (plural), ``{N} queued`` (throttle mode),
          or the flash copy.
        - Right-aligned muted-italic ``Press t for Tasks`` hint.
        """
        queue = self.queue
        active = queue.active_tasks
        queued = queue.queued_tasks
        history = queue.history

        # Drop flashed-id entries for tasks the user has cleared from
        # history. Without this prune, the set grows unbounded over a
        # long session even though any id not in history can't re-flash.
        if self._flashed_ids:
            live_ids = {t.task_id for t in history}
            self._flashed_ids &= live_ids

        in_flash = self._flash_until_tick is not None and self._tick_count <= self._flash_until_tick
        if not in_flash:
            self._flash_until_tick = None
            self._flash_outcome = None
            # Flash on the freshest completion that hasn't been flashed
            # yet. History now persists (rows show as DONE in Task
            # Center until cleared), so we must gate by task_id instead
            # of "history is non-empty".
            if not active and not queued and history:
                last = history[-1]
                if last.task_id not in self._flashed_ids and last.status in (
                    TaskStatus.DONE,
                    TaskStatus.FAILED,
                ):
                    self._flashed_ids.add(last.task_id)
                    self._flash_until_tick = self._tick_count + int(
                        _DONE_FLASH_SECONDS / _POLL_INTERVAL_SECONDS
                    )
                    self._flash_outcome = last.status

        idle = not active and not queued and not in_flash and self._flash_outcome is None
        pending = self._controller.pending_sync_count if idle else 0
        spawning_roles = sorted(self._controller.spawning_roles) if idle else []
        if idle and pending == 0 and not spawning_roles:
            self.display = False
            self._last_render_fingerprint = None
            return

        self.display = True
        if idle and spawning_roles:
            dot_color = "$primary"
            summary = self._spawning_workers_template(spawning_roles)
        elif idle and pending > 0:
            dot_color = "$text-muted"
            key = self._pending_sync_template(pending)
            summary = key.format(count=pending)
        else:
            dot_color, summary = self._compose_segments(active, queued)
        hint_text = self._hint_copy()
        # Fingerprint captures every variable the label content depends
        # on. Recomputing it is essentially free; the win comes from
        # skipping ``Label.update`` when nothing visible has changed,
        # since update re-segments and re-styles the whole line.
        fingerprint: tuple[object, ...] = (
            dot_color,
            summary,
            hint_text,
            in_flash,
            self._flash_outcome,
            pending,
            tuple(spawning_roles),
        )
        if fingerprint == self._last_render_fingerprint:
            return
        self._last_render_fingerprint = fingerprint

        label_text = f" [{dot_color}]{_DOT_GLYPH}[/]  {summary}    [i dim]{hint_text}[/]"
        with contextlib.suppress(Exception):
            label = self.query_one("#task-status-label", Label)
            label.update(label_text)

    def _spawning_workers_template(self, roles: list[str]) -> str:
        """Render the active worker-warmup hint for the bottom bar."""
        labels = ", ".join(role.replace("_", " ") for role in roles)
        template = msg.TASKBAR_STARTING_WORKER if len(roles) == 1 else msg.TASKBAR_STARTING_WORKERS
        return template.format(labels=labels)

    def _pending_sync_template(self, pending: int) -> str:
        """Pick the singular/plural hint, swapping in the Esc-prefixed copy
        when a chat ``Input`` swallows printable characters before bindings fire.
        """
        from textual.widgets import Input

        try:
            input_focused = isinstance(self.app.focused, Input)
        except Exception:
            input_focused = False
        if pending == 1:
            return (
                msg.TASKBAR_SYNC_PENDING_ONE_INPUT
                if input_focused
                else msg.TASKBAR_SYNC_PENDING_ONE
            )
        return (
            msg.TASKBAR_SYNC_PENDING_PLURAL_INPUT
            if input_focused
            else msg.TASKBAR_SYNC_PENDING_PLURAL
        )

    def _hint_copy(self) -> str:
        """Return the right-aligned hint, context-aware.

        When a chat ``Input`` (or similar) is focused the ``t`` keypress is
        eaten before the app-level binding fires, so the user needs
        ``Esc then t``. Every other screen (wizard grid, catalog,
        settings, task center) lets ``t`` bubble, so a shorter ``Press t
        for Tasks`` is accurate and easier to scan.
        """
        from textual.widgets import Input

        try:
            focused = self.app.focused
        except Exception:
            return msg.TASKBAR_HINT
        if isinstance(focused, Input):
            return msg.TASKBAR_HINT_INPUT
        return msg.TASKBAR_HINT

    def _compose_segments(self, active: list, queued: list) -> tuple[str, str]:
        """Return (dot color, text summary) for the current state."""
        # Pulsing even/odd cadence, shared with TaskRow's rail pulse.
        on_beat = (self._tick_count // _DOT_PULSE_HALF_TICKS) % 2 == 0

        if self._flash_outcome == TaskStatus.DONE:
            return "$success", msg.TASKBAR_ALL_DONE
        if self._flash_outcome == TaskStatus.FAILED:
            count = sum(1 for t in self.queue.history if t.status == TaskStatus.FAILED)
            key = msg.TASKBAR_FAILED if count == 1 else msg.TASKBAR_FAILED_PLURAL
            return "$error", key.format(count=count)

        parts: list[str] = []
        if active:
            count = len(active)
            task = active[0]
            if count == 1 and not queued:
                pct = "" if task.indeterminate else f"  [b]{task.progress:.1f}%[/b]"
                parts.append(f"[b]{task.name}[/b]{pct}")
            else:
                key = msg.TASKBAR_ONE if count == 1 else msg.TASKBAR_MULTIPLE
                parts.append(key.format(count=count))
                parts.append(f"[b]{task.name}[/b]")
        if queued:
            parts.append(f"[dim]{msg.TASKBAR_QUEUED_COUNT.format(count=len(queued))}[/dim]")

        dot_color = ("$primary" if on_beat else "$primary-lighten-2") if active else "$text-muted"
        return dot_color, "  ·  ".join(parts)
