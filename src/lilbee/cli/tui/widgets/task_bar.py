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
from lilbee.providers.warm_progress import WarmPhase, WarmProgress

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

log = logging.getLogger(__name__)

_CSS_FILE = Path(__file__).parent / "task_bar.tcss"

_DONE_FLASH_SECONDS = 2.0
_POLL_INTERVAL_ACTIVE_S = 0.1
_POLL_INTERVAL_IDLE_S = 1.0

# Pulsing-dot cadence: on/off flip at half of this tick count.
# 10 Hz poll x 5 = 500 ms per half cycle, which is a 1 Hz dot pulse,
# matching the active-row rail pulse in the Task Center.
_DOT_PULSE_HALF_TICKS = 5
_DOT_GLYPH = "●"

# Warm progress bar: filled/track glyphs and width, matching the fleet panel bars.
_WARM_BAR_FILL = "▓"
_WARM_BAR_TRACK = "░"
_WARM_BAR_WIDTH = 12
# Indeterminate sweep (loading engine has no byte signal): a lit window that walks
# the track so the bar reads as "working", not stalled. Advanced by the poll tick.
_WARM_SWEEP_WIDTH = 3


def _progress_bar(fraction: float) -> str:
    """A determinate fill bar for the byte-progress (reading-weights) phase."""
    filled = round(max(0.0, min(1.0, fraction)) * _WARM_BAR_WIDTH)
    return _WARM_BAR_FILL * filled + _WARM_BAR_TRACK * (_WARM_BAR_WIDTH - filled)


def _sweep_bar(tick: int) -> str:
    """An indeterminate bar with a lit window of fixed width walking (and wrapping)
    across the track, keyed to *tick*, so it always shows motion, never a blank."""
    start = tick % _WARM_BAR_WIDTH
    lit = {(start + offset) % _WARM_BAR_WIDTH for offset in range(_WARM_SWEEP_WIDTH)}
    return "".join(_WARM_BAR_FILL if i in lit else _WARM_BAR_TRACK for i in range(_WARM_BAR_WIDTH))


def _warm_detail(progress: WarmProgress | None, tick: int = 0) -> str | None:
    """Phase word plus a progress bar for the cold-start chat warm line.

    A determinate byte bar while paging weights, an indeterminate sweep while the
    engine loads (no byte signal). None once past an active phase, so the line
    reads as live progress, not a stalled spinner.
    """
    if progress is None:
        return None
    if progress.phase is WarmPhase.STARTING:
        return f"{_sweep_bar(tick)}  {msg.TASKBAR_WARM_STARTING}"
    if progress.phase is WarmPhase.LOADING_ENGINE:
        return f"{_sweep_bar(tick)}  {msg.TASKBAR_WARM_LOADING}"
    if progress.phase is WarmPhase.READING_WEIGHTS:
        fraction = progress.bytes_done / progress.bytes_total if progress.bytes_total else 0.0
        return (
            f"{_progress_bar(fraction)}  {msg.TASKBAR_WARM_READING.format(pct=int(fraction * 100))}"
        )
    return None


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
        # Failures among the just-finished batch (not all of persistent history).
        self._flash_failed_count: int = 0
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
        # True when no work is visible: the timer drops to 1 Hz here since
        # the fingerprint cache short-circuits the render path. Flips
        # back on the first non-idle event.
        self._idle_mode: bool = True

    def compose(self) -> ComposeResult:
        yield Label("", id="task-status-label")

    def on_mount(self) -> None:
        self._refresh_display()
        # Capture the handle so we can cancel the poll on unmount. Without
        # this, a screen push/pop cycle leaves the previous TaskBar's
        # interval firing against a detached widget, racing with the new
        # TaskBar and occasionally setting ``display=False`` on the live
        # instance. Start at the idle cadence; the first tick re-arms at
        # 10 Hz if work is already in flight.
        self._interval = self.set_interval(_POLL_INTERVAL_IDLE_S, self._tick)

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
        """Poll the shared queue and re-render."""
        self._tick_count += 1
        self._refresh_display()

    def _sync_poll_cadence(self, fully_idle: bool) -> None:
        """Re-arm the poll timer at idle/active cadence on state transitions."""
        if fully_idle == self._idle_mode:
            return
        self._idle_mode = fully_idle
        if self._interval is not None:
            self._interval.stop()
        interval = _POLL_INTERVAL_IDLE_S if fully_idle else _POLL_INTERVAL_ACTIVE_S
        self._interval = self.set_interval(interval, self._tick)

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
                new_done = [
                    t
                    for t in history
                    if t.task_id not in self._flashed_ids
                    and t.status in (TaskStatus.DONE, TaskStatus.FAILED)
                ]
                if new_done:
                    for t in new_done:
                        self._flashed_ids.add(t.task_id)
                    self._flash_until_tick = self._tick_count + int(
                        _DONE_FLASH_SECONDS / _POLL_INTERVAL_ACTIVE_S
                    )
                    self._flash_failed_count = sum(
                        1 for t in new_done if t.status == TaskStatus.FAILED
                    )
                    self._flash_outcome = (
                        TaskStatus.FAILED if self._flash_failed_count else TaskStatus.DONE
                    )

        idle = not active and not queued and not in_flash and self._flash_outcome is None
        pending = self._controller.pending_sync_count if idle else 0
        spawning_roles = sorted(self._controller.spawning_roles) if idle else []
        warm_line = self._warm_line() if idle else None
        fully_idle = idle and pending == 0 and not spawning_roles and warm_line is None
        self._sync_poll_cadence(fully_idle)
        if fully_idle:
            self.display = False
            self._last_render_fingerprint = None
            return

        self.display = True
        dot_color, summary = self._status_line(
            active, queued, spawning_roles, pending, warm_line, idle=idle
        )
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
            warm_line,
        )
        if fingerprint == self._last_render_fingerprint:
            return
        self._last_render_fingerprint = fingerprint

        label_text = f" [{dot_color}]{_DOT_GLYPH}[/]  {summary}    [i dim]{hint_text}[/]"
        with contextlib.suppress(Exception):
            label = self.query_one("#task-status-label", Label)
            label.update(label_text)

    def _status_line(
        self,
        active: list,  # type: ignore[type-arg]
        queued: list,  # type: ignore[type-arg]
        spawning_roles: list[str],
        pending: int,
        warm_line: str | None,
        *,
        idle: bool,
    ) -> tuple[str, str]:
        """Pick the dot color and summary text for the current bar state."""
        if idle and warm_line is not None:
            return "$primary", warm_line
        if idle and spawning_roles:
            return "$primary", self._spawning_workers_template(spawning_roles)
        if idle and pending > 0:
            return "$text-muted", self._pending_sync_template(pending).format(count=pending)
        return self._compose_segments(active, queued)

    def _warm_line(self) -> str | None:
        """The cold-start chat warm line, or None when chat isn't warming."""
        from lilbee.app.placement import active_chat_warm_progress

        detail = _warm_detail(active_chat_warm_progress(), self._tick_count)
        return msg.TASKBAR_WARM.format(detail=detail) if detail else None

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
            count = self._flash_failed_count
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
