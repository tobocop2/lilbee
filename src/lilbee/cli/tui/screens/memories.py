"""Memories management screen: browse, delete, toggle-shared, confirm.

A single :class:`DataTable` of the human's (``owner=local``) stored memories
with vim-style navigation. ``d`` deletes the highlighted memory (through the
shared :class:`ConfirmDialog`), ``s`` toggles whether it is shared with agents,
and ``c`` confirms an auto-extracted memory so it can be recalled. ``q`` / Esc
backs out. Mirrors :class:`WikiDraftsScreen`'s structure and keymap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Input

from lilbee.app.memory import forget, list_memories, memory_enabled, set_memory_flags
from lilbee.cli.tui import messages as msg

if TYPE_CHECKING:
    from lilbee.data.store import MemoryRow

log = logging.getLogger(__name__)


def _flag_label(value: bool) -> str:
    """Render a boolean memory flag as a human yes/no."""
    return msg.MEMORIES_FLAG_YES if value else msg.MEMORIES_FLAG_NO


class MemoriesScreen(Screen[None]):
    """Review-surface screen for the human's long-term memories."""

    CSS_PATH = "memories.tcss"
    AUTO_FOCUS = "#memories-table"
    HELP = "Manage memories. j/k navigate, d delete, s toggle shared, c confirm, / search, q back."

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "dismiss_or_back", "Back", show=False),
        Binding("d", "delete", "Delete", show=True),
        Binding("s", "toggle_shared", "Shared", show=True),
        Binding("c", "confirm", "Confirm", show=True),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("j", "cursor_down", "Nav", show=False),
        Binding("k", "cursor_up", "Nav", show=False),
        Binding("g", "jump_top", "Top", show=False),
        Binding("G", "jump_bottom", "End", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._memories: list[MemoryRow] = []
        self._filter: str = ""

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        table: DataTable[str] = DataTable(id="memories-table")
        table.cursor_type = "row"
        yield Vertical(
            Input(placeholder=msg.MEMORIES_SEARCH_PLACEHOLDER, id="memories-search"),
            table,
            id="memories-layout",
        )
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#memories-table", DataTable)
        table.add_columns(
            msg.MEMORIES_COLUMN_KIND,
            msg.MEMORIES_COLUMN_SHARED,
            msg.MEMORIES_COLUMN_CONFIRMED,
            msg.MEMORIES_COLUMN_TEXT,
        )
        self._load_memories()

    def _load_memories(self) -> None:
        """Fetch memories and populate the table, respecting the active filter."""
        table = self.query_one("#memories-table", DataTable)
        table.clear()
        if not memory_enabled():
            self.notify(msg.MEMORIES_DISABLED, severity="warning")
            self._memories = []
            return
        try:
            self._memories = list_memories()
        except Exception as exc:
            log.debug("Failed to list memories", exc_info=True)
            self._memories = []
            self.notify(msg.MEMORIES_LOAD_FAILED.format(error=exc), severity="error")
            return

        visible = self._visible_memories()
        if not visible:
            self.notify(msg.MEMORIES_EMPTY)
            return
        for m in visible:
            table.add_row(
                m.kind.value,
                _flag_label(m.shared),
                _flag_label(m.confirmed),
                m.text,
                key=m.id,
            )

    def _visible_memories(self) -> list[MemoryRow]:
        """Apply the current text filter to the loaded memory list."""
        if not self._filter:
            return self._memories
        needle = self._filter.lower()
        return [m for m in self._memories if needle in m.text.lower()]

    def _highlighted_id(self) -> str | None:
        """Return the id of the highlighted row, or ``None`` when empty."""
        table = self.query_one("#memories-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        except Exception:
            return None
        if row_key is None or row_key.value is None:
            return None
        return str(row_key.value)

    @on(Input.Changed, "#memories-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Filter memories as the user types."""
        self._filter = event.value.strip()
        self._load_memories()

    def action_focus_search(self) -> None:
        """Focus the search input (``/`` keybinding)."""
        self.query_one("#memories-search", Input).focus()

    def action_dismiss_or_back(self) -> None:
        """Clear the search if active, otherwise back out."""
        search = self.query_one("#memories-search", Input)
        if search.value:
            search.value = ""
            return
        self.action_go_back()

    def action_go_back(self) -> None:
        """Pop back to the previous screen, unless this is the only one."""
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()

    def _table_or_none(self) -> DataTable[str] | None:
        """Return the memories table unless an Input is focused."""
        if isinstance(self.focused, Input):
            return None
        return self.query_one("#memories-table", DataTable)

    def action_cursor_down(self) -> None:
        table = self._table_or_none()
        if table is not None:
            table.action_cursor_down()

    def action_cursor_up(self) -> None:
        table = self._table_or_none()
        if table is not None:
            table.action_cursor_up()

    def action_jump_top(self) -> None:
        table = self._table_or_none()
        if table is not None:
            table.scroll_home()

    def action_jump_bottom(self) -> None:
        table = self._table_or_none()
        if table is not None:
            table.scroll_end()

    def action_delete(self) -> None:
        """Prompt for confirmation, then delete the highlighted memory."""
        memory_id = self._highlighted_id()
        if memory_id is None:
            return
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self._do_delete(memory_id)

        self.app.push_screen(
            ConfirmDialog(
                msg.MEMORIES_DELETE_CONFIRM_TITLE,
                msg.MEMORIES_DELETE_CONFIRM_MESSAGE,
            ),
            _on_confirm,
        )

    def _do_delete(self, memory_id: str) -> None:
        """Execute the delete and refresh the list."""
        try:
            forget(memory_id)
        except Exception as exc:
            log.debug("Delete failed for %s", memory_id, exc_info=True)
            self.notify(msg.MEMORIES_DELETE_FAILED.format(error=exc), severity="error")
            return
        self.notify(msg.MEMORIES_DELETED)
        self._load_memories()

    def action_toggle_shared(self) -> None:
        """Flip the highlighted memory's shared-with-agents flag."""
        memory_id = self._highlighted_id()
        if memory_id is None:
            return
        memory = self._memory_by_id(memory_id)
        if memory is None:
            return
        new_shared = not memory.shared
        try:
            set_memory_flags(memory_id, shared=new_shared)
        except Exception as exc:
            log.debug("Toggle shared failed for %s", memory_id, exc_info=True)
            self.notify(msg.MEMORIES_FLAG_FAILED.format(error=exc), severity="error")
            return
        self.notify(msg.MEMORIES_SHARED_ON if new_shared else msg.MEMORIES_SHARED_OFF)
        self._load_memories()

    def action_confirm(self) -> None:
        """Confirm the highlighted memory so it can be recalled."""
        memory_id = self._highlighted_id()
        if memory_id is None:
            return
        try:
            set_memory_flags(memory_id, confirmed=True)
        except Exception as exc:
            log.debug("Confirm failed for %s", memory_id, exc_info=True)
            self.notify(msg.MEMORIES_FLAG_FAILED.format(error=exc), severity="error")
            return
        self.notify(msg.MEMORIES_CONFIRMED)
        self._load_memories()

    def _memory_by_id(self, memory_id: str) -> MemoryRow | None:
        """Look up a loaded memory by id."""
        for m in self._memories:
            if m.id == memory_id:
                return m
        return None
