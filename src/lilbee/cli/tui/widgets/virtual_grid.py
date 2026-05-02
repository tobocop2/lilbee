"""VirtualGrid: viewport-virtualized card grid for the catalog.

Mounts only the rows of ``ModelCard`` widgets currently in the
viewport (plus a 1-row over-draw above and below), so the compositor
walks ~50 widgets regardless of dataset size. The grid behaves like
``OptionList``-backed scrolling does for the list view.

Keyboard cursor + Selected/LeaveUp/LeaveDown messages mirror
``GridSelect``'s surface so the catalog screen's existing handlers
keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual import events
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui.screens.catalog_utils import CatalogRow
from lilbee.cli.tui.widgets.model_card import ModelCard

# Row of cards is 4 lines tall: name, pill line, specs, status (or hint).
# Card border adds 2 lines top+bottom, so each row visually occupies 6 lines.
_ROW_HEIGHT = 6
# Mount this many extra rows above and below the visible window so fast
# keyboard scrolls (PageUp/PageDown, Fn+Up/Down on Mac) and trackpad
# flings don't churn through mount/unmount on every step. Tuned so a
# single PageDown stays within the buffer so the user sees a smooth
# repaint instead of a frame of empty space.
_OVERDRAW_ROWS = 6
# Default columns when the container width is unknown (pre-mount).
_DEFAULT_COLUMNS = 4
# Coalesce rapid scroll events into a single layout pass. 50 ms keeps
# the buffer current at 20 fps, smooths key-repeat on macOS without
# adding visible lag on a one-off keystroke.
_SCROLL_DEBOUNCE_S = 0.05


class VirtualGrid(VerticalScroll, can_focus=True):
    """Viewport-virtualized grid of ``ModelCard`` widgets."""

    FOCUS_ON_CLICK = False
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("left", "cursor_left", "Left", show=False),
        Binding("right", "cursor_right", "Right", show=False),
        Binding("enter", "select", "Select", show=False),
    ]

    highlighted: reactive[int | None] = reactive(None)

    @dataclass
    class Selected(Message):
        grid: VirtualGrid
        widget: Widget

        @property
        def control(self) -> Widget:
            return self.grid

    @dataclass
    class LeaveUp(Message):
        grid: VirtualGrid

    @dataclass
    class LeaveDown(Message):
        grid: VirtualGrid

    def __init__(
        self,
        rows: list[CatalogRow] | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._rows: list[CatalogRow] = list(rows or [])
        self._cards_per_row: int = _DEFAULT_COLUMNS
        self._mounted_first: int = -1
        self._mounted_last: int = -1
        self._top_spacer: Static | None = None
        self._bot_spacer: Static | None = None
        self._row_widgets: dict[int, Horizontal] = {}
        self._scroll_debounce: Timer | None = None

    @property
    def rows(self) -> list[CatalogRow]:
        return list(self._rows)

    def set_rows(self, rows: list[CatalogRow]) -> None:
        """Replace the dataset; remount visible rows from scratch."""
        self._rows = list(rows)
        self.highlighted = None
        self.scroll_to(y=0, animate=False)
        self._tear_down_rows()
        self._update_layout()

    def on_mount(self) -> None:
        # Spacers + the row mount slot use absolute heights computed from
        # the dataset so the scrollbar tracks the full virtual height.
        self._top_spacer = Static("", classes="vg-spacer-top")
        self._bot_spacer = Static("", classes="vg-spacer-bot")
        self.mount(self._top_spacer)
        self.mount(self._bot_spacer)
        self._update_layout()

    def on_resize(self) -> None:
        # Width drives cards-per-row. Recompute and remount full visible
        # range with the new column count.
        self._update_layout()

    def watch_scroll_y(self, _old: float, _new: float) -> None:
        # Coalesce rapid scroll events (key repeat, trackpad fling) into
        # one layout pass so we don't spend more time mounting/unmounting
        # rows than rendering. Overdraw covers the in-between frames.
        if self._scroll_debounce is not None:
            self._scroll_debounce.stop()
        self._scroll_debounce = self.set_timer(_SCROLL_DEBOUNCE_S, self._update_layout)

    def _columns_for_width(self, width: int) -> int:
        """Compute cards per row from the container width.

        Card min-width 32 (border + padding included); pad of 1 between cards.
        """
        if width <= 0:
            return _DEFAULT_COLUMNS
        per_card = 32 + 1
        return max(1, width // per_card)

    def _total_rows(self) -> int:
        if not self._rows or self._cards_per_row <= 0:
            return 0
        return (len(self._rows) + self._cards_per_row - 1) // self._cards_per_row

    def _update_layout(self) -> None:
        if self._top_spacer is None or self._bot_spacer is None:
            return
        new_columns = self._columns_for_width(self.size.width)
        rebuild_all = new_columns != self._cards_per_row
        self._cards_per_row = new_columns
        if rebuild_all:
            self._tear_down_rows()
        total_rows = self._total_rows()

        if total_rows == 0:
            self._top_spacer.styles.height = 0
            self._bot_spacer.styles.height = 0
            return

        viewport_h = max(self.size.height, _ROW_HEIGHT)
        scroll_y = int(self.scroll_y)
        first = max(0, scroll_y // _ROW_HEIGHT - _OVERDRAW_ROWS)
        last = min(
            total_rows - 1,
            (scroll_y + viewport_h) // _ROW_HEIGHT + _OVERDRAW_ROWS,
        )

        if (first, last) == (self._mounted_first, self._mounted_last) and not rebuild_all:
            return

        # Unmount rows that left the window.
        for r in list(self._row_widgets):
            if r < first or r > last:
                with _suppress_not_found():
                    self._row_widgets[r].remove()
                del self._row_widgets[r]

        # Mount rows that entered the window.
        for r in range(first, last + 1):
            if r in self._row_widgets:
                continue
            row_widget = self._build_row(r)
            self._row_widgets[r] = row_widget
            # Mount before bot_spacer so spacer stays at the end.
            self.mount(row_widget, before=self._bot_spacer)

        self._top_spacer.styles.height = first * _ROW_HEIGHT
        self._bot_spacer.styles.height = max(0, total_rows - last - 1) * _ROW_HEIGHT
        self._mounted_first = first
        self._mounted_last = last
        self._sync_highlight()

    def _build_row(self, row_index: int) -> Horizontal:
        start = row_index * self._cards_per_row
        end = min(start + self._cards_per_row, len(self._rows))
        cards = [ModelCard(self._rows[i]) for i in range(start, end)]
        return Horizontal(*cards, classes="vg-row")

    def _tear_down_rows(self) -> None:
        for row in self._row_widgets.values():
            with _suppress_not_found():
                row.remove()
        self._row_widgets.clear()
        self._mounted_first = -1
        self._mounted_last = -1

    def _sync_highlight(self) -> None:
        for index, card in self._iter_mounted_cards():
            card.set_class(index == self.highlighted, "-highlight")
            card.selected = index == self.highlighted

    def _iter_mounted_cards(self):
        for row_index in sorted(self._row_widgets):
            row = self._row_widgets[row_index]
            for col_index, card in enumerate(row.query(ModelCard)):
                index = row_index * self._cards_per_row + col_index
                if index < len(self._rows):
                    yield index, card

    def _scroll_index_into_view(self, index: int) -> None:
        if not self._rows:
            return
        row = index // self._cards_per_row
        target_y = row * _ROW_HEIGHT
        viewport_h = self.size.height
        if target_y < self.scroll_y:
            self.scroll_to(y=target_y, animate=False)
        elif target_y + _ROW_HEIGHT > self.scroll_y + viewport_h:
            self.scroll_to(y=target_y - viewport_h + _ROW_HEIGHT, animate=False)

    def watch_highlighted(self, _old: int | None, new: int | None) -> None:
        if new is None:
            self._sync_highlight()
            return
        self._scroll_index_into_view(new)
        self._update_layout()

    def action_cursor_up(self) -> None:
        if self.highlighted is None:
            self.highlighted = 0
            return
        if self.highlighted < self._cards_per_row:
            self.post_message(self.LeaveUp(self))
            return
        self.highlighted = max(0, self.highlighted - self._cards_per_row)

    def action_cursor_down(self) -> None:
        if self.highlighted is None:
            self.highlighted = 0
            return
        next_index = self.highlighted + self._cards_per_row
        if next_index >= len(self._rows):
            self.post_message(self.LeaveDown(self))
            return
        self.highlighted = next_index

    def action_cursor_left(self) -> None:
        if self.highlighted is None:
            self.highlighted = 0
            return
        self.highlighted = max(0, self.highlighted - 1)

    def action_cursor_right(self) -> None:
        if self.highlighted is None:
            self.highlighted = 0
            return
        self.highlighted = min(len(self._rows) - 1, self.highlighted + 1)

    def action_select(self) -> None:
        if self.highlighted is None:
            return
        for index, card in self._iter_mounted_cards():
            if index == self.highlighted:
                self.post_message(self.Selected(self, card))
                return

    def on_click(self, event: events.Click) -> None:
        """Mouse click selects a card. First click highlights, second selects.

        Mirrors ``GridSelect.on_click`` semantics so the catalog screen's
        existing ``Selected`` handler keeps working under VirtualGrid.
        """
        if event.widget is None:
            return
        clicked_card: ModelCard | None = None
        for ancestor in event.widget.ancestors_with_self:
            if isinstance(ancestor, ModelCard):
                clicked_card = ancestor
                break
        if clicked_card is None:
            return
        clicked_index: int | None = None
        for index, card in self._iter_mounted_cards():
            if card is clicked_card:
                clicked_index = index
                break
        if clicked_index is None:
            return
        if self.highlighted == clicked_index:
            self.post_message(self.Selected(self, clicked_card))
        else:
            self.highlighted = clicked_index
        self.focus()

    def highlight_first(self) -> None:
        """Move highlight to the first card; mirrors ``GridSelect.highlight_first``."""
        if self._rows:
            self.highlighted = 0

    def highlight_last(self) -> None:
        """Move highlight to the last card; mirrors ``GridSelect.highlight_last``."""
        if self._rows:
            self.highlighted = len(self._rows) - 1


class _SuppressNotFound:
    """Context manager: swallow exceptions from removing already-removed widgets."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None


_suppress_not_found = _SuppressNotFound
