"""GridSelect — responsive grid with cursor navigation.

Ported from toad (https://github.com/batrachianai/toad).
Extends Textual's ItemGrid with keyboard cursor, highlight class, and messages.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import ClassVar

from textual import containers, events
from textual.binding import Binding, BindingType
from textual.layouts.grid import GridLayout
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget


class GridSelect(containers.ItemGrid, can_focus=True):
    """A responsive grid that supports arrow-key cursor navigation and selection."""

    FOCUS_ON_CLICK = False
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("left", "cursor_left", "Left", show=False),
        Binding("right", "cursor_right", "Right", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("tab", "tab_next", "Tab Next", show=False),
        Binding("shift+tab", "tab_previous", "Tab Previous", show=False),
    ]

    highlighted: reactive[int | None] = reactive(None)

    @dataclass
    class Selected(Message):
        grid_select: GridSelect
        widget: Widget

        @property
        def control(self) -> Widget:
            return self.grid_select

    @dataclass
    class Highlighted(Message):
        grid_select: GridSelect
        widget: Widget

        @property
        def control(self) -> Widget:
            return self.grid_select

    @dataclass
    class LeaveUp(Message):
        grid_select: GridSelect

    @dataclass
    class LeaveDown(Message):
        grid_select: GridSelect

    def __init__(
        self,
        *children: Widget,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        min_column_width: int = 30,
        max_column_width: int | None = None,
    ) -> None:
        super().__init__(
            *children,
            name=name,
            id=id,
            classes=classes,
            min_column_width=min_column_width,
            max_column_width=max_column_width,
        )

    @property
    def grid_size(self) -> tuple[int, int] | None:
        if not isinstance(self.layout, GridLayout):
            return None
        return self.layout.grid_size

    def highlight_first(self) -> None:
        self.highlighted = 0

    def highlight_last(self) -> None:
        if self.children:
            self.highlighted = len(self.children) - 1

    def on_focus(self) -> None:
        if self.highlighted is None:
            self.highlighted = 0
        self.reveal_highlight()

    def on_blur(self) -> None:
        self.highlighted = None

    def reveal_highlight(self) -> None:
        if self.highlighted is None:
            return
        try:
            widget = self.children[self.highlighted]
        except IndexError:
            return
        if not self.screen.can_view_entire(widget):
            self.screen.scroll_to_center(widget, origin_visible=True)

    def watch_highlighted(self, old_highlighted: int | None, highlighted: int | None) -> None:
        if old_highlighted is not None:
            with contextlib.suppress(IndexError):
                self.children[old_highlighted].remove_class("-highlight")
        if highlighted is not None:
            try:
                widget = self.children[highlighted]
                widget.add_class("-highlight")
                self.post_message(self.Highlighted(self, widget))
            except IndexError:
                pass
        self.reveal_highlight()

    def validate_highlighted(self, highlighted: int | None) -> int | None:
        if highlighted is None:
            return None
        if not self.children:
            return None
        if highlighted < 0:
            return 0
        if highlighted >= len(self.children):
            return len(self.children) - 1
        return highlighted

    def action_cursor_up(self) -> None:
        if (grid_size := self.grid_size) is None:
            self.post_message(self.LeaveUp(self))
            return
        if self.highlighted is None:
            self.highlighted = 0
        else:
            width, _height = grid_size
            if self.highlighted >= width:
                self.highlighted -= width
            else:
                self.post_message(self.LeaveUp(self))

    def action_cursor_down(self) -> None:
        if (grid_size := self.grid_size) is None:
            self.post_message(self.LeaveDown(self))
            return
        if self.highlighted is None:
            self.highlighted = 0
        else:
            width, _height = grid_size
            if self.highlighted + width < len(self.children):
                self.highlighted += width
            else:
                self.post_message(self.LeaveDown(self))

    def action_cursor_left(self) -> None:
        if self.highlighted is None:
            self.highlighted = 0
        else:
            self.highlighted -= 1

    def action_cursor_right(self) -> None:
        if self.highlighted is None:
            self.highlighted = 0
        else:
            self.highlighted += 1

    def on_click(self, event: events.Click) -> None:
        if event.widget is None:
            return
        highlighted_widget: Widget | None = None
        if self.highlighted is not None:
            with contextlib.suppress(IndexError):
                highlighted_widget = self.children[self.highlighted]
        for widget in event.widget.ancestors_with_self:
            if widget in self.children:
                if highlighted_widget is not None and highlighted_widget is widget:
                    self.action_select()
                else:
                    self.highlighted = self.children.index(widget)
                break
        self.focus()

    def action_select(self) -> None:
        if self.highlighted is not None:
            try:
                widget = self.children[self.highlighted]
            except IndexError:
                pass
            else:
                self.post_message(self.Selected(self, widget))

    def action_tab_next(self) -> None:
        """Advance highlight linearly; escape grid when past the last card."""
        if not self.children:
            self.post_message(self.LeaveDown(self))
            return
        if self.highlighted is None:
            self.highlighted = 0
        elif self.highlighted >= len(self.children) - 1:
            self.post_message(self.LeaveDown(self))
        else:
            self.highlighted += 1

    def action_tab_previous(self) -> None:
        """Retreat highlight linearly; escape grid when before the first card."""
        if not self.children:
            self.post_message(self.LeaveUp(self))
            return
        if self.highlighted is None:
            self.highlighted = len(self.children) - 1
        elif self.highlighted <= 0:
            self.post_message(self.LeaveUp(self))
        else:
            self.highlighted -= 1
