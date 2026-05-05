"""ModelGrid: single-render-surface grid of catalog cards.

The widget owns a single rendering surface: ``render_line(y)`` paints
one strip at a time so a fast scroll repaints visible cells rather than
cycling the compositor through mount + reflow + repaint frames.

Each card slot is composed by ``_render_card_strip(row, *, selected, width)``
on demand; ``ModelCard`` stays in the codebase because the setup wizard
still mounts it inside a ``GridSelect``.

Each ``ModelGrid`` section sizes itself to its full content height (via
``get_content_height``); the outer ``#catalog-grid`` ``VerticalScroll``
handles scrolling across sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from textual import events
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.geometry import Region, Size
from textual.message import Message
from textual.reactive import reactive
from textual.strip import Strip
from textual.style import Style
from textual.widget import Widget

from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.catalog_theme import MIDDLE_DOT, TASK_COLORS

_CSS_FILE = Path(__file__).parent / "model_grid.tcss"

# Card body is 4 visual lines (name / pills / specs / status) plus a 1-line
# hint slot painted only when a local card is highlighted-but-not-installed.
# We always pre-allocate the slot so highlight transitions don't reflow
# neighbours.
_CARD_BODY_HEIGHT = 5
"""Visual lines of card content: name / pills / specs / status / hint."""

_CARD_HEIGHT = _CARD_BODY_HEIGHT + 2
"""Total card height including the top and bottom border rows."""

_ROW_GUTTER = 0
_ROW_HEIGHT = _CARD_HEIGHT + _ROW_GUTTER
_DEFAULT_COLUMNS = 4
# Card body needs ~32 cells before pills wrap awkwardly.
_CARD_MIN_WIDTH = 32
_CARD_GUTTER = 1

# Box-drawing characters for the card frame. Drawn explicitly so the
# tile is visible regardless of theme background color (Solarized, etc.).
_BORDER_TL, _BORDER_TR, _BORDER_BL, _BORDER_BR = "╭", "╮", "╰", "╯"
_BORDER_H, _BORDER_V = "─", "│"


@dataclass
class _CardLines:
    """Pre-rendered content lines for one card. Each entry is one terminal row."""

    lines: list[Content]


class ModelGrid(Widget, can_focus=True):
    """Single-render-surface grid of ``CatalogRow`` cards."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

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
        """Posted when a card is activated. ``row`` is the underlying CatalogRow."""

        grid: ModelGrid
        row: CatalogRow

        @property
        def control(self) -> ModelGrid:
            return self.grid

    @dataclass
    class LeaveUp(Message):
        grid: ModelGrid

    @dataclass
    class LeaveDown(Message):
        grid: ModelGrid

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

    @property
    def rows(self) -> list[CatalogRow]:
        """The dataset backing this grid (defensive copy)."""
        return list(self._rows)

    @property
    def columns_per_row(self) -> int:
        """Current column count, derived from the container width on resize."""
        return self._cards_per_row

    def set_rows(self, rows: list[CatalogRow]) -> None:
        """Replace the dataset and reset the highlight."""
        self._rows = list(rows)
        self.highlighted = None
        self.refresh(layout=True)

    def on_resize(self) -> None:
        new_cols = self._columns_for_width(self.size.width)
        if new_cols != self._cards_per_row:
            self._cards_per_row = new_cols
            self.refresh(layout=True)

    @staticmethod
    def _columns_for_width(width: int) -> int:
        if width <= 0:
            return _DEFAULT_COLUMNS
        return max(1, width // (_CARD_MIN_WIDTH + _CARD_GUTTER))

    def _total_rows(self) -> int:
        if not self._rows or self._cards_per_row <= 0:
            return 0
        return (len(self._rows) + self._cards_per_row - 1) // self._cards_per_row

    def get_content_width(self, container: Size, viewport: Size) -> int:
        return container.width

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        # Recompute columns from the available width so the height stays
        # consistent with what ``render_line`` will draw.
        if not self._rows:
            return 0
        cols = self._columns_for_width(width)
        rows = (len(self._rows) + cols - 1) // cols
        return rows * _ROW_HEIGHT - 1

    def watch_highlighted(self, _old: int | None, new: int | None) -> None:
        """Repaint and scroll the highlighted cell into view.

        The scroll-into-view side effect lets the outer ``VerticalScroll``
        update ``scroll_y`` on every cursor move, which is what wakes the
        catalog screen's pagination watcher.
        """
        self.refresh()
        if new is None or self._cards_per_row <= 0 or self.size.width <= 0:
            return
        col_width = max(1, self.size.width // self._cards_per_row)
        row = new // self._cards_per_row
        col = new % self._cards_per_row
        self.scroll_to_region(Region(col * col_width, row * _ROW_HEIGHT, col_width, _CARD_HEIGHT))

    def on_focus(self) -> None:
        """Auto-highlight first card on focus so Tab navigation has visible feedback."""
        if self._rows and self.highlighted is None:
            self.highlighted = 0

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
        """Activate the highlighted card (post Selected with its row)."""
        if self.highlighted is None or not self._rows:
            return
        if 0 <= self.highlighted < len(self._rows):
            self.post_message(self.Selected(self, self._rows[self.highlighted]))

    def highlight_first(self) -> None:
        """Move highlight to the first card; mirrors the GridSelect surface."""
        if self._rows:
            self.highlighted = 0

    def highlight_last(self) -> None:
        """Move highlight to the last card; mirrors the GridSelect surface."""
        if self._rows:
            self.highlighted = len(self._rows) - 1

    def _cell_at(self, x: int, y: int) -> int | None:
        """Return the dataset index at terminal-local ``(x, y)`` or None."""
        if not self._rows or self._cards_per_row <= 0:
            return None
        if y < 0:
            return None
        row = y // _ROW_HEIGHT
        within_row = y - row * _ROW_HEIGHT
        if within_row >= _CARD_HEIGHT:
            # Click landed in the gutter between rows.
            return None
        col_width = max(1, self.size.width // self._cards_per_row)
        col = min(self._cards_per_row - 1, x // col_width)
        index = row * self._cards_per_row + col
        if index >= len(self._rows):
            return None
        return index

    def on_click(self, event: events.Click) -> None:
        """First click highlights, second click on the highlight posts Selected."""
        index = self._cell_at(event.x, event.y)
        if index is None:
            return
        if self.highlighted == index:
            self.post_message(self.Selected(self, self._rows[index]))
        else:
            self.highlighted = index
        self.focus()

    def render_line(self, y: int) -> Strip:
        """Compose one terminal line by stitching the per-column card slices."""
        if y < 0:
            return Strip.blank(self.size.width)
        grid_row, line_within = divmod(y, _ROW_HEIGHT)
        if grid_row >= self._total_rows() or line_within >= _CARD_HEIGHT:
            return Strip.blank(self.size.width)
        col_width = max(1, self.size.width // max(1, self._cards_per_row))
        segments: list[Content] = []
        for col in range(self._cards_per_row):
            index = grid_row * self._cards_per_row + col
            if index >= len(self._rows):
                segments.append(Content(" " * col_width))
                continue
            row = self._rows[index]
            selected = index == self.highlighted
            card = _render_card_strip(row, selected=selected, width=col_width)
            segments.append(card.lines[line_within])
        joined = Content("").join(segments)
        return Strip(joined.render_segments(Style.null())).simplify()


_NAME_MAX_CHARS = 28
"""Cap displayed model names so long refs don't blow up the grid layout."""

_ELLIPSIS = "…"


def _truncate_name(name: str) -> str:
    if len(name) <= _NAME_MAX_CHARS:
        return name
    return name[: _NAME_MAX_CHARS - 1].rstrip() + _ELLIPSIS


def _render_card_strip(row: CatalogRow, *, selected: bool, width: int) -> _CardLines:
    """Return the ``_CARD_HEIGHT`` content lines that make up one card slot.

    Borrows the styling decisions from ``model_card.py`` so the grid view
    looks identical to the wizard cards even though the grid never mounts
    a ``ModelCard`` widget. Each card draws an explicit box-drawing
    border so the tile separates clearly from the screen background even
    on themes whose surface color resolves to near-black.
    """
    if isinstance(row, FrontierCatalogRow):
        body = _frontier_lines(row)
    else:
        body = _local_lines(row, selected=selected)
    border_style = "$accent" if selected else "$surface-lighten-2"
    # Use the theme's $surface tone for unselected cards so tiles read
    # as theme-matching panels rather than transparent black rectangles.
    # Selected card layers $accent on top for unmistakable focus.
    fill_style = "on $accent 30%" if selected else "on $surface"
    outer_width = max(3, width - _CARD_GUTTER)
    inner_width = outer_width - 2  # subtract the two vertical border cells
    top = Content.styled(f"{_BORDER_TL}{_BORDER_H * inner_width}{_BORDER_TR}", border_style)
    bottom = Content.styled(f"{_BORDER_BL}{_BORDER_H * inner_width}{_BORDER_BR}", border_style)
    framed_body = [
        _frame_line(line, inner_width, border_style, fill_style)
        for line in body[:_CARD_BODY_HEIGHT]
    ]
    gap = Content(" " * _CARD_GUTTER) if _CARD_GUTTER else Content("")
    framed: list[Content] = [top, *framed_body, bottom]
    return _CardLines(lines=[Content.assemble(line, gap) for line in framed])


def _frame_line(content: Content, inner_width: int, border_style: str, fill_style: str) -> Content:
    """Wrap *content* with vertical border bars and pad to *inner_width*.

    *fill_style* may be empty (transparent interior; the screen bg shows
    through) or a Textual style string. When non-empty, the pad cells
    and the body inherit it as a base style so coloured pills + bold
    names keep their fg but pick up the tile background.
    """
    left = Content.styled(_BORDER_V, border_style)
    right = Content.styled(_BORDER_V, border_style)
    rendered_width = content.cell_length
    if rendered_width >= inner_width:
        body = content
    else:
        pad_text = " " * (inner_width - rendered_width)
        pad = Content.styled(pad_text, fill_style) if fill_style else Content(pad_text)
        body = Content.assemble(content, pad)
    if fill_style:
        body = body.stylize_before(fill_style)
    return Content.assemble(left, body, right)


def _local_lines(row: LocalCatalogRow, *, selected: bool) -> list[Content]:
    from lilbee.cli.tui import messages as msg

    bg = TASK_COLORS.get(row.task, "$primary")
    name = Content.styled(_truncate_name(row.name), "bold")
    pills: list[Content] = []
    if row.featured:
        pills.append(pill("pick", "$warning", "$text"))
    pills.append(pill(row.task, bg, "$text"))
    if row.backend:
        pills.append(pill(row.backend, "$accent", "$text"))
    pill_line = Content(" ").join(pills)
    specs = _build_specs(row.params, row.quant, row.size)
    status = _build_local_status(row)
    lines: list[Content] = [name, pill_line, specs]
    lines.append(status if status is not None else Content(""))
    if selected and not row.installed:
        lines.append(Content.styled(msg.SETUP_CARD_HINT, "$text-muted 40% italic"))
    else:
        lines.append(Content(""))
    return lines


def _frontier_lines(row: FrontierCatalogRow) -> list[Content]:
    name = Content.styled(_truncate_name(row.name), "bold")
    pill_line = Content(" ").join(
        [pill(row.provider, "$accent", "$text"), _key_status_pill(row.key_status)]
    )
    info = Content.styled(f"Cloud via {row.provider} API", "$text-muted")
    return [name, pill_line, info, Content(""), Content("")]


def _key_status_pill(status: KeyStatus) -> Content:
    if status == KeyStatus.READY:
        return pill("ready", "$success", "$text")
    return pill("needs key", "$warning", "$text")


def _build_specs(params: str, quant: str, size: str) -> Content:
    parts = [p for p in (params, quant, size) if p and p != "--"]
    if not parts:
        return Content("--")
    return Content(f" {MIDDLE_DOT} ".join(parts))


def _build_local_status(row: LocalCatalogRow) -> Content | None:
    if row.installed:
        return pill("installed", "$success", "$text")
    if row.sort_downloads > 0:
        return Content.styled(f"↓ {row.downloads}", "$text-muted")
    return None
