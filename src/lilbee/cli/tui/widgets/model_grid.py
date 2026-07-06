"""ModelGrid: single-render-surface grid of catalog cards.

Single render surface via ``render_line(y)``; one strip painted per
visible row keeps fast scrolls cheap. Decoration uses theme-token
strings (``"on $panel"`` / ``"$primary"``) so themes own their contrast.
"""

from __future__ import annotations

import time
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
    NATIVE_BACKEND,
    CatalogRow,
    CatalogRowKind,
    FrontierCatalogRow,
    LocalCatalogRow,
    SizeVariant,
)
from lilbee.cli.tui.widgets.catalog_card_shared import (
    _FIT_LEVEL_BACKGROUND,
    _build_local_status,
    _build_specs,
    _key_status_pill,
    _render_fit_pill,
    _truncate_name,
)
from lilbee.cli.tui.widgets.catalog_theme import MIDDLE_DOT, TASK_COLORS
from lilbee.runtime.hardware import FitChip, FitLevel

_CSS_FILE = Path(__file__).parent / "model_grid.tcss"

_CARD_BODY_HEIGHT = 6
"""Body lines per card: name / primary pills / secondary pills / specs / status / hint."""

_BORDER_RESERVED_LINES = 2
"""Top + bottom border slots; reserved on every card so layout stays stable."""

_CARD_HEIGHT = _CARD_BODY_HEIGHT + _BORDER_RESERVED_LINES

_ROW_GUTTER = 0
_ROW_HEIGHT = _CARD_HEIGHT + _ROW_GUTTER
_DEFAULT_COLUMNS = 4
_CARD_MIN_WIDTH = 32
_CARD_GUTTER = 1

_BORDER_TOP_LEFT = "╭"
_BORDER_TOP_RIGHT = "╮"
_BORDER_BOTTOM_LEFT = "╰"
_BORDER_BOTTOM_RIGHT = "╯"
_BORDER_HORIZONTAL = "─"
_BORDER_VERTICAL = "│"

# Theme-token style strings; resolved at render time on the active theme.
_CARD_BODY_STYLE = "on $panel"
# Every card draws a border at all times so the grid reads as discrete tiles.
# The default tone is dim; the selected card gets a brighter color depending
# on whether the grid has focus.
_DEFAULT_BORDER_STYLE = "$border-blurred on $panel"
_FOCUSED_BORDER_STYLE = "$primary on $panel"
_BLURRED_BORDER_STYLE = "$border-blurred on $panel"
# Inter-card gutter and empty slot fill: match the screen's surface so gaps
# read as theme background, not raw terminal black.
_GAP_STYLE = "on $background"


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

    @dataclass
    class Highlighted(Message):
        """Posted on every cursor move so the catalog can run keyboard-driven
        prefetch (mouse wheel triggers via the scroll watcher; cell-by-cell
        keyboard scrolling never crosses the 85 % threshold by itself).
        """

        grid: ModelGrid
        index: int

    # Window inside which a second click on the same card counts as a
    # double-click and posts Selected. Single click outside this window only
    # highlights so users can't accidentally trigger an install with one mis-tap.
    _DOUBLE_CLICK_WINDOW_S: ClassVar[float] = 0.5

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
        self._last_click_index: int | None = None
        self._last_click_at: float = 0.0
        # render_line is called once per terminal row, so each card is asked for
        # _CARD_HEIGHT times per repaint; cache the built lines so a card renders
        # once. Flushed on set_rows and highlight changes, and on a resize that
        # shifts the column count; col_width is part of the key, so a width change
        # at the same column count is served fresh without an explicit flush.
        self._card_cache: dict[tuple[int, int, bool, str], _CardLines] = {}

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
        self._card_cache.clear()
        self.highlighted = None
        self.refresh(layout=True)

    def on_resize(self) -> None:
        new_cols = self._columns_for_width(self.size.width)
        if new_cols != self._cards_per_row:
            self._cards_per_row = new_cols
            self._card_cache.clear()
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
        if not self._rows:
            return 0
        cols = self._columns_for_width(width)
        rows = (len(self._rows) + cols - 1) // cols
        return rows * _ROW_HEIGHT

    def watch_highlighted(self, _old: int | None, new: int | None) -> None:
        """Repaint, scroll the cell into view, post Highlighted.

        ModelGrid itself has ``height: auto`` so it isn't scrollable; the
        outer ``#catalog-grid`` VerticalScroll is. We translate the cell's
        local offset to the parent's doc coords (using ``virtual_region``)
        and ask the parent to scroll. The Highlighted message lets the
        catalog screen run keyboard-driven prefetch on every cursor move.
        """
        # The two cards whose selected state flipped must re-render; clearing
        # also bounds the cache to one repaint's worth of cards.
        self._card_cache.clear()
        self.refresh()
        if new is None or self._cards_per_row <= 0 or self.size.width <= 0:
            return
        self.post_message(self.Highlighted(self, new))
        parent = self.parent
        if not isinstance(parent, Widget):
            return
        col_width = max(1, self.size.width // self._cards_per_row)
        row = new // self._cards_per_row
        col = new % self._cards_per_row
        grid_doc = self.virtual_region
        parent.scroll_to_region(
            Region(
                grid_doc.x + col * col_width,
                grid_doc.y + row * _ROW_HEIGHT,
                col_width,
                _CARD_HEIGHT,
            ),
            animate=False,
        )

    def on_focus(self) -> None:
        """Auto-highlight first card on focus so Tab navigation has visible feedback."""
        if self._rows and self.highlighted is None:
            self.highlighted = 0

    def on_blur(self) -> None:
        # Mirrors toad's GridSelect: when the user crosses into a sibling grid,
        # this grid's cursor goes away entirely instead of lingering as a
        # blurred ghost. Otherwise stacked catalog sections show two cursors
        # simultaneously and the user can't tell which grid owns focus.
        self.highlighted = None

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
            return None
        col_width = max(1, self.size.width // self._cards_per_row)
        col = min(self._cards_per_row - 1, x // col_width)
        index = row * self._cards_per_row + col
        if index >= len(self._rows):
            return None
        return index

    def on_click(self, event: events.Click) -> None:
        """Single click only highlights; double-click on the same card posts Selected.

        The 'second click on a highlighted card installs' rule the previous
        catalog used was unsafe under the new auto-highlight-on-focus path:
        a fresh focus pre-highlights index 0, so a single mouse click on
        card 0 immediately fired Selected and started an install. Now we
        require two clicks on the same card within ``_DOUBLE_CLICK_WINDOW_S``
        to install; a stray click only highlights, matching the toad gesture
        users expect.
        """
        index = self._cell_at(event.x, event.y)
        if index is None:
            return
        now = time.monotonic()
        is_double_click = (
            self._last_click_index == index
            and now - self._last_click_at <= self._DOUBLE_CLICK_WINDOW_S
        )
        self._last_click_index = index
        self._last_click_at = now
        if is_double_click:
            self.post_message(self.Selected(self, self._rows[index]))
            return
        self.highlighted = index
        self.focus()

    def _card_lines(
        self, index: int, col_width: int, selected: bool, border_style: str
    ) -> _CardLines:
        """Build (and cache for this repaint) the card lines for one cell."""
        key = (index, col_width, selected, border_style)
        cached = self._card_cache.get(key)
        if cached is None:
            cached = _render_card_strip(
                self._rows[index], selected=selected, width=col_width, border_style=border_style
            )
            self._card_cache[key] = cached
        return cached

    def render_line(self, y: int) -> Strip:
        """Compose one terminal line by stitching the per-column card slices."""
        if y < 0:
            return Strip.blank(self.size.width)
        grid_row, line_within = divmod(y, _ROW_HEIGHT)
        if grid_row >= self._total_rows() or line_within >= _CARD_HEIGHT:
            return Strip.blank(self.size.width)
        col_width = max(1, self.size.width // max(1, self._cards_per_row))
        border_style = _FOCUSED_BORDER_STYLE if self.has_focus else _BLURRED_BORDER_STYLE
        segments: list[Content] = []
        for col in range(self._cards_per_row):
            index = grid_row * self._cards_per_row + col
            if index >= len(self._rows):
                # Empty slot in a partial last row -> match screen surface.
                segments.append(Content.styled(" " * col_width, _GAP_STYLE))
                continue
            selected = index == self.highlighted
            card = self._card_lines(index, col_width, selected, border_style)
            segments.append(card.lines[line_within])
        joined = Content("").join(segments)
        return Strip(joined.render_segments(Style.null())).simplify()


def _render_card_strip(
    row: CatalogRow, *, selected: bool, width: int, border_style: str
) -> _CardLines:
    """Return the ``_CARD_HEIGHT`` content lines that make up one card slot.

    Every card paints a ``$panel`` body fill plus a round box border in
    ``_DEFAULT_BORDER_STYLE``; the selected card swaps the border color for
    ``border_style`` (the focused / blurred token picked by ``render_line``).
    The body is always panel-tinted so cards read as discrete tiles even on
    dark themes.
    """
    body = (
        _frontier_lines(row)
        if row.kind == CatalogRowKind.FRONTIER
        else _local_lines(row, selected=selected)
    )

    inner_width = max(3, width - _CARD_GUTTER)
    body_width = inner_width - 2  # subtract the two side-border columns
    # Gap between cards on the same row; theme-tinted so it reads as a card
    # separator, not as raw black.
    gap = Content.styled(" " * _CARD_GUTTER, _GAP_STYLE) if _CARD_GUTTER else Content("")

    body_padded = [_pad_line(line, body_width) for line in body[:_CARD_BODY_HEIGHT]]
    while len(body_padded) < _CARD_BODY_HEIGHT:
        body_padded.append(Content(" " * body_width))

    border_color = border_style if selected else _DEFAULT_BORDER_STYLE
    top = Content.styled(
        _BORDER_TOP_LEFT + _BORDER_HORIZONTAL * body_width + _BORDER_TOP_RIGHT,
        border_color,
    )
    bottom = Content.styled(
        _BORDER_BOTTOM_LEFT + _BORDER_HORIZONTAL * body_width + _BORDER_BOTTOM_RIGHT,
        border_color,
    )
    side = Content.styled(_BORDER_VERTICAL, border_color)

    framed = [top]
    for line in body_padded:
        # Wrap each padded body line in side bars, then layer the panel
        # background across the whole inner_width so the body reads as a
        # single tile (the bg covers any unstyled padding inside `_pad_line`).
        wrapped = Content.assemble(side, line, side)
        framed.append(wrapped.stylize_before(_CARD_BODY_STYLE))
    framed.append(bottom)

    return _CardLines(lines=[Content.assemble(line, gap) for line in framed])


def _pad_line(content: Content, width: int) -> Content:
    """Right-pad *content* to *width* columns with plain spaces."""
    rendered_width = content.cell_length
    if rendered_width >= width:
        return content
    return Content.assemble(content, Content(" " * (width - rendered_width)))


def _local_lines(row: LocalCatalogRow, *, selected: bool) -> list[Content]:
    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.widgets.model_card import _compat_pill

    bg = TASK_COLORS.get(row.task, "$primary")
    name = Content.styled(_truncate_name(row.name), "bold")
    # Two pill rows so wide secondary chips (fit + 'unsupported') don't push
    # the card border out of alignment on narrow grid columns.
    primary_pills: list[Content] = []
    if row.featured:
        primary_pills.append(pill("pick", "$warning", "$text"))
    primary_pills.append(pill(row.task, bg, "$text"))
    # Drop the 'native' backend pill on cards to free horizontal space; the
    # backend is implied for local models. Remote backends (ollama, etc.)
    # still surface their pill since that's a meaningful distinction.
    if row.backend and row.backend != NATIVE_BACKEND:
        primary_pills.append(pill(row.backend, "$accent", "$text"))
    primary_line = Content(" ").join(primary_pills)

    secondary_pills: list[Content] = []
    if row.fit is not None:
        # Card uses the compact 'fits' / 'tight' / "won't run" label only;
        # the headroom GB lives in the detail drawer where the wider pane
        # can render it without competing for card width.
        secondary_pills.append(_fit_pill_compact(row.fit))
    compat_chip = _compat_pill(row.compat)
    if compat_chip is not None:
        secondary_pills.append(compat_chip)
    secondary_line = Content(" ").join(secondary_pills) if secondary_pills else Content("")

    # Family card with multiple quants: replace the simple specs line
    # with an inline chip strip so the user sees every available size
    # at a glance without expanding into the drawer.
    if len(row.size_variants) > 1:
        specs = _build_size_variant_strip(row.size_variants)
    else:
        specs = _build_specs(row.params, row.quant, row.size)
    status = _build_local_status(row)
    lines: list[Content] = [name, primary_line, secondary_line, specs]
    lines.append(status if status is not None else Content(""))
    if selected:
        hint = msg.INSTALLED_CARD_HINT if row.installed else msg.SETUP_CARD_HINT
        lines.append(Content.styled(hint, "$text-muted 40% italic"))
    else:
        lines.append(Content(""))
    return lines


def _build_size_variant_strip(variants: list[SizeVariant]) -> Content:
    """Inline chip strip showing every quant for a family-aggregated card.

    Renders compact 'Q4 · Q5 · F16' style chips so the eye reads the
    available sizes at a glance. Per-variant fit colors aren't applied
    here; the drawer (right pane) carries the full fit-per-size detail
    when a card is highlighted.
    """
    labels = [v.quant if v.quant != "--" else v.label for v in variants]
    return Content.styled(f" {MIDDLE_DOT} ".join(labels), "$text-muted")


def _frontier_lines(row: FrontierCatalogRow) -> list[Content]:
    name = Content.styled(_truncate_name(row.name), "bold")
    pill_line = Content(" ").join(
        [pill(row.provider, "$accent", "$text"), _key_status_pill(row.key_status)]
    )
    info = Content.styled(f"Cloud via {row.provider} API", "$text-muted")
    # Frontier cards have no secondary pill line, but pad to _CARD_BODY_HEIGHT
    # so they align with local cards in the same grid row.
    return [name, pill_line, Content(""), info, Content(""), Content("")]


_FIT_LEVEL_LABEL_COMPACT: dict[FitLevel, str] = {
    FitLevel.FITS: "fits",
    FitLevel.TIGHT: "tight",
    FitLevel.WONT_RUN: "won't run",
}

# The verbose drawer fit pill is the shared renderer.
_fit_pill = _render_fit_pill


def _fit_pill_compact(fit: FitChip) -> Content:
    """Card-side compact fit chip: just ``fits`` / ``tight`` / ``won't run``."""
    return pill(_FIT_LEVEL_LABEL_COMPACT[fit.level], _FIT_LEVEL_BACKGROUND[fit.level], "$text")


# _key_status_pill / _build_specs / _build_local_status live in
# catalog_card_shared and are re-imported above.
