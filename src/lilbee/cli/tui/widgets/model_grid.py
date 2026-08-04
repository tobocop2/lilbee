"""ModelGrid: single-render-surface grid of catalog cards.

Single render surface via ``render_line(y)``; one strip painted per
visible row keeps fast scrolls cheap. Decoration uses theme-token
strings (``"on $panel"`` / ``"$primary"``) so themes own their contrast.
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

# Body width of the narrowest grid column (GridSelect min_column_width 30,
# less the gutter and the two side borders). Used when a caller renders card
# lines without a concrete column width.
_DEFAULT_BODY_WIDTH = 27

_BORDER_TOP_LEFT = "╭"
_BORDER_TOP_RIGHT = "╮"
_BORDER_BOTTOM_LEFT = "╰"
_BORDER_BOTTOM_RIGHT = "╯"
_BORDER_HORIZONTAL = "─"
_BORDER_VERTICAL = "│"

_DOUBLE_CLICK_CHAIN = 2
"""events.Click.chain value for the second click of a double-click."""

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


def _row_key(row: CatalogRow) -> tuple[CatalogRowKind, str]:
    """Identity used to re-locate the highlighted row across a dataset swap."""
    return (row.kind, row.name)


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
        # A highlight assigned before layout (restore-after-remount, initial
        # focus) can't scroll into view yet; on_resize completes it.
        self._reveal_pending: bool = False
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
        """Replace the dataset, keeping the cursor on the same row when it survives.

        Background refreshes land through here constantly (HF pages arrive a
        few rows at a time), so the highlight follows the row's identity into
        the new dataset; resetting it would strand the cursor mid-navigation.
        """
        previous_index = self.highlighted
        previous_key = (
            _row_key(self._rows[previous_index])
            if previous_index is not None and 0 <= previous_index < len(self._rows)
            else None
        )
        self._rows = list(rows)
        self._card_cache.clear()
        self.highlighted = self._relocated_highlight(previous_index, previous_key)
        self.refresh(layout=True)

    def _relocated_highlight(
        self, previous_index: int | None, previous_key: tuple[CatalogRowKind, str] | None
    ) -> int | None:
        """Where the cursor lands after a dataset replacement."""
        if not self._rows:
            return None
        if previous_key is not None:
            for index, row in enumerate(self._rows):
                if _row_key(row) == previous_key:
                    return index
        if previous_index is not None:
            return min(previous_index, len(self._rows) - 1)
        # A focused grid always shows a cursor; an unfocused one stays bare.
        return 0 if self.has_focus else None

    def on_resize(self) -> None:
        new_cols = self._columns_for_width(self.size.width)
        if new_cols != self._cards_per_row:
            self._cards_per_row = new_cols
            self._card_cache.clear()
            self.refresh(layout=True)
        if self._reveal_pending:
            self._reveal_highlight()

    def on_show(self) -> None:
        if self._reveal_pending:
            self._reveal_highlight()

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
        """Repaint, post Highlighted, scroll the cell into view.

        The Highlighted message lets the catalog screen run keyboard-driven
        prefetch and drawer updates on every cursor move; it is posted even
        before layout so listeners never miss a move, while the scroll part
        waits for a real size (``_reveal_pending`` + ``on_resize``).
        """
        # The two cards whose selected state flipped must re-render; clearing
        # also bounds the cache to one repaint's worth of cards.
        self._card_cache.clear()
        self.refresh()
        if new is None:
            self._reveal_pending = False
            return
        self.post_message(self.Highlighted(self, new))
        self._reveal_highlight(new)

    def _reveal_highlight(self, index: int | None = None) -> None:
        """Scroll the highlighted cell into view, deferring until layout exists."""
        if index is None:
            index = self.highlighted
        if index is None:
            self._reveal_pending = False
            return
        if self._cards_per_row <= 0 or self.size.width <= 0:
            self._reveal_pending = True
            return
        self._reveal_pending = False
        col_width = max(1, self.size.width // self._cards_per_row)
        row, col = divmod(index, self._cards_per_row)
        cell = Region(col * col_width, row * _ROW_HEIGHT, col_width, _CARD_HEIGHT)
        self._scroll_region_into_view(cell)

    def _scroll_region_into_view(self, cell: Region) -> None:
        """Reveal *cell* (grid-local coords) by scrolling every ancestor that can.

        ModelGrid paints cards as strips, so there is no child widget to hand
        to ``Screen.scroll_to_widget``; this mirrors its ancestor walk for a
        region instead of assuming any particular ancestor is the scrollable.
        """
        region = cell.translate(self.virtual_region.offset)
        widget: Widget = self
        while isinstance(widget.parent, Widget):
            container = widget.parent
            scroll_offset = container.scroll_to_region(region, animate=False)
            widget = container
            if not region or not isinstance(widget.parent, Widget):
                break
            region = (
                region.translate(-scroll_offset)
                .translate(container.styles.margin.top_left)
                .translate(container.styles.border.spacing.top_left)
                .translate(container.virtual_region_with_margin.offset)
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
        """Single click only highlights; a double-click on the same card installs.

        A single click must never install (auto-highlight-on-focus made that
        a one-mis-tap hazard). ``event.chain`` carries the click multiplicity,
        so the double-click window follows the user's terminal settings.
        """
        index = self._cell_at(event.x, event.y)
        if index is None:
            return
        if event.chain >= _DOUBLE_CLICK_CHAIN and index == self.highlighted:
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
    inner_width = max(3, width - _CARD_GUTTER)
    body_width = inner_width - 2  # subtract the two side-border columns
    body = (
        _frontier_lines(row)
        if row.kind == CatalogRowKind.FRONTIER
        else _local_lines(row, selected=selected, body_width=body_width)
    )
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
    """Fit *content* to exactly *width* columns, padding or truncating.

    Truncating here rather than trusting each line builder keeps the card
    frame intact by construction: one over-wide line otherwise pushes its
    right border out and misaligns every card beside it in the row.
    """
    rendered_width = content.cell_length
    if rendered_width > width:
        return content.truncate(width, ellipsis=True)
    if rendered_width == width:
        return content
    return Content.assemble(content, Content(" " * (width - rendered_width)))


def _local_lines(
    row: LocalCatalogRow, *, selected: bool, body_width: int = _DEFAULT_BODY_WIDTH
) -> list[Content]:
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
        specs = _build_size_variant_strip(row.size_variants, body_width)
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


def _build_size_variant_strip(variants: list[SizeVariant], width: int) -> Content:
    """Inline chip strip showing the quants of a family-aggregated card.

    Renders compact 'Q4 · Q5 · F16' style chips so the eye reads the
    available sizes at a glance. A family that varies by parameter count
    rather than quant would render identical chips, so colliding quants
    fall back to the full per-variant label. Per-variant fit colors aren't
    applied here; the drawer (right pane) carries the full fit-per-size
    detail when a card is highlighted.

    Duplicate labels collapse, then chips that do not fit *width* are dropped
    and counted as ``+N``: a family can hold more variants than a card column
    has room for, and the long fallback labels reach that limit quickly.
    """
    quants = [v.quant if v.quant != "--" else v.label for v in variants]
    labels = quants if len(set(quants)) == len(quants) else [v.label for v in variants]
    # Two repos in one family can share a parameter count and quant, which
    # renders the same chip twice and reads as a bug.
    labels = list(dict.fromkeys(labels))
    sep = f" {MIDDLE_DOT} "
    shown = len(labels)
    while shown > 1:
        text = sep.join(labels[:shown])
        hidden = len(labels) - shown
        if hidden:
            text = f"{text}  +{hidden}"
        if len(text) <= width:
            return Content.styled(text, "$text-muted")
        shown -= 1
    return Content.styled(labels[0][:width] if labels else "", "$text-muted")


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
