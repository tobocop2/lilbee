"""Wiki drafts review screen: browse, diff, accept, or reject pending drafts.

The screen pairs a left-hand :class:`DataTable` of drafts with a
right-hand scrollable :class:`Static` that renders the unified diff of
the highlighted draft against its published counterpart. Accept and
reject are confirmed through the shared :class:`ConfirmDialog` modal.
Keybindings follow the rest of the TUI: vim j/k to navigate, ``/`` to
search, ``a`` / ``r`` for accept / reject, ``q`` / Esc to back out.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Input, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.nav_aware_input import NavAwareInput
from lilbee.cli.tui.widgets.task_bar import TaskBar
from lilbee.config import cfg
from lilbee.services import get_services
from lilbee.wiki.drafts import accept_draft, diff_draft, list_drafts, reject_draft

if TYPE_CHECKING:
    from lilbee.wiki.drafts import DraftInfo

log = logging.getLogger(__name__)


def _wiki_root() -> Path:
    """Resolve the wiki root directory from config."""
    return cfg.data_root / cfg.wiki_dir


def _format_drift(drift: float | None) -> str:
    """Render a drift ratio as a percentage, or ``-`` when absent."""
    return f"{drift:.0%}" if drift is not None else "-"


def _format_faithfulness(score: float | None) -> str:
    """Render a faithfulness score with two decimals, or ``-`` when absent."""
    return f"{score:.2f}" if score is not None else "-"


def _format_published(exists: bool) -> str:
    """Render the published-counterpart flag as a human yes/no."""
    return msg.WIKI_DRAFTS_PUBLISHED_YES if exists else msg.WIKI_DRAFTS_PUBLISHED_NO


def _kind_label(pending_kind: str | None) -> str:
    """Map a pending_kind value to its display label.

    ``None`` surfaces as "drift" because drift is the default review
    reason when no PENDING marker is present.
    """
    return pending_kind or msg.WIKI_DRAFTS_KIND_DRIFT


class WikiDraftsScreen(Screen[None]):
    """Review-surface screen for pending wiki drafts."""

    CSS_PATH = "wiki_drafts.tcss"
    AUTO_FOCUS = "#wiki-drafts-table"
    HELP = "Review pending wiki drafts. j/k navigate, a accept, r reject, / search, q back."

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "dismiss_or_back", "Back", show=False),
        Binding("a", "accept", "Accept", show=True),
        Binding("r", "reject", "Reject", show=True),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("j", "cursor_down", "Nav", show=False),
        Binding("k", "cursor_up", "Nav", show=False),
        Binding("g", "jump_top", "Top", show=False),
        Binding("G", "jump_bottom", "End", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._drafts: list[DraftInfo] = []
        self._filter: str = ""

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        table: DataTable[str] = DataTable(id="wiki-drafts-table")
        table.cursor_type = "row"
        yield Horizontal(
            Vertical(
                NavAwareInput(
                    placeholder=msg.WIKI_DRAFTS_SEARCH_PLACEHOLDER,
                    id="wiki-drafts-search",
                ),
                table,
                id="wiki-drafts-sidebar",
            ),
            Vertical(
                VerticalScroll(
                    Static(msg.WIKI_DRAFTS_DIFF_EMPTY, id="wiki-drafts-diff"),
                    id="wiki-drafts-diff-scroll",
                ),
                id="wiki-drafts-main",
            ),
            id="wiki-drafts-layout",
        )
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#wiki-drafts-table", DataTable)
        table.add_columns(
            msg.WIKI_DRAFTS_COLUMN_SLUG,
            msg.WIKI_DRAFTS_COLUMN_KIND,
            msg.WIKI_DRAFTS_COLUMN_DRIFT,
            msg.WIKI_DRAFTS_COLUMN_FAITHFULNESS,
            msg.WIKI_DRAFTS_COLUMN_PUBLISHED,
        )
        self._load_drafts()

    def _load_drafts(self) -> None:
        """Fetch drafts from disk and populate the table."""
        table = self.query_one("#wiki-drafts-table", DataTable)
        table.clear()
        try:
            self._drafts = list_drafts(_wiki_root())
        except Exception as exc:
            log.debug("Failed to list wiki drafts", exc_info=True)
            self._drafts = []
            self._show_diff(msg.WIKI_DRAFTS_LOAD_FAILED.format(error=exc))
            return

        visible = self._visible_drafts()
        if not visible:
            self._show_diff(msg.WIKI_DRAFTS_EMPTY)
            return

        for d in visible:
            table.add_row(
                d.slug,
                _kind_label(d.pending_kind),
                _format_drift(d.drift_ratio),
                _format_faithfulness(d.faithfulness_score),
                _format_published(d.published_exists),
                key=d.slug,
            )
        self._show_diff(msg.WIKI_DRAFTS_DIFF_EMPTY)

    def _visible_drafts(self) -> list[DraftInfo]:
        """Apply the current filter to the loaded draft list."""
        if not self._filter:
            return self._drafts
        needle = self._filter.lower()
        return [d for d in self._drafts if needle in d.slug.lower()]

    def _show_diff(self, text: str) -> None:
        """Update the diff pane with *text*."""
        self.query_one("#wiki-drafts-diff", Static).update(text)

    def _highlighted_slug(self) -> str | None:
        """Return the slug of the highlighted row, or ``None`` when empty."""
        table = self.query_one("#wiki-drafts-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        except Exception:
            return None
        if row_key is None or row_key.value is None:
            return None
        return str(row_key.value)

    @on(DataTable.RowHighlighted, "#wiki-drafts-table")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Load the diff for the newly highlighted row."""
        key = event.row_key.value if event.row_key is not None else None
        if key is None:
            return
        self._display_diff(str(key))

    def _display_diff(self, slug: str) -> None:
        """Compute and render the unified diff for *slug*."""
        try:
            diff = diff_draft(slug, _wiki_root())
        except FileNotFoundError:
            self._show_diff(msg.WIKI_DRAFTS_DIFF_EMPTY)
            return
        except Exception as exc:
            log.debug("Failed to compute diff for %s", slug, exc_info=True)
            self._show_diff(msg.WIKI_DRAFTS_DIFF_FAILED.format(error=exc))
            return
        self._show_diff(diff or msg.WIKI_DRAFTS_DIFF_NONE)

    @on(Input.Changed, "#wiki-drafts-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Filter drafts as the user types."""
        self._filter = event.value.strip()
        self._load_drafts()

    def action_focus_search(self) -> None:
        """Focus the search input (``/`` keybinding)."""
        self.query_one("#wiki-drafts-search", Input).focus()

    def action_dismiss_or_back(self) -> None:
        """Clear the search if active, otherwise back out to the wiki screen."""
        search = self.query_one("#wiki-drafts-search", Input)
        if search.value:
            search.value = ""
            return
        self.action_go_back()

    def action_go_back(self) -> None:
        """Pop back to the wiki screen (or the previous screen in tests)."""
        self.app.pop_screen()

    def _table_or_none(self) -> DataTable[str] | None:
        """Return the drafts table unless an Input is focused."""
        if isinstance(self.focused, Input):
            return None
        return self.query_one("#wiki-drafts-table", DataTable)

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

    def action_accept(self) -> None:
        """Prompt for confirmation, then accept the highlighted draft."""
        slug = self._highlighted_slug()
        if slug is None:
            return
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self._do_accept(slug)

        self.app.push_screen(
            ConfirmDialog(
                msg.WIKI_DRAFTS_ACCEPT_CONFIRM_TITLE,
                msg.WIKI_DRAFTS_ACCEPT_CONFIRM_MESSAGE.format(slug=slug),
            ),
            _on_confirm,
        )

    def _do_accept(self, slug: str) -> None:
        """Execute the accept call and refresh the list."""
        try:
            accept_draft(slug, _wiki_root(), get_services().store)
        except FileNotFoundError:
            self.notify(msg.WIKI_DRAFTS_ACCEPT_FAILED.format(error=f"missing: {slug}"))
            return
        except Exception as exc:
            log.debug("Accept failed for %s", slug, exc_info=True)
            self.notify(msg.WIKI_DRAFTS_ACCEPT_FAILED.format(error=exc))
            return
        self.notify(msg.WIKI_DRAFTS_ACCEPTED.format(slug=slug))
        self._load_drafts()

    def action_reject(self) -> None:
        """Prompt for confirmation, then reject the highlighted draft."""
        slug = self._highlighted_slug()
        if slug is None:
            return
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self._do_reject(slug)

        self.app.push_screen(
            ConfirmDialog(
                msg.WIKI_DRAFTS_REJECT_CONFIRM_TITLE,
                msg.WIKI_DRAFTS_REJECT_CONFIRM_MESSAGE.format(slug=slug),
            ),
            _on_confirm,
        )

    def _do_reject(self, slug: str) -> None:
        """Execute the reject call and refresh the list."""
        try:
            reject_draft(slug, _wiki_root())
        except FileNotFoundError:
            self.notify(msg.WIKI_DRAFTS_REJECT_FAILED.format(error=f"missing: {slug}"))
            return
        except Exception as exc:
            log.debug("Reject failed for %s", slug, exc_info=True)
            self.notify(msg.WIKI_DRAFTS_REJECT_FAILED.format(error=exc))
            return
        self.notify(msg.WIKI_DRAFTS_REJECTED.format(slug=slug))
        self._load_drafts()
