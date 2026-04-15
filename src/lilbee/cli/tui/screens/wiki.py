"""Wiki screen — browse wiki pages with sidebar navigation and markdown preview."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lilbee.wiki.browse import WikiPageInfo

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Input, Markdown, OptionList, Static
from textual.widgets.option_list import Option
from textual.worker import get_current_worker as _get_worker

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.nav_aware_input import NavAwareInput
from lilbee.cli.tui.wiki_worker import resolve_wiki_targets
from lilbee.config import cfg

log = logging.getLogger(__name__)


def _wiki_root() -> Path:
    """Resolve the wiki root directory from config."""
    return cfg.data_root / cfg.wiki_dir


def _format_page_header(
    title: str,
    page_type: str,
    source_count: int,
    created_at: str,
    faithfulness: float | None,
) -> str:
    """Build a header string for the content pane."""
    parts = [f"[bold]{title}[/]"]
    parts.append(f"  [dim]{page_type}[/]")
    if source_count > 0:
        parts.append(f"  [dim]{source_count} sources[/]")
    if created_at:
        parts.append(f"  [dim]{created_at}[/]")
    if faithfulness is not None:
        pct = int(faithfulness * 100)
        parts.append(f"  [dim]faithfulness {pct}%[/]")
    return "".join(parts)


class WikiScreen(Screen[None]):
    """Wiki page browser with sidebar and markdown content viewer."""

    CSS_PATH = "wiki.tcss"
    AUTO_FOCUS = "#wiki-page-list"
    HELP = "Browse wiki pages.\n\nUse / to search, Enter to select a page, Escape to clear search."

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "dismiss_or_back", "Back", show=False),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("r", "regenerate", "Regen", show=True, priority=True),
        Binding("j", "cursor_down", "Nav", show=False),
        Binding("k", "cursor_up", "Nav", show=False),
        Binding("g", "jump_top", "Top", show=False),
        Binding("G", "jump_bottom", "End", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._page_slugs: list[str] = []

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        yield Horizontal(
            Vertical(
                NavAwareInput(
                    placeholder=msg.WIKI_SEARCH_PLACEHOLDER,
                    id="wiki-search",
                ),
                OptionList(id="wiki-page-list"),
                id="wiki-sidebar",
            ),
            Vertical(
                Static("", id="wiki-page-header"),
                VerticalScroll(
                    Markdown("", id="wiki-content"),
                    id="wiki-content-scroll",
                ),
                id="wiki-main",
            ),
            id="wiki-layout",
        )
        yield TaskBar()
        yield ViewTabs()
        yield Footer()

    def on_mount(self) -> None:
        self._load_pages()

    def reload(self) -> None:
        """Refresh the sidebar from disk. Public entry point for external callers."""
        self._load_pages()

    def _load_pages(self, filter_text: str = "") -> None:
        """Populate the sidebar with wiki pages, optionally filtered."""
        from lilbee.wiki.browse import list_pages

        option_list = self.query_one("#wiki-page-list", OptionList)
        option_list.clear_options()
        self._page_slugs = []

        if not cfg.wiki:
            option_list.add_option(Option(msg.WIKI_EMPTY_STATE, disabled=True))
            self._show_placeholder()
            return

        root = _wiki_root()
        try:
            all_pages = list_pages(root)
        except Exception:
            log.debug("Failed to list wiki pages", exc_info=True)
            all_pages = []

        if filter_text:
            needle = filter_text.lower()
            all_pages = [p for p in all_pages if needle in p.title.lower()]

        if not all_pages:
            option_list.add_option(Option(msg.WIKI_EMPTY_STATE, disabled=True))
            self._show_placeholder()
            return

        grouped = _group_pages(all_pages)
        for page_type, pages in grouped:
            heading = msg.WIKI_TYPE_HEADINGS.get(page_type, page_type.capitalize())
            option_list.add_option(Option(f"── {heading} ──", disabled=True))
            for page in pages:
                self._page_slugs.append(page.slug)
                label = f"  {page.title}"
                option_list.add_option(Option(label, id=page.slug))

    def _show_placeholder(self) -> None:
        """Show the no-content placeholder in the main area."""
        self.query_one("#wiki-page-header", Static).update("")
        self.query_one("#wiki-content", Markdown).update(msg.WIKI_NO_CONTENT)

    @on(OptionList.OptionSelected, "#wiki-page-list")
    def _on_page_selected(self, event: OptionList.OptionSelected) -> None:
        """Load and display the selected wiki page."""
        slug = event.option.id
        if slug is None:
            return
        self._display_page(slug)

    def _display_page(self, slug: str) -> None:
        """Read and render a wiki page by slug."""
        from lilbee.wiki.browse import read_page

        root = _wiki_root()
        page = read_page(root, slug)
        if page is None:
            self.query_one("#wiki-page-header", Static).update("")
            self.query_one("#wiki-content", Markdown).update(msg.WIKI_NO_CONTENT)
            return

        faithfulness = page.frontmatter.get("faithfulness_score")
        faith_val = float(faithfulness) if faithfulness is not None else None

        page_type = ""
        parts = slug.split("/")
        if len(parts) >= 2:
            from lilbee.wiki.shared import SUBDIR_TO_TYPE

            page_type = SUBDIR_TO_TYPE.get(parts[0], "")

        source_count = page.frontmatter.get("source_count", 0)
        created_at = page.frontmatter.get("generated_at", "")
        if isinstance(created_at, (datetime, date)):
            created_at = created_at.isoformat()

        header_text = _format_page_header(
            title=page.title,
            page_type=page_type,
            source_count=int(source_count) if source_count else 0,
            created_at=str(created_at),
            faithfulness=faith_val,
        )
        self.query_one("#wiki-page-header", Static).update(header_text)
        self.query_one("#wiki-content", Markdown).update(page.content)

    @on(Input.Changed, "#wiki-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Filter pages when search input changes."""
        self._load_pages(filter_text=event.value.strip())

    def action_regenerate(self) -> None:
        """Regenerate wiki page(s). Selected page's source, or all sources."""
        if not cfg.wiki:
            self.notify(msg.CMD_WIKI_DISABLED, severity="warning")
            return

        requested: str | None = None
        option_list = self.query_one("#wiki-page-list", OptionList)
        highlighted = option_list.highlighted
        if highlighted is not None:
            slug = option_list.get_option_at_index(highlighted).id
            if slug is not None:
                requested = self._source_for_slug(slug)

        targets = resolve_wiki_targets(requested)
        if targets is None:
            if requested is not None:
                self.notify(msg.CMD_WIKI_NOT_FOUND.format(name=requested), severity="error")
            else:
                self.notify(msg.CMD_WIKI_NO_SOURCES, severity="warning")
            return

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        task_bar = self.query_one(TaskBar)
        task_id = task_bar.add_task(msg.TASK_NAME_WIKI.format(count=len(targets)), "wiki")
        task_bar.queue.advance("wiki")
        self.notify(msg.CMD_WIKI_STARTED.format(count=len(targets)))
        self._run_wiki_background(targets, task_id)

    def _source_for_slug(self, slug: str) -> str | None:
        """Extract the primary source filename from a wiki page's frontmatter."""
        from lilbee.wiki.browse import read_page

        root = _wiki_root()
        page = read_page(root, slug)
        if page is None:
            return None
        sources = page.frontmatter.get("sources")
        # frontmatter values are untyped (Any from YAML); guard against non-list shapes
        if isinstance(sources, list) and sources:
            return str(next(iter(sources)))
        return None

    @work(thread=True)
    def _run_wiki_background(self, sources: list[str], task_id: str) -> None:
        """Generate wiki pages in a background thread."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.wiki_worker import run_wiki_generation

        worker = _get_worker()
        task_bar = self.query_one(TaskBar)
        run_wiki_generation(
            sources=sources,
            task_id=task_id,
            widget=self,
            update_task=task_bar.update_task,
            complete_task=task_bar.complete_task,
            fail_task=task_bar.fail_task,
            notify=self.notify,
            on_complete=self.reload,
            is_cancelled=lambda: worker.is_cancelled,
        )

    def action_focus_search(self) -> None:
        """Focus the search input -- bound to / key."""
        self.query_one("#wiki-search", Input).focus()

    def action_dismiss_or_back(self) -> None:
        """Clear search if active, otherwise go back."""
        search = self.query_one("#wiki-search", Input)
        if search.value:
            search.value = ""
            return
        self.action_go_back()

    def action_go_back(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.app.switch_view("Chat")
        else:
            self.app.pop_screen()

    def action_cursor_down(self) -> None:
        if isinstance(self.focused, Input):
            return
        self.query_one("#wiki-page-list", OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        if isinstance(self.focused, Input):
            return
        self.query_one("#wiki-page-list", OptionList).action_cursor_up()

    def action_jump_top(self) -> None:
        if isinstance(self.focused, Input):
            return
        option_list = self.query_one("#wiki-page-list", OptionList)
        option_list.scroll_home()

    def action_jump_bottom(self) -> None:
        if isinstance(self.focused, Input):
            return
        option_list = self.query_one("#wiki-page-list", OptionList)
        option_list.scroll_end()


def _group_pages(
    pages: list[WikiPageInfo],
) -> list[tuple[str, list[WikiPageInfo]]]:
    """Group pages by page_type, maintaining order: summaries first, then synthesis."""
    from lilbee.wiki.shared import WikiPageType

    groups: dict[str, list[WikiPageInfo]] = {}
    type_order: tuple[str, ...] = (WikiPageType.SUMMARY, WikiPageType.SYNTHESIS)
    for t in type_order:
        group = [p for p in pages if p.page_type == t]
        if group:
            groups[t] = group
    for p in pages:
        if p.page_type not in groups:
            groups[p.page_type] = []
        if p.page_type not in type_order:
            groups[p.page_type].append(p)
    return [(k, v) for k, v in groups.items() if v]
