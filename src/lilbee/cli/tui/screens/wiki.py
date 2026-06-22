"""Wiki screen: browse wiki pages as a navigable tree with markdown preview."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.wiki.browse import WikiPageInfo

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Input, Markdown, Static, Tree
from textual.widgets.tree import TreeNode

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.task_bar import TaskBar
from lilbee.core.config import cfg
from lilbee.wiki.browse import read_page

log = logging.getLogger(__name__)

# Tree node data carries the full wiki-page slug when present. Group folders
# (page-type headings, per-source branches, inner-section branches) use None.
_INDEX_STEM = "index"
# Wiki slugs of the form ``<subdir>/<name>`` carry a meaningful page type;
# bare slugs (no slash) do not.
_SLUG_WITH_TYPE_MIN_PARTS = 2


def _wiki_root() -> Path:
    """Resolve the wiki root directory from config."""
    return cfg.data_root / cfg.wiki_dir


def _safe_float(value: object) -> float | None:
    """Coerce an untyped frontmatter value to float, or None if not numeric."""
    try:
        return float(cast("float", value))
    except (TypeError, ValueError):
        return None


def _safe_int(value: object, default: int = 0) -> int:
    """Coerce an untyped frontmatter value to int, or *default* if not numeric."""
    try:
        return int(cast("float", value))
    except (TypeError, ValueError):
        return default


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


def _short_label(slug_part: str) -> str:
    """Render a slug component as a human-friendly tree label."""
    return slug_part.replace("-", " ").replace("_", " ").strip()


def _breadcrumb_for_slug(slug: str, title: str) -> str:
    """Build a dim-themed breadcrumb string: chapter > section > page."""
    parts = slug.split("/")
    if len(parts) <= 1:
        return ""
    display_parts = [_short_label(p) for p in parts[:-1]]
    display_parts.append(title)
    return " [dim]>[/] ".join(display_parts)


class WikiScreen(Screen[None]):
    """Wiki page browser with a tree sidebar and markdown content viewer."""

    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "wiki.tcss"
    AUTO_FOCUS = "#wiki-page-list"
    HELP = "Browse wiki pages. h/l collapse/expand, j/k navigate, Enter opens a page, / searches."

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "dismiss_or_back", "Back", show=False),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("D", "open_drafts", "Drafts", show=True),
        Binding("j", "cursor_down", "Nav", show=False),
        Binding("k", "cursor_up", "Nav", show=False),
        Binding("h", "cursor_left", "Collapse", show=False),
        Binding("l", "cursor_right", "Expand", show=False),
        Binding("g", "jump_top", "Top", show=False),
        Binding("G", "jump_bottom", "End", show=False),
    ]

    _SEARCH_FILTER_DEBOUNCE_SECONDS = 0.12

    def __init__(self) -> None:
        super().__init__()
        self._page_slugs: list[str] = []
        self._search_filter_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        tree: Tree[str | None] = Tree("Wiki", id="wiki-page-list")
        tree.show_root = False
        yield Horizontal(
            Vertical(
                Input(
                    placeholder=msg.WIKI_SEARCH_PLACEHOLDER,
                    id="wiki-search",
                ),
                tree,
                id="wiki-sidebar",
            ),
            Vertical(
                Static("", id="wiki-breadcrumb"),
                Static("", id="wiki-page-header"),
                VerticalScroll(
                    Markdown("", id="wiki-content"),
                    id="wiki-content-scroll",
                ),
                id="wiki-main",
            ),
            id="wiki-layout",
        )
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        self._load_pages()

    def on_show(self) -> None:
        """Re-scan on focus so out-of-band builds (`lilbee wiki build` from a
        sibling shell) and incremental wiki updates land without a TUI restart.
        """
        self._load_pages()

    def reload(self) -> None:
        """Refresh the sidebar from disk. Public entry point for external callers."""
        self._load_pages()

    def _load_pages(self, filter_text: str = "") -> None:
        """Populate the sidebar tree with wiki pages, optionally filtered."""
        from lilbee.wiki.browse import list_pages

        tree = self.query_one("#wiki-page-list", Tree)
        tree.reset("Wiki")
        self._page_slugs = []

        if not cfg.wiki:
            tree.root.add_leaf(msg.wiki_empty_state_leaf())
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
            tree.root.add_leaf(msg.wiki_empty_state_leaf())
            self._show_placeholder()
            return

        self._populate_tree(tree, all_pages)

    def _populate_tree(self, tree: Tree[str | None], pages: list[WikiPageInfo]) -> None:
        """Build the sidebar tree from a flat list of wiki pages.

        Slugs like ``summaries/cv-manual/01-brakes/page-0042`` become nested
        branches under their page-type group, with leaves for leaf pages and
        expandable branches for intermediate heading folders. ``index.md``
        and ``log.md`` at the wiki root are surfaced as top-level leaves.
        """
        self._add_root_shortcut(tree, "index", msg.WIKI_INDEX_LABEL)
        self._add_root_shortcut(tree, "log", msg.WIKI_LOG_LABEL)
        grouped = _group_pages(pages)
        for page_type, group_pages in grouped:
            heading = msg.WIKI_TYPE_HEADINGS.get(page_type, page_type.capitalize())
            group_node = tree.root.add(heading, expand=True)
            for page in group_pages:
                self._page_slugs.append(page.slug)
                self._insert_page(group_node, page)

    def _add_root_shortcut(self, tree: Tree[str | None], slug: str, label: str) -> None:
        """Add a top-level leaf for an auto-generated page (index.md, log.md)."""
        if not (_wiki_root() / f"{slug}.md").is_file():
            return
        tree.root.add_leaf(label, data=slug)
        self._page_slugs.append(slug)

    def _insert_page(self, group_node: TreeNode[str | None], page: WikiPageInfo) -> None:
        """Walk the slug path and add/reuse branches until the leaf position.

        Slugs begin with the page-type prefix (``summaries/``/``synthesis/``),
        which is already reflected in the enclosing group node. The remaining
        path components form the nested tree inside the group.
        """
        parts = page.slug.split("/")
        if len(parts) <= 1:
            group_node.add_leaf(page.title, data=page.slug)
            return

        # Skip the leading page-type component since the group node represents it.
        inner_parts = parts[1:]
        node = group_node
        *branch_parts, leaf_part = inner_parts
        for part in branch_parts:
            node = _find_or_add_branch(node, part)

        if leaf_part == _INDEX_STEM:
            # An inner-node index.md file: show its title on the enclosing branch.
            node.label = page.title
            node.data = page.slug
            return

        label = _short_label(leaf_part)
        node.add_leaf(page.title if page.title else label, data=page.slug)

    def _show_placeholder(self) -> None:
        """Show the no-content placeholder in the main area."""
        self.query_one("#wiki-breadcrumb", Static).update("")
        self.query_one("#wiki-page-header", Static).update("")
        self.query_one("#wiki-content", Markdown).update(msg.wiki_empty_state_detail())

    @on(Tree.NodeSelected, "#wiki-page-list")
    def _on_node_selected(self, event: Tree.NodeSelected[str | None]) -> None:
        """Load and display the selected wiki page when the node carries a slug."""
        slug = event.node.data
        if not isinstance(slug, str):
            return
        self._display_page(slug)

    def _display_page(self, slug: str) -> None:
        """Read and render a wiki page by slug."""
        from lilbee.wiki.browse import read_page

        root = _wiki_root()
        page = read_page(root, slug)
        if page is None:
            self.query_one("#wiki-breadcrumb", Static).update("")
            self.query_one("#wiki-page-header", Static).update("")
            self.query_one("#wiki-content", Markdown).update(msg.WIKI_NO_CONTENT)
            return

        # Frontmatter is arbitrary parsed YAML; a non-numeric value must not
        # crash the node-select handler that calls this.
        faith_val = _safe_float(page.frontmatter.get("faithfulness_score"))

        page_type = ""
        parts = slug.split("/")
        if len(parts) >= _SLUG_WITH_TYPE_MIN_PARTS:
            from lilbee.wiki.shared import SUBDIR_TO_TYPE

            page_type = SUBDIR_TO_TYPE.get(parts[0], "")

        source_count = page.frontmatter.get("source_count", 0)
        created_at = page.frontmatter.get("generated_at", "")
        if isinstance(created_at, (datetime, date)):
            created_at = created_at.isoformat()

        header_text = _format_page_header(
            title=page.title,
            page_type=page_type,
            source_count=_safe_int(source_count),
            created_at=str(created_at),
            faithfulness=faith_val,
        )
        self.query_one("#wiki-breadcrumb", Static).update(_breadcrumb_for_slug(slug, page.title))
        self.query_one("#wiki-page-header", Static).update(header_text)
        self.query_one("#wiki-content", Markdown).update(page.content)

    @on(Input.Changed, "#wiki-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        """Re-filter after a short debounce so a multi-key term re-walks the wiki
        tree once on pause, not once per keystroke."""
        filter_text = event.value.strip()
        if self._search_filter_timer is not None:
            self._search_filter_timer.stop()
        self._search_filter_timer = self.set_timer(
            self._SEARCH_FILTER_DEBOUNCE_SECONDS,
            lambda: self._load_pages(filter_text=filter_text),
        )

    def _selected_source(self) -> str | None:
        """Return the source name for the highlighted wiki page, or None."""
        tree = self.query_one("#wiki-page-list", Tree)
        node = tree.cursor_node
        if node is None:
            return None
        slug = node.data
        if not isinstance(slug, str):
            return None
        return self._source_for_slug(slug)

    def _source_for_slug(self, slug: str) -> str | None:
        """Extract the primary source filename from a wiki page's frontmatter."""
        root = _wiki_root()
        page = read_page(root, slug)
        if page is None:
            return None
        sources = page.frontmatter.get("sources")
        # frontmatter values are untyped (Any from YAML); guard against non-list shapes
        if isinstance(sources, list) and sources:
            return str(sources[0])
        return None

    def action_focus_search(self) -> None:
        """Focus the search input -- bound to / key."""
        self.query_one("#wiki-search", Input).focus()

    def action_open_drafts(self) -> None:
        """Open the drafts review screen -- bound to capital D."""
        from lilbee.cli.tui.screens.wiki_drafts import WikiDraftsScreen

        self.app.push_screen(WikiDraftsScreen())

    def action_dismiss_or_back(self) -> None:
        """Clear search if active, otherwise go back."""
        search = self.query_one("#wiki-search", Input)
        if search.value:
            search.value = ""
            return
        self.action_go_back()

    def action_go_back(self) -> None:
        self.app.switch_view("Chat")

    def _tree_or_none(self) -> Tree[str | None] | None:
        if isinstance(self.focused, Input):
            return None
        return self.query_one("#wiki-page-list", Tree)

    def action_cursor_down(self) -> None:
        tree = self._tree_or_none()
        if tree is not None:
            tree.action_cursor_down()

    def action_cursor_up(self) -> None:
        tree = self._tree_or_none()
        if tree is not None:
            tree.action_cursor_up()

    def action_cursor_left(self) -> None:
        tree = self._tree_or_none()
        if tree is not None:
            tree.action_cursor_parent()

    def action_cursor_right(self) -> None:
        tree = self._tree_or_none()
        if tree is not None:
            tree.action_toggle_node()

    def action_jump_top(self) -> None:
        tree = self._tree_or_none()
        if tree is not None:
            tree.scroll_home()

    def action_jump_bottom(self) -> None:
        tree = self._tree_or_none()
        if tree is not None:
            tree.scroll_end()


def _find_or_add_branch(parent: TreeNode[str | None], label_part: str) -> TreeNode[str | None]:
    """Return the child branch whose raw label matches *label_part*, adding it if absent."""
    display = _short_label(label_part)
    for child in parent.children:
        existing = child.label.plain if hasattr(child.label, "plain") else str(child.label)
        if existing == display:
            return child
    return parent.add(display, expand=True)


def _group_pages(
    pages: list[WikiPageInfo],
) -> list[tuple[str, list[WikiPageInfo]]]:
    """Group pages by page_type in sidebar order: concepts, entities, then legacy."""
    from lilbee.wiki.shared import WikiPageType

    groups: dict[str, list[WikiPageInfo]] = {}
    type_order: tuple[str, ...] = (
        WikiPageType.CONCEPT,
        WikiPageType.ENTITY,
        WikiPageType.SUMMARY,
        WikiPageType.SYNTHESIS,
    )
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
