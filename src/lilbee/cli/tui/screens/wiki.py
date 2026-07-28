"""Wiki screen: browse wiki pages as a navigable tree with markdown preview."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
    from lilbee.runtime.progress import DetailedProgressCallback, ProgressEvent
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
from lilbee.cli.tui.task_queue import TaskType
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.task_bar import TaskBar
from lilbee.core.config import cfg
from lilbee.runtime.progress import EventType, WikiPageEvent, WikiPhaseEvent

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
    HELP = (
        "Browse wiki pages. h/l collapse/expand, j/k navigate, Enter opens a page, "
        "/ searches, b generates pages from your documents."
    )

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "dismiss_or_back", "Back", show=False),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("D", "open_drafts", "Drafts", show=True),
        Binding("b", "wikify", "Wikify", show=True),
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
        self._pages: list[WikiPageInfo] = []
        self._load_error: str | None = None
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
        self.reload()

    def on_show(self) -> None:
        """Re-scan on focus so out-of-band builds (`lilbee wiki build` from a
        sibling shell) and incremental wiki updates land without a TUI restart.
        """
        self.reload()

    def reload(self) -> None:
        """Re-walk the wiki from disk, then repaint under the live search filter.

        Public entry point for external callers (the task bar refreshes an
        open wiki screen after a sync or a wikify run). The walk parses every
        page's frontmatter, so it runs here and not on every filter keystroke.
        """
        from lilbee.wiki.browse import list_pages

        self._pages = []
        self._load_error = None
        if cfg.wiki:
            try:
                self._pages = list_pages(_wiki_root())
            except Exception as exc:
                log.warning("Failed to list wiki pages", exc_info=True)
                self._load_error = msg.WIKI_LOAD_FAILED.format(error=exc)
                self._show_load_failure(self._load_error)
                return
        self._load_pages(filter_text=self.query_one("#wiki-search", Input).value.strip())

    def _show_load_failure(self, detail: str) -> None:
        """Render the listing failure in both panes."""
        tree = self._empty_tree()
        tree.root.add_leaf(msg.WIKI_LOAD_FAILED_LEAF)
        self._show_detail(detail)

    def _empty_tree(self) -> Tree[str | None]:
        """Clear the sidebar tree and the slug list it indexes."""
        tree = self.query_one("#wiki-page-list", Tree)
        tree.reset("Wiki")
        self._page_slugs = []
        return tree

    def _load_pages(self, filter_text: str = "") -> None:
        """Populate the sidebar tree from the cached page list, optionally filtered."""
        if self._load_error is not None:
            # The cache is empty because listing failed, not because the wiki is.
            self._show_load_failure(self._load_error)
            return
        tree = self._empty_tree()
        if not self._pages:
            tree.root.add_leaf(msg.wiki_empty_state_leaf())
            self._show_placeholder()
            return

        needle = filter_text.lower()
        pages = [p for p in self._pages if needle in p.title.lower()]
        if not pages:
            # Pages exist but none match: leave the content pane untouched
            # rather than rendering the empty-wiki state.
            tree.root.add_leaf(msg.WIKI_NO_MATCHES.format(filter=filter_text))
            return

        self._populate_tree(tree, pages)

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
        branches: dict[str, TreeNode[str | None]] = {}
        for page_type, group_pages in grouped:
            heading = msg.WIKI_TYPE_HEADINGS.get(page_type, page_type.capitalize())
            group_node = tree.root.add(heading, expand=True)
            for page in group_pages:
                self._page_slugs.append(page.slug)
                self._insert_page(group_node, page, branches)

    def _add_root_shortcut(self, tree: Tree[str | None], slug: str, label: str) -> None:
        """Add a top-level leaf for an auto-generated page (index.md, log.md)."""
        if not (_wiki_root() / f"{slug}.md").is_file():
            return
        tree.root.add_leaf(label, data=slug)
        self._page_slugs.append(slug)

    def _insert_page(
        self,
        group_node: TreeNode[str | None],
        page: WikiPageInfo,
        branches: dict[str, TreeNode[str | None]],
    ) -> None:
        """Walk the slug path and add/reuse branches until the leaf position.

        Slugs begin with the page-type prefix (``summaries/``/``synthesis/``),
        which is already reflected in the enclosing group node. The remaining
        path components form the nested tree inside the group. *branches* is
        the per-build reuse map, keyed by raw slug path.
        """
        parts = page.slug.split("/")
        if len(parts) <= 1:
            group_node.add_leaf(page.title, data=page.slug)
            return

        # Skip the leading page-type component since the group node represents it.
        node = group_node
        *branch_parts, leaf_part = parts[1:]
        path = parts[0]
        for part in branch_parts:
            path = f"{path}/{part}"
            node = _find_or_add_branch(node, part, path, branches)

        if leaf_part == _INDEX_STEM:
            # An inner-node index.md file: show its title on the enclosing branch.
            node.label = page.title
            node.data = page.slug
            return

        label = _short_label(leaf_part)
        node.add_leaf(page.title if page.title else label, data=page.slug)

    def _show_detail(self, markdown: str) -> None:
        """Clear the breadcrumb and header rows, then render *markdown* alone."""
        self.query_one("#wiki-breadcrumb", Static).update("")
        self.query_one("#wiki-page-header", Static).update("")
        self.query_one("#wiki-content", Markdown).update(markdown)

    def _show_placeholder(self) -> None:
        """Show the no-content placeholder in the main area."""
        self._show_detail(msg.wiki_empty_state_detail())

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
            self._show_detail(msg.WIKI_NO_CONTENT)
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
        """Re-filter after a short debounce so a multi-key term repaints the
        tree once on pause, not once per keystroke."""
        filter_text = event.value.strip()
        if self._search_filter_timer is not None:
            self._search_filter_timer.stop()
        self._search_filter_timer = self.set_timer(
            self._SEARCH_FILTER_DEBOUNCE_SECONDS,
            lambda: self._load_pages(filter_text=filter_text),
        )

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
            tree.action_scroll_home()

    def action_jump_bottom(self) -> None:
        tree = self._tree_or_none()
        if tree is not None:
            tree.action_scroll_end()

    def action_wikify(self) -> None:
        """Generate wiki pages from the ingested corpus -- bound to b."""
        start_wikify(self.app)


def _build_progress(reporter: ProgressReporter) -> DetailedProgressCallback:
    """Map wiki build events onto task-bar progress updates."""

    def _on_progress(event_type: EventType, data: ProgressEvent) -> None:
        if event_type is EventType.WIKI_PHASE and isinstance(data, WikiPhaseEvent):
            reporter.update(
                0, msg.WIKI_BUILD_PHASE.format(phase=data.phase.value), indeterminate=True
            )
        elif event_type is EventType.WIKI_PAGE and isinstance(data, WikiPageEvent):
            percent = int(data.current * 100 / data.total) if data.total else 0
            reporter.update(
                percent,
                msg.WIKI_BUILD_PAGE.format(
                    label=data.label, current=data.current, total=data.total
                ),
                indeterminate=False,
            )

    return _on_progress


def start_wikify(app: LilbeeApp) -> None:
    """Run a full wiki build on the task bar.

    Shared by the wiki screen's ``b`` binding and the command palette so both
    surfaces get the same progress, serialization and completion refresh. The
    build takes the wiki mutex, so it must never run on the event loop.
    """
    if not cfg.wiki:
        app.notify(msg.CMD_WIKI_DISABLED, severity="warning")
        return

    queue = app.task_bar.queue
    pending = queue.active_tasks + queue.queued_tasks
    if any(t.task_type == TaskType.WIKI and t.name == msg.TASK_NAME_WIKI for t in pending):
        app.notify(msg.WIKI_ALREADY_ACTIVE, severity="warning")
        return

    def _target(reporter: ProgressReporter) -> None:
        from lilbee.wiki.generation import run_full_build

        try:
            summary = run_full_build(on_progress=_build_progress(reporter))
            reporter.check_cancelled()
        except Exception:
            # Pages already written must show; the done hook only fires on success.
            call_from_thread(app, app.task_bar.reload_wiki_screens)
            raise
        call_from_thread(app, app.notify, msg.WIKI_BUILD_DONE.format(count=summary["count"]))

    app.task_bar.start_task(msg.TASK_NAME_WIKI, TaskType.WIKI, _target, indeterminate=True)


def _find_or_add_branch(
    parent: TreeNode[str | None],
    label_part: str,
    path: str,
    branches: dict[str, TreeNode[str | None]],
) -> TreeNode[str | None]:
    """Return the branch registered for *path*, adding it under *parent* if absent.

    Reuse is keyed on the raw slug path, so components that render to the
    same display label ("cv-manual" and "cv_manual") stay separate, and a
    branch renamed by an inner index page is still found.
    """
    node = branches.get(path)
    if node is None:
        node = parent.add(_short_label(label_part), expand=True)
        branches[path] = node
    return node


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
