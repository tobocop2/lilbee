"""Interactive model catalog browser using Textual TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker, WorkerState

from lilbee.catalog import FEATURED_ALL, CatalogModel, get_catalog

TASK_TABS = ("All", "Chat", "Embedding", "Vision")
_TAB_TO_TASK: dict[str, str | None] = {
    "All": None,
    "Chat": "chat",
    "Embedding": "embedding",
    "Vision": "vision",
}


@dataclass
class BrowserState:
    """Cached catalog data for the browser."""

    featured: list[CatalogModel]
    hf_models: list[CatalogModel]

    @property
    def all_models(self) -> list[CatalogModel]:
        return self.featured + self.hf_models


class ModelRow(ListItem):
    """A single model row in the catalog list."""

    def __init__(self, model: CatalogModel) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        m = self.model
        star = " *" if m.featured else "  "
        size = f"{m.size_gb:.1f} GB"
        desc = m.description[:60] if m.description else ""
        yield Static(
            f"{star} {m.name:<30s} {m.task:<12s} {size:>8s}  {desc}",
            classes="model-row-text",
        )


class CatalogBrowser(App[CatalogModel | None]):
    """Full-screen TUI for browsing the model catalog."""

    TITLE = "lilbee model catalog"
    CSS_PATH = "browser.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit_browser", "Quit", show=True),
        Binding("escape", "quit_browser", "Quit", show=False),
        Binding("slash", "focus_search", "Search", show=True),
    ]

    search_text: reactive[str] = reactive("", layout=True)

    def __init__(
        self,
        initial_task: str | None = None,
        initial_search: str = "",
    ) -> None:
        super().__init__()
        self._initial_task = initial_task
        self._initial_search = initial_search
        self._state = BrowserState(featured=list(FEATURED_ALL), hf_models=[])
        self._selected: CatalogModel | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Type to filter...", id="search-input")
        with TabbedContent(*TASK_TABS, id="task-tabs"):
            for tab_label in TASK_TABS:
                with TabPane(tab_label, id=f"tab-{tab_label.lower()}"):
                    yield ListView(id=f"list-{tab_label.lower()}")
        yield Footer()

    def on_mount(self) -> None:
        if self._initial_search:
            search_input = self.query_one("#search-input", Input)
            search_input.value = self._initial_search
            self.search_text = self._initial_search

        if self._initial_task:
            tab_id = f"tab-{self._initial_task}"
            tabs = self.query_one("#task-tabs", TabbedContent)
            tabs.active = tab_id

        self._refresh_lists()
        self._fetch_hf_models()

    @work(thread=True)
    def _fetch_hf_models(self) -> list[CatalogModel]:
        """Fetch HuggingFace models in a background worker."""
        result = get_catalog(featured=False, limit=50)
        hf_only = [m for m in result.models if not m.featured]
        return hf_only

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.SUCCESS and event.worker.name == "_fetch_hf_models":
            hf_models = event.worker.result
            if isinstance(hf_models, list):
                self._state.hf_models = hf_models
                self._refresh_lists()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            self.search_text = event.value
            self._refresh_lists()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._refresh_lists()

    def _filter_models(self, models: list[CatalogModel], task: str | None) -> list[CatalogModel]:
        filtered = models
        if task:
            filtered = [m for m in filtered if m.task == task]
        search = self.search_text.strip().lower()
        if search:
            filtered = [
                m
                for m in filtered
                if search in m.name.lower()
                or search in m.hf_repo.lower()
                or search in m.description.lower()
            ]
        return filtered

    def _refresh_lists(self) -> None:
        """Rebuild every tab's ListView from current state and filters."""
        for tab_label in TASK_TABS:
            task = _TAB_TO_TASK[tab_label]
            list_id = f"list-{tab_label.lower()}"
            lv = self.query_one(f"#{list_id}", ListView)

            featured = self._filter_models(self._state.featured, task)
            hf = self._filter_models(self._state.hf_models, task)

            lv.clear()

            if featured:
                lv.append(ListItem(Label("FEATURED", classes="section-header")))
                for m in featured:
                    lv.append(ModelRow(m))

            if hf:
                lv.append(ListItem(Label("HUGGINGFACE", classes="section-header")))
                for m in hf:
                    lv.append(ModelRow(m))

            if not featured and not hf:
                lv.append(ListItem(Label("No models match your filters.")))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ModelRow):
            self._selected = item.model
            self.exit(item.model)

    def action_quit_browser(self) -> None:
        self.exit(None)

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()


def run_browser(task: str | None = None, search: str = "") -> CatalogModel | None:
    """Launch the catalog browser and return the selected model, or None."""
    app = CatalogBrowser(initial_task=task, initial_search=search)
    return app.run()
