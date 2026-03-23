"""Catalog screen — browse and install models inline."""

from __future__ import annotations

from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
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
from lilbee.model_manager import OllamaModel

TASK_TABS = ("All", "Chat", "Embedding", "Vision")
_TAB_TO_TASK: dict[str, str | None] = {
    "All": None,
    "Chat": "chat",
    "Embedding": "embedding",
    "Vision": "vision",
}

_HF_PAGE_SIZE = 10


class ModelRow(ListItem):
    """A catalog model row."""

    def __init__(self, model: CatalogModel) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        m = self.model
        star = "★" if m.featured else " "
        size = f"{m.size_gb:.1f} GB" if m.size_gb > 0 else "  ?   "
        dl = f"↓{_format_downloads(m.downloads)}" if m.downloads > 0 else ""
        desc = m.description[:50] if m.description else ""
        yield Static(
            f" {star} {m.name:<30s} {m.task:<10s} {size:>8s}  {dl:>8s}  {desc}",
            classes="model-row-text",
        )


class OllamaRow(ListItem):
    """An Ollama model (inference-only)."""

    def __init__(self, model: OllamaModel) -> None:
        super().__init__()
        self.ollama_model = model

    def compose(self) -> ComposeResult:
        m = self.ollama_model
        size = m.parameter_size or "?"
        yield Static(
            f"   {m.name:<30s} {m.task:<10s} {size:>8s}  (Ollama)",
            classes="model-row-text",
        )


class LoadMoreRow(ListItem):
    """A 'Load more...' pagination row."""

    def compose(self) -> ComposeResult:
        yield Static("   ↓ Load more models...", classes="model-row-text")


def _parse_param_size(name: str) -> str:
    """Extract parameter size category from model name."""
    import re

    match = re.search(r"(\d+\.?\d*)B", name, re.IGNORECASE)
    if not match:
        return "unknown"
    size = float(match.group(1))
    if size <= 3:
        return "Small (≤3B)"
    if size <= 8:
        return "Medium (3-8B)"
    if size <= 30:
        return "Large (8-30B)"
    return "Extra Large (30B+)"


def _format_downloads(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


class CatalogScreen(Screen[None]):
    """Model catalog with tabs, search, and inline install."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("space", "page_down", "Page Down", show=False),
        Binding("ctrl+d", "page_down", "½ Page Down", show=False),
        Binding("ctrl+u", "page_up", "½ Page Up", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._featured: list[CatalogModel] = list(FEATURED_ALL)
        self._hf_models: list[CatalogModel] = []
        self._ollama_models: list[OllamaModel] = []
        self._hf_offset = 0
        self._hf_has_more = True

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Filter models...", id="catalog-search")
        with TabbedContent(*TASK_TABS, id="catalog-tabs"):
            for tab_label in TASK_TABS:
                with TabPane(tab_label, id=f"cat-{tab_label.lower()}"):
                    yield ListView(id=f"catlist-{tab_label.lower()}")
        yield Static("", id="model-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_lists()
        self._fetch_hf_models()
        self._fetch_ollama_models()

    @work(thread=True)
    def _fetch_hf_models(self) -> list[CatalogModel]:
        result = get_catalog(
            featured=False, limit=_HF_PAGE_SIZE, offset=self._hf_offset, sort="downloads"
        )
        new_models = [m for m in result.models if not m.featured]
        self._hf_has_more = len(new_models) >= _HF_PAGE_SIZE
        return new_models

    @work(thread=True)
    def _fetch_ollama_models(self) -> list[OllamaModel]:
        from lilbee.config import cfg
        from lilbee.model_manager import classify_ollama_models

        return classify_ollama_models(cfg.ollama_url)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state != WorkerState.SUCCESS:
            return
        result = event.worker.result
        if event.worker.name == "_fetch_hf_models" and isinstance(result, list):
            self._hf_models = result
            self._refresh_lists()
        elif event.worker.name == "_fetch_more_hf" and isinstance(result, list):
            self._hf_models.extend(result)
            self._refresh_lists()
        elif event.worker.name == "_fetch_ollama_models" and isinstance(result, list):
            self._ollama_models = result
            self._refresh_lists()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "catalog-search":
            self._refresh_lists()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._refresh_lists()

    def _get_search_text(self) -> str:
        return self.query_one("#catalog-search", Input).value.strip().lower()

    def _refresh_lists(self) -> None:
        search = self._get_search_text()
        for tab_label in TASK_TABS:
            task = _TAB_TO_TASK[tab_label]
            lv = self.query_one(f"#catlist-{tab_label.lower()}", ListView)
            lv.clear()

            featured = _filter_catalog(self._featured, task, search)
            hf = _filter_catalog(self._hf_models, task, search)
            ollama = _filter_ollama(self._ollama_models, task, search)

            if featured:
                lv.append(ListItem(Label("★ FEATURED", classes="section-header")))
                for m in featured:
                    lv.append(ModelRow(m))

            if ollama:
                lv.append(ListItem(Label("INSTALLED (Ollama)", classes="section-header")))
                for om in ollama:
                    lv.append(OllamaRow(om))

            if hf:
                grouped = _group_by_size(sorted(hf, key=lambda x: x.downloads, reverse=True))
                for group_label, group_models in grouped:
                    lv.append(
                        ListItem(Label(f"HUGGINGFACE — {group_label}", classes="section-header"))
                    )
                    for m in group_models:
                        lv.append(ModelRow(m))

                if self._hf_has_more and not search:
                    lv.append(LoadMoreRow())

            if not featured and not ollama and not hf:
                lv.append(ListItem(Label("No models match your filters.")))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, LoadMoreRow):
            self._load_more()
            return
        if isinstance(item, ModelRow):
            self._install_model(item.model)
        elif isinstance(item, OllamaRow):
            from lilbee.config import cfg

            cfg.chat_model = item.ollama_model.name
            self.notify(f"Using {item.ollama_model.name} (Ollama)")
            self.app.title = f"lilbee — {item.ollama_model.name}"

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        detail = self.query_one("#model-detail", Static)
        if event.item and isinstance(event.item, ModelRow):
            m = event.item.model
            size = f"{m.size_gb:.1f} GB" if m.size_gb > 0 else "unknown"
            detail.update(
                f"{m.name} — {m.description}\nTask: {m.task}  Size: {size}  Repo: {m.hf_repo}"
            )
        elif event.item and isinstance(event.item, OllamaRow):
            om = event.item.ollama_model
            detail.update(f"{om.name} — {om.task}  Family: {om.family}  {om.parameter_size}")
        else:
            detail.update("")

    def _load_more(self) -> None:
        """Load next page of HF models."""
        self._hf_offset += _HF_PAGE_SIZE
        self._fetch_more_hf()

    @work(thread=True)
    def _fetch_more_hf(self) -> list[CatalogModel]:
        result = get_catalog(
            featured=False, limit=_HF_PAGE_SIZE, offset=self._hf_offset, sort="downloads"
        )
        new_models = [m for m in result.models if not m.featured]
        self._hf_has_more = len(new_models) >= _HF_PAGE_SIZE
        return new_models

    def _install_model(self, model: CatalogModel) -> None:
        from lilbee.model_manager import get_model_manager

        manager = get_model_manager()
        if manager.is_installed(model.name):
            self.notify(f"{model.name} is already installed")
            return

        from lilbee.cli.tui.widgets.download_modal import DownloadModal

        self.app.push_screen(DownloadModal(model))

    def action_focus_search(self) -> None:
        self.query_one("#catalog-search", Input).focus()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_page_down(self) -> None:
        for tab_label in TASK_TABS:
            lv = self.query_one(f"#catlist-{tab_label.lower()}", ListView)
            if lv.has_focus:
                for _ in range(10):
                    lv.action_cursor_down()
                return

    def action_page_up(self) -> None:
        for tab_label in TASK_TABS:
            lv = self.query_one(f"#catlist-{tab_label.lower()}", ListView)
            if lv.has_focus:
                for _ in range(10):
                    lv.action_cursor_up()
                return

    def key_j(self) -> None:
        if isinstance(self.focused, Input):
            return
        for tab_label in TASK_TABS:
            lv = self.query_one(f"#catlist-{tab_label.lower()}", ListView)
            if lv.has_focus:
                lv.action_cursor_down()
                return

    def key_k(self) -> None:
        if isinstance(self.focused, Input):
            return
        for tab_label in TASK_TABS:
            lv = self.query_one(f"#catlist-{tab_label.lower()}", ListView)
            if lv.has_focus:
                lv.action_cursor_up()
                return


def _filter_catalog(
    models: list[CatalogModel], task: str | None, search: str
) -> list[CatalogModel]:
    filtered = models
    if task:
        filtered = [m for m in filtered if m.task == task]
    if search:
        filtered = [
            m
            for m in filtered
            if search in m.name.lower()
            or search in m.hf_repo.lower()
            or search in m.description.lower()
        ]
    return filtered


def _filter_ollama(models: list[OllamaModel], task: str | None, search: str) -> list[OllamaModel]:
    filtered = models
    if task:
        filtered = [m for m in filtered if m.task == task]
    if search:
        filtered = [m for m in filtered if search in m.name.lower()]
    return filtered


def _group_by_size(models: list[CatalogModel]) -> list[tuple[str, list[CatalogModel]]]:
    """Group models by inferred parameter size."""
    groups: dict[str, list[CatalogModel]] = {}
    for m in models:
        category = _parse_param_size(m.name)
        groups.setdefault(category, []).append(m)

    order = ["Small (≤3B)", "Medium (3-8B)", "Large (8-30B)", "Extra Large (30B+)", "unknown"]
    return [(label, groups[label]) for label in order if label in groups]
