"""Catalog screen — browse and install models inline."""

from __future__ import annotations

import logging
import re
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

from lilbee.catalog import (
    FEATURED_ALL,
    CatalogModel,
    ModelFamily,
    ModelVariant,
    clean_display_name,
    get_catalog,
    get_families,
    quant_tier,
)
from lilbee.config import cfg
from lilbee.model_manager import RemoteModel

log = logging.getLogger(__name__)

TASK_TABS = ("All", "Chat", "Embedding", "Vision")
_TAB_TO_TASK: dict[str, str | None] = {
    "All": None,
    "Chat": "chat",
    "Embedding": "embedding",
    "Vision": "vision",
}

_HF_PAGE_SIZE = 25

_SORT_CYCLE = ("downloads", "name", "size_desc", "featured")
_SORT_LABELS = {
    "downloads": "Downloads ↓",
    "name": "Name A-Z",
    "size_desc": "Size ↓",
    "featured": "Featured first",
}


def _parse_param_label(name: str) -> str:
    """Extract parameter count label from model name (e.g. '8B', '0.6B')."""
    match = re.search(r"(\d+\.?\d*)B", name, re.IGNORECASE)
    return f"{match.group(1)}B" if match else "—"


def _parse_param_size(name: str) -> str:
    """Extract parameter size category from model name."""
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


def _format_row(m: CatalogModel, cached_size: float | None = None) -> str:
    """Format a model row string."""
    star = "★" if m.featured else " "
    display = clean_display_name(m.hf_repo)
    params = _parse_param_label(m.name)
    size_gb = cached_size if cached_size is not None else m.size_gb
    size = f"{size_gb:.1f} GB" if size_gb > 0 else "  —   "
    dl = f"↓{_format_downloads(m.downloads)}" if m.downloads > 0 else ""
    desc = m.description[:45] if m.description else ""
    return f" {star} {display:<30s} {m.task:<10s} {params:>5s} {size:>8s}  {dl:>8s}  {desc}"


def _format_size_mb(size_mb: int) -> str:
    """Format size in MB to a human-readable string."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb} MB"


def _format_variant_row(v: ModelVariant) -> str:
    """Format a variant row for display inside a family group."""
    star = "★ " if v.recommended else "  "
    quant_label = v.quant or "default"
    tier = quant_tier(v.quant)
    tier_tag = f" [{tier}]" if tier != "unknown" else ""
    size = _format_size_mb(v.size_mb)
    suffix = " — recommended" if v.recommended else ""
    return f"  {star}{v.param_count} {quant_label} ({size}){tier_tag}{suffix}"


def _format_family_header(f: ModelFamily) -> str:
    """Format a family header row."""
    return f"{f.name} — {f.description}"


class VariantRow(ListItem):
    """A model variant row within a family group."""

    def __init__(self, variant: ModelVariant, family: ModelFamily) -> None:
        super().__init__()
        self.variant = variant
        self.family = family

    def compose(self) -> ComposeResult:
        yield Static(_format_variant_row(self.variant), classes="model-row-text")


class ModelRow(ListItem):
    """A catalog model row."""

    def __init__(self, model: CatalogModel) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        yield Static(_format_row(self.model), classes="model-row-text")


class RemoteRow(ListItem):
    """A remote model (inference-only, managed by external tool)."""

    def __init__(self, model: RemoteModel) -> None:
        super().__init__()
        self.remote_model = model

    def compose(self) -> ComposeResult:
        m = self.remote_model
        size = m.parameter_size or "?"
        yield Static(
            f"   {m.name:<30s} {m.task:<10s} {size:>5s}           (remote)",
            classes="model-row-text",
        )


class LoadMoreRow(ListItem):
    """A 'Load more...' pagination row."""

    def compose(self) -> ComposeResult:
        yield Static("   ↓ Load more models...", classes="model-row-text")


class CatalogScreen(Screen[None]):
    """Model catalog with tabs, search, and inline install."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "pop_screen", "Back", show=True),
        Binding("escape", "pop_screen", "Back", show=False),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("s", "cycle_sort", "Sort", show=True),
        Binding("space", "page_down", "Page Down", show=False),
        Binding("ctrl+d", "page_down", "½ Page Down", show=False),
        Binding("ctrl+u", "page_up", "½ Page Up", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._featured: list[CatalogModel] = list(FEATURED_ALL)
        self._families: list[ModelFamily] = get_families()
        self._hf_models: list[CatalogModel] = []
        self._remote_models: list[RemoteModel] = []
        self._hf_offset = 0
        self._hf_has_more = True
        self._current_sort = "downloads"
        self._size_cache: dict[str, float] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Filter models...", id="catalog-search")
        yield Static(f"Sort: {_SORT_LABELS[self._current_sort]}", id="sort-label")
        with TabbedContent(*TASK_TABS, id="catalog-tabs"):
            for tab_label in TASK_TABS:
                with TabPane(tab_label, id=f"cat-{tab_label.lower()}"):
                    yield ListView(id=f"catlist-{tab_label.lower()}")
        yield Static("", id="model-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_lists()
        self._fetch_hf_models()
        self._fetch_remote_models()

    @work(thread=True)
    def _fetch_hf_models(self) -> list[CatalogModel]:
        result = get_catalog(
            featured=False, limit=_HF_PAGE_SIZE, offset=self._hf_offset, sort=self._current_sort
        )
        new_models = [m for m in result.models if not m.featured]
        self._hf_has_more = len(new_models) >= _HF_PAGE_SIZE
        return new_models

    @work(thread=True)
    def _fetch_remote_models(self) -> list[RemoteModel]:
        from lilbee.model_manager import classify_remote_models

        return classify_remote_models(cfg.litellm_base_url)

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
        elif event.worker.name == "_fetch_remote_models" and isinstance(result, list):
            self._remote_models = result
            self._refresh_lists()
        elif event.worker.name == "_fetch_model_size" and isinstance(result, tuple):
            repo, size_gb = result
            if size_gb > 0:
                self._size_cache[repo] = size_gb
                self._update_highlighted_detail()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "catalog-search":
            self._refresh_lists()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._hf_offset = 0
        self._hf_models = []
        self._hf_has_more = True
        self._refresh_lists()
        self._fetch_hf_models()

    def _get_search_text(self) -> str:
        return self.query_one("#catalog-search", Input).value.strip().lower()

    def _refresh_lists(self) -> None:
        search = self._get_search_text()
        for tab_label in TASK_TABS:
            task = _TAB_TO_TASK[tab_label]
            lv = self.query_one(f"#catlist-{tab_label.lower()}", ListView)
            lv.clear()

            families = _filter_families(self._families, task, search)
            hf = _filter_catalog(self._hf_models, task, search)
            remote = _filter_remote(self._remote_models, task, search)

            if families:
                lv.append(ListItem(Label("★ FEATURED", classes="section-header")))
                for fam in families:
                    lv.append(ListItem(Label(_format_family_header(fam), classes="section-header")))
                    for v in fam.variants:
                        lv.append(VariantRow(v, fam))

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

            if remote:
                lv.append(ListItem(Label("INSTALLED (Remote)", classes="section-header")))
                for rm in remote:
                    lv.append(RemoteRow(rm))

            if not families and not remote and not hf:
                lv.append(ListItem(Label("No models match your filters.")))

        n_featured = sum(len(f.variants) for f in self._families)
        n_hf = len(self._hf_models)
        n_remote = len(self._remote_models)
        total = n_featured + n_hf + n_remote
        more = "+" if self._hf_has_more else ""
        self.query_one("#sort-label", Static).update(
            f"Sort: {_SORT_LABELS[self._current_sort]}  |  "
            f"Showing {total}{more} models "
            f"({n_featured} featured, {n_hf} HF, {n_remote} remote)"
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, LoadMoreRow):
            self._load_more()
            return
        if isinstance(item, VariantRow):
            self._install_variant(item.variant, item.family)
        elif isinstance(item, ModelRow):
            self._install_model(item.model)
        elif isinstance(item, RemoteRow):
            cfg.chat_model = item.remote_model.name
            self.notify(f"Using {item.remote_model.name} (remote)")
            self.app.title = f"lilbee — {item.remote_model.name}"

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        self._update_highlighted_detail(event.item)

    def _update_highlighted_detail(self, item: ListItem | None = None) -> None:
        """Update detail panel, optionally triggering lazy size fetch."""
        detail = self.query_one("#model-detail", Static)

        if item is None:
            # Re-update for the currently highlighted item (after size fetch)
            for tab_label in TASK_TABS:
                lv = self.query_one(f"#catlist-{tab_label.lower()}", ListView)
                if lv.highlighted_child:
                    item = lv.highlighted_child
                    break
            if item is None:
                return

        if isinstance(item, VariantRow):
            v = item.variant
            fam = item.family
            size = _format_size_mb(v.size_mb)
            rec = " (recommended)" if v.recommended else ""
            detail.update(
                f"{fam.name} {v.param_count} — {fam.description}\n"
                f"Task: {fam.task}  Quant: {v.quant}  Size: {size}{rec}  Repo: {v.hf_repo}"
            )
        elif isinstance(item, ModelRow):
            m = item.model
            cached = self._size_cache.get(m.hf_repo)
            size_gb = cached if cached is not None else m.size_gb
            size = f"{size_gb:.1f} GB" if size_gb > 0 else "fetching..."
            params = _parse_param_label(m.name)
            detail.update(
                f"{m.name} — {m.description}\n"
                f"Task: {m.task}  Params: {params}  Size: {size}  Repo: {m.hf_repo}"
            )
            # Lazy-load file size if unknown and not cached
            if m.size_gb <= 0 and m.hf_repo not in self._size_cache:
                self._fetch_model_size(m.hf_repo)
        elif isinstance(item, RemoteRow):
            rm = item.remote_model
            detail.update(f"{rm.name} — {rm.task}  Family: {rm.family}  {rm.parameter_size}")
        else:
            detail.update("")

    @work(thread=True, exclusive=True, group="size_fetch")
    def _fetch_model_size(self, hf_repo: str) -> tuple[str, float]:
        """Lazy-load file size from HF tree API."""
        from lilbee.catalog import fetch_model_file_size

        size_gb = fetch_model_file_size(hf_repo)
        return (hf_repo, size_gb)

    def _load_more(self) -> None:
        """Load next page of HF models."""
        self._hf_offset += _HF_PAGE_SIZE
        self._fetch_more_hf()

    @work(thread=True)
    def _fetch_more_hf(self) -> list[CatalogModel]:
        result = get_catalog(
            featured=False, limit=_HF_PAGE_SIZE, offset=self._hf_offset, sort=self._current_sort
        )
        new_models = [m for m in result.models if not m.featured]
        self._hf_has_more = len(new_models) >= _HF_PAGE_SIZE
        return new_models

    def _install_variant(self, variant: ModelVariant, family: ModelFamily) -> None:
        """Convert a variant back to a CatalogModel and trigger install."""
        entry = CatalogModel(
            name=f"{family.name} {variant.param_count}",
            hf_repo=variant.hf_repo,
            gguf_filename=variant.filename,
            size_gb=variant.size_mb / 1024,
            min_ram_gb=max(2.0, (variant.size_mb / 1024) * 1.5),
            description=family.description,
            featured=True,
            downloads=0,
            task=family.task,
        )
        self._install_model(entry)

    def _install_model(self, model: CatalogModel) -> None:
        from lilbee.catalog import _resolve_filename

        try:
            filename = _resolve_filename(model)
            dest = cfg.models_dir / filename
            if dest.exists():
                self.notify(f"{model.name} is already installed")
                return
        except Exception:
            pass  # Can't resolve filename -- proceed with download

        self._enqueue_download(model)

    def _enqueue_download(self, model: CatalogModel) -> None:
        """Enqueue a model download in the ChatScreen's TaskBar."""
        task_bar = getattr(self.app, "_task_bar", None)
        if task_bar is None:
            self.notify("Cannot download: task bar not found", severity="error")
            return

        task_id = task_bar.add_task(f"Downloading {model.name}", "download")
        task_bar.queue.advance()
        self.notify(f"Queued download: {model.name}")
        self._run_download(model, task_id, task_bar)

    @work(thread=True)
    def _run_download(self, model: CatalogModel, task_id: str, task_bar: object) -> None:
        """Download a model in a background thread, reporting to TaskBar."""
        import time

        from lilbee.catalog import download_model
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        bar: TaskBar = task_bar  # type: ignore[assignment]

        try:
            last_update = 0.0

            def on_progress(downloaded: int, total: int) -> None:
                nonlocal last_update
                now = time.monotonic()
                if now - last_update < 0.25:
                    return
                last_update = now
                if total > 0:
                    pct = min(int(downloaded * 100 / total), 100)
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    self.app.call_from_thread(
                        bar.update_task, task_id, pct, f"{mb_done:.0f}/{mb_total:.0f} MB"
                    )

            download_model(model, on_progress=on_progress)
            self.app.call_from_thread(bar.complete_task, task_id)
            self.app.call_from_thread(self.notify, f"{model.name} installed")
        except PermissionError:
            msg = f"{model.name} requires login \u2014 run /login or lilbee login"
            log.warning("Gated repo: %s", model.hf_repo)
            self.app.call_from_thread(bar.fail_task, task_id, msg)
            self.app.call_from_thread(self.notify, msg, severity="warning")
        except Exception:
            log.warning("Download failed for %s", model.name, exc_info=True)
            msg = f"{model.name}: download failed"
            self.app.call_from_thread(bar.fail_task, task_id, msg)
            self.app.call_from_thread(self.notify, msg, severity="error")

    def action_focus_search(self) -> None:
        self.query_one("#catalog-search", Input).focus()

    def action_pop_screen(self) -> None:
        self.app.pop_screen()

    def action_cycle_sort(self) -> None:
        if isinstance(self.focused, Input):
            return
        idx = _SORT_CYCLE.index(self._current_sort)
        self._current_sort = _SORT_CYCLE[(idx + 1) % len(_SORT_CYCLE)]
        self._refresh_lists()

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


def _filter_remote(models: list[RemoteModel], task: str | None, search: str) -> list[RemoteModel]:
    filtered = models
    if task:
        filtered = [m for m in filtered if m.task == task]
    if search:
        filtered = [m for m in filtered if search in m.name.lower()]
    return filtered


def _filter_families(
    families: list[ModelFamily], task: str | None, search: str
) -> list[ModelFamily]:
    """Filter families by task and search text, preserving only matching variants."""
    filtered: list[ModelFamily] = []
    for fam in families:
        if task and fam.task != task:
            continue
        if search:
            matches_family = search in fam.name.lower() or search in fam.description.lower()
            if matches_family:
                filtered.append(fam)
                continue
            matching = tuple(
                v
                for v in fam.variants
                if search in v.hf_repo.lower()
                or search in v.param_count.lower()
                or search in v.quant.lower()
            )
            if matching:
                filtered.append(
                    ModelFamily(
                        name=fam.name,
                        task=fam.task,
                        description=fam.description,
                        variants=matching,
                    )
                )
        else:
            filtered.append(fam)
    return filtered


def _group_by_size(models: list[CatalogModel]) -> list[tuple[str, list[CatalogModel]]]:
    """Group models by inferred parameter size."""
    groups: dict[str, list[CatalogModel]] = {}
    for m in models:
        category = _parse_param_size(m.name)
        groups.setdefault(category, []).append(m)

    order = ["Small (≤3B)", "Medium (3-8B)", "Large (8-30B)", "Extra Large (30B+)", "unknown"]
    return [(label, groups[label]) for label in order if label in groups]
