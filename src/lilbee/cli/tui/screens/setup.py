"""First-run setup — single-screen model picker with RAM-based recommendations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ClassVar, NamedTuple

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalGroup, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Label, ProgressBar, Static

from lilbee.catalog import (
    FEATURED_CHAT,
    FEATURED_EMBEDDING,
    CatalogModel,
    DownloadProgress,
    download_model,
    make_download_callback,
)
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.catalog_utils import (
    TableRow,
    catalog_to_row,
    format_size_gb,
    parse_param_label,
)
from lilbee.cli.tui.widgets.grid_select import GridSelect
from lilbee.cli.tui.widgets.model_card import ModelCard
from lilbee.config import cfg
from lilbee.models import ModelTask, get_system_ram_gb
from lilbee.services import reset_services

log = logging.getLogger(__name__)


def _scan_installed_models() -> tuple[list[str], list[str]]:
    """List installed models from the registry, split into chat vs embedding."""
    try:
        from lilbee.registry import ModelRegistry

        registry = ModelRegistry(cfg.models_dir)
        chat: list[str] = []
        embed: list[str] = []
        for m in registry.list_installed():
            name = f"{m.name}:{m.tag}"
            if m.task == ModelTask.EMBEDDING:
                embed.append(name)
            elif m.task == ModelTask.CHAT:
                chat.append(name)
        return sorted(chat), sorted(embed)
    except Exception:
        return [], []


def _installed_name_to_row(name: str, task: str) -> TableRow:
    """Create a minimal TableRow for an already-installed model.
    ``name`` is a ref string like ``qwen3:0.6b``.  We store it in ``ref``
    (for config persistence) and also in ``name`` (for display) since we
    don't have a richer display label for already-installed models.
    """
    return TableRow(
        name=name,
        task=task,
        params=parse_param_label(name),
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=True,
        sort_downloads=0,
        sort_size=0.0,
        ref=name,
    )


def _pick_recommended(ram_gb: float) -> tuple[CatalogModel, CatalogModel]:
    """Pick chat + embedding models appropriate for system RAM.
    Selects the largest featured chat model whose min_ram_gb fits,
    and always picks the first embedding model (Nomic).
    """
    chat = FEATURED_CHAT[0]
    for m in reversed(FEATURED_CHAT):
        if m.min_ram_gb <= ram_gb:
            chat = m
            break
    embed = FEATURED_EMBEDDING[0]
    return chat, embed


def _card_download_size(card: ModelCard | None) -> float:
    """Return download size in GB for a non-installed card, or 0."""
    if card and not card.row.installed:
        cm = card.row.catalog_model
        if cm:
            return cm.size_gb
    return 0.0


class _DownloadRow(NamedTuple):
    """Per-model progress row: display name + the widgets that render its state."""

    display_name: str
    label: Label
    bar: ProgressBar


def _pending_download(card: ModelCard | None) -> CatalogModel | None:
    """Return the CatalogModel to download for a non-installed card, or None."""
    if card and not card.row.installed:
        return card.row.catalog_model
    return None


class SetupWizard(Screen[str | None]):
    """First-run setup — browsable single-screen model picker."""

    CSS_PATH = "setup.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._selections: dict[str, tuple[str | None, ModelCard | None]] = {
            ModelTask.CHAT: (None, None),
            ModelTask.EMBEDDING: (None, None),
        }
        self._chat_installed, self._embed_installed = _scan_installed_models()
        self._recommended_chat: CatalogModel | None = None
        self._recommended_embed: CatalogModel | None = None
        self._download_models: list[CatalogModel] = []
        self._download_rows: dict[str, _DownloadRow] = {}

    @property
    def _selected_chat(self) -> str | None:
        return self._selections[ModelTask.CHAT][0]

    @property
    def _selected_embed(self) -> str | None:
        return self._selections[ModelTask.EMBEDDING][0]

    def compose(self) -> ComposeResult:
        yield Static(msg.SETUP_WELCOME, id="setup-title")
        yield Static(msg.SETUP_SUBTITLE, id="setup-subtitle")
        yield VerticalScroll(id="setup-grid-container")
        with Horizontal(id="setup-footer"):
            yield Label(msg.SETUP_SLOT_EMPTY, id="setup-chat-slot")
            yield Label(msg.SETUP_SLOT_EMPTY, id="setup-embed-slot")
            yield Label("", id="setup-download-size")
        yield Button(msg.SETUP_INSTALL_BUTTON, id="setup-action", disabled=True)
        yield Button(msg.SETUP_BROWSE_CATALOG, id="setup-browse", variant="default")
        yield Button(msg.SETUP_SKIP_BUTTON, id="setup-skip", variant="default")
        yield VerticalGroup(id="setup-downloads")

    def on_mount(self) -> None:
        self._build_grid()

    def _build_section(
        self,
        heading: str,
        models: tuple[CatalogModel, ...],
        widgets_out: list[Static | GridSelect],
    ) -> list[ModelCard]:
        """Build a heading + GridSelect for a list of catalog models."""
        widgets_out.append(Static(heading, classes="section-heading"))
        cards = [ModelCard(catalog_to_row(m, installed=False)) for m in models]
        widgets_out.append(GridSelect(*cards, min_column_width=30, max_column_width=50))
        return cards

    def _build_grid(self) -> None:
        """Build all model sections and pre-select recommended combo."""
        ram_gb = get_system_ram_gb()
        rec_chat, rec_embed = _pick_recommended(ram_gb)
        self._recommended_chat = rec_chat
        self._recommended_embed = rec_embed

        container = self.query_one("#setup-grid-container", VerticalScroll)
        widgets_to_mount: list[Static | GridSelect] = []

        if self._chat_installed or self._embed_installed:
            widgets_to_mount.append(Static(msg.HEADING_INSTALLED, classes="section-heading"))
            installed_cards = [
                ModelCard(_installed_name_to_row(n, ModelTask.CHAT)) for n in self._chat_installed
            ] + [
                ModelCard(_installed_name_to_row(n, ModelTask.EMBEDDING))
                for n in self._embed_installed
            ]
            widgets_to_mount.append(
                GridSelect(*installed_cards, min_column_width=30, max_column_width=50)
            )

        chat_cards = self._build_section(msg.SETUP_HEADING_CHAT, FEATURED_CHAT, widgets_to_mount)
        embed_cards = self._build_section(
            msg.SETUP_HEADING_EMBED, FEATURED_EMBEDDING, widgets_to_mount
        )

        container.mount_all(widgets_to_mount)
        self._preselect_recommended(chat_cards, embed_cards)

    def _preselect_recommended(
        self, chat_cards: list[ModelCard], embed_cards: list[ModelCard]
    ) -> None:
        """Pre-select the RAM-appropriate recommended models."""
        for cards, recommended in [
            (chat_cards, self._recommended_chat),
            (embed_cards, self._recommended_embed),
        ]:
            if not recommended:
                continue
            for card in cards:
                cm = card.row.catalog_model
                if cm and cm.ref == recommended.ref:
                    self._select_card(card, card.row.task)
                    break

    def _select_card(self, card: ModelCard, task: str) -> None:
        """Select a card, deselecting the previous selection for that task."""
        _ref, prev_card = self._selections[task]
        if prev_card is not None:
            prev_card.selected = False
        ref = card.row.ref or card.row.name
        card.selected = True
        self._selections[task] = (ref, card)
        self._update_footer()

    def _update_footer(self) -> None:
        """Update footer slots, download size, and button states."""
        chat_slot = self.query_one("#setup-chat-slot", Label)
        embed_slot = self.query_one("#setup-embed-slot", Label)
        size_label = self.query_one("#setup-download-size", Label)
        action_btn = self.query_one("#setup-action", Button)
        skip_btn = self.query_one("#setup-skip", Button)

        chat_ref, chat_card = self._selections[ModelTask.CHAT]
        embed_ref, embed_card = self._selections[ModelTask.EMBEDDING]

        chat_slot.update(
            msg.SETUP_CHAT_SLOT.format(name=chat_ref) if chat_ref else msg.SETUP_SLOT_EMPTY
        )
        embed_slot.update(
            msg.SETUP_EMBED_SLOT.format(name=embed_ref) if embed_ref else msg.SETUP_SLOT_EMPTY
        )

        total_gb = _card_download_size(chat_card) + _card_download_size(embed_card)
        if total_gb > 0:
            size_label.update(msg.SETUP_TOTAL_DOWNLOAD.format(size=format_size_gb(total_gb)))
        else:
            size_label.update("")

        both_selected = chat_ref is not None and embed_ref is not None
        action_btn.disabled = not both_selected

        if both_selected:
            action_btn.label = msg.SETUP_INSTALL_BUTTON
            skip_btn.display = False
        elif chat_ref:
            skip_btn.label = msg.SETUP_CONTINUE_NO_SEARCH
            skip_btn.display = True
        else:
            skip_btn.label = msg.SETUP_SKIP_BUTTON
            skip_btn.display = True

    @on(GridSelect.Selected)
    def _on_grid_selected(self, event: GridSelect.Selected) -> None:
        if not isinstance(event.widget, ModelCard):
            return
        card = event.widget
        task = card.row.task
        if task in self._selections:
            self._select_card(card, task)

    @on(Button.Pressed, "#setup-browse")
    def _on_browse_catalog(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.dismiss("skipped")
            self.app.call_later(lambda: self.app.switch_view("Catalog"))

    @on(Button.Pressed, "#setup-action")
    def _on_install(self) -> None:
        """Start downloading selected models."""
        self._download_models = [
            cm
            for task in (ModelTask.CHAT, ModelTask.EMBEDDING)
            if (cm := _pending_download(self._selections[task][1]))
        ]

        if not self._download_models:
            self._save_and_dismiss("completed")
            return

        self.add_class("-downloading")
        self.query_one("#setup-action", Button).disabled = True
        self._run_downloads()

    def _mount_download_rows(self) -> None:
        """Mount one label+progress row per pending download before starting."""
        container = self.query_one("#setup-downloads", VerticalGroup)
        container.remove_children()
        self._download_rows = {}
        for model in self._download_models:
            label = Label(
                msg.SETUP_DOWNLOAD_WAITING.format(name=model.display_name),
                classes="download-label",
            )
            bar = ProgressBar(total=100, show_eta=False, classes="download-bar")
            container.mount(label, bar)
            self._download_rows[model.ref] = _DownloadRow(model.display_name, label, bar)

    @work(thread=True)
    def _run_downloads(self) -> None:
        """Download all selected models in a background thread."""
        self._download_loop(self.app.call_from_thread)  # pragma: no cover

    def _on_download_progress(
        self, notify: Callable[..., None], model_ref: str, p: DownloadProgress
    ) -> None:
        """Handle a single download progress update for the given model."""
        notify(self._update_row, model_ref, p.percent, p.detail)

    def _handle_download_error(
        self,
        notify: Callable[..., None],
        exc: Exception,
        model: CatalogModel,
        *,
        is_first: bool,
    ) -> None:
        """Mark the row as failed and, if the chat model succeeded first, save partial state."""
        log.warning("Download failed for %s", model.ref, exc_info=True)
        error_msg = str(exc)
        if "401" in error_msg or "PermissionError" in error_msg:
            error_msg = msg.SETUP_LOGIN_REQUIRED.format(name=model.display_name)
        notify(self._mark_row_failed, model.ref, error_msg)
        if not is_first:
            notify(self._on_partial_success)

    def _download_loop(self, notify: Callable[..., None]) -> None:
        """Download all selected models sequentially."""
        notify(self._mount_download_rows)
        for idx, model in enumerate(self._download_models, 1):
            is_first = idx == 1
            notify(self._update_row, model.ref, 0, msg.SETUP_DOWNLOAD_STARTING)

            def _progress(p: DownloadProgress, ref: str = model.ref) -> None:
                self._on_download_progress(notify, ref, p)

            callback = make_download_callback(_progress)
            try:
                download_model(model, on_progress=callback)
            except Exception as exc:
                self._handle_download_error(notify, exc, model, is_first=is_first)
                return
            notify(self._mark_row_done, model.ref)

        notify(self._on_all_downloads_complete)

    def _on_all_downloads_complete(self) -> None:
        self._save_and_dismiss("completed")

    def _on_partial_success(self) -> None:
        """Chat downloaded but embedding failed."""
        self._selections[ModelTask.EMBEDDING] = (None, None)
        self._save_and_dismiss("completed")

    def _update_row(self, model_ref: str, percent: int, detail: str) -> None:
        row = self._download_rows[model_ref]
        row.label.update(
            msg.SETUP_DOWNLOAD_ACTIVE.format(name=row.display_name, detail=detail, percent=percent)
        )
        row.bar.update(total=100, progress=percent)

    def _mark_row_done(self, model_ref: str) -> None:
        row = self._download_rows[model_ref]
        row.label.update(msg.SETUP_DOWNLOAD_DONE.format(name=row.display_name))
        row.bar.update(total=100, progress=100)

    def _mark_row_failed(self, model_ref: str, error: str) -> None:
        row = self._download_rows[model_ref]
        row.label.update(msg.SETUP_DOWNLOAD_FAILED.format(name=row.display_name, error=error))

    def _save_and_dismiss(self, result: str) -> None:
        """Persist selected models to config and dismiss."""
        from lilbee import settings

        chat_ref = self._selected_chat
        embed_ref = self._selected_embed
        if chat_ref:
            cfg.chat_model = chat_ref
            settings.set_value(cfg.data_root, "chat_model", chat_ref)
        if embed_ref:
            cfg.embedding_model = embed_ref
            settings.set_value(cfg.data_root, "embedding_model", embed_ref)
        reset_services()
        self.dismiss(result)

    @on(Button.Pressed, "#setup-skip")
    def _on_skip(self) -> None:
        """Skip setup — save any partial selection."""
        self._save_and_dismiss("skipped")

    def action_cancel(self) -> None:
        self.dismiss("skipped")
