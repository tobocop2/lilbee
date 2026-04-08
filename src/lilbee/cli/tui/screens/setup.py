"""First-run setup — single-screen model picker with RAM-based recommendations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Label, ProgressBar, Static

from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING, CatalogModel
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.grid_select import GridSelect
from lilbee.cli.tui.widgets.model_card import ModelCard
from lilbee.config import cfg

if TYPE_CHECKING:
    from lilbee.cli.tui.screens.catalog import TableRow

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
            if m.task == "embedding":
                embed.append(name)
            elif m.task == "chat":
                chat.append(name)
        return sorted(chat), sorted(embed)
    except Exception:
        return [], []


def _installed_name_to_row(name: str, task: str) -> TableRow:
    """Create a minimal TableRow for an already-installed model."""
    from lilbee.cli.tui.screens.catalog import TableRow, _parse_param_label

    return TableRow(
        name=name,
        task=task,
        params=_parse_param_label(name),
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=True,
        sort_downloads=0,
        sort_size=0.0,
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


def _format_download_size(size_gb: float) -> str:
    """Format a download size in GB to a human-readable string."""
    if size_gb < 1.0:
        return f"{size_gb * 1024:.0f} MB"
    return f"{size_gb:.1f} GB"


class SetupWizard(Screen[str | None]):
    """First-run setup — browsable single-screen model picker."""

    CSS_PATH = "setup.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._selected_chat: str | None = None
        self._selected_embed: str | None = None
        self._selected_chat_card: ModelCard | None = None
        self._selected_embed_card: ModelCard | None = None
        self._chat_installed, self._embed_installed = _scan_installed_models()
        self._recommended_chat: CatalogModel | None = None
        self._recommended_embed: CatalogModel | None = None
        self._download_models: list[CatalogModel] = []

    def compose(self) -> ComposeResult:
        yield Static(msg.SETUP_WELCOME, id="setup-title")
        yield Static(msg.SETUP_SUBTITLE, id="setup-subtitle")
        yield VerticalScroll(id="setup-grid-container")
        with Horizontal(id="setup-footer"):
            yield Label(msg.SETUP_SLOT_EMPTY, id="setup-chat-slot")
            yield Label(msg.SETUP_SLOT_EMPTY, id="setup-embed-slot")
            yield Label("", id="setup-download-size")
        yield Button(msg.SETUP_INSTALL_BUTTON, id="setup-action", disabled=True)
        yield Button(msg.SETUP_SKIP_BUTTON, id="setup-skip", variant="default")
        yield Label("", id="setup-status")
        yield ProgressBar(total=100, show_eta=False, id="setup-progress")

    def on_mount(self) -> None:
        self._build_grid()

    def _build_grid(self) -> None:
        """Build all model sections and pre-select recommended combo."""
        from lilbee.cli.tui.screens.catalog import _catalog_to_row
        from lilbee.models import get_system_ram_gb

        ram_gb = get_system_ram_gb()
        rec_chat, rec_embed = _pick_recommended(ram_gb)
        self._recommended_chat = rec_chat
        self._recommended_embed = rec_embed

        container = self.query_one("#setup-grid-container", VerticalScroll)
        widgets_to_mount: list[Static | GridSelect] = []

        # Installed section
        if self._chat_installed or self._embed_installed:
            widgets_to_mount.append(Static(msg.HEADING_INSTALLED, classes="section-heading"))
            installed_cards: list[ModelCard] = []
            for name in self._chat_installed:
                installed_cards.append(ModelCard(_installed_name_to_row(name, "chat")))
            for name in self._embed_installed:
                installed_cards.append(ModelCard(_installed_name_to_row(name, "embedding")))
            widgets_to_mount.append(
                GridSelect(*installed_cards, min_column_width=30, max_column_width=50)
            )

        # Chat models section
        widgets_to_mount.append(Static("Chat Models", classes="section-heading"))
        chat_cards: list[ModelCard] = []
        for m in FEATURED_CHAT:
            row = _catalog_to_row(m, installed=False)
            card = ModelCard(row)
            chat_cards.append(card)
        widgets_to_mount.append(GridSelect(*chat_cards, min_column_width=30, max_column_width=50))

        # Embedding models section
        widgets_to_mount.append(Static("Embedding Models", classes="section-heading"))
        embed_cards: list[ModelCard] = []
        for m in FEATURED_EMBEDDING:
            row = _catalog_to_row(m, installed=False)
            card = ModelCard(row)
            embed_cards.append(card)
        widgets_to_mount.append(GridSelect(*embed_cards, min_column_width=30, max_column_width=50))

        container.mount_all(widgets_to_mount)

        # Pre-select recommended models (cards already exist, just set state)
        self._preselect_recommended(chat_cards, embed_cards)

    def _preselect_recommended(
        self, chat_cards: list[ModelCard], embed_cards: list[ModelCard]
    ) -> None:
        """Pre-select the RAM-appropriate recommended models."""
        for card in chat_cards:
            if (
                card.row.catalog_model
                and self._recommended_chat
                and card.row.catalog_model.name == self._recommended_chat.name
            ):
                self._select_card(card, "chat")
                break
        for card in embed_cards:
            if (
                card.row.catalog_model
                and self._recommended_embed
                and card.row.catalog_model.name == self._recommended_embed.name
            ):
                self._select_card(card, "embedding")
                break

    def _select_card(self, card: ModelCard, task: str) -> None:
        """Select a card, deselecting the previous selection for that task."""
        if task == "chat":
            if self._selected_chat_card is not None:
                self._selected_chat_card.selected = False
            self._selected_chat_card = card
            self._selected_chat = card.row.name
            card.selected = True
        else:
            if self._selected_embed_card is not None:
                self._selected_embed_card.selected = False
            self._selected_embed_card = card
            self._selected_embed = card.row.name
            card.selected = True
        self._update_footer()

    def _update_footer(self) -> None:
        """Update footer slots, download size, and button states."""
        chat_slot = self.query_one("#setup-chat-slot", Label)
        embed_slot = self.query_one("#setup-embed-slot", Label)
        size_label = self.query_one("#setup-download-size", Label)
        action_btn = self.query_one("#setup-action", Button)
        skip_btn = self.query_one("#setup-skip", Button)

        if self._selected_chat:
            chat_slot.update(msg.SETUP_CHAT_SLOT.format(name=self._selected_chat))
        else:
            chat_slot.update(msg.SETUP_SLOT_EMPTY)

        if self._selected_embed:
            embed_slot.update(msg.SETUP_EMBED_SLOT.format(name=self._selected_embed))
        else:
            embed_slot.update(msg.SETUP_SLOT_EMPTY)

        # Calculate total download size from catalog models
        total_gb = 0.0
        if self._selected_chat_card and not self._selected_chat_card.row.installed:
            cm = self._selected_chat_card.row.catalog_model
            if cm:
                total_gb += cm.size_gb
        if self._selected_embed_card and not self._selected_embed_card.row.installed:
            cm = self._selected_embed_card.row.catalog_model
            if cm:
                total_gb += cm.size_gb

        if total_gb > 0:
            size_label.update(msg.SETUP_TOTAL_DOWNLOAD.format(size=_format_download_size(total_gb)))
        else:
            size_label.update("")

        both_selected = self._selected_chat is not None and self._selected_embed is not None
        action_btn.disabled = not both_selected

        if both_selected:
            action_btn.label = msg.SETUP_INSTALL_BUTTON
            skip_btn.display = False
        elif self._selected_chat:
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
        if task in ("chat", "embedding"):
            self._select_card(card, task)

    @on(Button.Pressed, "#setup-action")
    def _on_install(self) -> None:
        """Start downloading selected models."""
        self._download_models = []
        if self._selected_chat_card and not self._selected_chat_card.row.installed:
            cm = self._selected_chat_card.row.catalog_model
            if cm:
                self._download_models.append(cm)
        if self._selected_embed_card and not self._selected_embed_card.row.installed:
            cm = self._selected_embed_card.row.catalog_model
            if cm:
                self._download_models.append(cm)

        if not self._download_models:
            # Both already installed, just finish
            self._finish()
            return

        self.add_class("-downloading")
        self.query_one("#setup-action", Button).disabled = True
        self._run_downloads()

    @work(thread=True)
    def _run_downloads(self) -> None:
        """Download all selected models sequentially."""
        from lilbee.catalog import download_model

        total = len(self._download_models)
        for idx, model in enumerate(self._download_models, 1):
            self.app.call_from_thread(
                self._set_status,
                msg.SETUP_DOWNLOADING_N.format(name=model.name, current=idx, total=total),
            )
            self.app.call_from_thread(self._update_progress, 0)

            last_pct = -1

            def _on_progress(downloaded: int, total_bytes: int) -> None:
                nonlocal last_pct
                if total_bytes > 0:
                    pct = min(int(downloaded * 100 / total_bytes), 100)
                    if pct != last_pct:
                        last_pct = pct
                        self.app.call_from_thread(self._update_progress, pct)

            try:
                download_model(model, on_progress=_on_progress)
            except Exception as exc:
                log.warning("Download failed for %s", model.name, exc_info=True)
                error_msg = str(exc)
                if "401" in error_msg or "PermissionError" in error_msg:
                    error_msg = msg.SETUP_LOGIN_REQUIRED.format(name=model.name)
                self.app.call_from_thread(self._set_status, f"Error: {error_msg}")
                if idx == 1 and total > 1:
                    # Chat failed — don't continue to embedding
                    return
                if idx == 2:
                    # Embedding failed but chat succeeded — partial success
                    self.app.call_from_thread(self._on_partial_success)
                    return
                return

        self.app.call_from_thread(self._on_all_downloads_complete)

    def _on_all_downloads_complete(self) -> None:
        self._set_status(msg.SETUP_ALL_DONE)
        self._update_progress(100)
        self._finish()

    def _on_partial_success(self) -> None:
        """Chat downloaded but embedding failed."""
        self._set_status(msg.SETUP_PARTIAL_FAIL)
        self._selected_embed = None
        self._finish()

    def _set_status(self, text: str) -> None:
        self.query_one("#setup-status", Label).update(text)

    def _update_progress(self, percent: int) -> None:
        self.query_one("#setup-progress", ProgressBar).update(total=100, progress=percent)

    def _finish(self) -> None:
        from lilbee import settings
        from lilbee.services import reset_services

        if self._selected_chat:
            cfg.chat_model = self._selected_chat
            settings.set_value(cfg.data_root, "chat_model", self._selected_chat)
        if self._selected_embed:
            cfg.embedding_model = self._selected_embed
            settings.set_value(
                cfg.data_root,
                "embedding_model",
                self._selected_embed,
            )
        reset_services()
        self.dismiss("completed")

    @on(Button.Pressed, "#setup-skip")
    def _on_skip(self) -> None:
        """Skip setup — save any partial selection."""
        if self._selected_chat:
            from lilbee import settings
            from lilbee.services import reset_services

            cfg.chat_model = self._selected_chat
            settings.set_value(cfg.data_root, "chat_model", self._selected_chat)
            reset_services()
        self.dismiss("skipped")

    def action_cancel(self) -> None:
        self.dismiss("skipped")
