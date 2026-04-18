"""First-run setup — single-screen model picker with RAM-based recommendations.

The wizard mirrors the catalog's grid aesthetic: one ``GridSelect`` per
section (chat, embed), pressing Enter on a card installs that model
immediately via ``TaskBarController.start_download``. No separate
Install & Go button, no Browse, no Skip — pick what you want, press
Esc when done. Downloads continue under the app-level controller, so
dismissing the wizard while they're in flight is fine.
"""

from __future__ import annotations

import contextlib
import logging
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Label, Static

from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING, CatalogModel
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

SETUP_CHAT_GRID_ID = "setup-chat-grid"


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
    """Create a minimal TableRow for an already-installed model."""
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
    """Pick chat + embedding models appropriate for system RAM."""
    eligible = [m for m in FEATURED_CHAT if m.min_ram_gb <= ram_gb]
    chat = max(eligible, key=lambda m: m.size_gb) if eligible else FEATURED_CHAT[0]
    embed = FEATURED_EMBEDDING[0]
    return chat, embed


def _card_download_size(card: ModelCard | None) -> float:
    """Return download size in GB for a non-installed card, or 0."""
    if card and not card.row.installed:
        cm = card.row.catalog_model
        if cm:
            return cm.size_gb
    return 0.0


def _pending_download(card: ModelCard | None) -> CatalogModel | None:
    """Return the CatalogModel to download for a non-installed card, or None."""
    if card and not card.row.installed:
        return card.row.catalog_model
    return None


class SetupWizard(Screen[str | None]):
    """First-run setup — pick chat + embedding, Enter installs, Esc exits.

    Each card you press Enter on:
      1. Becomes the saved selection for its task (chat or embedding).
      2. Triggers a download via the app's ``TaskBarController`` unless
         the card is already installed.
      3. Leaves the wizard open so you can pick the other task next.

    Selections are persisted to settings eagerly (not at dismiss time),
    so Esc-ing out mid-wizard keeps your picks.
    """

    CSS_PATH = "setup.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Done", show=True),
        Binding("tab", "app.focus_next", "Next", show=False),
        Binding("shift+tab", "app.focus_previous", "Prev", show=False),
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
        # Model refs already submitted to the controller (avoid duplicate
        # start_download calls when a card is re-selected by arrow + Enter).
        self._submitted: set[str] = set()

    @property
    def _selected_chat(self) -> str | None:
        return self._selections[ModelTask.CHAT][0]

    @property
    def _selected_embed(self) -> str | None:
        return self._selections[ModelTask.EMBEDDING][0]

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        yield Static(msg.SETUP_WELCOME, id="setup-title")
        yield Static(msg.SETUP_SUBTITLE, id="setup-subtitle")
        yield VerticalScroll(id="setup-grid-container")
        with Horizontal(id="setup-footer"):
            yield Label(msg.SETUP_SLOT_EMPTY, id="setup-chat-slot")
            yield Label(msg.SETUP_SLOT_EMPTY, id="setup-embed-slot")
            yield Label("", id="setup-download-size")
            yield Label(msg.SETUP_ENTER_HINT, id="setup-enter-hint")
        yield TaskBar()

    def on_mount(self) -> None:
        self._build_grid()
        # Focus the chat-model grid so arrow keys / Enter work without a mouse.
        with contextlib.suppress(Exception):
            self.query_one(f"#{SETUP_CHAT_GRID_ID}", GridSelect).focus()

    def _build_section(
        self,
        heading: str,
        models: tuple[CatalogModel, ...],
        installed_refs: set[str],
        widgets_out: list[Static | GridSelect],
        grid_id: str | None = None,
    ) -> list[ModelCard]:
        """Build a heading + GridSelect for a list of catalog models."""
        widgets_out.append(Static(heading, classes="section-heading"))
        cards = [ModelCard(catalog_to_row(m, installed=m.ref in installed_refs)) for m in models]
        widgets_out.append(GridSelect(*cards, min_column_width=30, max_column_width=50, id=grid_id))
        return cards

    def _build_grid(self) -> None:
        """Build all model sections and pre-select recommended combo."""
        ram_gb = get_system_ram_gb()
        rec_chat, rec_embed = _pick_recommended(ram_gb)
        self._recommended_chat = rec_chat
        self._recommended_embed = rec_embed

        container = self.query_one("#setup-grid-container", VerticalScroll)
        widgets_to_mount: list[Static | GridSelect] = []
        installed_refs = set(self._chat_installed) | set(self._embed_installed)

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

        chat_cards = self._build_section(
            msg.SETUP_HEADING_CHAT,
            FEATURED_CHAT,
            installed_refs,
            widgets_to_mount,
            grid_id=SETUP_CHAT_GRID_ID,
        )
        embed_cards = self._build_section(
            msg.SETUP_HEADING_EMBED, FEATURED_EMBEDDING, installed_refs, widgets_to_mount
        )

        container.mount_all(widgets_to_mount)
        self._preselect_recommended(chat_cards, embed_cards)

    def _preselect_recommended(
        self, chat_cards: list[ModelCard], embed_cards: list[ModelCard]
    ) -> None:
        """Pre-select the RAM-appropriate recommended models (without installing)."""
        for cards, recommended in [
            (chat_cards, self._recommended_chat),
            (embed_cards, self._recommended_embed),
        ]:
            if not recommended:
                continue
            for card in cards:
                cm = card.row.catalog_model
                if cm and cm.ref == recommended.ref:
                    self._mark_selection(card, card.row.task)
                    break
        self._update_footer()

    def _mark_selection(self, card: ModelCard, task: str) -> None:
        """Record a selection and repaint its card. No download yet."""
        _ref, prev_card = self._selections[task]
        if prev_card is not None and prev_card is not card:
            prev_card.selected = False
        ref = card.row.ref or card.row.name
        card.selected = True
        self._selections[task] = (ref, card)

    def _commit_selection(self, card: ModelCard, task: str) -> None:
        """Persist the selection to settings and submit a download if pending.

        Called when the user presses Enter on a card. Saves the config
        fragment eagerly so Esc mid-wizard doesn't lose the pick.
        """
        from lilbee import settings
        from lilbee.cli.tui.app import LilbeeApp

        self._mark_selection(card, task)
        ref = self._selections[task][0]
        if ref is None:
            return
        # Write the fragment; embedding never overrides chat and vice versa.
        if task == ModelTask.CHAT:
            cfg.chat_model = ref
            settings.set_value(cfg.data_root, "chat_model", ref)
        elif task == ModelTask.EMBEDDING:
            cfg.embedding_model = ref
            settings.set_value(cfg.data_root, "embedding_model", ref)

        pending = _pending_download(card)
        if (
            pending is not None
            and pending.ref not in self._submitted
            and isinstance(self.app, LilbeeApp)
        ):
            self._submitted.add(pending.ref)
            self.app.task_bar.start_download(pending)
        self._update_footer()

    def _update_footer(self) -> None:
        """Update footer slots and download size strip."""
        chat_slot = self.query_one("#setup-chat-slot", Label)
        embed_slot = self.query_one("#setup-embed-slot", Label)
        size_label = self.query_one("#setup-download-size", Label)

        chat_ref, chat_card = self._selections[ModelTask.CHAT]
        embed_ref, embed_card = self._selections[ModelTask.EMBEDDING]

        chat_slot.update(
            msg.SETUP_CHAT_SLOT.format(name=chat_ref) if chat_ref else msg.SETUP_SLOT_EMPTY
        )
        embed_slot.update(
            msg.SETUP_EMBED_SLOT.format(name=embed_ref) if embed_ref else msg.SETUP_SLOT_EMPTY
        )

        total_gb = _card_download_size(chat_card) + _card_download_size(embed_card)
        size_label.update(
            msg.SETUP_TOTAL_DOWNLOAD.format(size=format_size_gb(total_gb)) if total_gb > 0 else ""
        )

    @on(GridSelect.Selected)
    def _on_grid_selected(self, event: GridSelect.Selected) -> None:
        """Enter on a card installs it (or records selection if already installed)."""
        if not isinstance(event.widget, ModelCard):
            return
        card = event.widget
        task = card.row.task
        if task in self._selections:
            self._commit_selection(card, task)

    @on(GridSelect.LeaveDown)
    def _on_grid_leave_down(self, event: GridSelect.LeaveDown) -> None:
        """Arrow-down past the last card walks to the next focusable widget."""
        self.focus_next()

    @on(GridSelect.LeaveUp)
    def _on_grid_leave_up(self, event: GridSelect.LeaveUp) -> None:
        """Arrow-up past the first card walks to the previous focusable widget."""
        self.focus_previous()

    def action_cancel(self) -> None:
        """Escape dismisses the wizard; any submitted downloads keep running.

        Selections are saved eagerly in ``_commit_selection``; we reset
        services here so the next screen pulls the updated config.
        """
        if self._selected_chat or self._selected_embed:
            reset_services()
            self.dismiss("completed")
        else:
            self.dismiss("skipped")
