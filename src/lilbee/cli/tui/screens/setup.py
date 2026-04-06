"""First-run setup wizard."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
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

_STEP_CHAT = 1
_STEP_EMBED = 2


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


class SetupWizard(Screen[str | None]):
    """First-run setup -- helps user configure chat and embedding models."""

    CSS_PATH = "setup.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._step = _STEP_CHAT
        self._selected_chat: str | None = None
        self._selected_embed: str | None = None
        self._chat_installed, self._embed_installed = _scan_installed_models()

    def compose(self) -> ComposeResult:
        yield Static(msg.SETUP_TITLE, id="setup-title")
        yield Label("", id="setup-step-label")
        yield VerticalScroll(id="setup-grid-container")
        yield Label("", id="setup-status")
        yield ProgressBar(total=100, show_eta=False, id="setup-progress")
        yield Button(msg.SETUP_BROWSE_CATALOG, id="setup-browse")
        yield Button(msg.SETUP_SKIP_BUTTON, id="setup-skip")

    def on_mount(self) -> None:
        self._show_step()

    def _show_step(self) -> None:
        container = self.query_one("#setup-grid-container", VerticalScroll)
        container.remove_children()
        label = self.query_one("#setup-step-label", Label)
        self.query_one("#setup-progress", ProgressBar).update(total=100, progress=0)
        self.query_one("#setup-status", Label).update("")

        if self._step == _STEP_CHAT:
            label.update(msg.SETUP_STEP_CHAT)
            installed_names = self._chat_installed
            featured_models = FEATURED_CHAT
            task = "chat"
        else:
            label.update(msg.SETUP_STEP_EMBED)
            installed_names = self._embed_installed
            featured_models = FEATURED_EMBEDDING
            task = "embedding"

        from lilbee.cli.tui.screens.catalog import _catalog_to_row

        widgets_to_mount: list[Static | GridSelect] = []

        if installed_names:
            installed_rows = [_installed_name_to_row(n, task) for n in installed_names]
            widgets_to_mount.append(
                Static(
                    msg.HEADING_INSTALLED,
                    classes="section-heading",
                )
            )
            cards = [ModelCard(row) for row in installed_rows]
            widgets_to_mount.append(
                GridSelect(
                    *cards,
                    min_column_width=30,
                    max_column_width=50,
                )
            )

        featured_rows = [_catalog_to_row(m, installed=False) for m in featured_models]
        widgets_to_mount.append(
            Static(
                msg.HEADING_OUR_PICKS,
                classes="section-heading",
            )
        )
        cards = [ModelCard(row) for row in featured_rows]
        widgets_to_mount.append(
            GridSelect(
                *cards,
                min_column_width=30,
                max_column_width=50,
            )
        )

        container.mount_all(widgets_to_mount)

    @on(GridSelect.Selected)
    def _on_grid_selected(self, event: GridSelect.Selected) -> None:
        if isinstance(event.widget, ModelCard):
            row = event.widget.row
            if row.installed:
                self._on_model_chosen(row.name)
            elif row.catalog_model:
                self._download_model(row.catalog_model)

    @on(Button.Pressed, "#setup-browse")
    def _on_browse_catalog(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        if isinstance(self.app, LilbeeApp):
            self.dismiss("skipped")
            self.app.call_later(lambda: self.app.switch_view("Catalog"))

    def _on_model_chosen(self, name: str) -> None:
        if self._step == _STEP_CHAT:
            self._selected_chat = name
            self._step = _STEP_EMBED
            self._show_step()
        else:
            self._selected_embed = name
            self._finish()

    @work(thread=True)
    def _download_model(self, model: CatalogModel) -> None:
        """Download via catalog API with TUI-native progress (no Rich)."""
        self.app.call_from_thread(self._set_status, msg.SETUP_CONNECTING)
        try:
            from lilbee.catalog import download_model

            last_update_time = 0.0
            last_pct = -1

            def _on_progress(downloaded: int, total: int) -> None:
                nonlocal last_update_time, last_pct
                import time

                current_time = time.time()
                if current_time - last_update_time < 0.1:
                    return
                try:
                    mb_done = downloaded / (1024 * 1024)
                    if total > 0:
                        pct = min(int(downloaded * 100 / total), 100)
                        mb_total = total / (1024 * 1024)
                        last_update_time = current_time
                        if pct == last_pct and pct > 0:
                            return
                        last_pct = pct
                        self.app.call_from_thread(self._update_progress, pct)
                        self.app.call_from_thread(
                            self._set_status,
                            f"Downloading... {mb_done:.0f} / {mb_total:.0f} MB ({pct}%)",
                        )
                    else:
                        self.app.call_from_thread(
                            self._set_status,
                            f"Downloading... {mb_done:.0f} MB",
                        )
                except Exception:
                    log.debug(
                        "Progress callback failed",
                        exc_info=True,
                    )

            dest = download_model(model, on_progress=_on_progress)
            self.app.call_from_thread(self._on_download_complete, dest.stem)
        except Exception as exc:
            log.warning(
                "Download failed for %s",
                model.name,
                exc_info=True,
            )
            error_msg = str(exc)
            if "401" in error_msg or "PermissionError" in error_msg:
                error_msg = msg.SETUP_LOGIN_REQUIRED.format(name=model.name)
            self.app.call_from_thread(self._set_status, f"Error: {error_msg}")

    def _set_status(self, text: str) -> None:
        self.query_one("#setup-status", Label).update(text)

    def _update_progress(self, percent: int) -> None:
        self.query_one("#setup-progress", ProgressBar).update(total=100, progress=percent)

    def _on_download_complete(self, name: str) -> None:
        self.query_one("#setup-progress", ProgressBar).update(total=100, progress=100)
        self._set_status(msg.SETUP_INSTALLED_STATUS.format(name=name))
        self._on_model_chosen(name)

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
        self.dismiss("skipped")

    def action_cancel(self) -> None:
        self.dismiss("skipped")
