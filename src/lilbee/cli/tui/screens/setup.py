"""First-run setup wizard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import Button, Label, ListItem, ListView, ProgressBar, Static

from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING, CatalogModel
from lilbee.config import cfg

log = logging.getLogger(__name__)

_STEP_CHAT = 1
_STEP_EMBED = 2


def _scan_installed_models(models_dir: Path) -> tuple[list[Path], list[Path]]:
    """Scan for installed GGUF models, split into chat vs embedding."""
    if not models_dir.exists():
        return [], []
    all_gguf = sorted(models_dir.glob("*.gguf"))
    embed = [p for p in all_gguf if "embed" in p.name.lower()]
    chat = [p for p in all_gguf if "embed" not in p.name.lower()]
    return chat, embed


class _InstalledRow(ListItem):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.model_path = path

    def compose(self) -> ComposeResult:
        size_mb = self.model_path.stat().st_size / (1024 * 1024)
        yield Static(f"  {self.model_path.stem}  ({size_mb:.0f} MB)  [installed]")


class _CatalogRow(ListItem):
    def __init__(self, model: CatalogModel) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        size = f"{self.model.size_gb:.1f} GB" if self.model.size_gb > 0 else "unknown"
        yield Static(f"  {self.model.name}  ({size})  {self.model.description}")


class SetupWizard(Screen[str | None]):
    """First-run setup -- helps user configure chat and embedding models."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._step = _STEP_CHAT
        self._selected_chat: str | None = None
        self._selected_embed: str | None = None
        self._chat_installed, self._embed_installed = _scan_installed_models(cfg.models_dir)

    def compose(self) -> ComposeResult:
        yield Static("Setup Wizard", id="setup-title")
        yield Label("", id="setup-step-label")
        yield ListView(id="setup-list")
        yield ProgressBar(total=100, show_eta=False, id="setup-progress")
        yield Label("", id="setup-status")
        yield Button("Skip -- chat only (no document search)", id="setup-skip")
        yield Button("Confirm", id="setup-confirm", disabled=True)

    def on_mount(self) -> None:
        self._show_step()

    def _show_step(self) -> None:
        lv = self.query_one("#setup-list", ListView)
        lv.clear()
        label = self.query_one("#setup-step-label", Label)
        self.query_one("#setup-confirm", Button).disabled = True
        self.query_one("#setup-progress", ProgressBar).update(total=100, progress=0)
        self.query_one("#setup-status", Label).update("")
        if self._step == _STEP_CHAT:
            label.update("Step 1/2: Choose a chat model")
            if self._chat_installed:
                lv.append(ListItem(Label("Installed locally:")))
                for p in self._chat_installed:
                    lv.append(_InstalledRow(p))
            lv.append(ListItem(Label("Featured models (download):")))
            for m in FEATURED_CHAT:
                lv.append(_CatalogRow(m))
        else:
            label.update("Step 2/2: Choose an embedding model")
            if self._embed_installed:
                lv.append(ListItem(Label("Installed locally:")))
                for p in self._embed_installed:
                    lv.append(_InstalledRow(p))
            lv.append(ListItem(Label("Featured models (download):")))
            for m in FEATURED_EMBEDDING:
                lv.append(_CatalogRow(m))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, _InstalledRow):
            self._on_model_chosen(item.model_path.stem)
        elif isinstance(item, _CatalogRow):
            self._download_model(item.model)

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
        self.app.call_from_thread(self._set_status, f"Downloading {model.name}...")
        try:
            from lilbee.catalog import download_model

            def _on_progress(downloaded: int, total: int) -> None:
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    self.app.call_from_thread(self._update_progress, pct)

            download_model(model, on_progress=_on_progress)
            self.app.call_from_thread(
                self._on_download_complete, model.gguf_filename.rsplit(".", 1)[0]
            )
        except Exception as exc:
            log.warning("Download failed for %s", model.name, exc_info=True)
            error_msg = str(exc)
            if "401" in error_msg:
                error_msg = f"{model.name} requires HuggingFace authentication"
            self.app.call_from_thread(self._set_status, f"Error: {error_msg}")

    def _set_status(self, text: str) -> None:
        self.query_one("#setup-status", Label).update(text)

    def _update_progress(self, percent: int) -> None:
        self.query_one("#setup-progress", ProgressBar).update(total=100, progress=percent)

    def _on_download_complete(self, name: str) -> None:
        self.query_one("#setup-progress", ProgressBar).update(total=100, progress=100)
        self._set_status(f"{name} installed!")
        self._on_model_chosen(name)

    def _finish(self) -> None:
        from lilbee import settings
        from lilbee.services import reset_services

        if self._selected_chat:
            cfg.chat_model = self._selected_chat
            settings.set_value(cfg.data_root, "chat_model", self._selected_chat)
        if self._selected_embed:
            cfg.embedding_model = self._selected_embed
            settings.set_value(cfg.data_root, "embedding_model", self._selected_embed)
        reset_services()
        self.dismiss("completed")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-skip":
            self.dismiss("skipped")
        elif event.button.id == "setup-confirm":
            self._finish()

    def action_cancel(self) -> None:
        self.dismiss("skipped")
