"""First-run setup wizard."""

from __future__ import annotations

import logging
from typing import ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import Button, Label, ListItem, ListView, ProgressBar, Static

from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING, CatalogModel
from lilbee.cli.tui import messages as msg
from lilbee.config import cfg

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
            # Skip vision and other types — not relevant for setup wizard
        return sorted(chat), sorted(embed)
    except Exception:
        return [], []


class _InstalledRow(ListItem):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.model_name = name

    def compose(self) -> ComposeResult:
        yield Static(f"  {self.model_name}  [installed]")


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
        self._chat_installed, self._embed_installed = _scan_installed_models()

    def compose(self) -> ComposeResult:
        yield Static(msg.SETUP_TITLE, id="setup-title")
        yield Label("", id="setup-step-label")
        yield ListView(id="setup-list")
        yield Label("", id="setup-status")
        yield ProgressBar(total=100, show_eta=False, id="setup-progress")
        yield Button(msg.SETUP_SKIP_BUTTON, id="setup-skip")
        yield Button(msg.SETUP_CONFIRM_BUTTON, id="setup-confirm", disabled=True)

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
            label.update(msg.SETUP_STEP_CHAT)
            if self._chat_installed:
                lv.append(ListItem(Label(msg.SETUP_INSTALLED_LABEL)))
                for name in self._chat_installed:
                    lv.append(_InstalledRow(name))
            lv.append(ListItem(Label(msg.SETUP_FEATURED_LABEL)))
            for m in FEATURED_CHAT:
                lv.append(_CatalogRow(m))
        else:
            label.update(msg.SETUP_STEP_EMBED)
            if self._embed_installed:
                lv.append(ListItem(Label(msg.SETUP_INSTALLED_LABEL)))
                for name in self._embed_installed:
                    lv.append(_InstalledRow(name))
            lv.append(ListItem(Label(msg.SETUP_FEATURED_LABEL)))
            for m in FEATURED_EMBEDDING:
                lv.append(_CatalogRow(m))

    @on(ListView.Selected, "#setup-list")
    def _on_list_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, _InstalledRow):
            self._on_model_chosen(item.model_name)
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
        """Download via catalog API with TUI-native progress (no Rich).

        Runs in a worker thread to avoid blocking the UI. Progress is reported
        via callbacks rather than tqdm to avoid multiprocessing lock issues
        with Textual's worker threads.

        Note: disable_progress_bars() is called in catalog.py during download.
        """
        self.app.call_from_thread(self._set_status, msg.SETUP_CONNECTING)
        try:
            from lilbee.catalog import download_model

            last_update_time = 0.0
            last_pct = -1

            def _on_progress(downloaded: int, total: int) -> None:
                nonlocal last_update_time, last_pct
                import time

                current_time = time.time()

                # Throttle updates to every 100ms to avoid UI spam
                if current_time - last_update_time < 0.1:
                    return

                try:
                    mb_done = downloaded / (1024 * 1024)

                    if total > 0:
                        pct = min(int(downloaded * 100 / total), 100)

                        # Always update status with MB downloaded (more informative than %)
                        mb_total = total / (1024 * 1024)
                        last_update_time = current_time

                        # Update progress bar - but allow 0% to show if truly at start
                        # Only throttle if we're at same percentage AND > 0%
                        if pct == last_pct and pct > 0:
                            return
                        last_pct = pct

                        self.app.call_from_thread(self._update_progress, pct)
                        self.app.call_from_thread(
                            self._set_status,
                            f"Downloading... {mb_done:.0f} / {mb_total:.0f} MB ({pct}%)",
                        )
                    else:
                        # Show raw MB if total is unknown
                        self.app.call_from_thread(
                            self._set_status,
                            f"Downloading... {mb_done:.0f} MB",
                        )
                except Exception:
                    log.debug("Progress callback failed", exc_info=True)

            dest = download_model(model, on_progress=_on_progress)
            self.app.call_from_thread(self._on_download_complete, dest.stem)
        except Exception as exc:
            log.warning("Download failed for %s", model.name, exc_info=True)
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
            settings.set_value(cfg.data_root, "embedding_model", self._selected_embed)
        reset_services()
        self.dismiss("completed")

    @on(Button.Pressed, "#setup-skip")
    def _on_skip(self) -> None:
        self.dismiss("skipped")

    @on(Button.Pressed, "#setup-confirm")
    def _on_confirm(self) -> None:
        self._finish()

    def action_cancel(self) -> None:
        self.dismiss("skipped")
