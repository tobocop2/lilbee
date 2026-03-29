"""Download modal — inline model installation with progress."""

from __future__ import annotations

import logging
from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ProgressBar

from lilbee.catalog import CatalogModel

log = logging.getLogger(__name__)


class DownloadModal(ModalScreen[bool]):
    """Modal overlay showing download progress for a model."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, model: CatalogModel) -> None:
        super().__init__()
        self._model = model
        self._dismiss_result: bool = False

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Installing {self._model.name}...")
            yield ProgressBar(total=100, show_eta=False, id="dl-progress")
            yield Label("Downloading...", id="dl-status")

    def on_mount(self) -> None:
        self._do_download()

    @work(thread=True)
    def _do_download(self) -> None:
        """Download the model in background."""
        try:
            self.app.call_from_thread(
                self._set_status, f"Downloading {self._model.name} from {self._model.hf_repo}..."
            )

            from lilbee.catalog import download_model

            def on_progress(downloaded: int, total: int) -> None:
                if total > 0:
                    pct = min(int(downloaded * 100 / total), 100)
                    self.app.call_from_thread(self._update_progress, pct)

            download_model(self._model, on_progress=on_progress)
            self.app.call_from_thread(self._on_success)
        except Exception as exc:
            log.warning("Download failed for %s", self._model.name, exc_info=True)
            self.app.call_from_thread(self._on_error, str(exc))

    def _set_status(self, text: str) -> None:
        self.query_one("#dl-status", Label).update(text)

    def _update_progress(self, pct: int) -> None:
        self.query_one("#dl-progress", ProgressBar).update(total=100, progress=pct)

    def _on_success(self) -> None:
        self._update_progress(100)
        self._set_status(f"{self._model.name} installed!")
        self._dismiss_result = True
        # Brief pause so user sees the "installed" message
        self.set_timer(1.0, self._do_dismiss)

    def _on_error(self, msg: str) -> None:
        self._set_status(f"Error: {msg}")
        self._dismiss_result = False
        self.call_later(self._do_dismiss)

    def _do_dismiss(self) -> None:
        self.dismiss(self._dismiss_result)

    def action_cancel(self) -> None:
        for worker in self.workers:
            worker.cancel()
        self.dismiss(False)
