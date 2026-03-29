"""Download modal — lightweight model installation with progress."""

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
    """Compact download overlay."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    DownloadModal {
        align: center middle;
    }
    DownloadModal > Vertical {
        width: 60;
        height: auto;
        max-height: 7;
        border: thick $surface-lighten-2;
        padding: 1 2;
    }
    """

    def __init__(self, model: CatalogModel) -> None:
        super().__init__()
        self._model = model
        self._dismiss_result: bool = False

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Connecting...", id="dl-status")
            yield ProgressBar(total=100, show_eta=False, id="dl-progress")

    def on_mount(self) -> None:
        self._do_download()

    @work(thread=True)
    def _do_download(self) -> None:
        """Download the model in background."""
        try:
            from lilbee.catalog import download_model

            def on_progress(downloaded: int, total: int) -> None:
                if total > 0:
                    pct = min(int(downloaded * 100 / total), 100)
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    self.app.call_from_thread(self._update_progress, pct)
                    self.app.call_from_thread(
                        self._set_status,
                        f"{self._model.name}  {mb_done:.0f}/{mb_total:.0f} MB  {pct}%",
                    )

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
        self._set_status(f"{self._model.name} installed")
        self._dismiss_result = True
        self.set_timer(1.0, self._do_dismiss)

    def _on_error(self, msg: str) -> None:
        self._set_status(f"Error: {msg}")
        self._dismiss_result = False
        self.set_timer(2.0, self._do_dismiss)

    def _do_dismiss(self) -> None:
        self.dismiss(self._dismiss_result)

    def action_cancel(self) -> None:
        for worker in self.workers:
            worker.cancel()
        self.dismiss(False)
