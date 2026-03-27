"""First-run setup modal — embedding model picker."""

from __future__ import annotations

import logging
from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, ProgressBar, Static

from lilbee.catalog import FEATURED_EMBEDDING, CatalogModel

log = logging.getLogger(__name__)


class _EmbeddingRow(ListItem):
    """A selectable embedding model row."""

    def __init__(self, model: CatalogModel, *, recommended: bool = False) -> None:
        super().__init__()
        self.model = model
        self._recommended = recommended

    def compose(self) -> ComposeResult:
        suffix = "  (Recommended)" if self._recommended else ""
        yield Static(f"  {self.model.name}  {self.model.size_gb:.1f} GB{suffix}")


class _OllamaRow(ListItem):
    """An Ollama embedding model (no download needed)."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.ollama_name = name

    def compose(self) -> ComposeResult:
        yield Static(f"  {self.ollama_name}  (Ollama, no download)")


class SetupModal(ModalScreen[str | None]):
    """First-run modal to pick an embedding model."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, ollama_embeddings: list[str] | None = None) -> None:
        super().__init__()
        self._ollama_embeddings = ollama_embeddings or []
        self._downloaded_name: str | None = None

    def compose(self) -> ComposeResult:
        items: list[ListItem] = []

        if self._ollama_embeddings:
            items.append(ListItem(Label("Found in Ollama:")))
            for name in self._ollama_embeddings:
                items.append(_OllamaRow(name))
            items.append(ListItem(Label("\nOr download:")))

        for i, model in enumerate(FEATURED_EMBEDDING):
            items.append(_EmbeddingRow(model, recommended=(i == 0)))

        with Vertical():
            yield Label("An embedding model is required to index and search your documents.\n")
            yield ListView(*items, id="embed-picker")
            yield ProgressBar(total=100, show_eta=False, id="setup-progress")
            yield Label("", id="setup-status")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, _OllamaRow):
            self.dismiss(item.ollama_name)
        elif isinstance(item, _EmbeddingRow):
            self._download_model(item.model)

    @work(thread=True)
    def _download_model(self, model: CatalogModel) -> None:
        """Download the selected embedding model."""
        self.app.call_from_thread(self._set_status, f"Downloading {model.name}...")
        try:
            from lilbee.models import pull_with_progress

            pull_with_progress(model.name)
            self.app.call_from_thread(self._on_downloaded, model.name)
        except Exception as exc:
            log.warning("Embedding download failed for %s", model.name, exc_info=True)
            self.app.call_from_thread(self._set_status, f"Error: {exc}")

    def _set_status(self, text: str) -> None:
        self.query_one("#setup-status", Label).update(text)

    def _on_downloaded(self, name: str) -> None:
        progress = self.query_one("#setup-progress", ProgressBar)
        progress.update(total=100, progress=100)
        self._set_status(f"{name} installed!")
        self._downloaded_name = name
        self.call_later(self._finish_dismiss)

    def _finish_dismiss(self) -> None:
        self.dismiss(self._downloaded_name)

    def action_cancel(self) -> None:
        self.dismiss(None)
