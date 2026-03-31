"""Model status bar — Select dropdowns for chat, embedding, and vision models."""

from __future__ import annotations

import logging

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Select

from lilbee import settings
from lilbee.config import cfg

log = logging.getLogger(__name__)

_DISABLED = Select.BLANK


class ModelBar(Widget):
    """Compact bar with Select dropdowns for active model assignments."""

    DEFAULT_CSS = """
    ModelBar {
        dock: top;
        height: 3;
        padding: 0 1;
    }
    ModelBar Horizontal {
        height: 3;
        width: 100%;
    }
    ModelBar Select {
        width: 1fr;
        margin: 0 1 0 0;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._populating = True  # Guard against change events during init

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Select[str](
                options=[],
                prompt="Chat model",
                id="chat-model-select",
                allow_blank=False,
            )
            yield Select[str](
                options=[],
                prompt="Embed model",
                id="embed-model-select",
                allow_blank=False,
            )
            yield Select[str](
                options=[],
                prompt="Vision (optional)",
                id="vision-model-select",
                allow_blank=True,
            )

    def on_mount(self) -> None:
        self._scan_models()

    @work(thread=True)
    def _scan_models(self) -> None:
        """Scan installed models in background, then populate dropdowns."""
        try:
            from lilbee.services import get_services

            provider = get_services().provider
            all_models = provider.list_models()
        except Exception:
            log.debug("Could not list models for dropdowns", exc_info=True)
            all_models = []

        # Split by type heuristic: "embed" in name = embedding, rest = chat
        embed_models = [m for m in all_models if "embed" in m.lower()]
        chat_models = [m for m in all_models if "embed" not in m.lower()]

        self.app.call_from_thread(
            self._populate, chat_models, embed_models, all_models
        )

    def _populate(
        self,
        chat_models: list[str],
        embed_models: list[str],
        all_models: list[str],
    ) -> None:
        """Populate Select widgets from scanned models (main thread)."""
        self._populating = True

        chat_sel = self.query_one("#chat-model-select", Select)
        embed_sel = self.query_one("#embed-model-select", Select)
        vision_sel = self.query_one("#vision-model-select", Select)

        chat_opts = [(m, m) for m in chat_models] if chat_models else [("(none)", "")]
        embed_opts = [(m, m) for m in embed_models] if embed_models else [("(none)", "")]
        vision_opts = [(m, m) for m in all_models]

        chat_sel.set_options(chat_opts)
        embed_sel.set_options(embed_opts)
        vision_sel.set_options(vision_opts)

        # Set current values
        if cfg.chat_model and any(v == cfg.chat_model for _, v in chat_opts):
            chat_sel.value = cfg.chat_model
        elif chat_models:
            chat_sel.value = chat_models[0]

        if cfg.embedding_model and any(v == cfg.embedding_model for _, v in embed_opts):
            embed_sel.value = cfg.embedding_model
        elif embed_models:
            embed_sel.value = embed_models[0]

        if cfg.vision_model and any(v == cfg.vision_model for _, v in vision_opts):
            vision_sel.value = cfg.vision_model
        else:
            vision_sel.value = _DISABLED

        self._populating = False

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle model selection changes."""
        if self._populating:
            return
        if event.value is _DISABLED or event.value is None:
            if event.select.id == "vision-model-select":
                cfg.vision_model = ""
                settings.set_value(cfg.data_root, "vision_model", "")
            return

        value = str(event.value)
        if event.select.id == "chat-model-select":
            cfg.chat_model = value
            settings.set_value(cfg.data_root, "chat_model", value)
        elif event.select.id == "embed-model-select":
            cfg.embedding_model = value
            settings.set_value(cfg.data_root, "embedding_model", value)
        elif event.select.id == "vision-model-select":
            cfg.vision_model = value
            settings.set_value(cfg.data_root, "vision_model", value)

        # Reset services to pick up new model
        from lilbee.services import reset_services

        reset_services()

    def refresh_models(self) -> None:
        """Re-scan models (called after downloads complete)."""
        self._scan_models()
