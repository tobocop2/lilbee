"""Model status bar — shows current chat, embedding, and vision model assignments."""

from __future__ import annotations

from textual.widgets import Static

from lilbee.config import cfg


class ModelBar(Static):
    """Compact status line showing active model assignments."""

    DEFAULT_CSS = """
    ModelBar {
        dock: top;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def on_mount(self) -> None:
        self.refresh_models()

    def refresh_models(self) -> None:
        """Update the display with current model config."""
        vision = cfg.vision_model or "off"
        self.update(f"chat: {cfg.chat_model}  │  embed: {cfg.embedding_model}  │  vision: {vision}")
