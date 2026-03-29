"""Chat message widgets — user and assistant bubbles."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Markdown, Static

# Minimum interval (seconds) between markdown widget updates during streaming
_MD_UPDATE_INTERVAL = 0.1


class UserMessage(Static):
    """A user's question in the chat log."""

    def __init__(self, text: str) -> None:
        super().__init__(f"You: {text}", classes="user-message")


class AssistantMessage(Vertical):
    """An assistant's response with streaming markdown, reasoning, and citations."""

    DEFAULT_CSS = """
    AssistantMessage {
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__(classes="assistant-message")
        self._reasoning_parts: list[str] = []
        self._content_parts: list[str] = []
        self._finished = False
        self._md_widget: Markdown | None = None
        self._reasoning_widget: Collapsible | None = None
        self._reasoning_static: Static | None = None
        self._citation_widget: Static | None = None
        self._last_md_update: float = 0.0

    def compose(self) -> ComposeResult:
        self._reasoning_static = Static("", classes="reasoning-text")
        self._reasoning_widget = Collapsible(
            self._reasoning_static,
            title="Thinking...",
            collapsed=True,
            classes="reasoning-block",
        )
        yield self._reasoning_widget
        self._md_widget = Markdown("", classes="response-md")
        yield self._md_widget
        self._citation_widget = Static("", classes="source-citation")
        yield self._citation_widget

    def append_reasoning(self, text: str) -> None:
        """Append reasoning token (shown in collapsible)."""
        self._reasoning_parts.append(text)
        if self._reasoning_widget is not None:
            self._reasoning_widget.collapsed = False
            self._reasoning_static.update("".join(self._reasoning_parts))

    def append_content(self, text: str) -> None:
        """Append response content token (debounced markdown updates)."""
        self._content_parts.append(text)
        now = time.monotonic()
        if self._md_widget is not None and now - self._last_md_update >= _MD_UPDATE_INTERVAL:
            self._last_md_update = now
            self._md_widget.update("".join(self._content_parts))
            self.refresh(layout=True)

    def finish(self, sources: list[str] | None = None) -> None:
        """Mark response as complete and show citations."""
        self._finished = True
        if self._md_widget is not None and self._content_parts:
            self._md_widget.update("".join(self._content_parts))
            self.refresh(layout=True)
        if self._reasoning_widget is not None and self._reasoning_parts:
            self._reasoning_widget.title = "Reasoning"
        elif self._reasoning_widget is not None:
            self._reasoning_widget.display = False

        if sources and self._citation_widget is not None:
            self._citation_widget.update("── " + ", ".join(sources))
        elif self._citation_widget is not None:
            self._citation_widget.display = False
