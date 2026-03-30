"""Chat message widgets — user and assistant bubbles."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Markdown, Static

from lilbee.config import cfg

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
        self._content_widget: Markdown | Static | None = None
        self._reasoning_widget: Collapsible | None = None
        self._reasoning_static: Static | None = None
        self._citation_widget: Static | None = None
        self._last_md_update: float = 0.0
        self._use_markdown: bool = cfg.markdown_rendering

    def compose(self) -> ComposeResult:
        self._reasoning_static = Static("", classes="reasoning-text")
        self._reasoning_widget = Collapsible(
            self._reasoning_static,
            title="Thinking...",
            collapsed=True,
            classes="reasoning-block",
        )
        yield self._reasoning_widget
        self._content_widget = self._build_content_widget()
        yield self._content_widget
        self._citation_widget = Static("", classes="source-citation")
        yield self._citation_widget

    def _build_content_widget(self) -> Markdown | Static:
        """Create the content widget based on the current rendering mode."""
        if self._use_markdown:
            return Markdown("", classes="response-md")
        return Static("", classes="response-md")

    @property
    def use_markdown(self) -> bool:
        """Whether this message is using Markdown rendering."""
        return self._use_markdown

    async def rebuild_content_widget(self, use_markdown: bool) -> None:
        """Replace the content widget with a different rendering mode."""
        if self._content_widget is None:
            return
        self._use_markdown = use_markdown
        old = self._content_widget
        new_widget = self._build_content_widget()
        text = "".join(self._content_parts)
        new_widget.update(text)
        await self.mount(new_widget, after=old)
        await old.remove()
        self._content_widget = new_widget

    def append_reasoning(self, text: str) -> None:
        """Append reasoning token (shown in collapsible)."""
        self._reasoning_parts.append(text)
        if self._reasoning_widget is not None:
            self._reasoning_widget.collapsed = False
            if self._reasoning_static is not None:
                self._reasoning_static.update("".join(self._reasoning_parts))

    def append_content(self, text: str) -> None:
        """Append response content token (debounced markdown updates)."""
        self._content_parts.append(text)
        now = time.monotonic()
        if self._content_widget is not None and now - self._last_md_update >= _MD_UPDATE_INTERVAL:
            self._last_md_update = now
            self._content_widget.update("".join(self._content_parts))
            self.refresh(layout=True)

    def finish(self, sources: list[str] | None = None) -> None:
        """Mark response as complete and show citations."""
        self._finished = True
        if self._content_widget is not None and self._content_parts:
            self._content_widget.update("".join(self._content_parts))
            self.refresh(layout=True)
        if self._reasoning_widget is not None and self._reasoning_parts:
            self._reasoning_widget.title = "Reasoning"
        elif self._reasoning_widget is not None:
            self._reasoning_widget.display = False

        if sources and self._citation_widget is not None:
            self._citation_widget.update("── " + ", ".join(sources))
        elif self._citation_widget is not None:
            self._citation_widget.display = False
