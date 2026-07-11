"""Chat message widgets: user and assistant bubbles."""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Collapsible, Markdown, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.widgets.thinking_header import ThinkingHeader
from lilbee.core.config import cfg

# Minimum interval (seconds) between markdown widget updates during streaming
_MD_UPDATE_INTERVAL = 0.1

_SPEAKER_YOU = "[bold $primary]you[/]"
_SPEAKER_LILBEE = "[bold $success]lilbee[/]"

_REASONING_BLOCK_CLASS = "reasoning-block"
_REASONING_STREAMING_CLASS = "-streaming"

_CSS_FILE = Path(__file__).parent / "message.tcss"
_MESSAGE_CSS = _CSS_FILE.read_text(encoding="utf-8")


class UserMessage(Vertical):
    """A user's question in the chat log."""

    DEFAULT_CSS: ClassVar[str] = _MESSAGE_CSS

    def __init__(self, text: str) -> None:
        super().__init__(classes="user-message")
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(_SPEAKER_YOU, classes="speaker-label")
        # Content() renders the question literally: a user asking about e.g. arr[0]
        # or "[/]" must not have it parsed as console markup (which would crash).
        yield Static(Content(self._text), classes="message-content")


class AssistantMessage(Vertical):
    """An assistant's response with streaming markdown, reasoning, and citations."""

    DEFAULT_CSS: ClassVar[str] = _MESSAGE_CSS

    def __init__(self) -> None:
        super().__init__(classes="assistant-message")
        self._reasoning_parts: list[str] = []
        self._content_parts: list[str] = []
        self._finished = False
        self._content_widget: Markdown | Static | None = None
        self._reasoning_widget: Collapsible | None = None
        self._reasoning_static: Static | None = None
        self._citation_widget: Static | None = None
        self._thinking_header: ThinkingHeader | None = None
        self._last_md_update: float = 0.0
        self._last_reasoning_update: float = 0.0
        self._use_markdown: bool = cfg.markdown_rendering

    def compose(self) -> ComposeResult:
        yield Static(_SPEAKER_LILBEE, classes="speaker-label")
        self._content_widget = self._build_content_widget()
        yield self._content_widget
        self._citation_widget = Static("", classes="source-citation")
        yield self._citation_widget

    def on_mount(self) -> None:
        """Mount the thinking header above the content widget.

        ``compose`` populates ``_content_widget`` before this hook runs.
        """
        if self._content_widget is None:
            return
        header = ThinkingHeader()
        self._thinking_header = header
        self.mount(header, before=self._content_widget)

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
        self._set_content(new_widget, text)
        await self.mount(new_widget, after=old)
        self._content_widget = new_widget
        await old.remove()

    @staticmethod
    def _set_content(widget: Markdown | Static, text: str) -> None:
        """Update a content widget with raw model text. A Markdown widget consumes
        the raw markdown string, but a Static parses console markup -- so wrap the
        text as literal Content. Otherwise a ``[..]`` in the answer (quoted code, an
        option like ``[/path]``) raises MarkupError and crashes the whole TUI.
        """
        if isinstance(widget, Markdown):
            widget.update(text)
        else:
            widget.update(Content(text))

    def append_reasoning(self, text: str) -> None:
        """Append a reasoning token; debounced at ``_MD_UPDATE_INTERVAL``."""
        first_token = not self._reasoning_parts
        self._reasoning_parts.append(text)
        if first_token and self._reasoning_widget is None:
            self._mount_reasoning_collapsible()
        now = time.monotonic()
        ready = now - self._last_reasoning_update >= _MD_UPDATE_INTERVAL
        if self._reasoning_static is not None and ready:
            self._last_reasoning_update = now
            self._reasoning_static.update(Content("".join(self._reasoning_parts)))

    def append_content(self, text: str) -> None:
        """Append response content token (debounced markdown updates)."""
        first_token = not self._content_parts
        self._content_parts.append(text)
        if first_token and not self._reasoning_parts:
            # No reasoning ever arrived; drop the standalone header.
            self._dismiss_thinking_header()
        now = time.monotonic()
        if self._content_widget is not None and now - self._last_md_update >= _MD_UPDATE_INTERVAL:
            self._last_md_update = now
            self._set_content(self._content_widget, "".join(self._content_parts))
            self.refresh()

    def finish(self, sources: list[str] | None = None) -> None:
        """Mark response as complete and show citations."""
        self._finished = True
        # Always retire the standalone header on finish; the reasoning fold
        # (if mounted) carries the post-stream title.
        self._dismiss_thinking_header()
        if self._content_widget is not None and self._content_parts:
            self._set_content(self._content_widget, "".join(self._content_parts))
            self.refresh()
        if self._reasoning_widget is not None and self._reasoning_parts:
            if self._reasoning_static is not None:
                self._reasoning_static.update(Content("".join(self._reasoning_parts)))
            token_count = len("".join(self._reasoning_parts).split())
            self._reasoning_widget.remove_class(_REASONING_STREAMING_CLASS)
            self._reasoning_widget.title = msg.CHAT_REASONING_FINISHED.format(tokens=token_count)
            self._reasoning_widget.collapsed = True

        if sources and self._citation_widget is not None:
            self._citation_widget.update(_build_citation_content(sources))
        elif self._citation_widget is not None:
            self._citation_widget.display = False

    def _mount_reasoning_collapsible(self) -> None:
        """Mount the reasoning Collapsible with the streaming-state class.

        Called from ``append_reasoning`` on the first reasoning token, after
        the message itself is mounted. The Collapsible slots in beneath the
        ``ThinkingHeader`` so the animator continues to drive the visual
        weight while the toggle row is hidden by the ``-streaming`` rule.
        """
        classes = f"{_REASONING_BLOCK_CLASS} {_REASONING_STREAMING_CLASS}"
        self._reasoning_static = Static("", classes="reasoning-text")
        collapsible = Collapsible(
            self._reasoning_static,
            title=msg.CHAT_REASONING_FINISHED.format(tokens=0),
            collapsed=False,
            classes=classes,
        )
        self._reasoning_widget = collapsible
        header = self._thinking_header
        if header is not None and header.is_mounted:
            self.mount(collapsible, after=header)
            return
        content = self._content_widget
        if content is not None:
            self.mount(collapsible, before=content)

    def _dismiss_thinking_header(self) -> None:
        """Stop the animator and remove the standalone header from the DOM."""
        header = self._thinking_header
        if header is None:
            return
        header.stop()
        if header.is_mounted:
            header.remove()
        self._thinking_header = None


def _build_citation_content(sources: list[str]) -> Content:
    """Build a 'sources: pill pill pill' content line from source paths."""
    parts: list[Content] = [Content.styled(msg.CHAT_SOURCES_LABEL, "$text-muted")]
    for src in sources:
        parts.append(Content("  "))
        parts.append(pill(Path(src).name, "$surface-lighten-2", "$text"))
    return Content.assemble(*parts)
