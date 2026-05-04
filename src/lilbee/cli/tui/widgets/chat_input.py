"""Multi-line chat prompt: TextArea with submit-on-Enter semantics.

Behaves like a chat input box: Enter submits, Shift+Enter inserts a literal
newline, paste preserves newlines so multi-line content (logs, code,
~/.zshrc, etc.) round-trips correctly. Posts a ``ChatInput.Submitted``
message on Enter so the screen handler can stay shaped like the previous
``Input.Submitted`` flow.

The completion overlay listens to :class:`textual.widgets.TextArea.Changed`
events from this widget; no additional event plumbing is required here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual import on
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.widgets import TextArea


class ChatInput(TextArea):
    """A TextArea variant where Enter submits and Shift+Enter inserts a newline."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "submit", "Send", show=False, priority=True),
        Binding("shift+enter", "newline", "Newline", show=False, priority=True),
    ]

    # Keys we deliberately let bubble up to the App-level help binding
    # even though the underlying TextArea is happy to type them. Without
    # this, Textual's binding-chain filter strips the App's `?` binding
    # whenever the chat input is focused (because TextArea consumes any
    # printable). Users who need a literal `?` can paste it.
    # ``"question_mark"`` is Textual's canonical name for `?`.
    _UNCONSUMED_KEYS: ClassVar[frozenset[str]] = frozenset({"question_mark"})

    # Per-keystroke layout cost is dominated by ``height: auto`` reflow.
    # Pin the visual height to a single row while the content has no
    # newline; flip to auto-grow only once a newline appears (Shift+Enter
    # or pasted multi-line text). The CSS hook is the ``-multiline``
    # class added by :meth:`_track_multiline`.

    @dataclass
    class Submitted(Message):
        """Posted when the user presses Enter to send the current text."""

        chat_input: ChatInput
        value: str

        @property
        def control(self) -> ChatInput:
            return self.chat_input

    def __init__(
        self,
        *,
        placeholder: str = "",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, placeholder=placeholder, soft_wrap=True)

    @property
    def value(self) -> str:
        """The current text, named for parity with ``Input.value`` callers."""
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        self.load_text(new_value)
        self.action_end()

    def check_consume_key(self, key: str, character: str | None = None) -> bool:
        """Pass App-level help/global keys back up to the binding chain."""
        if key in self._UNCONSUMED_KEYS:
            return False
        return super().check_consume_key(key, character)

    def action_submit(self) -> None:
        self.post_message(self.Submitted(chat_input=self, value=self.text))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_end(self) -> None:
        """Move cursor to end of all text (Input-compatible behavior)."""
        last_line = self.document.line_count - 1
        last_col = len(self.document.get_line(last_line))
        self.move_cursor((last_line, last_col))

    @on(TextArea.Changed)
    def _track_multiline(self, _event: TextArea.Changed) -> None:
        """Toggle the ``-multiline`` class so CSS can pin height for the
        single-line case and let it grow only when newlines are present."""
        self.set_class("\n" in self.text, "-multiline")
