"""Modal that picks the answer a fork of the conversation ends on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.clamped_option_list import ClampedOptionList
from lilbee.sessions import MessageRole, SessionMessage, derive_title


@dataclass(frozen=True, slots=True)
class ForkPoint:
    """A place to fork: after one answer, with the question it answers if there is one."""

    message_count: int
    answer: str
    question: str | None


def fork_points(messages: tuple[SessionMessage, ...]) -> list[ForkPoint]:
    """One point after each answer in the saved log, newest first."""
    points: list[ForkPoint] = []
    question: str | None = None
    for index, message in enumerate(messages):
        if message.role == MessageRole.USER:
            question = message.content
        else:
            points.append(ForkPoint(index + 1, message.content, question))
    return points[::-1]


def _label(point: ForkPoint) -> Content:
    """The answer's first line, and under it the question it answers."""
    answer = Content(derive_title(point.answer))
    if point.question is None:
        return answer
    question = msg.FORK_PICKER_ANSWER_TO.format(question=derive_title(point.question))
    return Content.assemble(answer, "\n", Content.styled(question, "$text-muted"))


class ForkPicker(ModalScreen[int | None]):
    """Dismisses with the number of leading messages to copy, or ``None`` on escape."""

    CSS_PATH = "fork_picker.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=True),
    ]

    def __init__(self, points: list[ForkPoint]) -> None:
        super().__init__()
        self._points = points

    def compose(self) -> ComposeResult:
        with Vertical(id="fork-root"):
            yield Static(msg.FORK_PICKER_TITLE, id="fork-title")
            yield ClampedOptionList(
                *(Option(_label(point)) for point in self._points), id="fork-list"
            )
            yield Static(msg.FORK_PICKER_HINT, id="fork-hint")

    def on_mount(self) -> None:
        self.query_one("#fork-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self._points[event.option_index].message_count)

    def action_cancel(self) -> None:
        self.dismiss(None)
