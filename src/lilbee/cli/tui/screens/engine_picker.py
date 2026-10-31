"""Modal that lists the engines a model can run on and returns the one picked."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from lilbee.app import engine_service
from lilbee.cli.tui import messages as msg
from lilbee.providers.fleet.engines.choice import AUTO


class EnginePickerModal(ModalScreen[str | None]):
    """One engine per row; Enter returns its name, Esc returns None."""

    CSS_PATH: ClassVar[str] = "engine_picker.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss(None)", "Cancel", show=True),
    ]

    def __init__(self, view: engine_service.RoleView, names: list[str]) -> None:
        super().__init__()
        self._view = view
        self._names = names

    def _label(self, name: str) -> str:
        text = msg.ENGINE_PICKER_AUTO if name == AUTO else name
        return msg.ENGINE_PICKER_CURRENT.format(name=text) if name == self._view.engine else text

    def compose(self) -> ComposeResult:
        view = self._view
        with Vertical(id="engine-picker-root"):
            yield Static(msg.ENGINE_PICKER_TITLE.format(role=view.role), id="engine-picker-title")
            yield Static(
                msg.ENGINE_PICKER_SUBTITLE.format(model=view.model, format=view.format),
                id="engine-picker-model",
                markup=False,
            )
            yield OptionList(
                *[Option(self._label(name), id=name) for name in self._names],
                id="engine-picker-list",
            )
            yield Static(msg.ENGINE_PICKER_HINT, id="engine-picker-hint")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id)
