"""Modal that lists the engines a model can run on and returns the one picked."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from lilbee.app.engine_service import EngineOption, InstallRoute
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_mark import engine_mark, format_size
from lilbee.providers.fleet.engines.choice import AUTO

_DETAIL_SEP = " · "


def option_detail(option: EngineOption) -> Content:
    """The state, docs link and backend of *option*, in the muted style the picker rows use."""
    parts = (state_text(option), option.docs_url, option.backend)
    text = _DETAIL_SEP.join(part for part in parts if part)
    return Content.styled(text, "$text-muted")


def state_text(option: EngineOption) -> str:
    """Where the engine of *option* stands: bundled, installed, or the size to download."""
    if option.bundled:
        return msg.ENGINE_STATE_BUNDLED
    if option.installed:
        return msg.ENGINE_STATE_INSTALLED
    if option.route is InstallRoute.NO_PACK:
        return msg.ENGINE_STATE_NO_DOWNLOAD
    if option.route is InstallRoute.BY_HAND:
        return msg.ENGINE_STATE_BY_HAND
    if option.download_bytes is None:
        return msg.ENGINE_STATE_NOT_INSTALLED
    return msg.ENGINE_STATE_DOWNLOAD.format(size=format_size(option.download_bytes))


class EnginePickerModal(ModalScreen[EngineOption | None]):
    """One engine per row; Enter returns its option, Esc returns None."""

    CSS_PATH: ClassVar[str] = "engine_picker.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss(None)", "Cancel", show=True),
    ]

    def __init__(
        self, title: str, subtitle: str, current: str | None, options: list[EngineOption]
    ) -> None:
        super().__init__()
        self._title = title
        self._subtitle = subtitle
        self._current = current
        self._options = {option.name: option for option in options}

    def _label(self, option: EngineOption) -> Content:
        if option.name == AUTO:
            text = (
                msg.ENGINE_PICKER_AUTO_NOW.format(engine=option.auto_now)
                if option.auto_now
                else msg.ENGINE_PICKER_AUTO
            )
            return Content(text)
        head = engine_mark(option.name)
        if option.name == self._current:
            head = Content.assemble(head, " (now)")
        return Content.assemble(head, "  ", option_detail(option))

    def compose(self) -> ComposeResult:
        with Vertical(id="engine-picker-root"):
            yield Static(self._title, id="engine-picker-title")
            yield Static(self._subtitle, id="engine-picker-model", markup=False)
            yield Static(msg.ENGINE_PICKER_NOTE, id="engine-picker-note")
            yield OptionList(
                *[
                    Option(
                        self._label(option),
                        id=name,
                        disabled=not option.installed and not option.offers_download,
                    )
                    for name, option in self._options.items()
                ],
                id="engine-picker-list",
            )
            yield Static(msg.ENGINE_PICKER_HINT, id="engine-picker-hint")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self._options[str(event.option.id)])
