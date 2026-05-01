"""Modal picker for chat / embedding model selection.

Replaces the flat ``Select`` widgets in ModelBar. Opens with focus on a
search input that filters a virtualized ``ModelList`` body. Returns the
selected ref via ``dismiss``; the caller routes it through
``apply_active_model`` so the persistence path is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Input, Label, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.model_bar import ModelOption
from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

PickerScope = Literal["chat", "embed"]


@dataclass
class _PickerOptions:
    """Bridge from the dropdown's ``ModelOption`` shape to ``CatalogRow``."""

    options: list[ModelOption]

    def to_sections(self, search: str) -> list[ModelListSection]:
        rows = [_option_to_row(o) for o in self.options if _matches(o, search)]
        return [ModelListSection(heading=None, rows=rows)] if rows else []


def _option_to_row(option: ModelOption) -> CatalogRow:
    return LocalCatalogRow(
        name=option.label,
        task="",
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=False,
        sort_downloads=0,
        sort_size=0.0,
        ref=option.ref,
        backend="",
    )


def _matches(option: ModelOption, search: str) -> bool:
    if not search:
        return True
    needle = search.lower()
    return needle in option.label.lower() or needle in option.ref.lower()


class ModelPickerModal(ModalScreen[str | None]):
    """Searchable model list. Returns the selected ref or None on cancel."""

    CSS_PATH: ClassVar[str] = "model_picker.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss(None)", "Cancel", show=True),
        Binding("slash", "focus_search", "Search", show=True),
    ]

    _SEARCH_DEBOUNCE_SECONDS = 0.08

    def __init__(
        self,
        *,
        scope: PickerScope,
        options: list[ModelOption],
    ) -> None:
        super().__init__()
        self._scope: PickerScope = scope
        self._options = _PickerOptions(options=options)
        self._search_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        title = (
            msg.MODEL_PICKER_TITLE_CHAT if self._scope == "chat" else msg.MODEL_PICKER_TITLE_EMBED
        )
        with Vertical(id="picker-root"):
            yield Label(title, id="picker-title")
            yield Input(placeholder=msg.MODEL_PICKER_SEARCH_PLACEHOLDER, id="picker-search")
            yield ModelList(id="picker-list")
            yield Static(msg.MODEL_PICKER_HINT, id="picker-hint")

    def on_mount(self) -> None:
        self._refresh_list("")
        self.query_one("#picker-search", Input).focus()

    def _refresh_list(self, search: str) -> None:
        ml = self.query_one("#picker-list", ModelList)
        ml.set_rows(self._options.to_sections(search))

    @on(Input.Changed, "#picker-search")
    def _on_search_changed(self, event: Input.Changed) -> None:
        # Debounce: rapid typing collapses to one rebuild after the user pauses.
        # Without this, each keystroke remounts every option (~50 ms each at 500 rows).
        if self._search_timer is not None:
            self._search_timer.stop()
        search = event.value.strip()
        self._search_timer = self.set_timer(
            self._SEARCH_DEBOUNCE_SECONDS, lambda: self._refresh_list(search)
        )

    @on(Input.Submitted, "#picker-search")
    def _on_search_submitted(self) -> None:
        ml = self.query_one("#picker-list", ModelList)
        if ml.option_count:
            ml.action_select()

    @on(ModelList.Selected)
    def _on_model_list_selected(self, event: ModelList.Selected) -> None:
        event.stop()
        self.dismiss(event.row.ref)

    def action_focus_search(self) -> None:
        self.query_one("#picker-search", Input).focus()
