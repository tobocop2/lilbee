"""Modal picker for chat / embedding model selection.

Replaces the flat ``Select`` widgets in ModelBar. Opens with focus on a
search input that filters a virtualized ``ModelList`` body. Returns the
selected ref via ``dismiss``; the caller routes it through
``apply_active_model`` so the persistence path is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Input, Label, Static

from lilbee.app import engine_service
from lilbee.app.engine_choice import engine_for_ref, stored_choices
from lilbee.catalog.refs import model_format
from lilbee.catalog.types import ModelCompat
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_install_flow import choose_for_model
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    LocalCatalogRow,
)
from lilbee.cli.tui.screens.engine_picker import EnginePickerModal
from lilbee.cli.tui.widgets.model_bar import ModelOption
from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet.engines.choice import AUTO, NoEngineError
from lilbee.providers.model_ref import routes_to_native_model

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

PickerScope = Literal["chat", "embed", "vision", "rerank"]

# Sentinel ref returned by the picker when the user picks the "Browse catalog
# to download" row. apply_model_pick intercepts it and navigates to the
# Catalog focused on the role's task tab; it is never persisted as a real ref.
BROWSE_CATALOG_REF = "__browse_catalog__"

# The engine step asks only when at least this many engines can load the model file.
_MIN_ENGINES_TO_ASK = 2


def _picker_title(scope: PickerScope) -> str:
    """Return the modal heading for the requested scope."""
    if scope == "embed":
        return msg.MODEL_PICKER_TITLE_EMBED
    if scope == "vision":
        return msg.MODEL_PICKER_TITLE_VISION
    if scope == "rerank":
        return msg.MODEL_PICKER_TITLE_RERANK
    return msg.MODEL_PICKER_TITLE_CHAT


def _browse_catalog_row() -> CatalogRow:
    """The 'Browse catalog' action row appended to every picker."""
    return LocalCatalogRow(
        name=msg.MODEL_PICKER_BROWSE_CATALOG,
        task="",
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=False,
        sort_downloads=0,
        sort_size=0.0,
        ref=BROWSE_CATALOG_REF,
        backend="",
        compat=ModelCompat.SUPPORTED,
    )


def _saved_engine(ref: str) -> str | None:
    """The engine saved for the model *ref*, or None when the user saved none."""
    return stored_choices(cfg.engine_choices).get(ref)


def _engine_of(ref: str) -> tuple[str | None, bool]:
    """The engine that runs *ref* now, and whether lilbee chose it.

    The engine is None for a row that has no engine to show (a remote model or an action
    row), and "" when no engine can load the model.
    """
    if not ref or not routes_to_native_model(ref):
        return None, False
    try:
        name = str(engine_for_ref(ref, cfg).name)
    except NoEngineError:
        return "", False
    auto = _saved_engine(ref) is None and cfg.preferred_engine is EngineSetting.AUTO
    return name, auto


def _display_labels(options: list[ModelOption]) -> dict[str, str]:
    """The label of each option; a model in two formats names its format after the name."""
    formats: dict[str, set[str]] = {}
    for option in options:
        formats.setdefault(option.label, set()).add(str(model_format(option.ref)))
    return {
        option.ref: f"{option.label} ({model_format(option.ref)})"
        if len(formats[option.label]) > 1
        else option.label
        for option in options
    }


@dataclass
class _PickerOptions:
    """Bridge from the dropdown's ``ModelOption`` shape to ``CatalogRow``."""

    options: list[ModelOption]
    labels: dict[str, str] = field(init=False)
    engines: dict[str, tuple[str | None, bool]] = field(init=False)

    def __post_init__(self) -> None:
        self.labels = _display_labels(self.options)
        self.refresh_engines()

    def refresh_engines(self) -> None:
        """Read the engine of every row again, after a choice changed one."""
        self.engines = {option.ref: _engine_of(option.ref) for option in self.options}

    def to_sections(self, search: str) -> list[ModelListSection]:
        rows = [
            _option_to_row(o, self.labels[o.ref], *self.engines[o.ref])
            for o in self.options
            if _matches(o, search)
        ]
        # The browse-catalog action row is always present (filter-agnostic) so
        # the on-ramp is reachable even when the search box is empty AND when
        # it has narrowed the list to zero installed matches.
        rows.append(_browse_catalog_row())
        return [ModelListSection(heading=None, rows=rows)]


def _option_to_row(
    option: ModelOption, label: str, engine: str | None, engine_auto: bool
) -> CatalogRow:
    # Picker options are already-available models; mark SUPPORTED so the list
    # doesn't tag every row with the "?" unknown-compat indicator.
    return LocalCatalogRow(
        name=label,
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
        compat=ModelCompat.SUPPORTED,
        engine=engine,
        engine_auto=engine_auto,
    )


def _matches(option: ModelOption, search: str) -> bool:
    if not search:
        return True
    needle = search.lower()
    return needle in option.label.lower() or needle in option.ref.lower()


class ModelPickerModal(ModalScreen[str | None]):
    """Searchable model list. Returns the selected ref, or None on cancel.

    A pick that changes the engine of the model returns None at once, and the role keeps
    its old model. *pick_later* gets the ref once the new engine is saved, which can wait
    on a download. The picker has closed by then, so it owns no action that ends late.
    """

    CSS_PATH: ClassVar[str] = "model_picker.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss(None)", "Cancel", show=True),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("e", "change_engine", "Engine", show=True),
        # The search field takes the plain letter and keeps ctrl+e for the end of the line,
        # so ctrl+l reaches the engine step from there and after a search.
        Binding("ctrl+l", "change_engine", "Engine", show=False, priority=True),
        # Arrow keys reach the list even while the search input has focus, so
        # open -> arrows -> Enter works without knowing about Tab. When the
        # list itself is focused its own cursor bindings win.
        Binding("down", "list_cursor_down", "Next", show=False),
        Binding("up", "list_cursor_up", "Prev", show=False),
    ]

    app: LilbeeApp  # type: ignore[assignment]

    _SEARCH_DEBOUNCE_SECONDS = 0.08

    def __init__(
        self,
        *,
        scope: PickerScope,
        options: list[ModelOption],
        pick_later: Callable[[str], object],
    ) -> None:
        super().__init__()
        self._scope: PickerScope = scope
        self._pick_later = pick_later
        self._options = _PickerOptions(options=options)
        self._search_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-root"):
            yield Label(_picker_title(self._scope), id="picker-title")
            # No select-all on focus: a letter typed on the list goes on after the search text.
            yield Input(
                placeholder=msg.MODEL_PICKER_SEARCH_PLACEHOLDER,
                id="picker-search",
                select_on_focus=False,
            )
            yield ModelList(id="picker-list")
            yield Static(msg.MODEL_PICKER_HINT, id="picker-hint")

    def on_mount(self) -> None:
        self._refresh_list("")
        self.query_one("#picker-search", Input).focus()

    def _refresh_list(self, search: str) -> None:
        ml = self.query_one("#picker-list", ModelList)
        sections = self._options.to_sections(search)
        ml.set_rows(sections)
        # A default highlight makes Enter select the first match immediately.
        # The ever-present browse-catalog action row is not a match, so Enter
        # with no real matches stays a no-op instead of jumping to the catalog.
        has_match = any(
            row.ref != BROWSE_CATALOG_REF for section in sections for row in section.rows
        )
        ml.highlighted = 0 if has_match else None

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
        ref = event.row.ref
        if cfg.choose_engine_on_pick and self._open_engine_picker(
            ref, lambda option: self._use_with_engine(ref, option)
        ):
            return
        self.dismiss(ref)

    def _open_engine_picker(
        self, ref: str, picked: Callable[[engine_service.EngineOption], object]
    ) -> bool:
        """Open the engine picker for *ref*; False when fewer than two engines can load it.

        The saved engine, or auto when none is saved, is the highlighted row. Esc leaves
        the picker and the model list stays as it was.
        """
        if not ref or not routes_to_native_model(ref):
            return False
        options = engine_service.engine_options(ref)
        if sum(option.name != AUTO for option in options) < _MIN_ENGINES_TO_ASK:
            return False
        modal = EnginePickerModal(
            msg.MODEL_INFO_PICKER_TITLE,
            self._options.labels[ref],
            self._options.engines[ref][0] or None,
            options,
            _saved_engine(ref) or AUTO,
        )

        def answered(option: engine_service.EngineOption | None) -> None:
            if option is not None:
                picked(option)

        self.app.push_screen(modal, answered)
        return True

    def _use_with_engine(self, ref: str, option: engine_service.EngineOption) -> None:
        """Return the model; a new engine is saved first, outside the picker.

        Keeping the saved engine writes nothing and returns the model at once. A new
        engine closes the picker with no model, and the model goes to *pick_later* once
        the engine is saved. A refused download or a failed install leaves the role as it was.
        """
        if option.name == (_saved_engine(ref) or AUTO):
            self.dismiss(ref)
            return
        app, pick_later = self.app, self._pick_later
        self.dismiss(None)
        choose_for_model(app, ref, option, lambda: pick_later(ref))

    def action_change_engine(self) -> None:
        """Open the engine picker for the highlighted model; the model stays unchanged."""
        row = self.query_one("#picker-list", ModelList).highlighted_row()
        ref = row.ref if row is not None else ""
        if not self._open_engine_picker(ref, lambda option: self._save_engine(ref, option)):
            self.notify(msg.MODEL_INFO_NO_ENGINE_CHOICE)

    def _save_engine(self, ref: str, option: engine_service.EngineOption) -> None:
        if option.name != (_saved_engine(ref) or AUTO):
            choose_for_model(self.app, ref, option, self._repaint)

    def _repaint(self) -> None:
        """Show the new engines in the rows and keep the highlight where it was."""
        if not self.is_attached:
            return
        ml = self.query_one("#picker-list", ModelList)
        highlighted = ml.highlighted
        self._options.refresh_engines()
        self._refresh_list(self.query_one("#picker-search", Input).value.strip())
        ml.highlighted = highlighted

    def on_key(self, event: events.Key) -> None:
        """A character typed on the list goes on into the search field.

        `/` keeps its binding. `e` opens the engine step only while the search field is
        empty; after the user starts to type, it is a letter of the search.
        """
        search = self.query_one("#picker-search", Input)
        if not event.is_printable or event.key == "slash" or self.app.focused is search:
            return
        if event.key == "e" and not search.value:
            return
        event.stop()
        event.prevent_default()
        search.focus()
        search.cursor_position = len(search.value)
        search.insert_text_at_cursor(event.character or "")

    def action_focus_search(self) -> None:
        self.query_one("#picker-search", Input).focus()

    def action_list_cursor_down(self) -> None:
        ml = self.query_one("#picker-list", ModelList)
        ml.action_cursor_down()
        ml.focus()

    def action_list_cursor_up(self) -> None:
        ml = self.query_one("#picker-list", ModelList)
        ml.action_cursor_up()
        ml.focus()
