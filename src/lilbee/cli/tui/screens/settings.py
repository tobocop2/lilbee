"""Settings screen. Grouped, type-aware configuration editor."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, VerticalGroup, VerticalScroll
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from lilbee.app.settings import OCR_SETTING_KEYS, setting_sources
from lilbee.app.settings_map import SETTINGS_MAP, SettingDef, SettingGroup
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.browse_bindings import BROWSE_LIST_BINDINGS, browse_back_bindings
from lilbee.cli.tui.screens.profile_tab import ProfileSnapshot, ProfileTab, load_snapshot
from lilbee.cli.tui.screens.settings_widgets import (
    ADVANCED_COLLAPSIBLE_CLASS,
    ADVANCED_COLLAPSIBLE_ID_PREFIX,
    API_KEYS_GROUP,
    API_KEYS_WARNING_CLASS,
    EDITOR_ID_PREFIX,
    EDITOR_KINDS,
    LIST_ERROR_ID_PREFIX,
    LIST_ERROR_VISIBLE_CLASS,
    LIST_RESTORE_PREFIX,
    MODEL_PICKER_BUTTON_PREFIX,
    RESET_BUTTON_ID_PREFIX,
    RESET_BUTTON_LABEL,
    ROW_ID_PREFIX,
    SettingEditor,
    active_profile_name,
    config_toml_path,
    displayed_text,
    effective_value,
    group_settings,
    help_content,
    list_editor_text,
    make_editor,
    model_field_to_picker_scope,
    model_picker_label,
    picker_scope_to_task,
    select_shown_value,
    set_widget_value,
    title_content,
)
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.list_text_area import ListTextArea
from lilbee.cli.tui.widgets.model_pick import apply_model_pick
from lilbee.cli.tui.widgets.profile_line import ProfileLine, ProfileLinePill
from lilbee.core.config import cfg
from lilbee.core.config.enums import SettingSource

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.model_picker import PickerScope
    from lilbee.cli.tui.widgets.model_bar import ModelOption

log = logging.getLogger(__name__)

PROFILE_PANE_ID = "settings-tab-profile"


@dataclass(frozen=True)
class _PaneGroup:
    """One settings tab: pane id, group label, ordered settings."""

    pane_id: str
    group_name: SettingGroup
    items: list[tuple[str, SettingDef]]


class _LazyGroupBody(VerticalScroll, can_focus=False):
    """Pane-body that mounts rows on first activation; scrolls when taller than viewport."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._populated = False

    @property
    def populated(self) -> bool:
        return self._populated

    def populate(self, build: Callable[[], list[Widget]]) -> None:
        """Build and mount this pane's row widgets exactly once."""
        if self._populated:
            return
        self._populated = True
        widgets = build()
        if widgets:
            self.mount_all(widgets)


class SettingsScreen(Screen[None]):
    """Interactive settings viewer with grouped, type-aware editors."""

    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "settings.tcss"
    # Target the TabbedContent's inner Tabs strip rather than the outer
    # #settings-scroll Container -- Container can't accept focus, so on
    # mount focus would otherwise stay at None and downstream Tab-cycling
    # has nowhere to start. The Tabs widget is the canonical entry point.
    AUTO_FOCUS = "#settings-tabs Tabs"
    HELP = (
        "Browse and edit configuration.\n\n"
        "Tab / Shift+Tab move between fields, > and < jump between groups, "
        "j / k scroll and g / G jump to the top / bottom, "
        "Ctrl+R resets the focused field and Ctrl+Shift+R resets every "
        "setting, and q or Escape goes back."
    )

    # < and > are one action in two directions, so a single "Tabs" label still
    # says what both keys do. Keys that do different things get their own cell
    # or move to the help panel.
    _TAB_GROUP = Binding.Group("Tabs", compact=True)

    BINDINGS: ClassVar[list[BindingType]] = [
        *browse_back_bindings(),
        # Tab cycles editors inside the active pane and rolls over to the
        # next group tab when you Tab past the last editor (and the
        # previous group tab on shift+Tab past the first editor). Use
        # > / < to jump straight to the next / previous group tab.
        Binding("tab", "next_field_or_pane", "Next field", show=False),
        Binding("shift+tab", "prev_field_or_pane", "Prev field", show=False),
        # Direct tab cycling, mirrored from CatalogScreen. priority=True
        # so the bindings win when an editor input has focus.
        Binding(
            "less_than_sign",
            "cycle_pane(-1)",
            "Prev tab",
            show=True,
            priority=True,
            group=_TAB_GROUP,
        ),
        Binding(
            "greater_than_sign",
            "cycle_pane(1)",
            "Next tab",
            show=True,
            priority=True,
            group=_TAB_GROUP,
        ),
        # Resetting the focused setting is what this screen is for, so it keeps
        # a cell. Reset-all is a rarer, wider-reaching action and lives in help;
        # the two are different actions, so they must not share one label.
        Binding("ctrl+r", "reset_focused", "Reset", show=True),
        Binding("ctrl+shift+r", "reset_all", "Reset all", show=False),
        *BROWSE_LIST_BINDINGS,
    ]

    def __init__(self) -> None:
        super().__init__()
        # Group definitions for lazy-mount on tab activation. Indexed
        # by pane id so the activated-pane handler can look up its
        # bundle in O(1). ``_eagerly_populate`` is the pane id whose
        # body gets populated in on_mount (the active-by-default first
        # pane); the rest fill in on first activation.
        self._pane_groups: dict[str, _PaneGroup] = {}
        self._pane_ids: list[str] = [PROFILE_PANE_ID]
        self._eagerly_populate = PROFILE_PANE_ID
        # Text each editor showed the moment it was last built or refreshed
        # (a field showing a model default renders that default's text, not
        # a user value). A save handler acts only when the new value differs
        # from this, not from cfg's stored value, so an untouched field
        # showing a model default is never mistaken for an edit, and only a
        # successful save updates it. Keyed by setting key; kept in sync by
        # ``_build_setting_row`` and ``_refresh_editor``.
        self._mount_display: dict[str, str] = {}
        self._stale_source_keys: set[str] = set()

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.stable_footer import StableFooter
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        yield ProfileLine(id="profile-line")
        # Container (not VerticalScroll) here -- each tab body is itself a
        # VerticalScroll, and stacking two scrollables on the same column
        # tears the layout when the inner one wheels past its top edge
        # (bb-...-wiki-tear). Only the inner pane scrolls; the outer just
        # reserves the flex row.
        with Container(id="settings-scroll"), TabbedContent(id="settings-tabs"):
            yield TabPane(
                msg.PROFILE_TAB_LABEL,
                _LazyGroupBody(id=f"{PROFILE_PANE_ID}-body"),
                id=PROFILE_PANE_ID,
            )
            yield from self._compose_group_tabs()
        with BottomBars():
            yield TaskBar()
            yield StableFooter()

    def _compose_group_tabs(self) -> ComposeResult:
        """Yield one TabPane per setting group; bodies populate on activation."""
        for group_name, items in group_settings().items():
            pane_id = f"settings-tab-{group_name.lower().replace('-', '_')}"
            self._pane_groups[pane_id] = _PaneGroup(
                pane_id=pane_id, group_name=group_name, items=items
            )
            self._pane_ids.append(pane_id)
            yield TabPane(
                group_name,
                _LazyGroupBody(id=f"{pane_id}-body"),
                id=pane_id,
            )

    def on_mount(self) -> None:
        """Defer first-pane content mount until after the screen has painted.

        ``_populate_pane`` calls ``mount_all`` for ~25 editor widgets which
        triggers a full Textual layout pass; running it inside ``on_mount``
        adds that pass to the screen-switch latency budget. ``call_after_refresh``
        moves it to the next event-loop tick so the user sees the empty pane
        skeleton immediately and the rows hydrate one frame later.
        """
        self.app.settings_changed_signal.subscribe(self, self._on_setting_changed)
        # The first pane is the one TabbedContent activates by default; populate
        # it eagerly so Settings shows content on first paint.
        self.call_after_refresh(self._populate_pane, self._eagerly_populate)

    def on_screen_resume(self) -> None:
        """Rescan the profiles, which the CLI or another terminal may have changed."""
        self.reload_profile()

    def _on_setting_changed(self, change: tuple[str, object]) -> None:
        """Queue the changed key's source pill and editor, and the profile state, for a refresh."""
        key, _value = change
        if not self._stale_source_keys:
            self.call_after_refresh(self._refresh_changed_rows)
        self._stale_source_keys.add(key)

    def _refresh_changed_rows(self) -> None:
        """Repaint every queued row's title and editor from one read of the setting sources."""
        keys, self._stale_source_keys = self._stale_source_keys, set()
        sources = setting_sources()
        profile_name = active_profile_name()
        for key in keys:
            for title in self.query(f"#{ROW_ID_PREFIX}{key} > .setting-title").results(Static):
                title.update(title_content(key, SETTINGS_MAP[key], sources[key], profile_name))
            self._sync_editor(key)
        self.reload_profile()

    def _sync_editor(self, key: str) -> None:
        """Show cfg's value in *key*'s editor, unless the user is typing in it."""
        defn = SETTINGS_MAP.get(key)
        editor = self.query(f"#{EDITOR_ID_PREFIX}{key}")
        if defn is None or not editor or editor.first().has_focus_within:
            return
        self._refresh_editor(key, defn, getattr(cfg, key))

    @work(thread=True, exclusive=True, group="profile-load", exit_on_error=False)
    def reload_profile(self) -> None:
        """Load the profile state off the loop, then show it on the line and the Profile tab."""
        call_from_thread(self, self._show_profile, load_snapshot())

    def _show_profile(self, snapshot: ProfileSnapshot) -> None:
        self.query_one(ProfileLine).show(snapshot)
        for tab in self.query(ProfileTab):
            tab.show(snapshot)

    @on(ProfileTab.Stale)
    def _on_profile_stale(self) -> None:
        self.reload_profile()

    @on(ProfileLinePill.Jump)
    def _on_profile_jump(self) -> None:
        self.show_profile_tab()

    def show_profile_tab(self) -> None:
        """Activate the Profile tab and put focus on its dropdown."""
        self.query_one("#settings-tabs", TabbedContent).active = PROFILE_PANE_ID
        self._populate_pane(PROFILE_PANE_ID)
        self.call_after_refresh(self._focus_profile_select)

    def _focus_profile_select(self) -> None:
        for tab in self.query(ProfileTab):
            tab.focus_select()

    @on(TabbedContent.TabActivated)
    def _on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Populate the activated pane's body on first activation."""
        pane = event.pane
        if pane is None or pane.id is None:
            return
        self._populate_pane(pane.id)
        if pane.id == PROFILE_PANE_ID:
            self.reload_profile()

    def populate_all_panes(self) -> None:
        """Force every tab body to populate now (test/agent helper)."""
        for pane_id in self._pane_ids:
            self._populate_pane(pane_id)

    def _populate_pane(self, pane_id: str) -> None:
        """Populate a pane's body if known and the body widget is mounted."""
        build = self._pane_builder(pane_id)
        if build is None:
            return
        try:
            body = self.query_one(f"#{pane_id}-body", _LazyGroupBody)
        except Exception:
            log.debug("pane body %s not yet mounted", pane_id, exc_info=True)
            return
        body.populate(build)

    def _pane_builder(self, pane_id: str) -> Callable[[], list[Widget]] | None:
        """The function that builds *pane_id*'s body widgets, or None for an unknown pane."""
        if pane_id == PROFILE_PANE_ID:
            return lambda: [ProfileTab()]
        group = self._pane_groups.get(pane_id)
        return None if group is None else lambda: self._build_pane_widgets(group)

    def _build_pane_widgets(self, group: _PaneGroup) -> list[Widget]:
        """Return the body widgets for one settings tab.

        Advanced settings are folded into one collapsed section at the
        bottom, after every regular row.
        """
        widgets: list[Widget] = []
        if group.group_name == API_KEYS_GROUP:
            widgets.append(
                Static(
                    msg.SETTINGS_API_KEYS_WARNING.format(path=config_toml_path()),
                    classes=API_KEYS_WARNING_CLASS,
                )
            )
        sources = setting_sources()
        profile_name = active_profile_name()
        basic = [(key, defn) for key, defn in group.items if not defn.advanced]
        advanced = [(key, defn) for key, defn in group.items if defn.advanced]
        for key, defn in basic:
            widgets.append(self._build_setting_row(key, defn, sources[key], profile_name))
        if advanced:
            rows = [
                self._build_setting_row(key, defn, sources[key], profile_name)
                for key, defn in advanced
            ]
            widgets.append(self._build_advanced_section(group.pane_id, rows))
        return widgets

    def _build_advanced_section(self, pane_id: str, rows: list[VerticalGroup]) -> Collapsible:
        """One collapsed section holding every advanced row for a tab."""
        title = msg.SETTINGS_ADVANCED_TITLE.format(count=len(rows))
        return Collapsible(
            *rows,
            title=title,
            collapsed=True,
            id=f"{ADVANCED_COLLAPSIBLE_ID_PREFIX}{pane_id}",
            classes=ADVANCED_COLLAPSIBLE_CLASS,
        )

    def _build_setting_row(
        self, key: str, defn: SettingDef, source: SettingSource, profile_name: str | None
    ) -> VerticalGroup:
        """Construct one setting row with its title, help, editor, and reset."""
        title = Static(title_content(key, defn, source, profile_name), classes="setting-title")
        help_widget = Static(help_content(key, defn), classes="setting-help")
        children: list[Widget] = [title, help_widget]
        if key in model_field_to_picker_scope():
            children.append(self._build_model_picker_row(key))
        elif defn.writable:
            editor = make_editor(key, defn)
            self._mount_display[key] = self._mount_baseline(key, defn, editor)
            editor_row = Horizontal(
                editor,
                Button(
                    RESET_BUTTON_LABEL,
                    id=f"{RESET_BUTTON_ID_PREFIX}{key}",
                    classes="setting-reset-button",
                    tooltip=msg.SETTINGS_RESET_TO_DEFAULT_TOOLTIP,
                ),
                classes="setting-editor-row",
            )
            children.append(editor_row)
        return VerticalGroup(
            *children,
            classes="setting-row",
            id=f"{ROW_ID_PREFIX}{key}",
        )

    @staticmethod
    def _mount_baseline(key: str, defn: SettingDef, editor: Collapsible | SettingEditor) -> str:
        """The text a freshly built editor will show once mounted.

        A ``Select``'s ``value`` reactive is not populated from its
        constructor kwarg until the widget mounts, so reading it off
        *editor* here (right after construction, before it is ever mounted)
        would see the pre-mount default rather than what it will display;
        this recomputes the same choice match ``make_select`` used instead.
        """
        # A list setting's editor is a Collapsible that holds the text area.
        if isinstance(editor, Collapsible):
            return list_editor_text(key)
        if defn.choices:
            return select_shown_value(defn, effective_value(key))
        return displayed_text(editor)

    def _build_model_picker_row(self, key: str) -> Horizontal:
        """A button-style row that opens the same ModelPickerModal as the chat bar."""
        return Horizontal(
            Button(
                model_picker_label(key),
                id=f"{MODEL_PICKER_BUTTON_PREFIX}{key}",
                classes="setting-model-picker-button",
            ),
            classes="setting-editor-row",
        )

    @on(Input.Submitted, ".setting-editor")
    @on(Input.Blurred, ".setting-editor")
    def _on_input_save(self, event: Input.Submitted | Input.Blurred) -> None:
        """Save string/number input on submit or blur, but only if it changed."""
        name = event.input.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        raw = event.value.strip()
        if self._mount_display.get(name) == raw:
            return
        if self._persist_value(name, defn, raw):
            self._mount_display[name] = raw

    @on(ListTextArea.Blurred, ".setting-multiline-editor")
    def _on_multiline_save(self, event: ListTextArea.Blurred) -> None:
        """Save multi-line string settings (system prompts) on blur, but only if changed."""
        ta = event.control
        name = ta.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        raw = ta.text
        if self._mount_display.get(name) == raw:
            return
        if self._persist_value(name, defn, raw):
            self._mount_display[name] = raw

    @on(Checkbox.Changed, ".setting-editor")
    def _on_checkbox_save(self, event: Checkbox.Changed) -> None:
        """Save boolean on toggle, but only if it still differs from the last saved value."""
        name = event.checkbox.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        value = str(event.checkbox.value)
        if self._mount_display.get(name) == value:
            return
        if self._persist_value(name, defn, value):
            self._mount_display[name] = value

    @on(Select.Changed, ".setting-editor")
    def _on_select_save(self, event: Select.Changed) -> None:
        """Save select choice on change, but only if it differs from the mount baseline."""
        name = event.select.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        value = str(event.value) if event.value != Select.BLANK else ""
        if self._mount_display.get(name) == value:
            return
        if self._persist_value(name, defn, value):
            self._mount_display[name] = value

    def _persist_value(self, key: str, defn: SettingDef, raw: str) -> bool:
        """Parse, apply, and persist a setting value. Returns whether it succeeded.

        Success is silent; a parse or apply error toasts and returns False.
        """
        try:
            parsed = self._parse_value(defn, raw)
            self.app.set_setting(key, parsed)
            self._refresh_help(key, defn)
            return True
        except (ValueError, TypeError) as exc:
            self.notify(msg.SETTINGS_INVALID_VALUE.format(error=exc), severity="error")
            return False

    def _parse_value(self, defn: SettingDef, raw: str) -> object:
        """Convert a raw string to the setting's target type."""
        if defn.nullable and raw.lower() in ("none", "null", ""):
            return None
        if defn.type is bool:
            return raw.lower() in ("true", "1", "yes", "on")
        if defn.type is list:
            return [line.strip() for line in raw.split("\n") if line.strip()]
        return defn.type(raw)

    @staticmethod
    def _validate_regex_list(lines: list[str]) -> tuple[int, str] | None:
        """Return the 1-indexed line number and error for the first bad regex, or None."""
        for i, line in enumerate(lines, 1):
            try:
                re.compile(line)
            except re.error as exc:
                return (i, str(exc))
        return None

    @on(ListTextArea.Blurred, ".setting-list-editor")
    def _on_list_blur_save(self, event: ListTextArea.Blurred) -> None:
        """Validate and save list values when a ListTextArea loses focus, but only if changed."""
        ta = event.control
        key = ta.name
        if key is None:
            return
        defn = SETTINGS_MAP.get(key)
        if defn is None:
            return
        raw = ta.text
        if self._mount_display.get(key) == raw:
            return
        parsed = self._parse_value(defn, raw)
        assert isinstance(parsed, list)  # noqa: S101 -- mypy narrowing, defn.type is list above
        err = self._validate_regex_list(parsed) if defn.validate_regex else None
        error_widget = self.query_one(f"#{LIST_ERROR_ID_PREFIX}{key}", Static)
        if err is not None:
            line_no, err_text = err
            error_widget.update(
                msg.SETTINGS_LIST_EDITOR_INVALID_REGEX.format(n=line_no, error=err_text)
            )
            error_widget.add_class(LIST_ERROR_VISIBLE_CLASS)
            return
        error_widget.remove_class(LIST_ERROR_VISIBLE_CLASS)
        if self._persist_value(key, defn, raw):
            self._mount_display[key] = raw
        self._refresh_list_title(key, len(parsed))

    @on(Button.Pressed, ".setting-list-restore")
    def _on_list_restore(self, event: Button.Pressed) -> None:
        """Reset a list setting."""
        btn_id = event.button.id
        if btn_id is None or not btn_id.startswith(LIST_RESTORE_PREFIX):
            return
        key = btn_id.removeprefix(LIST_RESTORE_PREFIX)
        if SETTINGS_MAP.get(key) is None or not self._reset_keys([key]):
            return
        error_widget = self.query_one(f"#{LIST_ERROR_ID_PREFIX}{key}", Static)
        error_widget.remove_class(LIST_ERROR_VISIBLE_CLASS)
        self._refresh_list_title(key, len(getattr(cfg, key)))

    def _refresh_list_title(self, key: str, count: int) -> None:
        """Update the Collapsible title to reflect the current line count."""
        try:
            collapsible = self.query_one(f"#collapsible-{key}", Collapsible)
            collapsible.title = msg.SETTINGS_LIST_EDITOR_TITLE.format(key=key, count=count)
        except Exception:
            log.debug("Failed to refresh collapsible title for %s", key, exc_info=True)

    def _refresh_help(self, key: str, defn: SettingDef) -> None:
        """Update the help text after a value change; an OCR key refreshes every OCR row."""
        if key not in OCR_SETTING_KEYS:
            self._update_help_row(key, defn)
            return
        for ocr_key in OCR_SETTING_KEYS:
            self._update_help_row(ocr_key, SETTINGS_MAP[ocr_key])

    def _update_help_row(self, key: str, defn: SettingDef) -> None:
        """Re-render one row's help text."""
        try:
            row = self.query_one(f"#{ROW_ID_PREFIX}{key}", VerticalGroup)
            help_widget = row.query_one(".setting-help", Static)
            help_widget.update(help_content(key, defn))
        except Exception:
            log.debug("Failed to refresh help for %s", key, exc_info=True)

    @on(Button.Pressed, ".setting-reset-button")
    def _on_reset_pressed(self, event: Button.Pressed) -> None:
        """Handle the small reset button embedded in each writable row."""
        button_id = event.button.id
        if button_id is None or not button_id.startswith(RESET_BUTTON_ID_PREFIX):
            return
        key = button_id[len(RESET_BUTTON_ID_PREFIX) :]
        self._reset_to_default(key)

    @on(Button.Pressed, ".setting-model-picker-button")
    def _on_model_picker_pressed(self, event: Button.Pressed) -> None:
        """Open ModelPickerModal for the model field this button represents."""
        button_id = event.button.id
        if button_id is None or not button_id.startswith(MODEL_PICKER_BUTTON_PREFIX):
            return
        key = button_id[len(MODEL_PICKER_BUTTON_PREFIX) :]
        scope = model_field_to_picker_scope().get(key)
        if scope is None:
            return
        self._discover_then_open_picker(key, scope)

    @work(thread=True, exit_on_error=False)
    def _discover_then_open_picker(self, key: str, scope: PickerScope) -> None:
        """Discover installed models off the UI thread, then push the picker.

        ``classify_installed_models_full`` probes the native registry,
        Ollama (HTTP), and litellm provider lists. Running it on the
        event loop blocks paint for hundreds of ms; the chat-bar uses
        the same worker pattern.
        """
        from lilbee.cli.tui.thread_safe import call_from_thread
        from lilbee.cli.tui.widgets.model_bar import classify_installed_models_full

        task = picker_scope_to_task(scope)
        buckets = classify_installed_models_full()
        options = list(buckets.get(task, []))
        call_from_thread(self, self._push_model_picker, key, scope, options)

    def _push_model_picker(self, key: str, scope: PickerScope, options: list[ModelOption]) -> None:
        """Push ModelPickerModal once the worker has resolved options."""
        from lilbee.cli.tui.screens.model_picker import ModelPickerModal
        from lilbee.cli.tui.widgets.model_bar import ModelOption

        # Bail out if the user navigated away from Settings while the
        # discovery worker was still running; otherwise we'd push the
        # modal onto whatever screen is now on top.
        if not self.is_mounted:
            return
        if not options:
            options = [ModelOption(label=msg.MODEL_VALUE_NONE, ref="")]
        # Nullable model fields (vision_model, reranker_model) need an
        # explicit "disable this model" pick. The picker's empty-input
        # cancel returns None; this row returns "" so the dismiss
        # handler can distinguish "cancel" from "set to none".
        defn = SETTINGS_MAP.get(key)
        if defn is not None and defn.nullable:
            options = [
                ModelOption(label=msg.MODEL_PICKER_DISABLE_LABEL, ref=""),
                *options,
            ]
        self.app.push_screen(
            ModelPickerModal(scope=scope, options=options),
            lambda ref: self._on_model_picker_dismissed(key, ref),
        )

    def _on_model_picker_dismissed(self, key: str, ref: str | None) -> None:
        """Persist the picker selection, refresh the button, and reload the role's server."""
        # apply_model_pick owns the reload (off the event loop in a worker); the
        # on_done callback only repaints the button once the swap is applied.
        apply_model_pick(self, key=key, ref=ref, on_done=lambda: self._after_model_pick(key))

    def _after_model_pick(self, key: str) -> None:
        """Refresh the picker button after a swap.

        The role reload is owned by ``apply_model_pick`` (it runs off the event
        loop in a worker), so this on_done only repaints the label. Reloading
        here too would double the fleet restart AND block the UI on the main
        thread, which is the freeze this path is meant to avoid.
        """
        self._refresh_picker_button(key)
        self._refresh_help(key, SETTINGS_MAP[key])

    def _refresh_picker_button(self, key: str) -> None:
        try:
            button = self.query_one(f"#{MODEL_PICKER_BUTTON_PREFIX}{key}", Button)
            button.label = model_picker_label(key)
        except Exception:
            log.debug("Failed to refresh model picker label for %s", key, exc_info=True)

    def action_reset_all(self) -> None:
        """Bound to Ctrl+Shift+R; opens the destructive-confirm dialog."""
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        self.app.push_screen(
            ConfirmDialog(
                title=msg.SETTINGS_RESET_ALL_CONFIRM_TITLE,
                message=msg.SETTINGS_RESET_ALL_CONFIRM_MESSAGE,
            ),
            self._on_reset_all_confirmed,
        )

    def _on_reset_all_confirmed(self, confirmed: bool | None) -> None:
        """Reset every writable setting in one batch."""
        if not confirmed:
            return

        writable = [key for key, defn in SETTINGS_MAP.items() if defn.writable]
        if self._reset_keys(writable, skip_unresettable=True):
            self.notify(msg.SETTINGS_RESET_ALL_SUCCESS)

    def action_reset_focused(self) -> None:
        """Reset the setting whose row holds focus."""
        focused = self.focused
        if focused is None:
            return
        for ancestor in focused.ancestors_with_self:
            ancestor_id = getattr(ancestor, "id", None)
            if ancestor_id and ancestor_id.startswith(ROW_ID_PREFIX):
                key = ancestor_id[len(ROW_ID_PREFIX) :]
                self._reset_to_default(key)
                return

    def _reset_to_default(self, key: str) -> None:
        """Reset one writable setting."""
        defn = SETTINGS_MAP.get(key)
        if defn is None or not defn.writable:
            return
        self._reset_keys([key])

    def _reset_keys(self, keys: list[str], *, skip_unresettable: bool = False) -> bool:
        """Reset *keys*, then show each one's resolved value. Returns whether it succeeded."""
        try:
            reset = self.app.reset_settings(keys, skip_unresettable=skip_unresettable)
        except (ValueError, OSError) as exc:
            self.notify(msg.SETTINGS_INVALID_VALUE.format(error=exc), severity="error")
            return False
        for key in reset:
            defn = SETTINGS_MAP[key]
            self._refresh_editor(key, defn, getattr(cfg, key))
            self._refresh_help(key, defn)
        return True

    def _refresh_editor(self, key: str, defn: SettingDef, value: object) -> None:
        """Update the editor widget to reflect a new value (e.g. after reset)."""
        try:
            widget = self.query_one(f"#{EDITOR_ID_PREFIX}{key}")
        except Exception:
            log.debug("Failed to refresh editor for %s", key, exc_info=True)
            return
        # prevent() only covers a Changed message this call posts, not one already queued.
        with widget.prevent(Checkbox.Changed):
            set_widget_value(widget, value)
        # Every writable key with an editor is seeded into _mount_display at row
        # construction, so a key that is absent has no baseline to refresh.
        if key in self._mount_display and isinstance(widget, EDITOR_KINDS):
            self._mount_display[key] = displayed_text(widget)

    def action_go_back(self) -> None:
        self.app.go_back()

    def _active_pane_body(self) -> _LazyGroupBody | None:
        """Resolve the currently-active settings tab body (a VerticalScroll).

        j/k/g/G key actions scroll this body directly because the outer
        ``#settings-scroll`` is a Container, not a scroller -- one column
        of scrolling per screen, the active tab's pane.
        """
        try:
            tabs = self.query_one("#settings-tabs", TabbedContent)
        except Exception:
            return None
        active = tabs.active
        if not active:
            return None
        try:
            return self.query_one(f"#{active}-body", _LazyGroupBody)
        except Exception:
            return None

    def action_cursor_down(self) -> None:
        if (body := self._active_pane_body()) is not None:
            body.scroll_down()

    def action_cursor_up(self) -> None:
        if (body := self._active_pane_body()) is not None:
            body.scroll_up()

    def action_jump_top(self) -> None:
        if (body := self._active_pane_body()) is not None:
            body.scroll_home()

    def action_jump_bottom(self) -> None:
        if (body := self._active_pane_body()) is not None:
            body.scroll_end()

    def action_next_field_or_pane(self) -> None:
        """Tab inside a pane; on overflow advance to the next group tab."""
        self._move_focus_within_pane(direction=1)

    def action_prev_field_or_pane(self) -> None:
        """Shift+Tab inside a pane; on underflow retreat to the previous group tab."""
        self._move_focus_within_pane(direction=-1)

    def action_cycle_pane(self, delta: int) -> None:
        """Step the active settings tab by *delta*, wrapping around the strip.

        Shortcut for users who don't want to Tab through every field to
        reach the next group. Mirrors CatalogScreen.action_cycle_tab. A
        focused editor is not a concern here: a focused Input/TextArea
        consumes printable keys before even priority bindings see them
        (verified empirically), so < and > always type into editors.
        """
        try:
            tabs = self.query_one("#settings-tabs", TabbedContent)
        except Exception:
            return
        pane_ids = self._pane_ids
        try:
            current = pane_ids.index(tabs.active)
        except ValueError:
            current = 0
        next_id = pane_ids[(current + delta) % len(pane_ids)]
        if tabs.active != next_id:
            tabs.active = next_id

    def _focus_adjacent(self, direction: int) -> None:
        """Move focus to the next/previous widget app-wide (direction 1 / -1)."""
        if direction == 1:
            self.app.action_focus_next()
        else:
            self.app.action_focus_previous()

    def _move_focus_within_pane(self, *, direction: int) -> None:
        focused = self.app.focused
        tabs = self.query_one("#settings-tabs", TabbedContent)
        active_pane_id = tabs.active
        try:
            body = self.query_one(f"#{active_pane_id}-body", _LazyGroupBody)
        except Exception:
            self._focus_adjacent(direction)
            return
        focusables = [w for w in body.query("*") if w.focusable]
        if not focusables or focused is None or focused not in focusables:
            self._focus_adjacent(direction)
            return
        index = focusables.index(focused)
        next_index = index + direction
        if 0 <= next_index < len(focusables):
            focusables[next_index].focus()
            return
        # At the boundary: advance to the next/previous pane.
        pane_ids = self._pane_ids
        if active_pane_id not in pane_ids:
            return
        target_index = (pane_ids.index(active_pane_id) + direction) % len(pane_ids)
        target_pane = pane_ids[target_index]
        tabs.active = target_pane
        self._populate_pane(target_pane)
        # Park focus on the first/last field of the new pane so the next
        # Tab keeps moving in the same direction.
        self.call_after_refresh(self._focus_pane_edge, target_pane, direction)

    def _focus_pane_edge(self, pane_id: str, direction: int) -> None:
        try:
            body = self.query_one(f"#{pane_id}-body", _LazyGroupBody)
        except Exception:
            return
        focusables = [w for w in body.query("*") if w.focusable]
        if not focusables:
            return
        focusables[0 if direction == 1 else -1].focus()
