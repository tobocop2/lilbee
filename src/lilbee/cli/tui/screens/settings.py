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

from lilbee.app.services import get_services
from lilbee.cli.settings_map import SETTINGS_MAP, SettingDef, get_default
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.settings_widgets import (
    API_KEYS_GROUP,
    API_KEYS_WARNING_CLASS,
    EDITOR_ID_PREFIX,
    LIST_ERROR_ID_PREFIX,
    LIST_ERROR_VISIBLE_CLASS,
    LIST_RESTORE_PREFIX,
    MODEL_PICKER_BUTTON_PREFIX,
    RESET_BUTTON_ID_PREFIX,
    RESET_BUTTON_LABEL,
    ROW_ID_PREFIX,
    config_toml_path,
    group_settings,
    help_content,
    make_editor,
    model_field_to_picker_scope,
    model_picker_label,
    picker_scope_to_task,
    set_widget_value,
    stringify_default,
    title_content,
)
from lilbee.cli.tui.widgets.list_text_area import ListTextArea
from lilbee.core import settings
from lilbee.core.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS, cfg
from lilbee.providers.worker.transport import WorkerRole

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.model_picker import PickerScope
    from lilbee.cli.tui.widgets.model_bar import ModelOption

log = logging.getLogger(__name__)


_MODEL_KEY_TO_WORKER_ROLE: dict[str, WorkerRole] = {
    "chat_model": WorkerRole.CHAT,
    "embedding_model": WorkerRole.EMBED,
    "reranker_model": WorkerRole.RERANK,
    "vision_model": WorkerRole.VISION,
}
"""Picker key -> worker pool role. Lets the Settings picker respawn the right
worker after a swap so the new ref actually takes effect on the next call.
"""


@dataclass(frozen=True)
class _PaneGroup:
    """One settings tab: pane id, group label, ordered settings."""

    pane_id: str
    group_name: str
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
        "Use / to search, Enter to confirm, Escape to return to the list."
    )

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        # Tab cycles editors inside the active pane and rolls over to the
        # next group tab when you Tab past the last editor (and the
        # previous group tab on shift+Tab past the first editor). Use
        # > / < to jump straight to the next / previous group tab.
        Binding("tab", "next_field_or_pane", "Next field", show=True),
        Binding("shift+tab", "prev_field_or_pane", "Prev field", show=True),
        # Direct tab cycling, mirrored from CatalogScreen. priority=True
        # so the bindings win when an editor input has focus.
        Binding("greater_than_sign", "cycle_pane(1)", "Next tab", show=True, priority=True),
        Binding("less_than_sign", "cycle_pane(-1)", "Prev tab", show=True, priority=True),
        Binding("ctrl+r", "reset_focused", "Reset field", show=False),
        Binding("ctrl+shift+r", "reset_all", "Reset all", show=True),
        Binding("j", "scroll_down", "Down", show=False),
        Binding("k", "scroll_up", "Up", show=False),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("G", "scroll_end", "End", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        # Group definitions for lazy-mount on tab activation. Indexed
        # by pane id so the activated-pane handler can look up its
        # bundle in O(1). ``_eagerly_populate`` is the pane id whose
        # body gets populated in on_mount (the active-by-default first
        # pane); the rest fill in on first activation.
        self._pane_groups: dict[str, _PaneGroup] = {}
        self._eagerly_populate: str | None = None

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        # Container (not VerticalScroll) here -- each tab body is itself a
        # VerticalScroll, and stacking two scrollables on the same column
        # tears the layout when the inner one wheels past its top edge
        # (bb-...-wiki-tear). Only the inner pane scrolls; the outer just
        # reserves the flex row.
        with Container(id="settings-scroll"), TabbedContent(id="settings-tabs"):
            yield from self._compose_group_tabs()
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def _compose_group_tabs(self) -> ComposeResult:
        """Yield one TabPane per setting group; bodies populate on activation."""
        first = True
        for group_name, items in group_settings().items():
            pane_id = f"settings-tab-{group_name.lower().replace('-', '_')}"
            self._pane_groups[pane_id] = _PaneGroup(
                pane_id=pane_id, group_name=group_name, items=items
            )
            yield TabPane(
                group_name,
                _LazyGroupBody(id=f"{pane_id}-body"),
                id=pane_id,
            )
            # The first pane is the one TabbedContent activates by
            # default; populate it eagerly so a user landing on
            # Settings sees content on first paint instead of an empty
            # active pane that fills in one frame later.
            if first:
                first = False
                self._eagerly_populate = pane_id

    def on_mount(self) -> None:
        """Defer first-pane content mount until after the screen has painted.

        ``_populate_pane`` calls ``mount_all`` for ~25 editor widgets which
        triggers a full Textual layout pass; running it inside ``on_mount``
        adds that pass to the screen-switch latency budget. ``call_after_refresh``
        moves it to the next event-loop tick so the user sees the empty pane
        skeleton immediately and the rows hydrate one frame later.
        """
        if self._eagerly_populate is not None:
            self.call_after_refresh(self._populate_pane, self._eagerly_populate)

    @on(TabbedContent.TabActivated)
    def _on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Populate the activated pane's body on first activation."""
        pane = event.pane
        if pane is None or pane.id is None:
            return
        self._populate_pane(pane.id)

    def populate_all_panes(self) -> None:
        """Force every tab body to populate now (test/agent helper)."""
        for pane_id in self._pane_groups:
            self._populate_pane(pane_id)

    def _populate_pane(self, pane_id: str) -> None:
        """Populate a pane's body if known and the body widget is mounted."""
        group = self._pane_groups.get(pane_id)
        if group is None:
            return
        try:
            body = self.query_one(f"#{pane_id}-body", _LazyGroupBody)
        except Exception:
            log.debug("pane body %s not yet mounted", pane_id, exc_info=True)
            return
        body.populate(lambda: self._build_pane_widgets(group))

    def _build_pane_widgets(self, group: _PaneGroup) -> list[Widget]:
        """Return the body widgets for one settings tab."""
        widgets: list[Widget] = []
        if group.group_name == API_KEYS_GROUP:
            widgets.append(
                Static(
                    msg.SETTINGS_API_KEYS_WARNING.format(path=config_toml_path()),
                    classes=API_KEYS_WARNING_CLASS,
                )
            )
        for key, defn in group.items:
            widgets.append(self._build_setting_row(key, defn))
        return widgets

    def _build_setting_row(self, key: str, defn: SettingDef) -> VerticalGroup:
        """Construct one setting row with its title, help, editor, and reset."""
        title = Static(title_content(key, defn), classes="setting-title")
        help_widget = Static(help_content(key, defn), classes="setting-help")
        children: list[Widget] = [title, help_widget]
        if key in model_field_to_picker_scope():
            children.append(self._build_model_picker_row(key))
        elif defn.writable:
            editor_row = Horizontal(
                make_editor(key, defn),
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
        """Save string/number input on submit or blur."""
        name = event.input.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        raw = event.value.strip()
        current = str(getattr(cfg, name, ""))
        if raw == current:
            return
        self._persist_value(name, defn, raw)

    @on(ListTextArea.Blurred, ".setting-multiline-editor")
    def _on_multiline_save(self, event: ListTextArea.Blurred) -> None:
        """Save multi-line string settings (system prompts) on blur."""
        ta = event.control
        name = ta.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        raw = ta.text
        current = str(getattr(cfg, name, ""))
        if raw == current:
            return
        self._persist_value(name, defn, raw)

    @on(Checkbox.Changed, ".setting-editor")
    def _on_checkbox_save(self, event: Checkbox.Changed) -> None:
        """Save boolean on toggle."""
        name = event.checkbox.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        self._persist_value(name, defn, str(event.checkbox.value))

    @on(Select.Changed, ".setting-editor")
    def _on_select_save(self, event: Select.Changed) -> None:
        """Save select choice on change."""
        name = event.select.name
        if name is None:
            return
        defn = SETTINGS_MAP.get(name)
        if defn is None:
            return
        value = str(event.value) if event.value != Select.BLANK else ""
        current = str(getattr(cfg, name, ""))
        if value == current:
            return
        self._persist_value(name, defn, value)

    def _persist_value(self, key: str, defn: SettingDef, raw: str, *, quiet: bool = False) -> None:
        """Parse, apply, and persist a setting value.

        No success toast: the editor already shows the new value and the
        write is silently persisted. Tab-cycling between sub-tabs blurs
        the focused input, which fires Input.Blurred -> _on_input_save
        en masse; one toast per blur is just noise. Errors still toast
        so the user sees why a value didn't take.
        """
        try:
            parsed = self._parse_value(defn, raw)
            # set_setting handles theme live-apply, signal publish, etc.
            self.app.set_setting(key, parsed)
            self._refresh_help(key, defn)
            _ = quiet  # accepted for API compatibility; success path is now always silent
        except (ValueError, TypeError) as exc:
            self.notify(msg.SETTINGS_INVALID_VALUE.format(error=exc), severity="error")

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
        """Validate and save list values when a ListTextArea loses focus."""
        ta = event.control
        key = ta.name
        if key is None:
            return
        defn = SETTINGS_MAP.get(key)
        if defn is None:
            return
        raw = ta.text
        parsed = self._parse_value(defn, raw)
        assert isinstance(parsed, list)  # noqa: S101 -- mypy narrowing, defn.type is list above
        err = self._validate_regex_list(parsed)
        error_widget = self.query_one(f"#{LIST_ERROR_ID_PREFIX}{key}", Static)
        if err is not None:
            line_no, err_text = err
            error_widget.update(
                msg.SETTINGS_LIST_EDITOR_INVALID_REGEX.format(n=line_no, error=err_text)
            )
            error_widget.add_class(LIST_ERROR_VISIBLE_CLASS)
            return
        error_widget.remove_class(LIST_ERROR_VISIBLE_CLASS)
        self._persist_value(key, defn, raw)
        self._refresh_list_title(key, len(parsed))

    @on(Button.Pressed, ".setting-list-restore")
    def _on_list_restore(self, event: Button.Pressed) -> None:
        """Restore defaults for a LIST_COLLAPSED setting."""
        btn_id = event.button.id
        if btn_id is None or not btn_id.startswith(LIST_RESTORE_PREFIX):
            return
        key = btn_id.removeprefix(LIST_RESTORE_PREFIX)
        defn = SETTINGS_MAP.get(key)
        if defn is None:
            return
        defaults = list(DEFAULT_CRAWL_EXCLUDE_PATTERNS)
        text = "\n".join(defaults)
        ta = self.query_one(f"#ed-{key}", ListTextArea)
        ta.load_text(text)
        self._persist_value(key, defn, text)
        error_widget = self.query_one(f"#{LIST_ERROR_ID_PREFIX}{key}", Static)
        error_widget.remove_class(LIST_ERROR_VISIBLE_CLASS)
        self._refresh_list_title(key, len(defaults))

    def _refresh_list_title(self, key: str, count: int) -> None:
        """Update the Collapsible title to reflect the current line count."""
        try:
            collapsible = self.query_one(f"#collapsible-{key}", Collapsible)
            collapsible.title = msg.SETTINGS_LIST_EDITOR_TITLE.format(key=key, count=count)
        except Exception:
            log.debug("Failed to refresh collapsible title for %s", key, exc_info=True)

    def _refresh_help(self, key: str, defn: SettingDef) -> None:
        """Update the help text after a value change."""
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
        """Persist the picker selection and refresh the button label.

        ``ref is None`` means the user cancelled (Esc); leave the field
        alone. ``ref == ""`` for a nullable field means the user picked
        the explicit "disabled" row; clear the field. Any other value is
        a real model ref. Embedding-model swaps against a populated store
        route through a confirm modal first so the user is not surprised
        by the rebuild requirement.
        """
        if ref is None:
            return
        defn = SETTINGS_MAP.get(key)
        if not ref and (defn is None or not defn.nullable):
            return
        if key == "embedding_model" and ref:
            self._maybe_confirm_embedding_swap(key, ref)
            return
        self._apply_picker_choice(key, ref, True)

    @work(thread=True, name="settings_has_chunks_check", exit_on_error=False)
    def _maybe_confirm_embedding_swap(self, key: str, ref: str) -> None:
        """Check store chunk-count off the UI thread; confirm-modal if non-empty.

        ``store.has_chunks`` hits LanceDB. On Windows with Defender that
        read used to stall the picker-dismiss handler on the UI thread;
        run it in a worker and dispatch back to push the confirm dialog
        or apply the choice directly.
        """
        from lilbee.cli.tui.thread_safe import call_from_thread

        if get_services().store.has_chunks():
            call_from_thread(self, self._push_embed_swap_confirm, key, ref)
        else:
            call_from_thread(self, self._apply_picker_choice, key, ref, True)

    def _push_embed_swap_confirm(self, key: str, ref: str) -> None:
        """Show the embed-swap confirm dialog from the UI thread."""
        if not self.is_mounted:
            return
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        self.app.push_screen(
            ConfirmDialog(msg.EMBED_SWAP_CONFIRM_TITLE, msg.EMBED_SWAP_CONFIRM_MESSAGE),
            lambda confirmed: self._apply_picker_choice(key, ref, confirmed),
        )

    def _apply_picker_choice(self, key: str, ref: str, confirmed: bool | None) -> None:
        """Commit the picker choice or notify cancel; ``confirmed`` mirrors ConfirmDialog."""
        if not confirmed:
            self.app.notify(msg.EMBED_SWAP_CANCELLED)
            return
        from lilbee.cli.tui.app import apply_active_model

        apply_active_model(self.app, key, ref)
        role = _MODEL_KEY_TO_WORKER_ROLE.get(key)
        if role is not None:
            get_services().reload_role(role)
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
        """Reset every writable setting to its cfg default atomically."""
        if not confirmed:
            return
        writable = [(key, defn) for key, defn in SETTINGS_MAP.items() if defn.writable]
        snapshot = {key: getattr(cfg, key) for key, _ in writable}
        updates, signal_payload, skipped = self._apply_batch_defaults(writable)
        if updates and not self._persist_batch(writable, snapshot, updates):
            return
        self._refresh_batch(writable, skipped)
        self._publish_batch_signals(signal_payload)
        self._notify_batch_result(skipped)

    def _apply_batch_defaults(
        self, writable: list[tuple[str, SettingDef]]
    ) -> tuple[dict[str, str], list[tuple[str, object]], list[str]]:
        """Mutate cfg in-memory for every writable key; track updates + skips."""
        updates: dict[str, str] = {}
        signal_payload: list[tuple[str, object]] = []
        skipped: list[str] = []
        for key, _defn in writable:
            default = get_default(key)
            try:
                setattr(cfg, key, default)
            except (ValueError, TypeError) as exc:
                log.warning("Default for %s rejected by cfg (%s); skipping", key, exc)
                skipped.append(key)
                continue
            updates[key] = stringify_default(default)
            signal_payload.append((key, default))
        return updates, signal_payload, skipped

    def _persist_batch(
        self,
        writable: list[tuple[str, SettingDef]],
        snapshot: dict[str, object],
        updates: dict[str, str],
    ) -> bool:
        """Persist the batch; roll back cfg + UI on disk error. Returns True on success."""
        try:
            settings.update_values(cfg.data_root, updates)
        except OSError as exc:
            self._rollback_batch(writable, snapshot)
            self.notify(msg.SETTINGS_INVALID_VALUE.format(error=exc), severity="error")
            return False
        return True

    def _rollback_batch(
        self, writable: list[tuple[str, SettingDef]], snapshot: dict[str, object]
    ) -> None:
        """Restore cfg and editor widgets from snapshot after a failed persist."""
        for key, prev in snapshot.items():
            try:
                setattr(cfg, key, prev)
            except (ValueError, TypeError):
                log.exception("Failed to roll back cfg.%s", key)
        for key, defn in writable:
            self._refresh_editor(key, defn, snapshot[key])
            self._refresh_help(key, defn)

    def _refresh_batch(self, writable: list[tuple[str, SettingDef]], skipped: list[str]) -> None:
        """Refresh editor + help for each successfully-reset writable key."""
        for key, defn in writable:
            if key in skipped:
                continue
            default = get_default(key)
            self._refresh_editor(key, defn, default)
            self._refresh_help(key, defn)

    def _publish_batch_signals(self, signal_payload: list[tuple[str, object]]) -> None:
        """Fan out settings_changed signals for every successfully-reset key."""
        for pub_key, pub_parsed in signal_payload:
            self.app.settings_changed_signal.publish((pub_key, pub_parsed))

    def _notify_batch_result(self, skipped: list[str]) -> None:
        """Surface a single summary toast; warning severity when any key skipped."""
        if skipped:
            self.notify(
                msg.SETTINGS_RESET_ALL_PARTIAL.format(skipped=", ".join(skipped)),
                severity="warning",
            )
        else:
            self.notify(msg.SETTINGS_RESET_ALL_SUCCESS)

    def action_reset_focused(self) -> None:
        """Reset the currently-focused setting row to its cfg default."""
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
        """Restore a single setting to its cfg default."""
        defn = SETTINGS_MAP.get(key)
        if defn is None or not defn.writable:
            return
        default = get_default(key)
        stringified = stringify_default(default)
        self._persist_value(key, defn, stringified)
        self._refresh_editor(key, defn, default)

    def _refresh_editor(self, key: str, defn: SettingDef, value: object) -> None:
        """Update the editor widget to reflect a new value (e.g. after reset)."""
        try:
            widget = self.query_one(f"#{EDITOR_ID_PREFIX}{key}")
        except Exception:
            log.debug("Failed to refresh editor for %s", key, exc_info=True)
            return
        set_widget_value(widget, value)

    def action_go_back(self) -> None:
        self.app.switch_view("Chat")

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

    def action_scroll_down(self) -> None:
        if (body := self._active_pane_body()) is not None:
            body.scroll_down()

    def action_scroll_up(self) -> None:
        if (body := self._active_pane_body()) is not None:
            body.scroll_up()

    def action_scroll_home(self) -> None:
        if (body := self._active_pane_body()) is not None:
            body.scroll_home()

    def action_scroll_end(self) -> None:
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
        reach the next group. Mirrors CatalogScreen.action_cycle_tab.
        """
        try:
            tabs = self.query_one("#settings-tabs", TabbedContent)
        except Exception:
            return
        pane_ids = list(self._pane_groups)
        if not pane_ids:
            return
        try:
            current = pane_ids.index(tabs.active)
        except ValueError:
            current = 0
        next_id = pane_ids[(current + delta) % len(pane_ids)]
        if tabs.active != next_id:
            tabs.active = next_id

    def _move_focus_within_pane(self, *, direction: int) -> None:
        focused = self.app.focused
        tabs = self.query_one("#settings-tabs", TabbedContent)
        active_pane_id = tabs.active
        try:
            body = self.query_one(f"#{active_pane_id}-body", _LazyGroupBody)
        except Exception:
            self.app.action_focus_next() if direction == 1 else self.app.action_focus_previous()
            return
        focusables = [w for w in body.query("*") if w.focusable]
        if not focusables or focused is None or focused not in focusables:
            self.app.action_focus_next() if direction == 1 else self.app.action_focus_previous()
            return
        index = focusables.index(focused)
        next_index = index + direction
        if 0 <= next_index < len(focusables):
            focusables[next_index].focus()
            return
        # At the boundary: advance to the next/previous pane.
        pane_ids = list(self._pane_groups.keys())
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
